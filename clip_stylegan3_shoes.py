import os
import sys
import time
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
sys.path.append('./CLIP')
import clip
import unicodedata
import re
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from einops import rearrange
import argparse

# Add CLIP to path


class StyleGAN3CLIPGenerator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Load StyleGAN3 model
        print("Loading StyleGAN3 model...")
        with open(model_path, 'rb') as fp:
            self.G = pickle.load(fp)['G_ema'].to(self.device)
        
        # Calculate w_stds for latent space normalization
        print("Calculating latent space statistics...")
        zs = torch.randn([10000, self.G.mapping.z_dim], device=self.device)
        self.w_stds = self.G.mapping(zs, None).std(0)
        
        # Load CLIP model
        print("Loading CLIP model...")
        self.clip_model = CLIP(self.device)
        
        # Initialize cutouts
        self.make_cutouts = MakeCutouts(224, 32, 0.5)
        
        # Transform for image processing
        self.tf = Compose([
            Resize(224),
            lambda x: torch.clamp((x+1)/2, min=0, max=1),
        ])
        
        print("Models loaded successfully!")

    def generate_images(self, text_prompt, steps=500, seed=-1, num_candidates=32, 
                       save_dir="outputs", save_intermediate=False, quality_threshold=0.6):
        """
        Generate images from text prompt with quality filtering
        
        Args:
            text_prompt: Text description for image generation
            steps: Number of optimization steps
            seed: Random seed (-1 for random)
            num_candidates: Number of initial candidates to evaluate
            save_dir: Directory to save images
            save_intermediate: Whether to save intermediate images
            quality_threshold: Threshold for filtering high-quality images (higher = more strict)
        """
        
        if seed == -1:
            seed = np.random.randint(0, 9e9)
        print(f"Using seed: {seed}")
        
        # Parse multiple prompts
        texts = [phrase.strip() for phrase in text_prompt.split("|") if phrase]
        targets = [self.clip_model.embed_text(text) for text in texts]
        print(f"Prompts: {texts}")
        
        # Create output directory
        timestring = time.strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(save_dir, f"{slugify(text_prompt)}_{timestring}")
        os.makedirs(output_dir, exist_ok=True)
        
        torch.manual_seed(seed)
        
        # Initialize with best candidate from multiple samples
        print("Finding best initial candidate...")
        best_q = self._find_best_initial_candidate(targets, num_candidates)
        
        # Optimization loop
        print("Starting optimization...")
        best_images, best_losses, final_q_ema, final_loss = self._optimize_latent(
            best_q, targets, steps, output_dir, save_intermediate, quality_threshold
        )
        
        # Save best images and also save the final optimized result
        self._save_best_images(best_images, best_losses, output_dir, text_prompt)
        
        # Always save the final optimized result regardless of quality threshold
        with torch.no_grad():
            final_w = final_q_ema * self.w_stds + self.G.mapping.w_avg
            final_image = self.G.synthesis(final_w, noise_mode='const')
            pil_image = TF.to_pil_image(final_image[0].add(1).div(2).clamp(0, 1))
            pil_image.save(os.path.join(output_dir, "final_optimized.jpg"))
            print(f"Final optimized image saved with loss: {final_loss:.4f}")
        
        print(f"Generation complete! Best images saved in: {output_dir}")
        return output_dir

    def _find_best_initial_candidate(self, targets, num_candidates):
        """Find the best initial latent code from multiple candidates"""
        with torch.no_grad():
            best_loss = float('inf')
            best_q = None
            
            # Generate candidates in batches to manage memory
            batch_size = 4
            num_batches = num_candidates // batch_size
            
            for batch in tqdm(range(num_batches), desc="Evaluating candidates"):
                q_batch = (self.G.mapping(torch.randn([batch_size, self.G.mapping.z_dim], 
                                                    device=self.device), None, truncation_psi=0.7) 
                          - self.G.mapping.w_avg) / self.w_stds
                
                images = self.G.synthesis(q_batch * self.w_stds + self.G.mapping.w_avg)
                embeds = self._embed_image(images.add(1).div(2))
                losses = prompts_dist_loss(embeds, targets, spherical_dist_loss).mean(0)
                
                # Find best in this batch
                min_idx = torch.argmin(losses)
                if losses[min_idx] < best_loss:
                    best_loss = losses[min_idx]
                    best_q = q_batch[min_idx].clone()
            
            return best_q.unsqueeze(0).requires_grad_()

    def _optimize_latent(self, q, targets, steps, output_dir, save_intermediate, quality_threshold):
        """Optimize latent code with quality tracking"""
        q_ema = q.clone()
        opt = torch.optim.AdamW([q], lr=0.03, betas=(0.0, 0.999))
        
        best_images = []
        best_losses = []
        final_loss = 0
        
        loop = tqdm(range(steps))
        for i in loop:
            opt.zero_grad()
            w = q * self.w_stds
            image = self.G.synthesis(w + self.G.mapping.w_avg, noise_mode='const')
            embed = self._embed_image(image.add(1).div(2))
            loss = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
            loss.backward()
            opt.step()
            
            final_loss = loss.item()  # Store the current loss
            loop.set_postfix(loss=loss.item(), q_magnitude=q.std().item())
            
            # Update EMA
            q_ema = q_ema * 0.9 + q * 0.1
            
            # Generate image with EMA weights
            with torch.no_grad():
                ema_image = self.G.synthesis(q_ema * self.w_stds + self.G.mapping.w_avg, 
                                           noise_mode='const')
                
                # Check if this is a high-quality image (lower loss = better quality)
                if loss.item() < quality_threshold:
                    best_images.append(ema_image[0].clone())
                    best_losses.append(loss.item())
            
            # Save intermediate images if requested
            if save_intermediate and i % 50 == 0:
                pil_image = TF.to_pil_image(ema_image[0].add(1).div(2).clamp(0, 1))
                pil_image.save(os.path.join(output_dir, f"step_{i:04d}.jpg"))
        
        return best_images, best_losses, q_ema, final_loss

    def _save_best_images(self, best_images, best_losses, output_dir, text_prompt):
        """Save the best generated images"""
        if not best_images:
            print("No high-quality images found, saving final result...")
            # Generate final image using proper latent mapping
            with torch.no_grad():
                z = torch.randn([1, self.G.mapping.z_dim], device=self.device)
                w = self.G.mapping(z, None)
                final_image = self.G.synthesis(w, noise_mode='const')
                pil_image = TF.to_pil_image(final_image[0].add(1).div(2).clamp(0, 1))
                pil_image.save(os.path.join(output_dir, "final_result.jpg"))
            return
        
        # Sort by loss (best first)
        sorted_indices = np.argsort(best_losses)
        
        # Save top 5 best images
        num_to_save = min(5, len(best_images))
        for i, idx in enumerate(sorted_indices[:num_to_save]):
            image_tensor = best_images[idx]
            pil_image = TF.to_pil_image(image_tensor.add(1).div(2).clamp(0, 1))
            
            filename = f"best_{i+1}_loss_{best_losses[idx]:.4f}.jpg"
            pil_image.save(os.path.join(output_dir, filename))
        
        print(f"Saved {num_to_save} best images with losses: {[best_losses[i] for i in sorted_indices[:num_to_save]]}")

    def _embed_image(self, image):
        """Embed image using CLIP"""
        n = image.shape[0]
        cutouts = self.make_cutouts(image)
        embeds = self.clip_model.embed_cutout(cutouts)
        embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
        return embeds


# Helper classes and functions
def slugify(value, allow_unicode=False):
    """Convert to filename-safe string"""
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def norm1(prompt):
    """Normalize to the unit sphere"""
    return prompt / prompt.square().sum(dim=-1, keepdim=True).sqrt()

def spherical_dist_loss(x, y):
    """Spherical distance loss"""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def prompts_dist_loss(x, targets, loss):
    """Distance loss for multiple prompts"""
    if len(targets) == 1:
        return loss(x, targets[0])
    distances = [loss(x, target) for target in targets]
    return torch.stack(distances, dim=-1).sum(dim=-1)

class MakeCutouts(torch.nn.Module):
    """Generate random cutouts from images"""
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)

class CLIP:
    """CLIP model wrapper"""
    def __init__(self, device):
        self.device = device
        clip_model = "ViT-B/32"
        self.model, _ = clip.load(clip_model, device=device)
        self.model = self.model.requires_grad_(False)
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

    @torch.no_grad()
    def embed_text(self, prompt):
        """Normalized CLIP text embedding"""
        return norm1(self.model.encode_text(clip.tokenize(prompt).to(self.device)).float())

    def embed_cutout(self, image):
        """Normalized CLIP image embedding"""
        return norm1(self.model.encode_image(self.normalize(image)))


def main():
    parser = argparse.ArgumentParser(description='StyleGAN3 + CLIP Text-to-Image Generation')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to StyleGAN3 model pickle file')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt for image generation')
    parser.add_argument('--steps', type=int, default=500,
                       help='Number of optimization steps')
    parser.add_argument('--seed', type=int, default=-1,
                       help='Random seed (-1 for random)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for generated images')
    parser.add_argument('--save_intermediate', action='store_true',
                       help='Save intermediate images during optimization')
    parser.add_argument('--quality_threshold', type=float, default=0.6,
                       help='Quality threshold for saving best images (lower = stricter)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = StyleGAN3CLIPGenerator(args.model_path, args.device)
    
    # Generate images
    output_path = generator.generate_images(
        text_prompt=args.prompt,
        steps=args.steps,
        seed=args.seed,
        save_dir=args.output_dir,
        save_intermediate=args.save_intermediate,
        quality_threshold=args.quality_threshold
    )
    
    print(f"Images saved to: {output_path}")


if __name__ == "__main__":
    # Example usage when running directly
    if len(sys.argv) == 1:
        # Default example run
        model_path = "D:\\Amjad+Memoona+Riaz\\Stylegan3\\shoe-generator\\stylegan3\\experiments\\00013-stylegan3-t-shoedataset-128x128-gpus1-batch32-gamma0.5\\network-snapshot-007132.pkl"  # Update this path
        prompt = "blue sneaker with white sole"
        
        generator = StyleGAN3CLIPGenerator(model_path)
        generator.generate_images(prompt, steps=300, seed=42)
    else:
        main()