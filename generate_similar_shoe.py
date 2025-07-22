import os
import torch
from PIL import Image
import numpy as np
import lpips
import sys
sys.path.append('D:/Amjad+Memo+Riaz/Stylegan3_pytorch_complete/notebook-sg3/stylegan3')

import dnnlib
import legacy
import imageio

# Path to the trained StyleGAN3 model
network_pkl = "D:\\Amjad+Memoona+Riaz\\Stylegan3\\shoe-generator\\stylegan3\\experiments\\00013-stylegan3-t-shoedataset-128x128-gpus1-batch32-gamma0.5\\network-snapshot-007132.pkl"

# Load pre-trained StyleGAN3 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model
print(f"Loading network from {network_pkl}...")
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # Load generator

# Print model information to help debug
print(f"Model z_dim: {G.z_dim}")
print(f"Model c_dim: {G.c_dim}")
print(f"Model img_resolution: {G.img_resolution}")
if hasattr(G.mapping, 'num_ws'):
    print(f"Model num_ws: {G.mapping.num_ws}")
else:
    print("Model doesn't have a num_ws attribute in mapping")

# Initialize LPIPS model for perceptual loss
lpips_fn = lpips.LPIPS(net='vgg').to(device)

# Load and preprocess the shoe image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((G.img_resolution, G.img_resolution))
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.tensor(img.transpose([2, 0, 1]), device=device)
    img = img.unsqueeze(0) * 2 - 1  # Normalize to [-1, 1]
    return img

# Generate an image from the z latent vector (using Z space instead of W space)
def generate_image_from_z(z, truncation_psi=0.7, noise_mode='const'):
    with torch.no_grad():
        img = G(z, None, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img + 1) * 127.5  # Scale back to [0, 255]
        img = img.clamp(0, 255).to(torch.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C] format
    return img

# Project the input shoe image into the latent space (Z space approach)
def project_image(img_tensor, num_steps=1000):
    # Start with random Z latent
    z_opt = torch.randn(1, G.z_dim, device=device, requires_grad=True)
    
    # Hyperparameters for optimization
    optimizer = torch.optim.Adam([z_opt], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    best_z = None
    best_loss = float('inf')
    
    # Run optimization
    for step in range(num_steps):
        # Generate image
        synth_image = G(z_opt, None, truncation_psi=0.7, noise_mode='const')
        
        # Compute losses
        mse_loss = torch.nn.functional.mse_loss(synth_image, img_tensor)
        perceptual_loss = lpips_fn(synth_image, img_tensor).mean()
        
        # Combined loss
        loss = mse_loss + perceptual_loss
        
        # Track best result
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_z = z_opt.detach().clone()
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Print progress
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}: MSE={mse_loss.item():.4f}, LPIPS={perceptual_loss.item():.4f}, Total={loss.item():.4f}")
    
    # Return the best result found
    return best_z if best_z is not None else z_opt.detach()

# Generate a variation of the projected image
def create_variation(z_projected, strength=0.3):
    # Create variation by adding noise to the projected z
    with torch.no_grad():
        z_noise = torch.randn_like(z_projected) * strength
        z_variation = z_projected + z_noise
    return z_variation

# Main function to load image, project, and generate similar but unique image
def generate_similar_shoe(image_path, variation_strength=0.3):
    img_tensor = preprocess_image(image_path)
    
    # Project the image into latent space
    print("Projecting image into latent space...")
    z_projected = project_image(img_tensor)
    
    # Save the reconstructed image (should be close to original)
    recon_img = generate_image_from_z(z_projected)
    recon_path = "D:\\Amjad+Memoona+Riaz\\Stylegan3\\New folder\\test3reconstruction.jpg"
    imageio.imwrite(recon_path, recon_img)
    print(f"Reconstructed image saved at: {recon_path}")
    
    # Create variations with different strengths
    for i, strength in enumerate([0.1, 0.2, 0.3, 0.5]):
        # Create variation
        z_variation = create_variation(z_projected, strength=strength)
        
        # Generate and save the variation
        variation_img = generate_image_from_z(z_variation)
        output_path = f"D:\\Amjad+Memoona+Riaz\\Stylegan3\\New folder\\variation_{i+1}_strength_{strength}.jpg"
        imageio.imwrite(output_path, variation_img)
        print(f"Generated variation {i+1} (strength {strength}) saved at: {output_path}")
# def generate_similar_shoe(image_path, variation_strength=0.3):
#     img_tensor = preprocess_image(image_path)
   
#     # Project the image into latent space
#     print("Projecting image into latent space...")
#     z_projected = project_image(img_tensor)
   
#     # Save the reconstructed image (should be close to original)
#     recon_img = generate_image_from_z(z_projected)
#     recon_path = "D:\\Amjad+Memoona+Riaz\\Stylegan3\\New folder\\test2reconstruction.jpg"
#     imageio.imwrite(recon_path, recon_img)
#     print(f"Reconstructed image saved at: {recon_path}")
#     import random
#     # Create 200 variations with different strengths
#     for i in range(100000):
#         # Vary the strength for more diversity (you can adjust this range)
#         # This creates variations between 0.1 and 0.8 strength
#         # base_strenght = 0.1 + (i / 199) * 0.7 # Linear progression from 0.1 to 0.8
#         strength =   random.uniform(1.0,500000.00000)
       
#         # Alternative: Random strength for each variation
#         # import random
#         # strength = random.uniform(0.1, 0.8)
       
#         # Create variation
#         z_variation = create_variation(z_projected, strength=strength)
       
#         # Generate and save the variation
#         variation_img = generate_image_from_z(z_variation)
#         output_path = f"D:\\Amjad+Memoona+Riaz\\Stylegan3\\New folder\\200_part2\\variation_{i+1:03d}_strength_{strength:.3f}.jpg"
#         imageio.imwrite(output_path, variation_img)
#         print(f"Generated variation {i+1}/200 (strength {strength:.3f}) saved at: {output_path}")
       
#         # Optional: Progress indicator for every 10 images
#         if (i + 1) % 10 == 0:
#             print(f"Progress: {i+1}/200 variations completed")
            
# Provide the path to the shoe image
shoe_image_path = "test_IMAGES\\test9.jpg"
# shoe_image_path = "test 5.jpg"
generate_similar_shoe(shoe_image_path)
