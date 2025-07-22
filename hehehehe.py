import os
import torch
from PIL import Image
import numpy as np
import lpips
import sys
import imageio

# StyleGAN3 dependencies
sys.path.append('D:/Amjad+Memo+Riaz/Stylegan3_pytorch_complete/notebook-sg3/stylegan3')
import dnnlib
import legacy

# Load model
network_pkl = "D:\\Amjad+Memoona+Riaz\\Stylegan3\\shoe-generator\\stylegan3\\experiments\\00013-stylegan3-t-shoedataset-128x128-gpus1-batch32-gamma0.5\\network-snapshot-007132.pkl"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

# Print model info
print(f"Model z_dim: {G.z_dim}, c_dim: {G.c_dim}, resolution: {G.img_resolution}, num_ws: {G.mapping.num_ws}")

# LPIPS model
lpips_fn = lpips.LPIPS(net='vgg').to(device)

# Image preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((G.img_resolution, G.img_resolution))
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.tensor(img.transpose([2, 0, 1]), device=device)
    img = img.unsqueeze(0) * 2 - 1
    return img

# Generate image from W+ latent
def generate_image_from_w(w, noise_mode='const'):
    with torch.no_grad():
        img = G.synthesis(w, noise_mode=noise_mode)
        img = (img + 1) * 127.5
        img = img.clamp(0, 255).to(torch.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()
    return img

# Project image into W+ latent space
def project_to_w_plus(img_tensor, num_steps=1000):
    z = torch.randn(1, G.z_dim, device=device)
    w = G.mapping(z, None)  # [1, num_ws, z_dim]
    w_opt = w.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([w_opt], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    best_w = None
    best_loss = float('inf')

    for step in range(num_steps):
        synth_img = G.synthesis(w_opt, noise_mode='const')
        mse_loss = torch.nn.functional.mse_loss(synth_img, img_tensor)
        perceptual_loss = lpips_fn(synth_img, img_tensor).mean()
        loss = mse_loss + perceptual_loss

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_w = w_opt.detach().clone()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"Step {step}/{num_steps}: MSE={mse_loss.item():.4f}, LPIPS={perceptual_loss.item():.4f}, Total={loss.item():.4f}")

    return best_w if best_w is not None else w_opt.detach()

# Create variation in W+ space
def create_variation_w(w_projected, strength=0.3):
    with torch.no_grad():
        noise = torch.randn_like(w_projected) * strength
        return w_projected + noise

# Main pipeline
def generate_similar_shoe_w_plus(image_path):
    img_tensor = preprocess_image(image_path)

    print("Projecting image into W+ latent space...")
    w_projected = project_to_w_plus(img_tensor)

    recon_img = generate_image_from_w(w_projected)
    recon_path = "D:\\Amjad+Memoona+Riaz\\Stylegan3\\New folder\\reconstruction_wplus2.jpg"
    imageio.imwrite(recon_path, recon_img)
    print(f"Reconstructed image saved at: {recon_path}")

    for i, strength in enumerate([0.1, 0.2, 0.3, 0.5]):
        w_variation = create_variation_w(w_projected, strength)
        variation_img = generate_image_from_w(w_variation)
        out_path = f"D:\\Amjad+Memoona+Riaz\\Stylegan3\\New folder\\variation_wplus2_{i+1}_strength_{strength}.jpg"
        imageio.imwrite(out_path, variation_img)
        print(f"Generated variation {i+1} saved at: {out_path}")

# Run
shoe_image_path = "test_IMAGES\\test13.jpg"
generate_similar_shoe_w_plus(shoe_image_path)
