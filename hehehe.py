import os
import torch
from PIL import Image
import numpy as np
import lpips  # Perceptual loss library
import sys
import dnnlib
import legacy
import imageio

sys.path.append('D:/Amjad+Memo+Riaz/Stylegan3_pytorch_complete/notebook-sg3/stylegan3')

# Path to the trained StyleGAN3 model
# network_pkl = "D:\\Amjad+Memoona+Riaz\\Stylegan3\\stylegan3-r-ffhq-1024x1024.pkl"
network_pkl = "D:\\Amjad+Memoona+Riaz\\Stylegan3\\stylegan3\\experiments\\00005-stylegan3-t-shoedataset-128x128-gpus1-batch32-gamma0.5\\network-snapshot-002936.pkl"
print("step1")
# Load StyleGAN3 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("step2")
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)
    print("step3")
# Load and preprocess the shoe image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((G.img_resolution, G.img_resolution))
    img = np.array(img).astype(np.float32) / 255.0
    print("step4")
    img = torch.tensor(img.transpose([2, 0, 1]), device=device)
    print("step5")
    img = img.unsqueeze(0) * 2 - 1  # Normalize to [-1, 1]
    return img
print("step6")
# Generate an image from the latent vector
def generate_image(latent_vector, truncation_psi=0.7):
    img = G.synthesis(latent_vector, noise_mode='const')
    img = (img + 1) * 127.5  # Scale back to [0, 255]
    img = img.clamp(0, 255).to(torch.uint8)
    img = img[0].permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C] format
    print("step7")
    return img

# Project the input shoe image into the latent space
def project_image(img_tensor, num_steps=500, diversity_factor=0.3):
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)  # Perceptual loss model
    print("step8")
    latent_in = torch.randn([1, G.z_dim], device=device).unsqueeze(1).repeat(1, G.mapping.num_ws, 1)
    latent_in = latent_in.detach().requires_grad_(True)  # Detach and set requires_grad
    print("step9")
    optimizer = torch.optim.Adam([latent_in], lr=0.1)
    
    for step in range(num_steps):
        generated_img = G.synthesis(latent_in, noise_mode='const')
        
        # Compute perceptual loss + MSE loss
        perceptual_loss = loss_fn_vgg(generated_img, img_tensor).mean()
        mse_loss = torch.nn.functional.mse_loss(generated_img, img_tensor)
        reg_loss = 0.01 * torch.norm(latent_in)  # Regularization to avoid overfitting
        
        loss = mse_loss + perceptual_loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Inject some uniqueness
    random_latent = torch.randn_like(latent_in) * diversity_factor
    final_latent = latent_in + random_latent  # Blend the optimized latent with a random one
    
    return final_latent.detach()

# Main function to load image, project, and generate similar but unique image
def generate_similar_shoe(image_path):
    img_tensor = preprocess_image(image_path)
    latent_vector = project_image(img_tensor, num_steps=500, diversity_factor=0.3)
    similar_img = generate_image(latent_vector) 
    
    # Save the generated image
    output_path = "D:\\Amjad+Memoona+Riaz\\Stylegan3\\New folder\\2.2.jpg"
    imageio.imwrite(output_path, similar_img)
    print(f"Generated unique shoe saved at: {output_path}")

# Provide the path to the shoe image
shoe_image_path = "2.jpg"
generate_similar_shoe(shoe_image_path)
