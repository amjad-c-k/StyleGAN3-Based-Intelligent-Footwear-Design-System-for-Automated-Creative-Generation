import os
import torch
import numpy as np
import PIL.Image
from torchvision import transforms
import dnnlib
import legacy
from models.encoders import psp_encoders  # Importing the e4e encoder
import sys
import os



# Load the pre-trained StyleGAN model
def load_stylegan(network_pkl):
    print(f'Loading network from "{network_pkl}"...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # Load generator
    return G, device

# Load the pre-trained e4e encoder for latent inversion
def load_encoder(encoder_path, device):
    print(f'Loading encoder from "{encoder_path}"...')
    encoder = psp_encoders.Encoder4Editing(50).to(device)  # e4e encoder
    encoder.load_state_dict(torch.load(encoder_path, map_location=device)['state_dict'])
    encoder.eval()
    return encoder

# Convert input shoe image to latent vector using e4e
def encode_image(image_path, encoder, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    img = PIL.Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        latent_code = encoder(img_tensor)
    
    return latent_code

# Generate a shoe variation from the encoded latent vector
def generate_variation(G, latent_code, truncation_psi, noise_mode, outdir, variation_strength=0.2):
    os.makedirs(outdir, exist_ok=True)
    device = latent_code.device
    
    # Add small variations to the latent vector
    variation = torch.randn_like(latent_code) * variation_strength
    modified_latent = latent_code + variation

    with torch.no_grad():
        img = G(modified_latent, None, truncation_psi=truncation_psi, noise_mode=noise_mode)
    
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_path = os.path.join(outdir, 'face_variation.png')
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(img_path)
    
    print(f"Generated face design saved at: {img_path}")

# Main function
def main(network_pkl, encoder_path, input_image, outdir, truncation_psi=0.7, noise_mode='const'):
    G, device = load_stylegan(network_pkl)
    encoder = load_encoder(encoder_path, device)
    
    latent_code = encode_image(input_image, encoder, device)
    
    generate_variation(G, latent_code, truncation_psi, noise_mode, outdir)

# Example usage
if __name__ == "__main__":
    main(
        network_pkl="D:\\Amjad+Memoona+Riaz\\Stylegan3\\stylegan3-r-ffhq-1024x1024.pkl",  # Your pre-trained shoe StyleGAN model
        encoder_path="D:\\Amjad+Memoona+Riaz\\Stylegan3\\e4e_ffhq_encode.pt",  # Pre-trained e4e encoder
        input_image="stylegan3\\TOM.jpg",  # Path to the input shoe image
        outdir="D:\\Amjad+Memoona+Riaz\\Stylegan3\\New folder"  # Directory to save generated shoe designs
    )
