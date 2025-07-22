
# import torch
# import dnnlib
# import legacy

# with open('D:\\Amjad+Memoona+Riaz\\Stylegan3\\Dataset and SG3 final\\models\\network-snapshot-007132.pkl', 'rb') as f:
#     G = legacy.load_network_pkl(f)['G_ema']
#     print(G.synthesis)  # or G.init_kwargs to inspect the architecture config



# import torch
# import pickle
# import argparse

# def convert_pkl_to_pt(pkl_path, output_path):
#     print(f"Loading: {pkl_path}")
#     with open(pkl_path, 'rb') as f:
#         data = pickle.load(f)
    
#     # Handle different formats
#     if 'G_ema' in data:
#         G = data['G_ema']  # For standard StyleGAN3 training
#     elif hasattr(data, 'G_ema'):
#         G = data.G_ema     # For training results as SimpleNamespace
#     else:
#         raise ValueError("Could not find G_ema in the .pkl file")

#     print("Saving as .pt...")
#     torch.save(G.state_dict(), output_path)
#     print(f"Model saved to {output_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pkl', required=True, help="D:\\Amjad+Memoona+Riaz\\Stylegan3\\Dataset and SG3 final\\models\\network-snapshot-007132.pkl")
#     parser.add_argument('--pt', required=True, help="D:\\Amjad+Memoona+Riaz\\Stylegan3\\Dataset and SG3 final\\models\\updated.pt")
#     args = parser.parse_args()

#     convert_pkl_to_pt(args.pkl, args.pt)


# import torch
# import pickle

# # Load pkl file
# with open("D:\\Amjad+Memoona+Riaz\\Stylegan3\\Dataset and SG3 final\\models\\network-snapshot-007132.pkl", "rb") as f:
#     pkl_data = pickle.load(f)

# # Load pt file
# pt_data = torch.load("D:\\Amjad+Memoona+Riaz\\Stylegan3\\Dataset and SG3 final\\models\\shoe_stylegan3.pt", map_location="cpu")

# # Print top-level keys or attributes
# print("PKL keys:", list(pkl_data.keys()) if isinstance(pkl_data, dict) else dir(pkl_data))
# print("PT keys:", list(pt_data.keys()) if isinstance(pt_data, dict) else dir(pt_data))




# # Extract generator state_dicts
# pkl_G = pkl_data['G_ema'].state_dict()
# pt_G = pt_data['G'].state_dict() if 'G' in pt_data else pt_data['G_ema'].state_dict()

# # Compare keys
# print("Missing in PT:", set(pkl_G.keys()) - set(pt_G.keys()))
# print("Extra in PT:", set(pt_G.keys()) - set(pkl_G.keys()))

# # Optionally compare specific tensors
# for key in pkl_G:
#     if key in pt_G:
#         if not torch.allclose(pkl_G[key], pt_G[key], atol=1e-5):
#             print(f"Weight mismatch in: {key}")


# pkl_G = pkl_data['G_ema'].state_dict()
# pt_G = pt_data  # Already a state_dict

# # Compare keys
# print("Missing keys in PT:", set(pkl_G.keys()) - set(pt_G.keys()))
# print("Extra keys in PT:", set(pt_G.keys()) - set(pkl_G.keys()))

# # Compare values
# for k in pkl_G:
#     if k in pt_G and not torch.allclose(pkl_G[k], pt_G[k], atol=1e-5):
#         print(f"Mismatch in weight: {k}")


import torch
import pickle
import os

# === CONFIGURATION ===
pkl_path = 'D:\\Amjad+Memoona+Riaz\\Stylegan3\\Dataset and SG3 final\\models\\network-snapshot-007132.pkl'  # <-- Update this path
pt_path = 'D:\\Amjad+Memoona+Riaz\\Stylegan3\\Dataset and SG3 final\\models\\NEW_UPDATED_SHOE.pt'  # <-- Output path

# === LOAD .PKL CHECKPOINT ===
print(f"🔍 Loading: {pkl_path}")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# === VERIFY KEY COMPONENTS ===
required_keys = ['G', 'G_ema']
missing = [key for key in required_keys if key not in data]
if missing:
    raise KeyError(f"❌ Missing keys in .pkl: {missing}")

print("✅ Found components:")
for key in ['G', 'G_ema', 'D', 'augment_pipe', 'training_set_kwargs']:
    print(f"  {key}: {'Found' if key in data else 'Not found'}")

# === CONVERT TO .pt FORMAT ===
pt_dict = {
    'G': data['G'].state_dict(),
    'G_ema': data['G_ema'].state_dict(),
}

# Optionally include discriminator and metadata if available
if 'D' in data and data['D'] is not None:
    pt_dict['D'] = data['D'].state_dict()
if 'augment_pipe' in data:
    pt_dict['augment_pipe'] = data['augment_pipe']
if 'training_set_kwargs' in data:
    pt_dict['training_set_kwargs'] = data['training_set_kwargs']

# === SAVE TO .PT FILE ===
os.makedirs(os.path.dirname(pt_path), exist_ok=True)
torch.save(pt_dict, pt_path)
print(f"💾 Model saved successfully to: {pt_path}")

