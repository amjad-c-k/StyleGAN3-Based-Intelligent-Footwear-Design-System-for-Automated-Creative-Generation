# #------------------------cloudinary app implementation with text-to-image------------------------------------
import random
import json
import os
import uuid
import sys
import torch
from PIL import Image
import numpy as np
import tempfile
import shutil
import io
import lpips
from zipfile import ZipFile
import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory, send_file
from flask_mail import Mail, Message
import pyrebase

# Import Cloudinary and Firebase Admin
import cloudinary
import cloudinary.uploader
import cloudinary.api
import firebase_admin
from firebase_admin import credentials, firestore, auth

# Add the path to the StyleGAN3 directory
sys.path.append('D:/Amjad+Memo+Riaz/Stylegan3_pytorch_complete/notebook-sg3/stylegan3')
print("Hello3")
import dnnlib
import legacy
import imageio

# Text-to-image imports
import pickle
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
import time
print("Hello1")
# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"
print("Hello")
# Global flag for Firestore availability
FIRESTORE_AVAILABLE = False

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
TEXT_TO_IMAGE_OUTPUT_FOLDER = 'text_to_image_outputs'
FAVORITES_FOLDER = 'favorites'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEXT_TO_IMAGE_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FAVORITES_FOLDER, exist_ok=True)

# Path to the trained StyleGAN3 model
NETWORK_PKL = "D:\\Amjad+Memoona+Riaz\\Stylegan3\\shoe-generator\\stylegan3\\experiments\\00013-stylegan3-t-shoedataset-128x128-gpus1-batch32-gamma0.5\\network-snapshot-007132.pkl"

# Global variables for text-to-image
G = None
w_stds = None
clip_model = None
make_cutouts = None
tf = None
device = None

# Load cloudinary and firebase admin configurations
with open('cloud_config.json', 'r') as f:
    cloud_config = json.load(f)

# Initialize Cloudinary
cloudinary.config( 
    cloud_name = cloud_config['cloudinary']['cloud_name'], 
    api_key = cloud_config['cloudinary']['api_key'], 
    api_secret = cloud_config['cloudinary']['api_secret'],
    secure = True
)

# Initialize Firebase Admin SDK and check Firestore availability
try:
    # Check if any Firebase Admin app is already initialized
    default_app = firebase_admin.get_app()
except ValueError:
    # If no app exists, initialize with credentials
    cred = credentials.Certificate(cloud_config['firebase_admin'])
    firebase_admin.initialize_app(cred)

# Initialize Firestore client and check availability
try:
    db = firestore.client()
    # Test Firestore connection
    test_ref = db.collection('test').document('test')
    test_ref.set({'timestamp': datetime.datetime.now()})
    FIRESTORE_AVAILABLE = True
    print("✅ Firestore connection successful and API is enabled!")
except Exception as e:
    FIRESTORE_AVAILABLE = False
    print(f"⚠️ Firestore unavailable or disabled: {e}")
    print("Using local storage fallback for favorites")

# Firebase Configuration for Authentication (Pyrebase)
with open('config.json', 'r') as f:
    params = json.load(f)['params']

firebaseConfig = {
    'apiKey': "AIzaSyCxCo10M1wSmHuqkTKDmPt-0wDgHX1IDtw",
    'authDomain': "generativeaishoedesigner-94f9f.firebaseapp.com",
    'databaseURL': "https://generativeaishoedesigner-94f9f-default-rtdb.firebaseio.com",
    'projectId': "generativeaishoedesigner-94f9f",
    'storageBucket': "generativeaishoedesigner-94f9f.firebasestorage.app",
    'messagingSenderId': "601866275210",
    'appId': "1:601866275210:web:ce1f35f7423c58f6a4e073",
    'measurementId': "G-HCQ0EQF9QC"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth_pyrebase = firebase.auth()

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = params['gmail-user']
app.config['MAIL_PASSWORD'] = params['gmail-password']
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)

# Store OTP for verification
otp_store = {}

# Text-to-image helper functions
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
            # Fix for int32 bounds error
            offsetx = torch.randint(0, max(1, sideX - size + 1), (), dtype=torch.int32)
            offsety = torch.randint(0, max(1, sideY - size + 1), (), dtype=torch.int32)
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)

class CLIPModel:
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

def initialize_text_to_image_models():
    """Initialize models for text-to-image generation"""
    global G, w_stds, clip_model, make_cutouts, tf, device
    
    if G is not None:
        return  # Already initialized
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Initializing text-to-image models on device: {device}')
    
    # Load StyleGAN3 model
    print("Loading StyleGAN3 model...")
    with open(NETWORK_PKL, 'rb') as fp:
        G = pickle.load(fp)['G_ema'].to(device)
    
    # Calculate w_stds for latent space normalization
    print("Calculating latent space statistics...")
    zs = torch.randn([10000, G.mapping.z_dim], device=device)
    w_stds = G.mapping(zs, None).std(0)
    
    # Load CLIP model
    print("Loading CLIP model...")
    clip_model = CLIPModel(device)
    
    # Initialize cutouts
    make_cutouts = MakeCutouts(224, 32, 0.5)
    
    # Transform for image processing
    tf = Compose([
        Resize(224),
        lambda x: torch.clamp((x+1)/2, min=0, max=1),
    ])
    
    print("Text-to-image models loaded successfully!")

def embed_image(image):
    """Embed image using CLIP"""
    n = image.shape[0]
    cutouts = make_cutouts(image)
    embeds = clip_model.embed_cutout(cutouts)
    embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
    return embeds

def find_best_initial_candidate(targets, num_candidates=32):
    """Find the best initial latent code from multiple candidates"""
    with torch.no_grad():
        best_loss = float('inf')
        best_q = None
        
        # Generate candidates in batches to manage memory
        batch_size = 4
        num_batches = num_candidates // batch_size
        
        for batch in range(num_batches):
            q_batch = (G.mapping(torch.randn([batch_size, G.mapping.z_dim], 
                                           device=device), None, truncation_psi=0.7) 
                      - G.mapping.w_avg) / w_stds
            
            images = G.synthesis(q_batch * w_stds + G.mapping.w_avg)
            embeds = embed_image(images.add(1).div(2))
            losses = prompts_dist_loss(embeds, targets, spherical_dist_loss).mean(0)
            
            # Find best in this batch
            min_idx = torch.argmin(losses)
            if losses[min_idx] < best_loss:
                best_loss = losses[min_idx]
                best_q = q_batch[min_idx].clone()
        
        return best_q.unsqueeze(0).requires_grad_()

def optimize_latent(q, targets, steps, quality_threshold=0.6):
    """Optimize latent code with quality tracking"""
    q_ema = q.clone()
    opt = torch.optim.AdamW([q], lr=0.03, betas=(0.0, 0.999))
    
    best_images = []
    best_losses = []
    final_loss = 0
    
    for i in range(steps):
        opt.zero_grad()
        w = q * w_stds
        image = G.synthesis(w + G.mapping.w_avg, noise_mode='const')
        embed = embed_image(image.add(1).div(2))
        loss = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
        loss.backward()
        opt.step()
        
        final_loss = loss.item()
        
        # Update EMA
        q_ema = q_ema * 0.9 + q * 0.1
        
        # Generate image with EMA weights
        with torch.no_grad():
            ema_image = G.synthesis(q_ema * w_stds + G.mapping.w_avg, 
                                   noise_mode='const')
            
            # Check if this is a high-quality image (lower loss = better quality)
            if loss.item() < quality_threshold:
                best_images.append(ema_image[0].clone())
                best_losses.append(loss.item())
    
    return best_images, best_losses, q_ema, final_loss

def generate_text_to_image(text_prompt, steps=300, seed=-1, num_candidates=32, 
                          quality_threshold=0.6, user_id=None):
    """
    Generate images from text prompt
    """
    initialize_text_to_image_models()
    
    if seed == -1:
        seed = np.random.randint(0, 2**31 - 1)
    print(f"Using seed: {seed}")
    
    # Parse multiple prompts
    texts = [phrase.strip() for phrase in text_prompt.split("|") if phrase]
    targets = [clip_model.embed_text(text) for text in texts]
    print(f"Prompts: {texts}")
    
    torch.manual_seed(seed)
    
    # Initialize with best candidate from multiple samples
    print("Finding best initial candidate...")
    best_q = find_best_initial_candidate(targets, num_candidates)
    
    # Optimization loop
    print("Starting optimization...")
    best_images, best_losses, final_q_ema, final_loss = optimize_latent(
        best_q, targets, steps, quality_threshold
    )
    
    # Generate output images
    output_images = []
    
    # Save best images if any
    if best_images:
        # Sort by loss (best first)
        sorted_indices = np.argsort(best_losses)
        
        # Save top 3 best images
        num_to_save = min(3, len(best_images))
        for i, idx in enumerate(sorted_indices[:num_to_save]):
            image_tensor = best_images[idx]
            pil_image = TF.to_pil_image(image_tensor.add(1).div(2).clamp(0, 1))
            
            # Save locally
            unique_id = str(uuid.uuid4())
            output_filename = f"text_to_image_{unique_id}.jpg"
            output_path = os.path.join(TEXT_TO_IMAGE_OUTPUT_FOLDER, output_filename)
            pil_image.save(output_path)
            
            # Upload to Cloudinary if user_id is provided
            cloudinary_info = {}
            if user_id:
                try:
                    cloudinary_result = upload_to_cloudinary(
                        output_path, 
                        f"text_to_image/{user_id}", 
                        f"text_gen_{unique_id}"
                    )
                    
                    if cloudinary_result['success']:
                        cloudinary_info = {
                            'cloudinary_url': cloudinary_result['url'],
                            'cloudinary_public_id': cloudinary_result['public_id']
                        }
                        
                        # Create design data
                        design_data = {
                            'uid': user_id,
                            'design_id': unique_id,
                            'created_at': datetime.datetime.now(),
                            'cloudinary_url': cloudinary_result['url'],
                            'cloudinary_public_id': cloudinary_result['public_id'],
                            'is_favorite': False,
                            'loss': float(best_losses[idx]),
                            'prompt': text_prompt,
                            'type': 'text_to_image'
                        }
                        
                        # Try to save to Firestore if available
                        if FIRESTORE_AVAILABLE:
                            try:
                                db.collection('designs').document(unique_id).set(design_data)
                                print(f"Successfully saved text-to-image design {unique_id} to Firestore")
                            except Exception as e:
                                print(f"Error saving to Firestore: {e}")
                        
                        # Always save locally as backup
                        user_designs_dir = os.path.join(FAVORITES_FOLDER, user_id, 'designs')
                        os.makedirs(user_designs_dir, exist_ok=True)
                        
                        # Convert datetime to string for JSON serialization
                        design_data_json = dict(design_data)
                        for key, value in design_data_json.items():
                            if isinstance(value, datetime.datetime):
                                design_data_json[key] = value.isoformat()
                        
                        with open(os.path.join(user_designs_dir, f"{unique_id}.json"), 'w') as f:
                            json.dump(design_data_json, f)
                except Exception as e:
                    print(f"Error in cloud storage process: {e}")
            
            # Add to the output list
            image_info = {
                'id': unique_id,
                'url': f"/text_to_image_outputs/{output_filename}",
                'path': output_path,
                'is_favorite': False,
                'loss': best_losses[idx],
                'type': 'text_to_image'
            }
            
            # Add Cloudinary info if available
            image_info.update(cloudinary_info)
            output_images.append(image_info)
    
    # Always save the final optimized result
    with torch.no_grad():
        final_w = final_q_ema * w_stds + G.mapping.w_avg
        final_image = G.synthesis(final_w, noise_mode='const')
        pil_image = TF.to_pil_image(final_image[0].add(1).div(2).clamp(0, 1))
        
        unique_id = str(uuid.uuid4())
        output_filename = f"text_to_image_final_{unique_id}.jpg"
        output_path = os.path.join(TEXT_TO_IMAGE_OUTPUT_FOLDER, output_filename)
        pil_image.save(output_path)
        
        # Upload to Cloudinary if user_id is provided
        cloudinary_info = {}
        if user_id:
            try:
                cloudinary_result = upload_to_cloudinary(
                    output_path, 
                    f"text_to_image/{user_id}", 
                    f"text_gen_final_{unique_id}"
                )
                
                if cloudinary_result['success']:
                    cloudinary_info = {
                        'cloudinary_url': cloudinary_result['url'],
                        'cloudinary_public_id': cloudinary_result['public_id']
                    }
                    
                    # Create design data
                    design_data = {
                        'uid': user_id,
                        'design_id': unique_id,
                        'created_at': datetime.datetime.now(),
                        'cloudinary_url': cloudinary_result['url'],
                        'cloudinary_public_id': cloudinary_result['public_id'],
                        'is_favorite': False,
                        'loss': float(final_loss),
                        'prompt': text_prompt,
                        'type': 'text_to_image'
                    }
                    
                    # Try to save to Firestore if available
                    if FIRESTORE_AVAILABLE:
                        try:
                            db.collection('designs').document(unique_id).set(design_data)
                        except Exception as e:
                            print(f"Error saving to Firestore: {e}")
                    
                    # Always save locally as backup
                    user_designs_dir = os.path.join(FAVORITES_FOLDER, user_id, 'designs')
                    os.makedirs(user_designs_dir, exist_ok=True)
                    
                    # Convert datetime to string for JSON serialization
                    design_data_json = dict(design_data)
                    for key, value in design_data_json.items():
                        if isinstance(value, datetime.datetime):
                            design_data_json[key] = value.isoformat()
                    
                    with open(os.path.join(user_designs_dir, f"{unique_id}.json"), 'w') as f:
                        json.dump(design_data_json, f)
            except Exception as e:
                print(f"Error in cloud storage process: {e}")
        
        # Add final image to output
        final_image_info = {
            'id': unique_id,
            'url': f"/text_to_image_outputs/{output_filename}",
            'path': output_path,
            'is_favorite': False,
            'loss': final_loss,
            'type': 'text_to_image'
        }
        final_image_info.update(cloudinary_info)
        output_images.append(final_image_info)
    
    print(f"Text-to-image generation complete! Generated {len(output_images)} images")
    return output_images

# Helper Functions (existing ones remain the same)
def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def verify_firebase_token(id_token):
    """Verify Firebase ID token and return user information"""
    try:
        # Verify the ID token while checking if the token is revoked
        decoded_token = auth.verify_id_token(id_token, check_revoked=True)
        # Extract user information from the token
        uid = decoded_token['uid']
        return {'success': True, 'uid': uid}
    except auth.RevokedIdTokenError:
        # Token has been revoked
        return {'success': False, 'error': 'Token revoked'}
    except auth.InvalidIdTokenError:
        # Token is invalid
        return {'success': False, 'error': 'Invalid token'}
    except Exception as e:
        # Other errors
        return {'success': False, 'error': str(e)}

def upload_to_cloudinary(file_path, folder, public_id=None):
    """Upload file to Cloudinary in the specified folder"""
    try:
        result = cloudinary.uploader.upload(
            file_path,
            folder=folder,
            public_id=public_id,
            overwrite=True,
            resource_type="auto"
        )
        return {
            'success': True, 
            'url': result['secure_url'],
            'public_id': result['public_id']
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def save_to_firestore(collection, document_id, data):
    """Save data to Firestore collection with fallback"""
    if not FIRESTORE_AVAILABLE:
        return {'success': False, 'error': 'Firestore not available'}
    
    try:
        db.collection(collection).document(document_id).set(data)
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_from_firestore(collection, query=None):
    """Get data from Firestore collection with optional query and fallback"""
    if not FIRESTORE_AVAILABLE:
        return {'success': False, 'error': 'Firestore not available'}
    
    try:
        if query:
            # Expecting query as a dict with field, operator, value
            # e.g., {'field': 'uid', 'operator': '==', 'value': 'user123'}
            ref = db.collection(collection)
            results = ref.where(query['field'], query['operator'], query['value']).get()
        else:
            results = db.collection(collection).get()
        
        return {'success': True, 'data': [doc.to_dict() for doc in results]}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def delete_from_firestore(collection, document_id):
    """Delete document from Firestore collection with fallback"""
    if not FIRESTORE_AVAILABLE:
        return {'success': False, 'error': 'Firestore not available'}
    
    try:
        db.collection(collection).document(document_id).delete()
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def delete_from_cloudinary(public_id):
    """Delete file from Cloudinary"""
    try:
        result = cloudinary.uploader.destroy(public_id)
        return {'success': True if result['result'] == 'ok' else False}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def generate_variations(image_path, num_variations=4, user_id=None):
    """Generate variations of the input shoe image using StyleGAN3"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained StyleGAN3 model
    print(f"Loading network from {NETWORK_PKL}...")
    with dnnlib.util.open_url(NETWORK_PKL) as f:
        print("Network loading complete")
        G_local = legacy.load_network_pkl(f)['G_ema'].to(device)
        print("Network loaded to device")
    
    # Initialize LPIPS model for perceptual loss
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    # Load and preprocess the shoe image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((G_local.img_resolution, G_local.img_resolution))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_array.transpose([2, 0, 1]), device=device)
    img_tensor = img_tensor.unsqueeze(0) * 2 - 1  # Normalize to [-1, 1]
    
    # Project the image into latent space
    print("Projecting image into latent space...")
    z_projected = project_image_z_space(G_local, img_tensor, lpips_fn)
    
    # Create variations with different strengths
    output_images = []
    strengths = [0.1, 0.2, 0.3, 0.5]
    
    # Adjust the number of variations based on request
    used_strengths = strengths[:num_variations] if num_variations <= len(strengths) else strengths + [0.3] * (num_variations - len(strengths))
    
    for i, strength in enumerate(used_strengths):
        # Create variation
        z_variation = create_variation_z_space(z_projected, strength=strength)
        
        # Generate the variation
        variation_img = generate_image_from_z_space(G_local, z_variation)
        
        # Save the variation locally first
        unique_id = str(uuid.uuid4())
        output_filename = f"variation_{unique_id}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        imageio.imwrite(output_path, variation_img)
        
        # If user_id is provided, upload to Cloudinary and save metadata
        cloudinary_info = {}
        if user_id:
            try:
                # Upload to Cloudinary
                cloudinary_result = upload_to_cloudinary(
                    output_path, 
                    f"shoe_designs/{user_id}", 
                    f"variation_{unique_id}"
                )
                
                if cloudinary_result['success']:
                    cloudinary_info = {
                        'cloudinary_url': cloudinary_result['url'],
                        'cloudinary_public_id': cloudinary_result['public_id']
                    }
                    
                    # Create design data
                    design_data = {
                        'uid': user_id,
                        'design_id': unique_id,
                        'created_at': datetime.datetime.now(),
                        'cloudinary_url': cloudinary_result['url'],
                        'cloudinary_public_id': cloudinary_result['public_id'],
                        'is_favorite': False,
                        'strength': float(strength),
                        'type': 'variation'
                    }
                    
                    # Try to save to Firestore if available
                    if FIRESTORE_AVAILABLE:
                        try:
                            db.collection('designs').document(unique_id).set(design_data)
                            print(f"Successfully saved design {unique_id} to Firestore")
                        except Exception as e:
                            print(f"Error saving to Firestore: {e}")
                    
                    # Always save locally as backup
                    user_designs_dir = os.path.join(FAVORITES_FOLDER, user_id, 'designs')
                    os.makedirs(user_designs_dir, exist_ok=True)
                    
                    # Convert datetime to string for JSON serialization
                    design_data_json = dict(design_data)
                    for key, value in design_data_json.items():
                        if isinstance(value, datetime.datetime):
                            design_data_json[key] = value.isoformat()
                    
                    with open(os.path.join(user_designs_dir, f"{unique_id}.json"), 'w') as f:
                        json.dump(design_data_json, f)
            except Exception as e:
                print(f"Error in cloud storage process: {e}")
        
        # Add to the output list
        image_info = {
            'id': unique_id,
            'url': f"/outputs/{output_filename}",  # Local URL for immediate display
            'path': output_path,
            'is_favorite': False,
            'strength': strength,
            'type': 'variation'
        }
        
        # Add Cloudinary info if available
        image_info.update(cloudinary_info)
        
        output_images.append(image_info)
    
    return output_images

def project_image_z_space(G, img_tensor, lpips_fn, num_steps=1000):
    """Project the input shoe image into the latent space (Z space approach) - matches console code"""
    device = img_tensor.device
    
    # Start with random Z latent
    z_opt = torch.randn(1, G.z_dim, device=device, requires_grad=True)
    
    # Hyperparameters for optimization
    optimizer = torch.optim.Adam([z_opt], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    best_z = None
    best_loss = float('inf')
    
    # For production use, reduce the number of steps to speed up the process
    num_steps = 1000  # Same as console version
    
    # Run optimization
    for step in range(num_steps):
        # Generate image using Z space (not W space)
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
        
        # Print progress every 100 steps (like console version)
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}: MSE={mse_loss.item():.4f}, LPIPS={perceptual_loss.item():.4f}, Total={loss.item():.4f}")
    
    # Return the best result found
    return best_z if best_z is not None else z_opt.detach()

def create_variation_z_space(z_projected, strength=0.3):
    """Create a variation by adding noise to the projected z - matches console code"""
    with torch.no_grad():
        z_noise = torch.randn_like(z_projected) * strength
        z_variation = z_projected + z_noise
    return z_variation

def generate_image_from_z_space(G, z, truncation_psi=0.7, noise_mode='const'):
    """Generate an image from the z latent vector - matches console code"""
    with torch.no_grad():
        img = G(z, None, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img + 1) * 127.5  # Scale back to [0, 255]
        img = img.clamp(0, 255).to(torch.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C] format
    return img

# Helper functions for design history
def filter_designs_by_date(designs, date_range, start_date=None, end_date=None):
    """Filter designs by date range"""
    import datetime
    
    filtered_designs = []
    now = datetime.datetime.now()
    
    for design in designs:
        # Parse the creation date
        created_at = design.get('created_at')
        if isinstance(created_at, str):
            try:
                created_at = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except ValueError:
                # If parsing fails, use current time as fallback
                created_at = now
        
        # Apply date filters
        if date_range == 'today':
            # Today only
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if created_at >= today:
                filtered_designs.append(design)
                
        elif date_range == 'week':
            # Last 7 days
            week_ago = now - datetime.timedelta(days=7)
            if created_at >= week_ago:
                filtered_designs.append(design)
                
        elif date_range == 'month':
            # Last 30 days
            month_ago = now - datetime.timedelta(days=30)
            if created_at >= month_ago:
                filtered_designs.append(design)
                
        elif date_range == 'custom' and (start_date or end_date):
            # Custom date range
            valid = True
            
            if start_date:
                start = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if created_at < start:
                    valid = False
                    
            if end_date and valid:
                end = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                end = end.replace(hour=23, minute=59, second=59)
                if created_at > end:
                    valid = False
                    
            if valid:
                filtered_designs.append(design)
                
        else:
            # No date filter or 'all'
            filtered_designs.append(design)
    
    return filtered_designs

def sort_designs(designs, sort_order):
    """Sort designs by creation date"""
    import datetime
    
    # Ensure each design has a created_at datetime
    for design in designs:
        if 'created_at' not in design:
            design['created_at'] = datetime.datetime.now()
        elif isinstance(design['created_at'], str):
            try:
                design['created_at'] = datetime.datetime.fromisoformat(design['created_at'].replace('Z', '+00:00'))
            except ValueError:
                design['created_at'] = datetime.datetime.now()
    
    # Sort the designs
    if sort_order == 'oldest':
        return sorted(designs, key=lambda x: x['created_at'])
    else:  # newest first
        return sorted(designs, key=lambda x: x['created_at'], reverse=True)

def paginate_designs(designs, page, limit):
    """Paginate the designs list"""
    start = (page - 1) * limit
    end = start + limit
    return designs[start:end]

# Authentication Routes
@app.route('/')
def loginsignup():
    return render_template('loginsignup.html')

@app.route('/verify', methods=['POST'])
def verify():
    email = request.form['email_signup']
    username = request.form['username']
    password = request.form['password']
    confirm_password = request.form['verify_password']

    # Validate passwords
    if password != confirm_password:
        return render_template('loginsignup.html', msg="Passwords do not match. Please try again.")

    try:
        auth_pyrebase.get_account_info(auth_pyrebase.sign_in_with_email_and_password(email, password)['idToken'])
        return render_template('loginsignup.html', msg="Email already exists. Please log in.")
    except:
        pass

    # Generate OTP
    otp = random.randint(1000, 9999)
    otp_store[email] = {"otp": otp, "password": password, "username": username}

    # Send OTP
    subject = "Your OTP for Shoe Designer Registration"
    msg = Message(subject=subject, sender=app.config['MAIL_USERNAME'], recipients=[email])
    msg.body = f"Hello {username},\n\nYour OTP for registration is: {otp}.\n\nThank you!"
    mail.send(msg)

    return render_template('verify.html', email=email)

@app.route('/validate', methods=['POST'])
def validate():
    data = request.get_json()
    email = data.get('email')
    user_otp = data.get('otp')

    if email not in otp_store:
        return jsonify({'status': 'error', 'message': 'Invalid session. Please sign up again.'})

    if otp_store[email]['otp'] == int(user_otp):
        password = otp_store[email]['password']
        username = otp_store[email]['username']
        
        try:
            # Create user in Firebase Authentication
            user = auth_pyrebase.create_user_with_email_and_password(email, password)
            user_id = user['localId']
            
            # Create user directory for favorites
            user_dir = os.path.join(FAVORITES_FOLDER, user_id)
            os.makedirs(user_dir, exist_ok=True)
            
            # Save user profile in Firestore if available
            if FIRESTORE_AVAILABLE:
                profile_data = {
                    'uid': user_id,
                    'email': email,
                    'username': username,
                    'created_at': datetime.datetime.now()
                }
                
                try:
                    db.collection('users').document(user_id).set(profile_data)
                except Exception as e:
                    print(f"Error saving user profile to Firestore: {e}")
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error creating user, Email Already Exists'})

        del otp_store[email]
        return jsonify({'status': 'success', 'message': 'Email verified successfully!'})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid OTP. Please try again.'})

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email-login']
    password = request.form['password-login']
    try:
        user = auth_pyrebase.sign_in_with_email_and_password(email, password)
        session['user'] = user['localId']
        session['id_token'] = user['idToken']
        
        # Create user directory for favorites if it doesn't exist
        user_dir = os.path.join(FAVORITES_FOLDER, user['localId'])
        os.makedirs(user_dir, exist_ok=True)
        
        return redirect(url_for('dashboard'))
    except:
        return render_template('loginsignup.html', msg="Invalid email or password.")

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return render_template('dashboard.html')
    return redirect(url_for('loginsignup'))

@app.route('/index.html', methods=['GET'])
def index_html():
    """Alias for dashboard"""
    if 'user' in session:
        return render_template('dashboard.html')
    return redirect(url_for('loginsignup'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('id_token', None)
    return redirect(url_for('loginsignup'))

# User profile route
@app.route('/user-profile', methods=['GET'])
def user_profile():
    """Get the current user's profile information"""
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user']
    
    if FIRESTORE_AVAILABLE:
        try:
            # Try to get user data from Firestore
            user_doc = db.collection('users').document(user_id).get()
            
            if user_doc.exists:
                user_data = user_doc.to_dict()
                return jsonify({
                    'success': True,
                    'profile': {
                        'uid': user_id,
                        'username': user_data.get('username', ''),
                        'email': user_data.get('email', '')
                    }
                })
            else:
                # If no user document found, return just the user ID
                return jsonify({
                    'success': True,
                    'profile': {
                        'uid': user_id
                    }
                })
        except Exception as e:
            print(f"Error fetching user profile from Firestore: {e}")
    
    # If Firestore not available or failed, return just the user ID
    return jsonify({
        'success': True,
        'profile': {
            'uid': user_id
        }
    })

# Design history route
@app.route('/design-history', methods=['GET'])
def design_history():
    """Get the user's design history with optional filters"""
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user']
    
    # Get filter parameters
    date_range = request.args.get('date_range', 'all')
    design_type = request.args.get('type', 'all')
    status = request.args.get('status', 'all')
    sort_order = request.args.get('sort_order', 'newest')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 12))
    
    # Initialize empty designs list
    designs = []
    
    # Fetch designs from Firestore if available
    if FIRESTORE_AVAILABLE:
        try:
            # Start with a base query for this user's designs
            query = db.collection('designs').where('uid', '==', user_id)
            
            # Apply type filter if specified
            if design_type != 'all':
                query = query.where('type', '==', design_type)
                
            # Apply status filter if specified
            if status != 'all':
                is_favorite = (status == 'production')
                query = query.where('is_favorite', '==', is_favorite)
            
            # Execute query
            results = query.get()
            
            # Convert to list of dicts
            designs = [doc.to_dict() for doc in results]
            
        except Exception as e:
            print(f"Error fetching design history from Firestore: {e}")
    
    # If Firestore failed or not available, try to use local backup
    if not designs and user_id:
        try:
            user_designs_dir = os.path.join(FAVORITES_FOLDER, user_id, 'designs')
            if os.path.exists(user_designs_dir):
                for filename in os.listdir(user_designs_dir):
                    if filename.endswith('.json'):
                        try:
                            with open(os.path.join(user_designs_dir, filename), 'r') as f:
                                design_data = json.load(f)
                                designs.append(design_data)
                        except:
                            pass
        except Exception as e:
            print(f"Error loading designs from local storage: {e}")
    
    # Apply additional filters that are not applied in the Firestore query
    # Date filter
    if date_range != 'all' or (start_date and end_date):
        designs = filter_designs_by_date(designs, date_range, start_date, end_date)
    
    # Sort designs
    designs = sort_designs(designs, sort_order)
    
    # Get total count before pagination
    total_designs = len(designs)
    
    # Paginate results
    paginated_designs = paginate_designs(designs, page, limit)
    
    return jsonify({
        'success': True,
        'designs': paginated_designs,
        'total': total_designs,
        'page': page,
        'pages': (total_designs + limit - 1) // limit
    })

# Shoe Generator Routes
@app.route('/firebase-config', methods=['GET'])
def firebase_config():
    """Return the Firebase configuration to the client"""
    return jsonify(firebaseConfig)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate variations of the uploaded shoe image"""
    # Check if user is authenticated
    if 'user' not in session:
        return jsonify({'error': 'Please login to use this feature'}), 401
        
    if 'shoe_image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['shoe_image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Get number of variations to generate
    variations_count = int(request.form.get('variations_count', 4))
    
    # Save the uploaded file
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # Get user ID from session
        user_id = session.get('user')
        
        # Upload the original image to Cloudinary
        upload_result = upload_to_cloudinary(
            filepath, 
            f"shoe_originals/{user_id}", 
            f"original_{filename.split('.')[0]}"
        )
        
        if upload_result['success']:
            # Create original metadata
            original_data = {
                'uid': user_id,
                'image_id': filename.split('.')[0],
                'created_at': datetime.datetime.now(),
                'cloudinary_url': upload_result['url'],
                'cloudinary_public_id': upload_result['public_id'],
                'is_original': True
            }
            
            # Save original image metadata to Firestore if available
            if FIRESTORE_AVAILABLE:
                try:
                    db.collection('originals').document(filename.split('.')[0]).set(original_data)
                except Exception as e:
                    print(f"Error saving original to Firestore: {e}")
            
            # Also save original metadata locally
            user_originals_dir = os.path.join(FAVORITES_FOLDER, user_id, 'originals')
            os.makedirs(user_originals_dir, exist_ok=True)
            
            # Convert datetime for JSON serialization
            original_data_json = dict(original_data)
            for key, value in original_data_json.items():
                if isinstance(value, datetime.datetime):
                    original_data_json[key] = value.isoformat()
            
            with open(os.path.join(user_originals_dir, f"{filename.split('.')[0]}.json"), 'w') as f:
                json.dump(original_data_json, f)
        
        # Generate variations with user ID
        output_images = generate_variations(filepath, variations_count, user_id)
        
        # Return the paths to the generated images
        return jsonify({'images': output_images})
    
    except Exception as e:
        app.logger.error(f"Error generating variations: {str(e)}")
        return jsonify({'error': str(e)}), 500

# TEXT-TO-IMAGE ROUTE
@app.route('/generate-text-to-image', methods=['POST'])
def generate_text_to_image_route():
    """Generate images from text prompt"""
    # Check if user is authenticated
    if 'user' not in session:
        return jsonify({'error': 'Please login to use this feature'}), 401
    
    data = request.get_json()
    text_prompt = data.get('text_prompt', '').strip()
    
    if not text_prompt:
        return jsonify({'error': 'Please provide a text prompt'}), 400
    
    # Get optional parameters
    steps = int(data.get('steps', 300))
    quality_threshold = float(data.get('quality_threshold', 0.6))
    
    try:
        # Get user ID from session
        user_id = session.get('user')
        
        # Generate images from text
        output_images = generate_text_to_image(
            text_prompt=text_prompt,
            steps=steps,
            quality_threshold=quality_threshold,
            user_id=user_id
        )
        
        # Return the generated images
        return jsonify({'images': output_images, 'prompt': text_prompt})
    
    except Exception as e:
        app.logger.error(f"Error generating text-to-image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    """Download a generated image"""
    # Check if user is authenticated
    if 'user' not in session:
        return jsonify({'error': 'Please login to use this feature'}), 401
        
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(filepath, as_attachment=True)

@app.route('/download-all', methods=['POST'])
def download_all():
    """Download all generated images as a zip file"""
    # Check if user is authenticated
    if 'user' not in session:
        return jsonify({'error': 'Please login to use this feature'}), 401
        
    image_paths = request.json.get('image_paths', [])
    
    if not image_paths:
        return jsonify({'error': 'No images to download'}), 400
    
    # Create a zip file in memory
    memory_file = io.BytesIO()
    with ZipFile(memory_file, 'w') as zf:
        for i, path in enumerate(image_paths):
            if os.path.exists(path):
                zf.write(path, f'design-{i+1}.jpg')
    
    memory_file.seek(0)
    
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='shoe-designs.zip'
    )

# Favorites Management (Production Management)
@app.route('/favorites', methods=['GET'])
def list_favorites():
    """Get list of user's favorite designs (production designs)"""
    if 'user' not in session:
        return jsonify({'error': 'Please login to view favorites'}), 401
    
    user_id = session['user']
    print(f"Listing favorites for user: {user_id}")
    
    favorites = []
    
    if FIRESTORE_AVAILABLE:
        # Try Firestore first
        try:
            designs_ref = db.collection('designs').where('uid', '==', user_id).where('is_favorite', '==', True).get()
            
            for doc in designs_ref:
                design = doc.to_dict()
                # Convert datetime objects to strings for JSON
                for key, value in dict(design).items():
                    if isinstance(value, datetime.datetime):
                        design[key] = value.isoformat()
                favorites.append(design)
                print(f"Found Firestore favorite: {design.get('design_id')}")
        except Exception as e:
            print(f"Error retrieving favorites from Firestore: {e}")
            # Fall back to local storage
    
    # Use local storage as fallback or supplement
    user_favorites_dir = os.path.join(FAVORITES_FOLDER, user_id)
    
    if os.path.exists(user_favorites_dir):
        for filename in os.listdir(user_favorites_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(user_favorites_dir, filename), 'r') as f:
                        favorite_data = json.load(f)
                        
                        # Check if this favorite is already in the list from Firestore
                        exists = False
                        for existing in favorites:
                            if existing.get('design_id') == favorite_data.get('design_id'):
                                exists = True
                                break
                        
                        if not exists and favorite_data.get('is_favorite', False):
                            favorites.append(favorite_data)
                            print(f"Found local favorite: {favorite_data.get('design_id')}")
                except Exception as e:
                    print(f"Error reading favorite file {filename}: {e}")
    
    print(f"Total favorites found: {len(favorites)}")
    return jsonify({'favorites': favorites})

@app.route('/favorites/<design_id>', methods=['POST'])
def toggle_favorite(design_id):
    """Toggle favorite status for a design (send to production)"""
    if 'user' not in session:
        print("Authentication error: User not in session")
        return jsonify({'error': 'Please login to manage favorites'}), 401
    
    user_id = session['user']
    print(f"Processing favorite toggle for design_id: {design_id}, user_id: {user_id}")
    
    # Create user directory for favorites if needed
    user_favorites_dir = os.path.join(FAVORITES_FOLDER, user_id)
    os.makedirs(user_favorites_dir, exist_ok=True)
    favorite_json_path = os.path.join(user_favorites_dir, f"{design_id}.json")
    
    if FIRESTORE_AVAILABLE:
        # Firestore approach
        try:
            # Check if the design exists in Firestore
            design_doc = db.collection('designs').document(design_id).get()
            
            if not design_doc.exists:
                print(f"Design {design_id} not found in Firestore")
                
                # Look for local file instead
                local_file_found = False
                local_path = None
                
                # Check both output folders for the file
                for folder in [OUTPUT_FOLDER, TEXT_TO_IMAGE_OUTPUT_FOLDER]:
                    for filename in os.listdir(folder):
                        if design_id in filename:
                            local_file_found = True
                            local_path = os.path.join(folder, filename)
                            break
                    if local_file_found:
                        break
                
                if not local_file_found:
                    return jsonify({'error': 'Design not found'}), 404
                
                # Create new document in Firestore
                design_data = {
                    'uid': user_id,
                    'design_id': design_id,
                    'created_at': datetime.datetime.now(),
                    'is_favorite': True,
                    'type': 'text_to_image' if 'text_to_image' in local_path else 'variation'
                }
                
                try:
                    # Try to upload to Cloudinary if local file exists
                    if local_file_found:
                        folder_name = "text_to_image" if design_data['type'] == 'text_to_image' else "shoe_designs"
                        upload_result = upload_to_cloudinary(
                            local_path,
                            f"{folder_name}/{user_id}",
                            f"design_{design_id}"
                        )
                        
                        if upload_result['success']:
                            design_data['cloudinary_url'] = upload_result['url']
                            design_data['cloudinary_public_id'] = upload_result['public_id']
                except Exception as e:
                    print(f"Warning: Cloudinary upload failed: {e}")
                
                # Save to Firestore
                db.collection('designs').document(design_id).set(design_data)
                print(f"Created new design document in Firestore for {design_id}")
                
                # Also save locally as backup
                design_data_json = dict(design_data)
                for key, value in design_data_json.items():
                    if isinstance(value, datetime.datetime):
                        design_data_json[key] = value.isoformat()
                
                with open(favorite_json_path, 'w') as f:
                    json.dump(design_data_json, f)
                
                return jsonify({
                    'success': True,
                    'design_id': design_id,
                    'is_favorite': True
                })
            
            # Design exists in Firestore
            design_data = design_doc.to_dict()
            print(f"Retrieved design data from Firestore: {design_data}")
            
            # Verify the design belongs to this user
            if design_data.get('uid') != user_id:
                print(f"Authorization error: Design {design_id} belongs to {design_data.get('uid')}, not {user_id}")
                return jsonify({'error': 'Unauthorized access to this design'}), 403
            
            # Toggle favorite status
            new_status = not design_data.get('is_favorite', False)
            print(f"Toggling favorite status from {design_data.get('is_favorite', False)} to {new_status}")
            
            # Update in Firestore
            db.collection('designs').document(design_id).update({
                'is_favorite': new_status,
                'updated_at': datetime.datetime.now()
            })
            print(f"Successfully updated favorite status in Firestore")
            
            # Update local copy for backup
            if new_status:
                # Save locally
                design_data['is_favorite'] = True
                design_data['updated_at'] = datetime.datetime.now()
                
                # Convert datetime objects to strings for JSON
                design_data_json = dict(design_data)
                for key, value in design_data_json.items():
                    if isinstance(value, datetime.datetime):
                        design_data_json[key] = value.isoformat()
                
                with open(favorite_json_path, 'w') as f:
                    json.dump(design_data_json, f)
            else:
                # Remove local copy if exists
                if os.path.exists(favorite_json_path):
                    os.remove(favorite_json_path)
            
            return jsonify({
                'success': True, 
                'design_id': design_id, 
                'is_favorite': new_status
            })
            
        except Exception as e:
            print(f"Error in Firestore toggle_favorite: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to local storage
            print("Falling back to local storage")
            # Continue to local storage approach below
    
    # Local storage approach (used as fallback or when Firestore is unavailable)
    try:
        # Check if favorite exists locally
        is_favorite = os.path.exists(favorite_json_path)
        
        if is_favorite:
            # Remove from favorites
            os.remove(favorite_json_path)
            print(f"Removed favorite {design_id} from local storage")
            
            return jsonify({
                'success': True,
                'design_id': design_id,
                'is_favorite': False
            })
        else:
            # Find the design in outputs
            local_path = None
            cloudinary_url = None
            
            # Check both output folders
            for folder in [OUTPUT_FOLDER, TEXT_TO_IMAGE_OUTPUT_FOLDER]:
                for filename in os.listdir(folder):
                    if design_id in filename:
                        local_path = os.path.join(folder, filename)
                        break
                if local_path:
                    break
            
            if not local_path:
                return jsonify({'error': 'Design not found'}), 404
            
            # Try to upload to Cloudinary
            try:
                folder_name = "text_to_image" if TEXT_TO_IMAGE_OUTPUT_FOLDER in local_path else "shoe_designs"
                upload_result = upload_to_cloudinary(
                    local_path,
                    f"{folder_name}/{user_id}",
                    f"design_{design_id}"
                )
                
                if upload_result['success']:
                    cloudinary_url = upload_result['url']
                    cloudinary_public_id = upload_result['public_id']
            except Exception as e:
                print(f"Warning: Cloudinary upload failed: {e}")
            
            # Save favorite metadata
            favorite_data = {
                'uid': user_id,
                'design_id': design_id,
                'created_at': datetime.datetime.now().isoformat(),
                'is_favorite': True,
                'local_path': local_path,
                'type': 'text_to_image' if TEXT_TO_IMAGE_OUTPUT_FOLDER in local_path else 'variation'
            }
            
            if cloudinary_url:
                favorite_data['cloudinary_url'] = cloudinary_url
                favorite_data['cloudinary_public_id'] = cloudinary_public_id
            
            # Save to local JSON file
            with open(favorite_json_path, 'w') as f:
                json.dump(favorite_data, f)
            
            print(f"Added favorite {design_id} to local storage")
            return jsonify({
                'success': True,
                'design_id': design_id,
                'is_favorite': True
            })
            
    except Exception as e:
        print(f"Error in local storage toggle_favorite: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to update favorite status: {str(e)}'}), 500

@app.route('/designs/<design_id>', methods=['DELETE'])
def delete_design(design_id):
    """Delete a design"""
    if 'user' not in session:
        return jsonify({'error': 'Please login to delete designs'}), 401
    
    user_id = session['user']
    
    # Check if design exists in Firestore
    if FIRESTORE_AVAILABLE:
        try:
            # Get the design document
            design_doc = db.collection('designs').document(design_id).get()
            
            if design_doc.exists:
                design_data = design_doc.to_dict()
                
                # Verify the design belongs to this user
                if design_data.get('uid') != user_id:
                    return jsonify({'error': 'Unauthorized access to this design'}), 403
                
                # Delete from Cloudinary if public_id exists
                if 'cloudinary_public_id' in design_data:
                    delete_from_cloudinary(design_data['cloudinary_public_id'])
                
                # Delete from Firestore
                db.collection('designs').document(design_id).delete()
        except Exception as e:
            print(f"Error deleting from Firestore: {e}")
    
    # Also delete local file if it exists
    user_favorites_dir = os.path.join(FAVORITES_FOLDER, user_id)
    favorite_json_path = os.path.join(user_favorites_dir, f"{design_id}.json")
    
    if os.path.exists(favorite_json_path):
        os.remove(favorite_json_path)
    
    # Try to delete from both output folders
    for folder in [OUTPUT_FOLDER, TEXT_TO_IMAGE_OUTPUT_FOLDER]:
        for filename in os.listdir(folder):
            if design_id in filename:
                try:
                    os.remove(os.path.join(folder, filename))
                except:
                    pass
    
    return jsonify({'success': True, 'message': 'Design deleted successfully'})
    
# File Serving Routes
@app.route('/static/images/<path:filename>')
def serve_static(filename):
    return send_from_directory('static/images', filename)

@app.route('/outputs/<filename>')
def serve_output(filename):
    """Serve generated images"""
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

@app.route('/text_to_image_outputs/<filename>')
def serve_text_to_image_output(filename):
    """Serve text-to-image generated images"""
    return send_file(os.path.join(TEXT_TO_IMAGE_OUTPUT_FOLDER, filename))

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded images"""
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/favorites/<user_id>/<filename>')
def serve_favorite(user_id, filename):
    """Serve favorite images from local storage"""
    # Check if user is authenticated and requesting their own favorites
    if 'user' not in session or session['user'] != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
        
    return send_file(os.path.join(FAVORITES_FOLDER, user_id, filename))

# Add these imports at the top of app.py if not already present
# import random
# from flask_mail import Mail, Message

# Complete implementation for direct password reset functionality

@app.route('/forgot.html', methods=['GET'])
def forgot_password():
    """Render the forgot password page"""
    print("heheeeeeeeeeeeeeeeeeeeeeeeeeee")
    return render_template('forgot.html')

@app.route('/request-reset-otp', methods=['POST'])
def request_reset_otp():
    """Handle request for password reset OTP"""
    data = request.get_json()
    email = data.get('email')
    
    if not email:
        return jsonify({'status': 'error', 'message': 'Email is required'})
    
    # Check if the email exists in Firebase
    try:
        # Using Firebase Admin SDK to check if user exists
        try:
            user = auth.get_user_by_email(email)
            # User exists, store the UID for later use
            user_exists = True
            user_id = user.uid
        except Exception as e:
            if 'NO_USER_RECORD_FOUND' in str(e):
                # User doesn't exist
                return jsonify({'status': 'error', 'message': 'No account found with this email. Please sign up first.'})
            else:
                # Some other error occurred with Admin SDK
                # Try alternative check with Pyrebase
                try:
                    # This will throw an exception if user doesn't exist
                    auth_pyrebase.send_password_reset_email(email)
                    user_exists = True
                except:
                    return jsonify({'status': 'error', 'message': 'No account found with this email. Please sign up first.'})
    except Exception as e:
        print(f"Error checking user existence: {e}")
        return jsonify({'status': 'error', 'message': 'Error verifying account. Please try again.'})
    
    try:
        # Generate OTP
        otp = random.randint(1000, 9999)
        otp_store[email] = {
            "otp": otp, 
            "type": "reset",
            "timestamp": datetime.datetime.now()
        }
        
        # Send OTP
        subject = "Password Reset OTP for Shoe Designer"
        msg = Message(subject=subject, sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f"""Hello,

Your OTP for password reset is: {otp}

This OTP will expire in 10 minutes.

Thank you!
Shoe Designer Team
"""
        mail.send(msg)
        
        return jsonify({'status': 'success', 'message': 'OTP sent successfully'})
    except Exception as e:
        print(f"Error sending reset OTP: {e}")
        return jsonify({'status': 'error', 'message': f'Error sending OTP: {str(e)}'})

@app.route('/reset-password', methods=['POST'])
def reset_password():
    """Handle password reset with OTP verification"""
    data = request.get_json()
    email = data.get('email')
    new_password = data.get('newPassword')
    user_otp = data.get('otp')
    
    if not email or not new_password or not user_otp:
        return jsonify({'status': 'error', 'message': 'All fields are required'})
    
    if email not in otp_store:
        return jsonify({'status': 'error', 'message': 'Invalid session. Please request a new OTP.'})
    
    # Check OTP expiration (10 minutes)
    otp_timestamp = otp_store[email].get('timestamp')
    if otp_timestamp:
        current_time = datetime.datetime.now()
        time_diff = current_time - otp_timestamp
        if time_diff.total_seconds() > 600:  # 10 minutes in seconds
            del otp_store[email]
            return jsonify({'status': 'error', 'message': 'OTP has expired. Please request a new one.'})
    
    # Verify OTP
    if otp_store[email]['otp'] != int(user_otp) or otp_store[email]['type'] != 'reset':
        return jsonify({'status': 'error', 'message': 'Invalid OTP. Please try again.'})
    
    try:
        # Try direct password reset with Firebase Auth
        try:
            # First, try with Admin SDK if available
            user = auth.get_user_by_email(email)
            auth.update_user(user.uid, password=new_password)
            
            # Password updated successfully
            del otp_store[email]
            return jsonify({'status': 'success', 'message': 'Password reset successfully'})
        except Exception as admin_error:
            print(f"Admin password reset failed: {admin_error}")
            
            # Fall back to Pyrebase approach
            try:
                # Get current auth data (might throw if we don't know current password)
                user_auth = None
                
                try:
                    # This will most likely fail since we don't know the user's current password
                    user_auth = auth_pyrebase.sign_in_with_email_and_password(email, "temporary")
                except:
                    pass
                
                if user_auth:
                    # Change password if we somehow got user auth
                    auth_pyrebase.change_password(user_auth['idToken'], new_password)
                    del otp_store[email]
                    return jsonify({'status': 'success', 'message': 'Password reset successfully'})
                else:
                    # Send password reset email as fallback
                    auth_pyrebase.send_password_reset_email(email)
                    del otp_store[email]
                    return jsonify({
                        'status': 'success', 
                        'message': 'A password reset link has been sent to your email. Please check your inbox to complete the reset process.'
                    })
            except Exception as pyrebase_error:
                print(f"Pyrebase password reset failed: {pyrebase_error}")
                raise pyrebase_error
    except Exception as e:
        print(f"Error in password reset process: {e}")
        return jsonify({'status': 'error', 'message': f'Error resetting password: {str(e)}'})
    
    
if __name__ == '__main__':
    app.run(debug=True)