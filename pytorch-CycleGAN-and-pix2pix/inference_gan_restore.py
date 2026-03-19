import os
import sys
import torch
import cv2
import numpy as np
import glob
from tqdm.auto import tqdm
from PIL import Image
from models import create_model
# Helper import for transforms
import torchvision.transforms as transforms

# --- CONFIGURATION ---
# Path to noisy images (Input)
INPUT_ROOT = "../wbc_dataset_noise"

# Path to clean images (Output)
OUTPUT_ROOT = "../dataset_gan_clean"

# Experiment name used during training (to locate checkpoint)
# Note: Checkpoint must exist at ./checkpoints/wbc_denoise/latest_net_G.pth
EXPERIMENT_NAME = "wbc_denoise"

# Image sizes
ORIG_SIZE = 224  # Original and target image size
GAN_SIZE = 256   # Required input size for GAN

# --- MOCK OPTIONS CLASS (CONFIGURATION WRAPPER) ---
class Opt:
    def __init__(self):
        self.aspect_ratio = 1.0
        self.batch_size = 1
        self.checkpoints_dir = './checkpoints'
        self.crop_size = 256
        self.gpu_ids = [0]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataroot = ''  # Dummy
        self.dataset_mode = 'single'  # Single mode: no paired A-B required
        self.direction = 'AtoB'       # Direction: A (Noisy) -> B (Clean)
        self.display_id = -1
        self.display_winsize = 256
        self.epoch = 'latest'         # Load latest checkpoint
        self.eval = True              # Evaluation mode (disable dropout if any)
        self.gpu_ids = [0]            # Use GPU 0
        self.init_gain = 0.02
        self.init_type = 'normal'
        self.input_nc = 3
        self.isTrain = False          # Important flag
        self.load_iter = 0
        self.load_size = 256
        self.max_dataset_size = float("inf")
        self.model = 'pix2pix'        # Model type
        self.n_layers_D = 3
        self.name = EXPERIMENT_NAME
        self.ndf = 64
        self.netD = 'basic'
        self.netG = 'unet_256'        # Generator architecture (must match training)
        self.ngf = 64
        self.no_dropout = False
        self.no_flip = True
        self.norm = 'batch'
        self.num_threads = 0
        self.output_nc = 3
        self.phase = 'test'
        self.preprocess = 'none'      # Manual resizing
        self.serial_batches = True
        self.suffix = ''
        self.use_wandb = False
        self.verbose = False

def tensor2im(input_image, imtype=np.uint8):
    """Convert tensor [-1, 1] to numpy image [0, 255]."""
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image

    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    
    # Denormalize: (x + 1) / 2 * 255
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def run_restoration():
    print("Starting image restoration (denoising) pipeline...")
    
    # 1. Load model
    opt = Opt()

    try:
        model = create_model(opt)
        model.setup(opt)
        model.eval()
        print(f"Model loaded successfully: {EXPERIMENT_NAME}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Scan input files
    print(f"Scanning files from: {INPUT_ROOT}")
    # Recursively find all JPG images
    files = glob.glob(os.path.join(INPUT_ROOT, "**", "*.jpg"), recursive=True)
    if not files:
        print("No images found.")
        return
    
    print(f"Found {len(files)} images to process.")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 3. Processing loop
    for path in tqdm(files, desc="Denoising"):
        try:
            # --- PREPROCESS ---
            # Read image in BGR format (OpenCV)
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Resize from 224 -> 256 (GAN input size)
            img_resized = cv2.resize(
                img_rgb,
                (GAN_SIZE, GAN_SIZE),
                interpolation=cv2.INTER_CUBIC
            )
            
            # Normalize to [-1, 1] and convert to tensor (1, C, H, W)
            img_tensor = transforms.ToTensor()(Image.fromarray(img_resized))
            img_tensor = transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )(img_tensor)
            img_tensor = img_tensor.unsqueeze(0).to(opt.device)
            
            # --- INFERENCE ---
            # Pix2Pix requires a dict input {'A': ..., 'A_paths': ...}
            # In 'single' test mode, the model uses 'A' as input and produces 'fake_B'
            model.set_input({
                'A': img_tensor,
                'B': img_tensor,      # Dummy input to avoid KeyError
                'A_paths': path,
                'B_paths': path
            })
            model.test()
            
            # Get output
            visuals = model.get_current_visuals()
            fake_B = visuals['fake_B']  # Denoised image
            
            # --- POSTPROCESS ---
            # Tensor -> Numpy [0, 255]
            restored_img = tensor2im(fake_B)
            
            # Resize from 256 -> 224 (desired output size)
            restored_img = cv2.resize(
                restored_img,
                (ORIG_SIZE, ORIG_SIZE),
                interpolation=cv2.INTER_CUBIC
            )
            
            # Convert back to BGR for saving
            restored_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
            
            # --- SAVE ---
            # Recreate subdirectory structure
            # Example: INPUT_ROOT/train/BA/img.jpg -> OUTPUT_ROOT/train/BA/img.jpg
            rel_path = os.path.relpath(path, INPUT_ROOT)
            save_path = os.path.join(OUTPUT_ROOT, rel_path)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, restored_bgr)
            
        except Exception as e:
            print(f"Error processing file {path}: {e}")

    print(f"\nCOMPLETED! Clean images are saved at: {OUTPUT_ROOT}")



if __name__ == "__main__":
    run_restoration()
