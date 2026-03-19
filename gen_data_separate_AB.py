import os
import cv2
import numpy as np
import glob
import random
import shutil
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

# --- CONFIGURATION ---
SOURCE_CLEAN_DIR = "wbc_dataset_clean"          # Source directory containing clean images
OUTPUT_ROOT = "pytorch-CycleGAN-and-pix2pix/datasets/wbc_denoising_separate"  # Temporary output root for separated A/B datasets
IMG_SIZE = 256
VAL_RATIO = 0.1

# White background threshold
BG_THRESHOLD = 245

def create_cell_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray < BG_THRESHOLD).astype(np.uint8)
    return mask

def add_vibrant_noise(
    image,
    mask,
    min_intensity=0.95,
    max_intensity=0.99,
    base_sigma=80,
    blur_ksize=1
):
    """
    Standard noise generation algorithm (Final Version).
    """
    h, w, c = image.shape
    img_f = image.astype(np.float32)

    # 1. Random density
    intensity = random.uniform(min_intensity, max_intensity)

    # 2. Gaussian colored noise
    noise = np.random.normal(0, base_sigma, (h, w, c)).astype(np.float32)

    # 3. Luminance modulation (preserve spatial structure)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mod = 0.6 + 0.8 * gray
    mod = mod[..., np.newaxis]
    noise *= mod

    # 4. Low-frequency blur
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    noise = cv2.GaussianBlur(noise, (blur_ksize, blur_ksize), 0)

    # 5. Density control
    prob_map = np.random.rand(h, w)
    apply_mask = (prob_map < intensity) & (mask == 1)

    # 6. Additive blending
    noisy = img_f.copy()
    noisy[apply_mask] += 1.7 * noise[apply_mask]

    # 7. Darken cell region (important for realistic appearance)
    noisy[mask == 1] *= 0.7

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def process_single(args):
    src_path, dest_root, split_type = args
    # split_type: 'train', 'val', 'test'
    
    try:
        # 1. Read image
        img_bgr = cv2.imread(src_path)
        if img_bgr is None:
            return
        
        # 2. Resize image
        img_bgr = cv2.resize(
            img_bgr,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )
        
        # 3. Create Clean image (B) - keep original image
        img_clean = img_bgr
        
        # 4. Create Noisy image (A) - apply synthetic noise
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = create_cell_mask(img_rgb)
        img_noisy_rgb = add_vibrant_noise(img_rgb, mask)
        img_noisy = cv2.cvtColor(img_noisy_rgb, cv2.COLOR_RGB2BGR)
        
        # 5. Save into separate A and B directories
        # Keep filenames identical for paired training
        filename = os.path.basename(src_path)
        
        # Folder A (Noisy)
        path_A = os.path.join(dest_root, f"{split_type}A", filename)
        cv2.imwrite(path_A, img_noisy)
        
        # Folder B (Clean)
        path_B = os.path.join(dest_root, f"{split_type}B", filename)
        cv2.imwrite(path_B, img_clean)
        
    except Exception as e:
        print(f"Error processing {src_path}: {e}")

def main():
    print("Starting creation of separated A/B datasets...")
    
    # 1. Collect source images
    all_files = glob.glob(
        os.path.join(SOURCE_CLEAN_DIR, "**/*.jpg"),
        recursive=True
    )
    if not all_files:
        print("No source images found.")
        return
        
    # 2. Train / Validation split
    random.seed(42)
    random.shuffle(all_files)
    val_count = int(len(all_files) * VAL_RATIO)
    
    train_files = all_files[val_count:]
    val_files = all_files[:val_count]
    
    # 3. Create A/B directory structure
    # Structure: trainA, trainB, valA, valB, testA, testB
    subfolders = ['trainA', 'trainB', 'valA', 'valB', 'testA', 'testB']
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    
    for sub in subfolders:
        os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)
        
    # 4. Prepare tasks
    tasks = []
    for f in train_files:
        tasks.append((f, OUTPUT_ROOT, 'train'))
    for f in val_files:
        tasks.append((f, OUTPUT_ROOT, 'val'))
    for f in val_files:
        tasks.append((f, OUTPUT_ROOT, 'test'))  # Temporarily reuse val set as test
    
    # 5. Run processing
    print(f"⚡ Processing {len(tasks)} images...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_single, tasks), total=len(tasks)))
        
    print("\nDONE! Output directory created at:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
