import os
import cv2
import numpy as np
import shutil
from glob import glob
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

# --- CONFIGURATION ---
INPUT_DIR = "wbc_cellpose_clean_224"   # Directory containing the current dataset
CLEAN_DIR = "wbc_dataset_clean"            # Output directory for clean images
NOISY_DIR = "wbc_dataset_noisy"            # Output directory for noisy images
NOISE_THRESHOLD = 20.0                 # Threshold for noise classification

# --- 1. NOISE SCORE CALCULATION (CORE LOGIC) ---
def calculate_noise_score(img_path):
    try:
        # Read image in grayscale for faster processing
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        
        # Mask: remove white background (pixels > 245 are treated as background)
        mask = (img < 245).astype(np.uint8)
        if np.sum(mask) == 0:
            return 0.0
        
        # Median blur to suppress salt-and-pepper noise
        img_median = cv2.medianBlur(img, 3)
        
        # Absolute difference between original and median-blurred image
        diff = cv2.absdiff(img, img_median)
        
        # Compute mean difference over the cell region only
        score = cv2.mean(diff, mask=mask)[0]
        return score

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return 0.0

# --- 2. WORKER FUNCTION (PROCESS SINGLE FILE) ---
def process_and_copy(file_path):
    # Compute noise score
    score = calculate_noise_score(file_path)
    
    # Determine destination path
    # Preserve subdirectory structure (e.g., train/BA/img.jpg)
    rel_path = os.path.relpath(file_path, INPUT_DIR)
    
    if score < NOISE_THRESHOLD:
        target_root = CLEAN_DIR
        result_type = "clean"
    else:
        target_root = NOISY_DIR
        result_type = "noisy"
        
    target_path = os.path.join(target_root, rel_path)
    
    # Create parent directories if needed
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Copy file (copy2 preserves metadata timestamps)
    shutil.copy2(file_path, target_path)
    
    return result_type

# --- 3. MAIN EXECUTION ---
def run_split():
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        return

    # Collect all image files (Train, Val, Test)
    print("Scanning image files...")
    files = glob(os.path.join(INPUT_DIR, "**/*.jpg"), recursive=True)
    print(f"Total files to process: {len(files)}")
    
    # Remove old output directories if you want to start fresh (optional)
    # shutil.rmtree(CLEAN_DIR, ignore_errors=True)
    # shutil.rmtree(NOISY_DIR, ignore_errors=True)

    print(f"Starting dataset split with Threshold = {NOISE_THRESHOLD}")
    print(f"   -> Clean  : score <  {NOISE_THRESHOLD} → {CLEAN_DIR}")
    print(f"   -> Noisy  : score >= {NOISE_THRESHOLD} → {NOISY_DIR}")

    # Parallel execution
    clean_count = 0
    noisy_count = 0
    
    # Use ProcessPoolExecutor to leverage multiple CPU cores
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(
            tqdm(executor.map(process_and_copy, files), total=len(files))
        )
        
    # Statistics
    clean_count = results.count("clean")
    noisy_count = results.count("noisy")
    
    print("\nDATASET SPLITTING COMPLETED")
    print("=" * 40)
    print(f"Clean dataset : {clean_count} images ({(clean_count / len(files) * 100):.1f}%)")
    print(f"Noisy dataset : {noisy_count} images ({(noisy_count / len(files) * 100):.1f}%)")
    print("=" * 40)
    
    # Quick check for Test subset
    test_clean = len(glob(os.path.join(CLEAN_DIR, "test_images", "*.jpg")))
    test_noisy = len(glob(os.path.join(NOISY_DIR, "test_images", "*.jpg")))
    print(
        "Test subset summary:\n"
        f"   - Clean: {test_clean}\n"
        f"   - Noisy: {test_noisy}"
    )

if __name__ == "__main__":
    run_split()
