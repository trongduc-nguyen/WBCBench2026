import os
import shutil
from glob import glob
from collections import defaultdict

# --- CONFIGURATION ---

# Source datasets (same structure)
DATASET_CLEAN = "wbc_dataset_clean"
DATASET_GAN_CLEAN = "dataset_gan_clean"

# Output merged dataset
MERGED_OUTPUT = "dataset_merged"


IMAGE_EXT = ".jpg"

# --- HELPER FUNCTIONS ---



def copy_with_rename(src_path, dst_path):
    """
    Copy file from src to dst.
    """
    if not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)
    else:
        base, ext = os.path.splitext(dst_path)
        new_path = base + ext
        shutil.copy2(src_path, new_path)

def collect_image_counts(root_dir):
    """
    Count number of images per subfolder (relative path).
    """
    counts = defaultdict(int)
    files = glob(os.path.join(root_dir, "**", f"*{IMAGE_EXT}"), recursive=True)

    for f in files:
        rel_dir = os.path.relpath(os.path.dirname(f), root_dir)
        counts[rel_dir] += 1

    return counts



def merge_datasets():
    print("🚀 Starting dataset merge...")
    print(f"   Source A: {DATASET_CLEAN}")
    print(f"   Source B: {DATASET_GAN_CLEAN}")
    print(f"   Output  : {MERGED_OUTPUT}")

    if os.path.exists(MERGED_OUTPUT):
        shutil.rmtree(MERGED_OUTPUT)
    os.makedirs(MERGED_OUTPUT, exist_ok=True)

    # Merge dataset_clean
    files_clean = glob(os.path.join(DATASET_CLEAN, "**", f"*{IMAGE_EXT}"), recursive=True)
    for src in files_clean:
        rel_path = os.path.relpath(src, DATASET_CLEAN)
        dst = os.path.join(MERGED_OUTPUT, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    # Merge dataset_gan_clean
    files_gan = glob(os.path.join(DATASET_GAN_CLEAN, "**", f"*{IMAGE_EXT}"), recursive=True)
    for src in files_gan:
        rel_path = os.path.relpath(src, DATASET_GAN_CLEAN)
        dst = os.path.join(MERGED_OUTPUT, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        copy_with_rename(src, dst)

    print("✅ Dataset merge completed.\n")



def main():
    merge_datasets()

if __name__ == "__main__":
    main()
