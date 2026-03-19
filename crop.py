import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= CONFIGURATION =================
# Root directory containing the dataset
# Structure example: train/CLASS/..., val/CLASS/..., test_images/...
INPUT_ROOT = "dataset_merged"

# Output directory (will preserve the same directory tree structure)
OUTPUT_ROOT = "dataset_merged_cropped"

# Target image size
TARGET_SIZE = (224, 224)
# =================================================


def crop_symmetric_and_resize(image_path, output_path, target_size):
    """
    Image processing function:
    Symmetrically crop maximum white borders and resize with high-quality interpolation.
    """
    try:
        # 1. Read image
        img_color = cv2.imread(image_path)
        if img_color is None:
            return False
        
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        H, W = img_gray.shape

        # 2. Find non-white pixels (not equal to pure white)
        # Use threshold < 250 to better suppress faint white noise if present
        non_white_pixels = np.argwhere(img_gray < 250)

        if len(non_white_pixels) == 0:
            # Fully white image -> fallback to resizing original image
            h_crop = 0
        else:
            # 3. Compute bounding box of foreground
            y_min, y_max = non_white_pixels[:, 0].min(), non_white_pixels[:, 0].max()
            x_min, x_max = non_white_pixels[:, 1].min(), non_white_pixels[:, 1].max()

            # 4. Compute distances from bounding box to image borders
            dist_top = y_min
            dist_bottom = H - 1 - y_max
            dist_left = x_min
            dist_right = W - 1 - x_max

            # 5. Use the minimum distance to ensure symmetric cropping
            h_crop = min(dist_top, dist_bottom, dist_left, dist_right)

        # 6. Perform cropping
        y_start = max(0, h_crop)
        y_end = min(H, H - h_crop)
        x_start = max(0, h_crop)
        x_end = min(W, W - h_crop)

        if y_start >= y_end or x_start >= x_end:
            # Fallback if cropping computation fails
            cropped_img = img_color
        else:
            cropped_img = img_color[y_start:y_end, x_start:x_end]

        # 7. Resize (Lanczos4 gives best quality for downscaling)
        resized_img = cv2.resize(
            cropped_img, target_size, interpolation=cv2.INTER_LANCZOS4
        )

        # 8. Save image
        cv2.imwrite(output_path, resized_img)
        return True

    except Exception as e:
        print(f"\n[Exception] {image_path}: {e}")
        return False


def main():
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ Input directory not found: {INPUT_ROOT}")
        return

    print(f"🚀 Starting processing from: {INPUT_ROOT}")
    print(f"📂 Output will be saved to: {OUTPUT_ROOT}")

    # Count total files to make progress bar accurate
    total_files = sum([len(files) for r, d, files in os.walk(INPUT_ROOT)])
    processed_count = 0
    
    # Use os.walk to recursively traverse all subdirectories
    with tqdm(total=total_files, desc="Processing") as pbar:
        for root, dirs, files in os.walk(INPUT_ROOT):
            # root: current directory being traversed
            # e.g., dataset_merged/train/BA
            
            # Compute relative path to reconstruct directory structure in output
            # e.g., rel_path = train/BA
            rel_path = os.path.relpath(root, INPUT_ROOT)
            
            # Create corresponding output directory
            # e.g., dataset_merged_cropped/train/BA
            current_output_dir = os.path.join(OUTPUT_ROOT, rel_path)
            os.makedirs(current_output_dir, exist_ok=True)

            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(current_output_dir, file)
                    
                    if crop_symmetric_and_resize(
                        input_path, output_path, TARGET_SIZE
                    ):
                        processed_count += 1
                    
                    pbar.update(1)

    print("\n" + "=" * 50)
    print("✅ COMPLETED!")
    print(f"   Processed: {processed_count}/{total_files} images.")
    print(f"   Output dataset located at: {OUTPUT_ROOT}")
    print("=" * 50)


if __name__ == "__main__":
    main()
