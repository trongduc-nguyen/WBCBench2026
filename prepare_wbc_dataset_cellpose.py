import os
import cv2
import numpy as np
import pandas as pd
import torch
from cellpose import models
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageFile

# --- FIX TRUNCATED JPEG ISSUE ---
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CONFIGURATION ---
PHASE1_CSV = "dataset_wbc/phase1_label.csv"
PHASE2_TRAIN_CSV = "dataset_wbc/phase2_train.csv"
PHASE2_EVAL_CSV = "dataset_wbc/phase2_eval.csv"
TEST_CSV = "dataset_wbc/phase2_test.csv"

OUTPUT_DIR = "wbc_cellpose_clean_224"
CROP_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 8

# --- CHECK GPU ---
USE_GPU = torch.cuda.is_available()
print(f"⚙️ Hardware Config: GPU={USE_GPU}, Batch Size={BATCH_SIZE}")

# --- 1. BUILD IMAGE PATH MAP ---
def get_image_path_dict(root_dir="."):
    target_dirs = [
        "dataset_wbc/phase1",
        "dataset_wbc/phase2/train",
        "dataset_wbc/phase2/eval",
        "dataset_wbc/phase2/test"
    ]
    path_map = {}
    print(f"Indexing images from root: {os.path.abspath(root_dir)}")

    for d in target_dirs:
        full_dir = os.path.join(root_dir, d)
        if not os.path.exists(full_dir):
            continue

        files = [
            os.path.join(full_dir, f)
            for f in os.listdir(full_dir)
            if f.endswith(".jpg")
        ]

        for f in files:
            path_map[os.path.basename(f)] = f

    return path_map

PATH_MAP = get_image_path_dict()

# --- 2. LOAD AND MERGE CSV FILES ---
print("Loading phase CSV files...")

df_p1 = pd.read_csv(PHASE1_CSV)
df_p2_train = pd.read_csv(PHASE2_TRAIN_CSV)
df_p2_eval = pd.read_csv(PHASE2_EVAL_CSV)

# Keep only required columns
df_p1 = df_p1[['ID', 'labels', 'split']]
df_p2_train = df_p2_train[['ID', 'labels', 'split']]
df_p2_eval = df_p2_eval[['ID', 'labels', 'split']]

# Merge all
df_all = pd.concat([df_p1, df_p2_train, df_p2_eval], ignore_index=True)

# Map original split to train / val
def map_split(split_name):
    if split_name in ['phase1_train', 'phase2_train']:
        return 'train'
    elif split_name == 'phase2_eval':
        return 'val'
    else:
        return None

df_all['tv_split'] = df_all['split'].apply(map_split)

# Keep valid rows only
df_train = df_all[df_all['tv_split'].notnull()].reset_index(drop=True)

print("Train / Val distribution:")
print(df_train['tv_split'].value_counts())

# --- LOAD TEST CSV ---
try:
    df_test = pd.read_csv(TEST_CSV)
    print(f"Loaded Test CSV: {len(df_test)} samples.")
except Exception:
    print(f"Warning: {TEST_CSV} not found.")
    df_test = pd.DataFrame({'ID': []})

# --- 3. CREATE OUTPUT DIRECTORIES ---
splits = ['train', 'val']
classes = sorted(df_train['labels'].unique())

for sp in splits:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, sp, cls), exist_ok=True)

TEST_OUT_DIR = os.path.join(OUTPUT_DIR, "test_images")
os.makedirs(TEST_OUT_DIR, exist_ok=True)

# --- 4. LOAD CELLPOSE MODEL ---
print("Loading CellPose model...")
model = models.CellposeModel(gpu=USE_GPU, model_type='cpsam')

# --- 5. IMAGE PROCESSING LOGIC ---
def process_batch_masks(images_rgb, masks_batch):
    processed_images = []

    for i, img in enumerate(images_rgb):
        H, W = img.shape[:2]
        center_x, center_y = W // 2, H // 2
        masks = masks_batch[i]

        # If no mask detected, fall back to image center
        if masks.max() == 0:
            centroid_x, centroid_y = center_x, center_y
            mask_binary = np.ones((H, W), dtype=np.uint8)
        else:
            unique_masks = np.unique(masks)
            unique_masks = unique_masks[unique_masks != 0]

            best_dist = float('inf')
            for m in unique_masks:
                ys, xs = np.where(masks == m)
                if len(xs) == 0:
                    continue
                cy, cx = np.mean(ys), np.mean(xs)
                dist = (cx - center_x) ** 2 + (cy - center_y) ** 2
                if dist < best_dist:
                    best_dist = dist
                    centroid_x, centroid_y = cx, cy
                    target_mask = m

            mask_binary = (masks == target_mask).astype(np.uint8)

        # Apply white background
        white_bg = np.ones_like(img) * 255
        clean_img = np.where(mask_binary[..., None] == 1, img, white_bg)

        # Crop around centroid
        cx, cy = int(centroid_x), int(centroid_y)
        half = CROP_SIZE // 2
        x1, y1 = cx - half, cy - half
        x2, y2 = x1 + CROP_SIZE, y1 + CROP_SIZE

        pad_top = max(0, -y1)
        pad_left = max(0, -x1)
        pad_bottom = max(0, y2 - H)
        pad_right = max(0, x2 - W)

        if any([pad_top, pad_bottom, pad_left, pad_right]):
            clean_img = cv2.copyMakeBorder(
                clean_img,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            )
            x1 += pad_left
            y1 += pad_top

        crop = clean_img[y1:y1 + CROP_SIZE, x1:x1 + CROP_SIZE]

        # Safety resize
        if crop.shape[:2] != (CROP_SIZE, CROP_SIZE):
            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

        processed_images.append(crop)

    return processed_images

def save_worker(args):
    img_rgb, save_path = args
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)

# --- 6. MAIN LOOP ---
def run():
    data_items = []

    print("Adding Train / Val images...")
    for _, row in df_train.iterrows():
        src = PATH_MAP.get(row['ID'])
        if not src:
            continue

        dst = os.path.join(
            OUTPUT_DIR,
            row['tv_split'],
            row['labels'],
            row['ID']
        )

        if not os.path.exists(dst):
            data_items.append({'src': src, 'dst': dst})

    print("Adding Test images...")
    for _, row in df_test.iterrows():
        src = PATH_MAP.get(row['ID'])
        if not src:
            continue

        dst = os.path.join(TEST_OUT_DIR, row['ID'])
        if not os.path.exists(dst):
            data_items.append({'src': src, 'dst': dst})

    if not data_items:
        print("All images have already been processed.")
        return

    print(f"Processing {len(data_items)} images...")
    io_executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

    for i in tqdm(range(0, len(data_items), BATCH_SIZE)):
        batch = data_items[i:i + BATCH_SIZE]

        batch_imgs = []
        valid_indices = []

        for idx, item in enumerate(batch):
            try:
                with Image.open(item['src']) as img:
                    img = img.convert("RGB")
                    batch_imgs.append(np.array(img))
                    valid_indices.append(idx)
            except Exception:
                continue

        if not batch_imgs:
            continue

        masks, _, _ = model.eval(
            batch_imgs,
            diameter=None,
            channels=None,
            normalize=True,
            batch_size=BATCH_SIZE
        )

        crops = process_batch_masks(batch_imgs, masks)

        save_tasks = []
        for j, crop in enumerate(crops):
            dst = batch[valid_indices[j]]['dst']
            save_tasks.append((crop, dst))

        list(io_executor.map(save_worker, save_tasks))

    io_executor.shutdown()
    print("DONE!")

if __name__ == "__main__":
    run()
