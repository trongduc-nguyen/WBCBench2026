import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModel
from torch.utils.data import Dataset, DataLoader

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') 


MODEL_ID = "./pretrained/medsiglip-448"
DATA_ROOT = "dataset_merged_cropped" 
TRAIN_VAL_DIRS = [
    os.path.join(DATA_ROOT, "train"), 
    os.path.join(DATA_ROOT, "val")
]
TEST_DIR = os.path.join(DATA_ROOT, "test_images")

TARGET_CLASSES = [
    'BA', 'BL', 'BNE', 'EO', 'LY',
    'MMY', 'MO', 'MY', 'PC',
    'PLY', 'PMY', 'SNE', 'VLY'
]

OUTPUT_DIR = "embeddings_medsiglip_final"
BATCH_SIZE = 128 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def resize_tf(image):

    img_array = np.array(image)
    
    resized = tf.image.resize(
        images=img_array, 
        size=[448, 448], 
        method='bilinear', 
        antialias=False
    )
    
    return Image.fromarray(resized.numpy().astype(np.uint8))

# ================= DATASET =================
class MedSigLIPDataset(Dataset):
    def __init__(self, image_list, processor):
        self.image_list = image_list
        self.processor = processor

    def __len__(self): 
        return len(self.image_list)

    def __getitem__(self, idx):
        path, name = self.image_list[idx]
        try:
            image = Image.open(path).convert("RGB")
            
            image_resized = resize_tf(image)
            

            inputs = self.processor(
                images=image_resized, 
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                "pixel_values": inputs.pixel_values.squeeze(0), 
                "name": name, 
                "valid": True
            }
        except Exception as e:
            return {"valid": False}

# ================= ENGINE =================
def extract_embeddings(model, processor, image_list, save_path, desc="Processing"):
    if len(image_list) == 0:
        print(f"⚠️ Không có ảnh nào cho {desc}.")
        return

    dataset = MedSigLIPDataset(image_list, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    
    all_embeddings = []
    all_names = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            if not torch.any(batch["valid"]): continue
            
            pixel_values = batch["pixel_values"][batch["valid"]].to(DEVICE)
            

            embeddings = model.get_image_features(pixel_values=pixel_values)
            
            # --- NORMALIZATION ---

            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
            
            all_embeddings.append(embeddings.cpu().numpy())
            
            names = np.array(batch["name"])
            all_names.extend(names[batch["valid"].cpu().numpy()])
            
    if len(all_embeddings) > 0:
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        final_names = np.array(all_names)

        np.savez_compressed(save_path, image_names=final_names, embeddings=final_embeddings)
        print(f"💾 Saved {len(final_embeddings)} embeddings -> {save_path}")
    else:
        print(f"❌ Failed to extract for {desc}")

def main():
    print(f"🏗️ Loading AutoModel: {MODEL_ID}...")
    
    # Load Model & Processor
    # processor = AutoProcessor.from_pretrained(MODEL_ID)
    # model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)

    print("✅ Model loaded successfully.")

    # 1. TRAIN + VAL 
    print("\n🚀 GENERATING TRAIN+VAL EMBEDDINGS...")
    for cls_name in TARGET_CLASSES:
        image_list = []
        for d in TRAIN_VAL_DIRS:
            cls_folder = os.path.join(d, cls_name)
            if not os.path.exists(cls_folder): continue
            
            files = glob.glob(os.path.join(cls_folder, "*.jpg")) + \
                    glob.glob(os.path.join(cls_folder, "*.png")) + \
                    glob.glob(os.path.join(cls_folder, "*.bmp"))
            
            for fpath in files:
                image_list.append((fpath, os.path.basename(fpath)))
        
        save_path = os.path.join(OUTPUT_DIR, f"{cls_name}.npz")
        extract_embeddings(model, processor, image_list, save_path, desc=f"Class {cls_name}")

    # 2. TEST 
    print("\n🚀 GENERATING TEST EMBEDDINGS...")
    test_images = []
    if os.path.exists(TEST_DIR):
        files = glob.glob(os.path.join(TEST_DIR, "*.jpg")) + \
                glob.glob(os.path.join(TEST_DIR, "*.png")) + \
                glob.glob(os.path.join(TEST_DIR, "*.bmp"))
        
        for fpath in files:
            test_images.append((fpath, os.path.basename(fpath)))
            
        save_path = os.path.join(OUTPUT_DIR, "TEST.npz")
        extract_embeddings(model, processor, test_images, save_path, desc="Test Images")
    else:
        print(f"Test dir not found: {TEST_DIR}")

    print("\n DONE! All embeddings saved.")

if __name__ == "__main__":
    main()