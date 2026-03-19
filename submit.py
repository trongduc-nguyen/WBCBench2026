import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from transformers import AutoModelForImageClassification
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
import glob
import torch.nn as nn
from scipy.spatial.distance import mahalanobis
# --- CONFIGURATION ---
CONFIG = {
    "checkpoint_dir": "checkpoints_final_submission_swin",
    "checkpoint_mode": "best", 
    "model_name": "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    
    "img_size": 224,
    "num_classes": 13,
    "batch_size": 64, 
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
    "test_dir": "dataset_merged_cropped/test_images",
    # orignal index file for testing
    "sample_sub": "pretrained/phase2_test.csv",

}

# Class Mapping
CLASSES = sorted(['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY'])
idx_to_class = {i: c for i, c in enumerate(CLASSES)}
class_to_idx = {c: i for i, c in enumerate(CLASSES)}


# --- HELPER FUNCTIONS ---

def get_checkpoint_paths(config):
    dir_path = config['checkpoint_dir']
    mode = config['checkpoint_mode']
    if not os.path.exists(dir_path): raise FileNotFoundError(f"Not found: {dir_path}")
    search_pattern = os.path.join(dir_path, f"fold*_{mode}.pth")
    paths = glob.glob(search_pattern)
    paths.sort()
    if not paths: raise FileNotFoundError(f"Not found: {search_pattern}")
    print(f"Found {len(paths)} checkpoints in '{dir_path}'")
    return paths

def load_model(model_name, num_classes, device):
    if "facebook/" in model_name:
        model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.to(device)
    model.eval()
    return model

# --- DATASET ---
class TTATestDataset(Dataset):
    def __init__(self, df, img_dir, img_size):
        self.df = df
        self.img_dir = img_dir
        self.img_size = img_size
        self.base_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['ID']
        path = os.path.join(self.img_dir, img_id)
        try:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        images_tta = []
        def add_aug(aug_img): images_tta.append(self.base_transform(image=aug_img)['image'])
        
        # 8 TTA variants
        add_aug(image)
        add_aug(cv2.flip(image, 1))
        add_aug(cv2.flip(image, 0))
        add_aug(cv2.flip(cv2.flip(image, 1), 0))
        add_aug(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
        add_aug(cv2.rotate(image, cv2.ROTATE_180))
        add_aug(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
        add_aug(cv2.transpose(image))
        return torch.stack(images_tta)




_EMBEDDING_CONTEXT = {
    "model": None,
    "name_to_emb": None,
    "device": None
}

class AdvancedContrastiveHead(nn.Module):
    def __init__(self, input_dim, hidden_dim_1=1024, hidden_dim_2=512, output_dim=256, num_classes=13, dropout_rate=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LayerNorm(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.projection = nn.Linear(hidden_dim_2, output_dim)
        self.classifier = nn.Linear(hidden_dim_2, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        proj = F.normalize(self.projection(feat), p=2, dim=1)
        logits = self.classifier(feat)
        return proj, logits

def infer_embedding_model(candidate_ids, target_class_name):
    """
    Predict the probability of the target class for a list of image IDs.

    Args:
        candidate_ids (list): List of image file names (e.g., ['img1.jpg', 'img2.jpg'])
        target_class_name (str): Name of the target class to retrieve the probability

    Returns:
        dict: {image_id: probability}
    """

    EMB_PATH = "embeddings_medsiglip_final/TEST.npz"
    MODEL_PATH = "pretrained/best_model_contrastive_head.pth" 
    
    CLASSES = sorted(['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY'])
    
    if _EMBEDDING_CONTEXT["model"] is None:
        print(f"[Embedding] Loading Model & Embeddings into Memory...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _EMBEDDING_CONTEXT["device"] = device
        
        # A. Load Embeddings (.npz)
        if not os.path.exists(EMB_PATH):
            print(f"[Embedding] Not found: {EMB_PATH}")
            return {}
        
        data = np.load(EMB_PATH)
        test_embs = data['embeddings']      # Shape: (N, Dim)
        test_names = data['image_names']    # Shape: (N,)
        

        _EMBEDDING_CONTEXT["name_to_emb"] = {
            os.path.basename(n): test_embs[i] for i, n in enumerate(test_names)
        }
        
        # B. Load Model Weights (.pth)
        input_dim = test_embs.shape[1]
        model = AdvancedContrastiveHead(input_dim=input_dim).to(device)
        
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            _EMBEDDING_CONTEXT["model"] = model
            print("[Embedding] Model Loaded Successfully.")
        else:
            print(f"[Embedding] Not found checkpoint: {MODEL_PATH}")
            return {}

    model = _EMBEDDING_CONTEXT["model"]
    device = _EMBEDDING_CONTEXT["device"]
    name_to_emb = _EMBEDDING_CONTEXT["name_to_emb"]
    
    try:
        target_idx = CLASSES.index(target_class_name)
    except ValueError:
        print(f"Class '{target_class_name}' invalid.")
        return {}

    # Filter
    valid_ids = []
    vectors = []
    
    results = {}
    
    for img_id in candidate_ids:
        if img_id in name_to_emb:
            valid_ids.append(img_id)
            vectors.append(name_to_emb[img_id])
        else:

            results[img_id] = -1.0
            
    if not valid_ids:
        return results

    # 3. Inference (Batch Processing)
    input_tensor = torch.tensor(np.array(vectors), dtype=torch.float32).to(device)
    
    
    with torch.no_grad():
        _, logits = model(input_tensor)
        probs = F.softmax(logits, dim=1) # [Batch, Num_Classes]

        target_probs = probs[:, target_idx].cpu().numpy()
        
    for i, img_id in enumerate(valid_ids):
        results[img_id] = float(target_probs[i])
        
    return results

def get_spikiness_score(img_path):
    """
    Compute the 'Spikiness' index.

    Logic: Max_Distance / Median_Distance from the center.
    """

    if not os.path.exists(img_path): return 0.0
    
    img = cv2.imread(img_path)
    if img is None: return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50: return 0.0
    
    M = cv2.moments(cnt)
    if M["m00"] == 0: return 0.0
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    pts = cnt.squeeze()
    if len(pts.shape) < 2: return 0.0
    
    diff = pts - np.array([cx, cy])
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    if len(distances) == 0: return 0.0
    
    max_dist = np.max(distances)
    median_dist = np.median(distances)
    
    # Higher score -> more distorted / more buds
    return max_dist / (median_dist + 1e-6)
def refine_ply_candidates(candidate_ids, cnn_df):
    """
    PLY Branch Refinement: Morphological distinction based on Spikiness Score.
    
    Logic & Biological Reasoning:
    1. Scope: Only targets 'LY' predictions as this class is most frequently 
       confused with PLY due to their similar mononuclear structure.
    2. Confidence Gate: Refines only low-confidence cases 
       where the CNN model shows uncertainty.
    3. Trait Analysis: Based on training set observations, 'protrusion' or 'blebbing' 
       is a key discriminant feature. Mature LYs are typically smooth/circular, 
       while PLY/VLY exhibit irregular boundaries.
    """
    final_ply_ids = []
    
    subset = cnn_df[cnn_df['ID'].isin(candidate_ids)].copy()
    

    suspicious_mask = (subset['cnn_pred'] == 'LY') & (subset['cnn_max_prob'] < 0.8)
    
    target_ids = subset[suspicious_mask]['ID'].tolist()
    
    
    SPIKE_THRESHOLD = 1.35 
    
    count_pass = 0
    test_dir = CONFIG['test_dir'] 
    
    for img_id in target_ids:
        path = os.path.join(test_dir, img_id)
        
        score = get_spikiness_score(path)
        
        if score > SPIKE_THRESHOLD:
            final_ply_ids.append(img_id)
            count_pass += 1
                 
    return final_ply_ids

def get_features_for_pc(img_path):
    """
    Specialized feature extraction for Plasma Cells (PC) using K-Means clustering.
    
    Logic:
    1. Nuclear Segmentation: Employs K-Means (K=2) to robustly separate the 
       dense nucleus from the cytoplasm based on intensity distributions.
    2. Feature Computation: Calculates area-based (NC Ratio), intensity-based, 
       and spatial (Offset/Centroid deviation) metrics.
    3. Output: Returns a dictionary of features compatible with the downstream 
       Box Filter and Mahalanobis refinement modules.
    """
    if not os.path.exists(img_path): return None
    img = cv2.imread(img_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Masking
    _, cell_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    if cv2.countNonZero(cell_mask) == 0: return None
    cell_pixels = gray[cell_mask == 255]
    if len(cell_pixels) < 10: return None
    
    # K-Means Clustering
    Z = np.float32(cell_pixels.reshape((-1, 1)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    c0, c1 = float(center[0]), float(center[1])
    kmeans_thresh = (c0 + c1) / 2
    
    _, nuc_mask = cv2.threshold(gray, kmeans_thresh, 255, cv2.THRESH_BINARY_INV)
    nuc_mask = cv2.bitwise_and(nuc_mask, cell_mask) 
    nuc_area = cv2.countNonZero(nuc_mask)
    cell_area = cv2.countNonZero(cell_mask)
    
    # Features
    nc_ratio = nuc_area / cell_area if cell_area > 0 else 0
    mean_int = np.mean(cell_pixels)
    std_int = np.std(cell_pixels) 
    
    img_f = img.astype(np.float32)
    b, g, r = cv2.split(img_f)
    rb_ratio = np.mean(r[cell_mask==255]) / (np.mean(b[cell_mask==255]) + 1e-6)
    
    # Offset
    M_c = cv2.moments(cell_mask)
    M_n = cv2.moments(nuc_mask)
    if M_c['m00'] == 0 or M_n['m00'] == 0: norm_offset = 0
    else:
        cx_c, cy_c = int(M_c['m10']/M_c['m00']), int(M_c['m01']/M_c['m00'])
        cx_n, cy_n = int(M_n['m10']/M_n['m00']), int(M_n['m01']/M_n['m00'])
        offset = np.sqrt((cx_c - cx_n)**2 + (cy_c - cy_n)**2)
        norm_offset = offset / (np.sqrt(cell_area / np.pi) + 1e-6)
        
    # Cyto Int
    cyto_mask = cv2.bitwise_xor(cell_mask, nuc_mask)
    cyto_int = np.mean(gray[cyto_mask==255]) if cv2.countNonZero(cyto_mask)>0 else 0

    return {
        'Area': cell_area,
        'Intensity': mean_int,
        'StdDev': std_int, 
        'RB_Ratio': rb_ratio,
        'NC_Ratio': nc_ratio,
        'Offset': norm_offset,
        'Cyto_Int': cyto_int
    }


_PC_FILTER_PARAMS = None

def refine_pc_candidates(candidate_ids, cnn_df):
    """
    PC Branch Refinement: Adaptive Divide & Conquer strategy for OOD samples.
    
    1. Misclassification Analysis: Validation results indicate that Plasma Cells (PC) 
       often overlap morphologically with 'LY' and 'VLY' (eccentric mononuclear cells).
    2. Adaptive Filtering: Given the Out-of-Distribution (OOD) nature of PC samples, 
       a single-model approach is insufficient. We divide candidates into sub-clusters 
       to enhance sensitivity:
       - LY-group Filter: Specialized for PC samples with lymphocyte-like features.
       - VLY-group Filter: Specialized for PC samples with variant-like features.

    """
    global _PC_FILTER_PARAMS
    final_pc_ids = []
    

    PARAM_FILE = "pretrained/params.npz" 
    
    if _PC_FILTER_PARAMS is None:
        if not os.path.exists(PARAM_FILE):
            print(f"Warning: Missing {PARAM_FILE}. Skipping PC refinement.")
            return []
        
        try:
            data = np.load(PARAM_FILE, allow_pickle=True)
            _PC_FILTER_PARAMS = {
                "LY": {
                    "mean_vec": data["LY_mean_vec"],
                    "inv_cov": data["LY_inv_cov"],
                    "thresh": float(data["LY_thresh"]),
                    "box": data["LY_box"].item()
                },
                "VLY": {
                    "mean_vec": data["VLY_mean_vec"],
                    "inv_cov": data["VLY_inv_cov"],
                    "thresh": float(data["VLY_thresh"]),
                    "box": data["VLY_box"].item()
                }
            }
        except Exception as e:
            print(f"Error loading {PARAM_FILE}: {e}")
            return []

    subset = cnn_df[cnn_df['ID'].isin(candidate_ids)].copy()
    
    valid_classes = ['LY', 'VLY']
    target_subset = subset[subset['cnn_pred'].isin(valid_classes)]
    target_ids = target_subset['ID'].tolist()
    id_to_group = target_subset.set_index('ID')['cnn_pred'].to_dict()
    PC_RAW_IMG_DIR = "dataset_merged/test_images" 
    maha_cols = ['Area', 'Intensity', 'RB_Ratio', 'NC_Ratio', 'Offset', 'Cyto_Int']
    
    count_pass = 0
    
    for img_id in target_ids:
        group = id_to_group.get(img_id)
        if group not in _PC_FILTER_PARAMS: continue 
        params = _PC_FILTER_PARAMS[group]
        mean_vec = params["mean_vec"]
        inv_cov = params["inv_cov"]
        maha_thresh = params["thresh"]
        box = params["box"]
        
        path = os.path.join(PC_RAW_IMG_DIR, img_id)
        ft = get_features_for_pc(path) 
        
        if ft:
            passed_box = True
            if not (box['Area_min'] <= ft['Area'] <= box['Area_max']): passed_box = False
            if not (box['Intensity_min'] <= ft['Intensity'] <= box['Intensity_max']): passed_box = False
            if not (box['StdDev_min'] <= ft['StdDev'] <= box['StdDev_max']): passed_box = False
            if not (box['RB_Ratio_min'] <= ft['RB_Ratio'] <= box['RB_Ratio_max']): passed_box = False
            if not (box['NC_Ratio_min'] <= ft['NC_Ratio'] <= box['NC_Ratio_max']): passed_box = False
            
            if passed_box:
                # B. Mahalanobis Filter 
                vec = np.array([ft[c] for c in maha_cols])
                dist = mahalanobis(vec, mean_vec, inv_cov)
                
                if dist <= maha_thresh:
                    final_pc_ids.append(img_id)
                    count_pass += 1
    return final_pc_ids
# --- MAIN INFERENCE MODIFIED ---
def run_inference():
    print(f"\nSTARTING TTA ENSEMBLE INFERENCE WITH BOOST & EMBEDDING FILTER")
    
    # --- PHASE 1: CNN PREDICTION ---
    try:
        model_paths = get_checkpoint_paths(CONFIG)
    except Exception as e:
        print(e); return

    models = []
    for path in model_paths:
        m = load_model(CONFIG['model_name'], CONFIG['num_classes'], CONFIG['device'])
        checkpoint = torch.load(path, map_location=CONFIG['device'])
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        try: m.load_state_dict(state_dict)
        except: m.load_state_dict(state_dict, strict=False)
        models.append(m)
    
    if os.path.exists(CONFIG['test_dir']):
        test_files = glob.glob(os.path.join(CONFIG['test_dir'], "*"))
        df = pd.DataFrame({'ID': [os.path.basename(f) for f in test_files]})
    else:
        df = pd.read_csv(CONFIG['sample_sub'])

    dataset = TTATestDataset(df, CONFIG['test_dir'], CONFIG['img_size'])
    dataloader = DataLoader(dataset, batch_size=max(1, CONFIG['batch_size']//8), shuffle=False, num_workers=4)
    
    all_probs = []
    
    with torch.no_grad():
        for imgs_tta in tqdm(dataloader, desc="CNN Inferencing"):
            B, N_TTA, C, H, W = imgs_tta.shape
            inputs = imgs_tta.view(-1, C, H, W).to(CONFIG['device'])
            batch_ensemble_probs = torch.zeros((B, CONFIG['num_classes']), device=CONFIG['device'])
            
            for model in models:
                if "facebook/" in CONFIG['model_name']: outputs = model(inputs).logits
                else: outputs = model(inputs)
                probs = F.softmax(outputs, dim=1) 
                probs = probs.view(B, N_TTA, -1).mean(dim=1)
                batch_ensemble_probs += probs
            
            batch_ensemble_probs /= len(models)
            all_probs.append(batch_ensemble_probs.cpu().numpy())

    # Probabilities
    final_probs = np.concatenate(all_probs, axis=0)
    for i, cls in enumerate(CLASSES):
        df[f'prob_{cls}'] = final_probs[:, i]

    # --- PHASE 2: OVERWRITE CNN LABEL ---

    top2_vals, top2_indices = torch.tensor(final_probs).topk(k=2, dim=1)
    
    current_preds = []
    current_max_probs = []
    
    idx_pc = class_to_idx['PC']
    idx_ply = class_to_idx['PLY']
    
    for i in range(len(df)):
        top1_idx = top2_indices[i, 0].item()
        top1_prob = top2_vals[i, 0].item()
        
        final_idx = top1_idx
        final_prob = top1_prob
        
        if top1_idx == idx_pc or top1_idx == idx_ply:
            final_idx = top2_indices[i, 1].item()
            final_prob = top2_vals[i, 1].item()
            
        current_preds.append(idx_to_class[final_idx])
        current_max_probs.append(final_prob)
        
    df['cnn_pred'] = current_preds
    df['cnn_max_prob'] = current_max_probs
    
    # --- PHASE 3: BOOST & FILTER ---
    
    # Compensate for extreme class imbalance (rare classes).
    # Applied to 'surface' potential candidates for subsequent rigorous filtering.
    BOOST_PC = 3750.0
    BOOST_PLY = 150.0
    
    df['boosted_prob_PC'] = df['prob_PC'] * BOOST_PC
    df['boosted_prob_PLY'] = df['prob_PLY'] * BOOST_PLY
    
    potential_pc_mask = df['boosted_prob_PC'] > df['cnn_max_prob']
    potential_ply_mask = df['boosted_prob_PLY'] > df['cnn_max_prob']
    
    potential_pc_ids = df[potential_pc_mask]['ID'].tolist()
    potential_ply_ids = df[potential_ply_mask]['ID'].tolist()
    

    # --- PHASE 4: EMBEDDING FILTER & REFINE ---
    
    if len(potential_ply_ids) > 0:
        
        ply_emb_probs = infer_embedding_model(potential_ply_ids, "PLY")
        
        thresh_ply = 0.35
        
        filtered_ply_ids = [pid for pid in potential_ply_ids if ply_emb_probs.get(pid, 0) > thresh_ply]
        
        final_ply_ids = refine_ply_candidates(filtered_ply_ids, df)
        
        if final_ply_ids:
            df.loc[df['ID'].isin(final_ply_ids), 'cnn_pred'] = 'PLY'
        
    # --- EMBEDDING FILTER & REFINE (PROCESS PC NEXT) ---
    
    current_ply_preds = df[df['cnn_pred'] == 'PLY']['ID'].tolist()
    potential_pc_ids = [pid for pid in potential_pc_ids if pid not in current_ply_preds]
    
    if len(potential_pc_ids) > 0:
        pc_emb_probs = infer_embedding_model(potential_pc_ids, "PC")
        
        thresh_pc = 0.01
        
        filtered_pc_ids = [pid for pid in potential_pc_ids if pc_emb_probs.get(pid, 0) > thresh_pc]
        final_pc_ids = refine_pc_candidates(filtered_pc_ids, df)
        
        if final_pc_ids:
            df.loc[df['ID'].isin(final_pc_ids), 'cnn_pred'] = 'PC'
    
    # --- SAVE FINAL SUBMISSION ---
    sub_df = df[['ID', 'cnn_pred']].rename(columns={'cnn_pred': 'labels'})
    out_file = f"submission.csv"
    sub_df.to_csv(out_file, index=False)
    
    print(f"\nFinal Submission Saved: {out_file}")
    


if __name__ == "__main__":
    run_inference()