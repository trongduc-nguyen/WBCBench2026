import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import timm  
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
import random
import warnings
from PIL import Image, ImageFile
import glob
import matplotlib.pyplot as plt
import seaborn as sns

ImageFile.LOAD_TRUNCATED_IMAGES = True 
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
CONFIG = {
    "data_dir": "dataset_merged_cropped", 
    "model_name": "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    "img_size": 224,
    "num_classes": 13,
    "batch_size": 128,       
    "epochs": 20,           
    "lr": 1e-4,             
    "n_folds": 5,
    "seed": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "output_dir": "checkpoints_final_submission_swin",
    "log_dir": "logs_and_visuals" # Thư mục mới cho paper
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['log_dir'], exist_ok=True)

# --- UTILS ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])

CLASSES = sorted(['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY'])
cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
idx_to_cls = {i: c for i, c in enumerate(CLASSES)}

# --- VISUALIZATION HELPERS ---
def save_confusion_matrix(y_true, y_pred, fold, epoch):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Fold {fold} Epoch {epoch}')
    plt.savefig(f"{CONFIG['log_dir']}/cm_fold{fold}_epoch{epoch}.png")
    plt.close()

def save_training_curves(history, fold):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='tab:red', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('F1 Score', color='tab:blue')
    ax2.plot(epochs, history['val_f1'], label='Val Macro-F1', color='tab:blue', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(f'Training Metrics - Fold {fold}')
    fig.tight_layout()
    plt.savefig(f"{CONFIG['log_dir']}/curves_fold{fold}.png")
    plt.close()

# --- 2. LOSS FUNCTION ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()

# --- 3. DATASET ---
class WBCDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples 
        self.transform = transform
        
    def __len__(self): return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        label = int(label)
        img_id = os.path.basename(path) 
        try:
            img = Image.open(path).convert('RGB')
            img = np.array(img)
        except Exception as e:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            
        if self.transform:
            img = self.transform(image=img)['image']
        return img, torch.tensor(label, dtype=torch.long), img_id

def get_data_from_folder(root_dir):
    all_samples = []
    print(f"🔍 Scanning data from: {root_dir}")
    subdirs = ['train', 'val']
    for subdir in subdirs:
        full_path = os.path.join(root_dir, subdir)
        if not os.path.exists(full_path): continue
        for cls in CLASSES:
            cls_dir = os.path.join(full_path, cls)
            if not os.path.isdir(cls_dir): continue
            files = glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.png"))
            for f in files: 
                all_samples.append((f, int(cls_to_idx[cls])))
    return all_samples 

def get_transforms(mode='train'):
    if mode == 'train':
        return A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(), 
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']), 
            A.Normalize(), 
            ToTensorV2()
        ])

# --- 4. ENGINE ---
def evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs) 
            preds = torch.argmax(outputs, dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.numpy())
    return labels_all, preds_all

def train_pipeline():
    samples_list = get_data_from_folder(CONFIG['data_dir'])
    if not samples_list: return

    labels = np.array([s[1] for s in samples_list])
    samples_obj = np.array(samples_list, dtype=object) 
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(samples_obj, labels)):
        print(f"\n🚀 STARTING FOLD {fold}")
        
        history = {'train_loss': [], 'val_f1': []}
        train_sub, val_sub = samples_obj[train_idx], samples_obj[val_idx]
        train_ds = WBCDataset(train_sub, transform=get_transforms('train'))
        val_ds = WBCDataset(val_sub, transform=get_transforms('val'))

        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels, minlength=CONFIG['num_classes'])
        weights = 1. / (np.log1p(class_counts) + 1e-5)
        samples_weights = torch.from_numpy(np.array([weights[t] for t in train_labels]))
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=sampler, num_workers=CONFIG['num_workers'])
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'] * 2, shuffle=False, num_workers=CONFIG['num_workers'])

        model = timm.create_model(CONFIG['model_name'], pretrained=True, num_classes=CONFIG['num_classes']).to(CONFIG['device'])
        
        cls_weights = 1.0 / (np.log1p(class_counts) + 1e-5)
        cls_weights = (torch.FloatTensor(cls_weights).to(CONFIG['device']) / cls_weights.sum()) * CONFIG['num_classes']
        criterion = FocalLoss(alpha=cls_weights, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
        scaler = GradScaler()
        
        best_f1 = 0.0

        for epoch in range(1, CONFIG['epochs'] + 1):
            model.train()
            total_loss = 0
            for imgs, lbls, _ in tqdm(train_loader, desc=f"Fold {fold} Ep {epoch}", leave=False):
                imgs, lbls = imgs.to(CONFIG['device']), lbls.to(CONFIG['device'])
                with autocast():
                    outputs = model(imgs) 
                    loss = criterion(outputs, lbls)
                optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                total_loss += loss.item()
            
            scheduler.step()
            v_true, v_pred = evaluate(model, val_loader, CONFIG['device'])
            val_f1 = f1_score(v_true, v_pred, average='macro')
            
            # Save history for curves
            history['train_loss'].append(total_loss/len(train_loader))
            history['val_f1'].append(val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), f"{CONFIG['output_dir']}/fold{fold}_best.pth")
                # Lưu CM của model tốt nhất
                save_confusion_matrix(v_true, v_pred, fold, epoch)
            
            print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | F1: {val_f1:.4f}")

        # Kết thúc fold, lưu training curves
        save_training_curves(history, fold)
        pd.DataFrame(history).to_csv(f"{CONFIG['log_dir']}/history_fold{fold}.csv", index=False)

if __name__ == "__main__":
    train_pipeline()