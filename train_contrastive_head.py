import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
import random
import warnings

warnings.filterwarnings("ignore")

CONFIG = {
    "emb_dir": "embeddings_medsiglip_final",
    
    # Model Params 
    "input_dim": None,     # Auto-detected
    "hidden_dim_1": 1024,  
    "hidden_dim_2": 512,   
    "output_dim": 256,     
    "num_classes": 13,     
    
    # Low Dropout
    "dropout_rate": 0.05,   
    
    # Training Params
    "batch_size": 256,     
    "epochs": 150,         
    "lr": 1e-4,            
    "weight_decay": 1e-5,  
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    "alpha": 0.3,          # Weight for SupCon Loss
    "seed": 42,
    
    "noise_level": 0.06,  # Noise injection for robustness
    
    "subsample_classes": ['SNE', 'LY'],
    "subsample_ratio": 0.5 
}

CLASSES = sorted(['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY'])
cls_to_idx = {c: i for i, c in enumerate(CLASSES)}

# OVERSAMPLING MAPPING
OVERSAMPLE_MAPPING = {
    'PC': 50,   
    'PLY': 50,  
    'MMY': 5,
    'BNE': 5
}

# ================= 1. LOSS FUNCTIONS =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask 
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = - (self.temperature / self.temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss

# ================= 2. DATASET =================
class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels, is_train=False, noise_level=0.0):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
        self.is_train = is_train
        self.noise_level = noise_level

    def __len__(self): return len(self.embeddings)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        label = self.labels[idx]
        
        # Apply noise only during training
        if self.is_train and self.noise_level > 0:
            noise = torch.randn_like(emb) * self.noise_level
            emb = emb + noise
            
        return emb, label

def load_data_only_train(emb_dir):

    all_embs = []
    all_labels = []
    
    for cls_name in CLASSES:
        path = os.path.join(emb_dir, f"{cls_name}.npz")
        if not os.path.exists(path): 
            print(f"⚠️ Warning: Missing file {path}")
            continue
            
        data = np.load(path)
        embs = data['embeddings']
        
        # A. Subsampling logic 
        if cls_name in CONFIG['subsample_classes']:
            n_keep = int(len(embs) * CONFIG['subsample_ratio'])
            indices = np.random.choice(len(embs), n_keep, replace=False)
            embs = embs[indices]
        
        # B. Oversampling logic 
        if cls_name in OVERSAMPLE_MAPPING:
            factor = OVERSAMPLE_MAPPING[cls_name]
            embs = np.tile(embs, (factor, 1)) # Duplicate N times
            print(f"   🚀 {cls_name}: Oversampled x{factor} -> {len(embs)} samples")
        else:
            if cls_name not in CONFIG['subsample_classes']:
                print(f"   ✅ {cls_name}: {len(embs)}")

        labels = np.full(len(embs), cls_to_idx[cls_name])
        
        all_embs.append(embs)
        all_labels.append(labels)

    # Concatenate all data
    X = np.concatenate(all_embs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    CONFIG['input_dim'] = X.shape[1]
    print(f"\n✅ Total Training Samples: {len(X)}")
    return X, y

# ================= 3. MODEL =================
class AdvancedContrastiveHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Deep MLP Encoder
        self.encoder = nn.Sequential(
            nn.Linear(CONFIG['input_dim'], CONFIG['hidden_dim_1']),
            nn.LayerNorm(CONFIG['hidden_dim_1']),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout_rate']), 
            
            nn.Linear(CONFIG['hidden_dim_1'], CONFIG['hidden_dim_2']),
            nn.LayerNorm(CONFIG['hidden_dim_2']),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout_rate'])
        )
        self.projection = nn.Linear(CONFIG['hidden_dim_2'], CONFIG['output_dim'])
        self.classifier = nn.Linear(CONFIG['hidden_dim_2'], CONFIG['num_classes'])

    def forward(self, x):
        feat = self.encoder(x)
        proj = F.normalize(self.projection(feat), p=2, dim=1)
        logits = self.classifier(feat)
        return proj, logits

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# ================= 4. MAIN =================
def main():
    seed_everything(CONFIG['seed'])
    
    # 1. Load Data 
    X, y = load_data_only_train(CONFIG['emb_dir'])
    
    # 2. Weights (Soft balancing calculation)
    unique, counts = np.unique(y, return_counts=True)
    weights = np.sqrt(max(counts) / counts) 
    
    # 3. Loaders
    # Train Loader: Shuffle, Noise Augmentation
    train_ds = EmbeddingsDataset(X, y, is_train=True, noise_level=CONFIG['noise_level'])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    
    # Eval Loader: No Shuffle, No Noise 
    eval_ds = EmbeddingsDataset(X, y, is_train=False)
    eval_loader = DataLoader(eval_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 4. Model Setup
    model = AdvancedContrastiveHead().to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # Losses
    criterion_ce = FocalLoss(gamma=2.0) 
    criterion_supcon = SupConLoss(temperature=0.07)
    
    best_global_acc = 0.0
    
    print(f"\nSTART TRAINING...")
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        train_loss = 0
        
        for embs, lbls in train_loader:
            embs, lbls = embs.to(CONFIG['device']), lbls.to(CONFIG['device'])
            feats, logits = model(embs)
            
            # Combined Loss
            loss = CONFIG['alpha'] * criterion_supcon(feats, lbls) + (1 - CONFIG['alpha']) * criterion_ce(logits, lbls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Evaluation Loop
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for embs, lbls in eval_loader:
                embs = embs.to(CONFIG['device'])
                _, logits = model(embs)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbls.numpy())
        
        global_acc = accuracy_score(all_labels, all_preds)
            
        print(f"Ep {epoch:02d} | Loss: {train_loss/len(train_loader):.4f} | Global Acc: {global_acc:.4f}")
        
        # Save Best Model based on Global Accuracy
        if global_acc >= best_global_acc:
            best_global_acc = global_acc
            torch.save(model.state_dict(), "pretrained/best_model_contrastive_head.pth")
            print("   SAVE BEST!")
    
    print("\nDONE.")
    


if __name__ == "__main__":
    main()