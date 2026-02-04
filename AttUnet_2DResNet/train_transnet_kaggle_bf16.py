"""
BraTS21 Training Pipeline (Train/Val/Test + Resume + Plots)
===========================================================
Optimized for Kaggle Dual T4 GPUs.

Features:
- 80/10/10 Data Split
- Resume Training Capability
- Best & Last Checkpoint Saving
- Training Plots (Loss & Dice)
- Final Test Set Evaluation
"""

import os
import sys
import random
import warnings
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
warnings.filterwarnings('ignore')
BRATS_DATA_PATH = Path("/kaggle/input/brain-tumor-segmentation-hackathon") 
OUTPUT_DIR = Path("/kaggle/working/pytorch_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_SHAPE = (96, 96, 96)
BATCH_SIZE = 2      # 2 GPUs = Batch size 2 (1 per GPU)
NUM_CLASSES = 3     # WT, TC, ET
NUM_WORKERS = 4     # 4 CPU cores on Kaggle
EPOCHS = 20         # Total target epochs
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5  # L2 regularization
VAL_INTERVAL = 1    # Validate every epoch
RESUME_FROM = None 

def is_valid_nifti(path: Path) -> bool:
    """Check if path is a valid NIfTI file."""
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= 100
    except Exception:
        return False

def _find_file(patient_dir: Path, patient_id: str, suffix: str) -> Path | None:
    """
    Find a file matching pattern {patient_id}_{suffix}.nii(.gz) in patient_dir.
    Handles multiple folder structures:
      1. Files directly in patient_dir: patient_dir/{patient_id}_{suffix}.nii.gz
      2. Fake-file folders: patient_dir/{patient_id}_{suffix}.nii/ containing actual .nii inside
      3. Files inside subfolders (one file per subfolder)
      4. Mixed: some files direct, some in subfolders
    
    Prioritizes .nii.gz over .nii to avoid Windows permission issues with uncompressed files.
    """
    # PRIORITY 1: Search for .nii.gz files (preferred format)
    gz_pattern = f"{patient_id}_{suffix}.nii.gz"
    
    # Check directly in patient_dir
    candidate = patient_dir / gz_pattern
    if candidate.exists() and candidate.is_file():
        return candidate
    
    # Search recursively in subfolders
    matches = [m for m in patient_dir.rglob(gz_pattern) if m.is_file()]
    if matches:
        return matches[0]
    
    # PRIORITY 2: Check for .nii files 
    nii_pattern = f"{patient_id}_{suffix}.nii"
    candidate = patient_dir / nii_pattern
    if candidate.exists() and candidate.is_file():
        return candidate
    
    matches = [m for m in patient_dir.rglob(nii_pattern) if m.is_file()]
    if matches:
        return matches[0]
    
    # PRIORITY 3: Handle "fake file folders" - folders named like *.nii containing actual data
    # e.g., patient_dir/BraTS2021_00000_flair.nii/ -> contains 00000057_brain_flair.nii
    fake_folder = patient_dir / nii_pattern
    if fake_folder.exists() and fake_folder.is_dir():
        # Look for any .nii file inside this folder
        inner_files = list(fake_folder.glob("*.nii"))
        if inner_files:
            return inner_files[0]
        # Also check for .nii.gz inside
        inner_files = list(fake_folder.glob("*.nii.gz"))
        if inner_files:
            return inner_files[0]
    
    # PRIORITY 4: Search for any .nii/.nii.gz with matching suffix pattern in filename
    # e.g., *_brain_flair.nii, *_flair.nii.gz, etc.
    suffix_patterns = [f"*_{suffix}.nii.gz", f"*_{suffix}.nii", f"*_brain_{suffix}.nii", f"*_brain_{suffix}.nii.gz"]
    for pattern in suffix_patterns:
        matches = [m for m in patient_dir.rglob(pattern) if m.is_file()]
        if matches:
            return matches[0]
    
    # Handle "seg" suffix variations (final_seg, seg)
    if suffix == "seg":
        seg_patterns = ["*_final_seg.nii", "*_final_seg.nii.gz", "*_seg.nii", "*_seg.nii.gz"]
        for pattern in seg_patterns:
            matches = [m for m in patient_dir.rglob(pattern) if m.is_file()]
            if matches:
                return matches[0]
    
    return None

def find_brats_files(base_path: Path):
    """
    Find all valid BraTS patients with all 4 modalities + segmentation.
    Handles multiple folder structures including nested and fake-file folders.
    """
    patients = []
    skipped = []
    modalities = ['flair', 't1', 't1ce', 't2']
    
    if not base_path.exists():
        print(f"[Error] path {base_path} does not exist.")
        return []

    # Find all patient folders (flexible matching)
    patient_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    print(f"[Data] Scanning {len(patient_dirs)} directories in {base_path}...")

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        img_files = []
        found_modalities = 0
        
        # Find each modality using robust search
        for mod in modalities:
            file_path = _find_file(patient_dir, patient_id, mod)
            if file_path is not None and is_valid_nifti(file_path):
                img_files.append(str(file_path))
                found_modalities += 1

        # Find Segmentation
        seg_path = _find_file(patient_dir, patient_id, "seg")
        seg_file = str(seg_path) if seg_path is not None and is_valid_nifti(seg_path) else None
        
        if found_modalities == 4 and seg_file:
            patients.append({'id': patient_id, 'images': img_files, 'label': seg_file})
        else:
            skipped.append(patient_id)
    
    if skipped:
        print(f"[Data] Skipped {len(skipped)} incomplete cases (missing modalities or seg)")
            
    return patients

class BraTSDataset(Dataset):
    def __init__(self, patient_list, patch_size, augment=True):
        self.patients = patient_list
        self.patch_size = patch_size
        self.augment = augment
        
    def __len__(self): return len(self.patients)
    
    def _load_nifti(self, path):
        import nibabel as nib
        return nib.load(path).get_fdata().astype(np.float32)
    
    def _normalize(self, image):
        brain_mask = image[0] > 0
        for c in range(image.shape[0]):
            channel = image[c]
            if brain_mask.sum() > 0:
                mean, std = channel[brain_mask].mean(), channel[brain_mask].std()
                if std > 0:
                    image[c] = (channel - mean) / std
                    image[c][~brain_mask] = 0
        return image
    
    def _to_regions(self, label):
        # WT: Whole Tumor (1+2+4), TC: Tumor Core (1+4), ET: Enhancing Tumor (4)
        wt = ((label == 1) | (label == 2) | (label == 4)).astype(np.float32)
        tc = ((label == 1) | (label == 4)).astype(np.float32)
        et = (label == 4).astype(np.float32)
        return np.stack([wt, tc, et], axis=0)
    
    def _smart_crop(self, image, label):
        """Ensures we don't just crop background 90% of the time."""
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        # 66% chance to center on tumor if it exists
        if random.random() < 0.66:
            tumor_mask = label[0] > 0 # WT channel
            coords = np.argwhere(tumor_mask)
            if len(coords) > 0:
                center = coords[random.randint(0, len(coords)-1)]
                d_start = min(max(0, center[0] - pd//2), d - pd)
                h_start = min(max(0, center[1] - ph//2), h - ph)
                w_start = min(max(0, center[2] - pw//2), w - pw)
                return image[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw], \
                       label[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]

        # Fallback to random crop
        d_start = random.randint(0, max(0, d - pd))
        h_start = random.randint(0, max(0, h - ph))
        w_start = random.randint(0, max(0, w - pw))
        return image[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw], \
               label[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]

    def _pad_if_needed(self, image, label):
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        if d < pd or h < ph or w < pw:
            pad_d, pad_h, pad_w = max(0, pd-d), max(0, ph-h), max(0, pw-w)
            image = np.pad(image, ((0,0), (0,pad_d), (0,pad_h), (0,pad_w)))
            label = np.pad(label, ((0,0), (0,pad_d), (0,pad_h), (0,pad_w)))
        return image, label

    def _augment(self, image, label):
        if random.random() > 0.5: # Flip D
            image, label = np.flip(image, 1).copy(), np.flip(label, 1).copy()
        if random.random() > 0.5: # Flip H
            image, label = np.flip(image, 2).copy(), np.flip(label, 2).copy()
        if random.random() > 0.5: # Flip W
            image, label = np.flip(image, 3).copy(), np.flip(label, 3).copy()
        return image, label

    def __getitem__(self, idx):
        try:
            patient = self.patients[idx]
            images = [self._load_nifti(p) for p in patient['images']]
            image = np.stack(images, axis=0)
            label = self._to_regions(self._load_nifti(patient['label']).astype(np.int32))
            
            image = self._normalize(image)
            image, label = self._pad_if_needed(image, label)
            
            if self.augment:
                image, label = self._smart_crop(image, label)
                image, label = self._augment(image, label)
            else:
                # Center crop for validation/test
                _, d, h, w = image.shape
                pd, ph, pw = self.patch_size
                d_s, h_s, w_s = max(0, (d-pd)//2), max(0, (h-ph)//2), max(0, (w-pw)//2)
                image = image[:, d_s:d_s+pd, h_s:h_s+ph, w_s:w_s+pw]
                label = label[:, d_s:d_s+pd, h_s:h_s+ph, w_s:w_s+pw]
            
            image, label = self._pad_if_needed(image, label)
            return torch.from_numpy(image), torch.from_numpy(label)
        except Exception:
            # Return dummy on failure
            return torch.zeros((4, *self.patch_size)), torch.zeros((3, *self.patch_size))


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, activation='leaky_relu'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.01, inplace=True) if activation == 'leaky_relu' else nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.01, inplace=True) if activation == 'leaky_relu' else nn.GELU(),
        )
    def forward(self, x): return self.conv(x)

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)

class TransUNetMLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class ViTEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = TransUNetMLP(embed_dim, mlp_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x

class LearnableQueries(nn.Module):
    def __init__(self, num_queries, embed_dim):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
    def forward(self, x):
        batch_size = x.shape[0]
        return self.queries.expand(batch_size, -1, -1)

class MaskedCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim=1024, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = TransUNetMLP(embed_dim, mlp_dim, dropout_rate)
    def forward(self, queries, kv, mask=None):
        q_norm = self.norm1(queries)
        attn_out, _ = self.cross_attn(q_norm, kv, kv, attn_mask=mask)
        queries = queries + attn_out
        queries = queries + self.mlp(self.norm2(queries))
        return queries

class TransUNet3D(nn.Module):
    def __init__(
        self, in_channels=4, num_classes=3, encoder_depth=4, 
        encoder_filters=(24, 48, 96, 192, 384),  # Lighter: was (32, 64, 128, 256, 512)
        embed_dim=192, num_vit_layers=4, num_heads=6, mlp_dim=768, num_queries=24,  # Lighter ViT
        dropout_rate=0.1, decoder_filters=(192, 96, 48, 24), use_p1_attention=False,  # Lighter decoder
        p2_attn_pool=2, p1_attn_pool=4
    ):
        super().__init__()
        self.encoder_depth = encoder_depth
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.use_p1_attention = use_p1_attention
        self.p2_attn_pool = p2_attn_pool
        self.p1_attn_pool = p1_attn_pool
        
        # CNN Encoder
        self.enc1 = ResBlock3D(in_channels, encoder_filters[0])
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ResBlock3D(encoder_filters[0], encoder_filters[1])
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ResBlock3D(encoder_filters[1], encoder_filters[2])
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ResBlock3D(encoder_filters[2], encoder_filters[3])
        self.pool4 = nn.MaxPool3d(2)
        self.bottleneck = ResBlock3D(encoder_filters[3], encoder_filters[4])
        
        # ViT
        self.patch_embed = nn.Sequential(
            nn.Conv3d(encoder_filters[4], embed_dim, kernel_size=1),
            nn.GroupNorm(8, embed_dim),
        )
        self.vit_blocks = nn.ModuleList([
            ViTEncoderBlock(embed_dim, num_heads, mlp_dim, dropout_rate)
            for _ in range(num_vit_layers)
        ])
        
        # Decoder
        self.learnable_queries = LearnableQueries(num_queries, embed_dim)
        self.proj_p4 = nn.Sequential(nn.Conv3d(encoder_filters[3], embed_dim, 1), nn.GroupNorm(8, embed_dim))
        self.proj_p3 = nn.Sequential(nn.Conv3d(encoder_filters[2], embed_dim, 1), nn.GroupNorm(8, embed_dim))
        self.proj_p2 = nn.Sequential(nn.Conv3d(encoder_filters[1], embed_dim, 1), nn.GroupNorm(8, embed_dim))
        self.proj_p1 = nn.Sequential(nn.Conv3d(encoder_filters[0], embed_dim, 1), nn.GroupNorm(8, embed_dim))
        
        self.xattn_p4 = MaskedCrossAttention(embed_dim, num_heads, mlp_dim, dropout_rate)
        self.xattn_p3 = MaskedCrossAttention(embed_dim, num_heads, mlp_dim, dropout_rate)
        self.xattn_p2 = MaskedCrossAttention(embed_dim, num_heads, mlp_dim, dropout_rate)
        self.xattn_p1 = MaskedCrossAttention(embed_dim, num_heads, mlp_dim, dropout_rate)
        
        self.class_head = nn.Linear(embed_dim, num_classes)
        
        self.up4 = nn.ConvTranspose3d(embed_dim, decoder_filters[0], 2, stride=2)
        self.dec4 = ConvBlock3D(decoder_filters[0] + encoder_filters[3], decoder_filters[0])
        self.up3 = nn.ConvTranspose3d(decoder_filters[0], decoder_filters[1], 2, stride=2)
        self.dec3 = ConvBlock3D(decoder_filters[1] + encoder_filters[2], decoder_filters[1])
        self.up2 = nn.ConvTranspose3d(decoder_filters[1], decoder_filters[2], 2, stride=2)
        self.dec2 = ConvBlock3D(decoder_filters[2] + encoder_filters[1], decoder_filters[2])
        self.up1 = nn.ConvTranspose3d(decoder_filters[2], decoder_filters[3], 2, stride=2)
        self.dec1 = ConvBlock3D(decoder_filters[3] + encoder_filters[0], decoder_filters[3])
        self.out_conv = nn.Conv3d(decoder_filters[3], num_classes, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    def _flatten_spatial(self, x):
        return x.flatten(2).transpose(1, 2)
    
    def _compute_coarse_mask(self, queries, encoder_output):
        mask_logits = torch.bmm(queries, encoder_output.transpose(1, 2))
        return torch.sigmoid(mask_logits)
    
    def _resize_mask(self, mask, target_tokens):
        B, Q, N = mask.shape
        mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(Q, target_tokens), mode='nearest')
        return mask.squeeze(1)
    
    def forward(self, x):
        p1 = self.enc1(x)
        p2 = self.enc2(self.pool1(p1))
        p3 = self.enc3(self.pool2(p2))
        p4 = self.enc4(self.pool3(p3))
        p5 = self.bottleneck(self.pool4(p4))
        
        vit_input = self.patch_embed(p5)
        encoder_tokens = self._flatten_spatial(vit_input)
        for vit_block in self.vit_blocks: encoder_tokens = vit_block(encoder_tokens)
        
        queries = self.learnable_queries(encoder_tokens)
        
        p4_proj = self._flatten_spatial(self.proj_p4(p4))
        p3_proj = self._flatten_spatial(self.proj_p3(p3))
        p2_proj_feat = self.proj_p2(p2)
        if self.p2_attn_pool > 1:
            p2_proj_feat = F.avg_pool3d(p2_proj_feat, kernel_size=self.p2_attn_pool, stride=self.p2_attn_pool)
        p2_proj = self._flatten_spatial(p2_proj_feat)
        
        queries = self.xattn_p4(queries, p4_proj, mask=None)
        queries = self.xattn_p3(queries, p3_proj, mask=None)
        queries = self.xattn_p2(queries, p2_proj, mask=None)
        
        if self.use_p1_attention:
             p1_proj_feat = self.proj_p1(p1)
             if self.p1_attn_pool > 1:
                 p1_proj_feat = F.avg_pool3d(p1_proj_feat, kernel_size=self.p1_attn_pool, stride=self.p1_attn_pool)
             p1_proj = self._flatten_spatial(p1_proj_feat)
             queries = self.xattn_p1(queries, p1_proj, mask=None)

        spatial_shape = p5.shape[2:]
        B = x.shape[0]
        decoder_feat = encoder_tokens.transpose(1, 2).view(B, self.embed_dim, *spatial_shape)
        
        d4 = self.dec4(torch.cat([self.up4(decoder_feat), p4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), p3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), p2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), p1], dim=1))
        
        return self.out_conv(d1)


class RegionDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        total_loss = 0.0
        for i in range(pred.shape[1]):
            p, t = pred[:, i].flatten(1), target[:, i].flatten(1)
            dice = (2. * (p * t).sum(1) + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
            total_loss += (1 - dice).mean()
        return total_loss / pred.shape[1]

def get_dice_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    scores = {}
    for i, name in enumerate(['WT', 'TC', 'ET']):
        p, t = pred[:, i].flatten(), target[:, i].flatten()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        scores[name] = ((2. * inter + 1e-5) / (union + 1e-5)).item()
    scores['Mean'] = np.mean(list(scores.values()))
    return scores

def plot_history(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 6))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dice Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_dice_mean'], label='Mean Dice', linewidth=2)
    plt.plot(epochs, history['val_dice_wt'], label='WT', linestyle='--')
    plt.plot(epochs, history['val_dice_tc'], label='TC', linestyle='--')
    plt.plot(epochs, history['val_dice_et'], label='ET', linestyle='--')
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} with {torch.cuda.device_count()} GPUs")
    
    # --- DATA SPLIT (80/10/10) ---
    all_patients = find_brats_files(BRATS_DATA_PATH)
    if not all_patients: return
    
    # First split: Train (80%) vs Rest (20%)
    train_patients, rest = train_test_split(all_patients, test_size=0.2, random_state=42)
    # Second split: Rest into Val (10% total) and Test (10% total)
    # 50% of 20% is 10%
    val_patients, test_patients = train_test_split(rest, test_size=0.5, random_state=42)
    
    print(f"Data Split: Train {len(train_patients)} | Val {len(val_patients)} | Test {len(test_patients)}")
    
    # --- LOADERS ---
    train_loader = DataLoader(
        BraTSDataset(train_patients, INPUT_SHAPE, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        BraTSDataset(val_patients, INPUT_SHAPE, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # --- MODEL & OPTIMIZER ---
    # NOTE: Ensure you are using the FULL TransUNet3D class definition in production!
    model = TransUNet3D(num_classes=NUM_CLASSES).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    criterion = RegionDiceLoss()
    
    # --- RESUME LOGIC ---
    start_epoch = 0
    best_dice = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_dice_mean': [], 'val_dice_wt': [], 'val_dice_tc': [], 'val_dice_et': []}
    
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Resuming from {RESUME_FROM}...")
        checkpoint = torch.load(RESUME_FROM, map_location=device)
        
        # Handle DataParallel loading
        state_dict = checkpoint['model_state_dict']
        if isinstance(model, nn.DataParallel) and 'module.' not in list(state_dict.keys())[0]:
            # If model is parallel but checkpoint isn't
            model.module.load_state_dict(state_dict)
        elif not isinstance(model, nn.DataParallel) and 'module.' in list(state_dict.keys())[0]:
            # If checkpoint is parallel but model isn't
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items(): new_state_dict[k.replace('module.', '')] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Optional: Load scheduler if it matches the new T_max, otherwise restart scheduler
        if 'scheduler_state_dict' in checkpoint:
             try: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
             except: pass
        
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        history = checkpoint.get('history', history)
        print(f"Resuming at Epoch {start_epoch + 1} with Best Dice: {best_dice:.4f}")

    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=True)
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = model(imgs)
                loss = criterion(preds, lbls)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION ---
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            val_loss = 0.0
            scores = {'WT': 0.0, 'TC': 0.0, 'ET': 0.0, 'Mean': 0.0}
            
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        preds = model(imgs)
                        val_loss += criterion(preds, lbls).item()
                    
                    batch_scores = get_dice_score(preds, lbls)
                    for k in scores: scores[k] += batch_scores[k]
            
            val_loss /= len(val_loader)
            for k in scores: scores[k] /= len(val_loader)
            
            # Update History
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_dice_mean'].append(scores['Mean'])
            history['val_dice_wt'].append(scores['WT'])
            history['val_dice_tc'].append(scores['TC'])
            history['val_dice_et'].append(scores['ET'])
            
            print(f"--> Val Loss: {val_loss:.4f} | Dice Mean: {scores['Mean']:.4f} (WT: {scores['WT']:.3f})")
            
            # Save checkpoints helper
            def save_ckpt(name, is_best=False):
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_dice': best_dice,
                    'history': history
                }
                torch.save(state, OUTPUT_DIR / name)
            
            # Save Last
            save_ckpt("last_checkpoint.pth")
            
            # Save Best
            if scores['Mean'] > best_dice:
                best_dice = scores['Mean']
                save_ckpt("best_checkpoint.pth", is_best=True)
                print("    [Saved New Best Model]")
            
            # Plot
            plot_history(history, OUTPUT_DIR / "training_plots.png")

    print("Training Complete.")
    return test_patients


def evaluate_test_set(test_patients):
    print("\n" + "="*50)
    print("STARTING TEST SET EVALUATION")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Best Model
    model = TransUNet3D(num_classes=NUM_CLASSES).to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    
    checkpoint_path = OUTPUT_DIR / "best_checkpoint.pth"
    if not checkpoint_path.exists():
        print("No best checkpoint found. Skipping evaluation.")
        return

    print(f"Loading best model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_ds = BraTSDataset(test_patients, INPUT_SHAPE, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False) # BS=1 for accurate metrics per patient
    
    final_scores = {'WT': [], 'TC': [], 'ET': [], 'Mean': []}
    
    print(f"Evaluating on {len(test_patients)} unseen patients...")
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = model(imgs)
            
            scores = get_dice_score(preds, lbls)
            for k in final_scores: final_scores[k].append(scores[k])

    print("\n=== FINAL TEST METRICS (AVG) ===")
    print(f"Mean Dice: {np.mean(final_scores['Mean']):.4f}")
    print(f"WT Dice:   {np.mean(final_scores['WT']):.4f}")
    print(f"TC Dice:   {np.mean(final_scores['TC']):.4f}")
    print(f"ET Dice:   {np.mean(final_scores['ET']):.4f}")
    
    # Save test results to JSON
    with open(OUTPUT_DIR / "test_results.json", "w") as f:
        json.dump({k: np.mean(v) for k, v in final_scores.items()}, f)

if __name__ == "__main__":
    # 1. Run Training
    test_patients_list = run_training()
    
    # 2. Run Evaluation on Test Set
    if test_patients_list:
        evaluate_test_set(test_patients_list)