"""
BraTS21 Training Script using PyTorch TransUNet-style Architecture
===================================================================
Pure PyTorch implementation for local GPU training.

This script:
1. Loads BraTS data with efficient preprocessing
2. Trains a TransUNet-inspired model using PyTorch
3. Uses region-based evaluation (WT, TC, ET)

Requirements:
    pip install torch torchvision nibabel scikit-learn tqdm scipy

Usage:
    python train_transnet_brats.py
"""

import os
import sys
import random
import warnings
import time
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split


BRATS_DATA_PATH = Path(r"C:\Users\Compumarts\Desktop\Orange Hackathon\data")
CACHE_DIR = Path(r"C:\Users\Compumarts\Desktop\Orange Hackathon\brats_cache")
OUTPUT_DIR = Path(r"C:\Users\Compumarts\Desktop\Orange Hackathon\pytorch_results")

INPUT_SHAPE = (96, 96, 96)  # 3D patch size (smaller = faster)
BATCH_SIZE = 1  # Limited by 16GB VRAM
NUM_CLASSES = 3  # 3 regions: WT, TC, ET
EPOCHS = 100
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
VAL_INTERVAL = 10  # Validate every N epochs
NUM_WORKERS = 0  # 0 for Windows

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def is_valid_nifti(path: str) -> bool:
    """Check if a NIfTI file exists and is not empty/corrupted."""
    try:
        p = Path(path)
        if not p.exists():
            return False
        # Check file size (empty or too small files are invalid)
        if p.stat().st_size < 100:  # NIfTI headers are at least 348 bytes
            return False
        return True
    except Exception:
        return False


def find_brats_files(base_path: Path):
    """Find all BraTS patient folders and their files, skipping corrupted ones."""
    patients = []
    skipped = []
    modalities = ['flair', 't1', 't1ce', 't2']
    
    for patient_dir in sorted(base_path.iterdir()):
        if not patient_dir.is_dir() or "BraTS" not in patient_dir.name:
            continue
        
        patient_id = patient_dir.name
        
        # Find modality files
        img_files = []
        valid = True
        for mod in modalities:
            found = False
            # Try direct file patterns first
            for ext in ['.nii.gz', '.nii']:
                path = patient_dir / f"{patient_id}_{mod}{ext}"
                if path.exists() and is_valid_nifti(str(path)):
                    img_files.append(str(path))
                    found = True
                    break
            
            # Check nested folder structure (e.g., BraTS2021_00000_flair.nii/file.nii)
            if not found:
                for ext in ['.nii.gz', '.nii']:
                    nested_dir = patient_dir / f"{patient_id}_{mod}{ext}"
                    if nested_dir.is_dir():
                        for f in nested_dir.iterdir():
                            if f.suffix in ['.nii', '.gz'] and is_valid_nifti(str(f)):
                                img_files.append(str(f))
                                found = True
                                break
                    if found:
                        break
            
            if not found:
                valid = False
                break
        
        # Find segmentation
        seg_file = None
        for ext in ['.nii.gz', '.nii']:
            path = patient_dir / f"{patient_id}_seg{ext}"
            if path.exists() and is_valid_nifti(str(path)):
                seg_file = str(path)
                break
        
        # Check nested folder for segmentation too
        if seg_file is None:
            for ext in ['.nii.gz', '.nii']:
                nested_dir = patient_dir / f"{patient_id}_seg{ext}"
                if nested_dir.is_dir():
                    for f in nested_dir.iterdir():
                        if f.suffix in ['.nii', '.gz'] and is_valid_nifti(str(f)):
                            seg_file = str(f)
                            break
                if seg_file:
                    break
        
        if valid and len(img_files) == 4 and seg_file:
            patients.append({
                'id': patient_id,
                'images': img_files,
                'label': seg_file
            })
        else:
            skipped.append(patient_id)
    
    if skipped:
        print(f"[Warning] Skipped {len(skipped)} patients with missing/corrupted files: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    
    return patients


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS data.
    Loads NIfTI files, applies preprocessing and augmentation.
    """
    
    def __init__(self, patient_list, patch_size, augment=True, cache_dir=None):
        self.patients = patient_list
        self.patch_size = patch_size
        self.augment = augment
        self.cache_dir = cache_dir
        
    def __len__(self):
        return len(self.patients)
    
    def _load_nifti(self, path):
        """Load NIfTI file."""
        import nibabel as nib
        nii = nib.load(path)
        return nii.get_fdata().astype(np.float32)
    
    def _normalize(self, image):
        """Z-score normalize per channel (brain region only)."""
        brain_mask = image[0] > 0  # Use first modality for mask
        for c in range(image.shape[0]):
            channel = image[c]
            brain_vals = channel[brain_mask]
            if len(brain_vals) > 0:
                mean, std = brain_vals.mean(), brain_vals.std()
                if std > 0:
                    image[c] = (channel - mean) / std
                    image[c][~brain_mask] = 0
        return image
    
    def _to_regions(self, label):
        """Convert BraTS labels to regions (WT, TC, ET)."""
        # WT = 1 + 2 + 4, TC = 1 + 4, ET = 4
        wt = ((label == 1) | (label == 2) | (label == 4)).astype(np.float32)
        tc = ((label == 1) | (label == 4)).astype(np.float32)
        et = (label == 4).astype(np.float32)
        return np.stack([wt, tc, et], axis=0)
    
    def _random_crop(self, image, label):
        """Random 3D crop."""
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        d_start = random.randint(0, max(0, d - pd))
        h_start = random.randint(0, max(0, h - ph))
        w_start = random.randint(0, max(0, w - pw))
        
        image = image[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        label = label[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        return image, label
    
    def _center_crop(self, image, label):
        """Center crop."""
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        d_start = max(0, (d - pd) // 2)
        h_start = max(0, (h - ph) // 2)
        w_start = max(0, (w - pw) // 2)
        
        image = image[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        label = label[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        return image, label
    
    def _pad_if_needed(self, image, label):
        """Pad to patch size if smaller."""
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        if d < pd or h < ph or w < pw:
            pad_d = max(0, pd - d)
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            
            image = np.pad(image, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))
            label = np.pad(label, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))
        
        return image, label
    
    def _augment(self, image, label):
        """Apply random augmentations."""
        # Random flips
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=2).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=3).copy()
            label = np.flip(label, axis=3).copy()
        
        # Random intensity shift
        if random.random() > 0.7:
            shift = (random.random() - 0.5) * 0.2
            image = image + shift
        
        return image, label
    
    def __getitem__(self, idx):
        # Try loading with error handling - fallback to random sample on failure
        max_retries = 3
        for attempt in range(max_retries):
            try:
                current_idx = idx if attempt == 0 else random.randint(0, len(self.patients) - 1)
                patient = self.patients[current_idx]
                
                # Load modalities and stack: (4, D, H, W)
                images = []
                for img_path in patient['images']:
                    img = self._load_nifti(img_path)
                    images.append(img)
                image = np.stack(images, axis=0)
                
                # Load and convert segmentation
                label = self._load_nifti(patient['label']).astype(np.int32)
                label = self._to_regions(label)  # (3, D, H, W)
                
                # Normalize
                image = self._normalize(image)
                
                # Pad if needed
                image, label = self._pad_if_needed(image, label)
                
                # Crop
                if self.augment:
                    image, label = self._random_crop(image, label)
                    image, label = self._augment(image, label)
                else:
                    image, label = self._center_crop(image, label)
                
                # Final pad check
                image, label = self._pad_if_needed(image, label)
                
                return torch.from_numpy(image.copy()), torch.from_numpy(label.copy())
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[Warning] Error loading patient {patient['id']}: {e}. Retrying with another sample...")
                else:
                    # Return zeros as last resort
                    print(f"[Error] Failed to load after {max_retries} attempts. Returning zero tensor.")
                    image = np.zeros((4,) + self.patch_size, dtype=np.float32)
                    label = np.zeros((3,) + self.patch_size, dtype=np.float32)
                    return torch.from_numpy(image), torch.from_numpy(label)



class ConvBlock3D(nn.Module):
    """Double convolution block with GroupNorm."""
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
    
    def forward(self, x):
        return self.conv(x)


class ResBlock3D(nn.Module):
    """Residual block for CNN encoder."""
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



class ViTPatchingAndEmbedding(nn.Module):
    """
    Patch embedding for ViT - converts spatial features to sequence of tokens.
    Equivalent to MedicAI's ViTPatchingAndEmbedding.
    """
    def __init__(self, image_size, patch_size, in_channels, embed_dim, use_patch_bias=True):
        super().__init__()
        self.image_size = image_size  # (D, H, W)
        self.patch_size = patch_size  # (pD, pH, pW)
        self.num_patches = (image_size[0] // patch_size[0]) * \
                          (image_size[1] // patch_size[1]) * \
                          (image_size[2] // patch_size[2])
        
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=use_patch_bias
        )
        self.embed_dim = embed_dim
    
    def forward(self, x):
        # x: (B, C, D, H, W) -> (B, embed_dim, nD, nH, nW)
        x = self.proj(x)
        # Flatten to sequence: (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransUNetMLP(nn.Module):
    """MLP block for transformer with GELU activation."""
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
    """
    Single ViT encoder block with multi-head self-attention and MLP.
    Equivalent to MedicAI's ViTEncoderBlock.
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = TransUNetMLP(embed_dim, mlp_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x



class LearnableQueries(nn.Module):
    """
    Learnable query tokens for the decoder.
    These act as "segmentation concepts" refined through cross-attention.
    """
    def __init__(self, num_queries, embed_dim):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
    
    def forward(self, x):
        # x is just used for batch size
        batch_size = x.shape[0]
        return self.queries.expand(batch_size, -1, -1)


class MaskedCrossAttention(nn.Module):
    """
    Masked cross-attention block for TransUNet decoder.
    Performs cross-attention between queries and CNN features with spatial masking.
    """
    def __init__(self, embed_dim, num_heads, mlp_dim=1024, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = TransUNetMLP(embed_dim, mlp_dim, dropout_rate)
    
    def forward(self, queries, kv, mask=None):
        """
        Args:
            queries: (B, num_queries, embed_dim)
            kv: (B, num_tokens, embed_dim) - CNN features as keys/values
            mask: (B, num_queries, num_tokens) - attention mask
        """
        # Cross-attention with residual
        q_norm = self.norm1(queries)
        attn_out, _ = self.cross_attn(q_norm, kv, kv, attn_mask=mask)
        queries = queries + attn_out
        
        # MLP with residual
        queries = queries + self.mlp(self.norm2(queries))
        return queries



class TransUNet3D(nn.Module):
    """
    Full TransUNet 3D model for medical image segmentation.
    
    Architecture:
    1. CNN Encoder: Multi-scale feature extraction (P1, P2, P3, P4, P5)
    2. ViT Encoder: Global context modeling on bottleneck features
    3. Hybrid Decoder:
       - Learnable queries refined via coarse-to-fine masked cross-attention
       - U-Net style upsampling with skip connections
    
    Based on MedicAI's TransUNet implementation.
    """
    
    def __init__(
        self,
        in_channels=4,
        num_classes=3,
        encoder_depth=4,
        encoder_filters=(32, 64, 128, 256, 512),
        embed_dim=256,
        num_vit_layers=6,
        num_heads=8,
        mlp_dim=1024,
        num_queries=32,
        dropout_rate=0.1,
        decoder_filters=(256, 128, 64, 32),
        use_p1_attention=False,
        p2_attn_pool=2,
        p1_attn_pool=4,
    ):
        super().__init__()
        
        self.encoder_depth = encoder_depth
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.use_p1_attention = use_p1_attention
        self.p2_attn_pool = p2_attn_pool
        self.p1_attn_pool = p1_attn_pool
        
        # =====================================================================
        # CNN Encoder - produces pyramid features P1, P2, P3, P4, (P5)
        # =====================================================================
        self.enc1 = ResBlock3D(in_channels, encoder_filters[0])
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = ResBlock3D(encoder_filters[0], encoder_filters[1])
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = ResBlock3D(encoder_filters[1], encoder_filters[2])
        self.pool3 = nn.MaxPool3d(2)
        
        self.enc4 = ResBlock3D(encoder_filters[2], encoder_filters[3])
        self.pool4 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = ResBlock3D(encoder_filters[3], encoder_filters[4])
        
        # =====================================================================
        # ViT Encoder - operates on bottleneck features
        # =====================================================================
        # Patch embedding: project bottleneck features to embed_dim
        self.patch_embed = nn.Sequential(
            nn.Conv3d(encoder_filters[4], embed_dim, kernel_size=1),
            nn.GroupNorm(8, embed_dim),
        )
        
        # ViT encoder blocks
        self.vit_blocks = nn.ModuleList([
            ViTEncoderBlock(embed_dim, num_heads, mlp_dim, dropout_rate)
            for _ in range(num_vit_layers)
        ])
        
        # =====================================================================
        # Hybrid Decoder with Coarse-to-Fine Attention
        # =====================================================================
        
        # Learnable queries
        self.learnable_queries = LearnableQueries(num_queries, embed_dim)
        
        # Project CNN skip features to embed_dim for cross-attention
        self.proj_p4 = nn.Sequential(
            nn.Conv3d(encoder_filters[3], embed_dim, 1),
            nn.GroupNorm(8, embed_dim)
        )
        self.proj_p3 = nn.Sequential(
            nn.Conv3d(encoder_filters[2], embed_dim, 1),
            nn.GroupNorm(8, embed_dim)
        )
        self.proj_p2 = nn.Sequential(
            nn.Conv3d(encoder_filters[1], embed_dim, 1),
            nn.GroupNorm(8, embed_dim)
        )
        self.proj_p1 = nn.Sequential(
            nn.Conv3d(encoder_filters[0], embed_dim, 1),
            nn.GroupNorm(8, embed_dim)
        )
        
        # Masked cross-attention blocks for coarse-to-fine refinement
        self.xattn_p4 = MaskedCrossAttention(embed_dim, num_heads, mlp_dim, dropout_rate)
        self.xattn_p3 = MaskedCrossAttention(embed_dim, num_heads, mlp_dim, dropout_rate)
        self.xattn_p2 = MaskedCrossAttention(embed_dim, num_heads, mlp_dim, dropout_rate)
        self.xattn_p1 = MaskedCrossAttention(embed_dim, num_heads, mlp_dim, dropout_rate)
        
        # Class prediction from queries
        self.class_head = nn.Linear(embed_dim, num_classes)
        
        # =====================================================================
        # U-Net Style Upsampling Path
        # =====================================================================
        self.up4 = nn.ConvTranspose3d(embed_dim, decoder_filters[0], 2, stride=2)
        self.dec4 = ConvBlock3D(decoder_filters[0] + encoder_filters[3], decoder_filters[0])
        
        self.up3 = nn.ConvTranspose3d(decoder_filters[0], decoder_filters[1], 2, stride=2)
        self.dec3 = ConvBlock3D(decoder_filters[1] + encoder_filters[2], decoder_filters[1])
        
        self.up2 = nn.ConvTranspose3d(decoder_filters[1], decoder_filters[2], 2, stride=2)
        self.dec2 = ConvBlock3D(decoder_filters[2] + encoder_filters[1], decoder_filters[2])
        
        self.up1 = nn.ConvTranspose3d(decoder_filters[2], decoder_filters[3], 2, stride=2)
        self.dec1 = ConvBlock3D(decoder_filters[3] + encoder_filters[0], decoder_filters[3])
        
        # Final output
        self.out_conv = nn.Conv3d(decoder_filters[3], num_classes, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _flatten_spatial(self, x):
        """Flatten spatial dims to tokens: (B, C, D, H, W) -> (B, N, C)"""
        B, C = x.shape[:2]
        return x.flatten(2).transpose(1, 2)  # (B, N, C)
    
    def _compute_coarse_mask(self, queries, encoder_output):
        """
        Compute coarse mask prediction from queries and encoder output.
        M = Q^T @ E  ->  (B, num_queries, num_tokens)
        """
        # queries: (B, num_queries, embed_dim)
        # encoder_output: (B, num_tokens, embed_dim)
        mask_logits = torch.bmm(queries, encoder_output.transpose(1, 2))
        return torch.sigmoid(mask_logits)
    
    def _resize_mask(self, mask, target_tokens):
        """Resize mask to match target number of tokens."""
        # mask: (B, num_queries, current_tokens)
        # Use interpolation to resize
        B, Q, N = mask.shape
        mask = mask.unsqueeze(1)  # (B, 1, Q, N)
        mask = F.interpolate(mask, size=(Q, target_tokens), mode='nearest')
        return mask.squeeze(1)  # (B, Q, target_tokens)
    
    def forward(self, x):
        # =====================================================================
        # 1. CNN Encoder - extract pyramid features
        # =====================================================================
        p1 = self.enc1(x)                    # (B, 32, D, H, W)
        p2 = self.enc2(self.pool1(p1))       # (B, 64, D/2, H/2, W/2)
        p3 = self.enc3(self.pool2(p2))       # (B, 128, D/4, H/4, W/4)
        p4 = self.enc4(self.pool3(p3))       # (B, 256, D/8, H/8, W/8)
        p5 = self.bottleneck(self.pool4(p4)) # (B, 512, D/16, H/16, W/16)
        
        # =====================================================================
        # 2. ViT Encoder - global context on bottleneck
        # =====================================================================
        # Project and flatten to tokens
        vit_input = self.patch_embed(p5)
        encoder_tokens = self._flatten_spatial(vit_input)  # (B, N, embed_dim)
        
        # Apply ViT blocks
        for vit_block in self.vit_blocks:
            encoder_tokens = vit_block(encoder_tokens)
        
        # =====================================================================
        # 3. Coarse-to-Fine Decoder with Masked Cross-Attention
        # =====================================================================
        
        # Initialize learnable queries
        queries = self.learnable_queries(encoder_tokens)  # (B, num_queries, embed_dim)
        
        # Initial coarse mask prediction
        coarse_mask = self._compute_coarse_mask(queries, encoder_tokens)
        
        # Project CNN features and flatten for cross-attention
        p4_proj = self._flatten_spatial(self.proj_p4(p4))
        p3_proj = self._flatten_spatial(self.proj_p3(p3))

        p2_proj_feat = self.proj_p2(p2)
        if self.p2_attn_pool and self.p2_attn_pool > 1:
            p2_proj_feat = F.avg_pool3d(p2_proj_feat, kernel_size=self.p2_attn_pool, stride=self.p2_attn_pool)
        p2_proj = self._flatten_spatial(p2_proj_feat)

        p1_proj = None
        if self.use_p1_attention:
            p1_proj_feat = self.proj_p1(p1)
            if self.p1_attn_pool and self.p1_attn_pool > 1:
                p1_proj_feat = F.avg_pool3d(p1_proj_feat, kernel_size=self.p1_attn_pool, stride=self.p1_attn_pool)
            p1_proj = self._flatten_spatial(p1_proj_feat)
        
        # Coarse-to-fine refinement: P4 -> P3 -> P2 -> P1
        # Level P4
        mask_p4 = self._resize_mask(coarse_mask, p4_proj.shape[1])
        attn_mask = torch.where(mask_p4 > 0.5, 0.0, -1e9)
        queries = self.xattn_p4(queries, p4_proj, mask=None)  # Skip mask for stability
        coarse_mask = self._compute_coarse_mask(queries, encoder_tokens)
        
        # Level P3
        queries = self.xattn_p3(queries, p3_proj, mask=None)
        coarse_mask = self._compute_coarse_mask(queries, encoder_tokens)
        
        # Level P2
        queries = self.xattn_p2(queries, p2_proj, mask=None)
        coarse_mask = self._compute_coarse_mask(queries, encoder_tokens)
        
        # Level P1
        if self.use_p1_attention and p1_proj is not None:
            queries = self.xattn_p1(queries, p1_proj, mask=None)
        
        # Final predictions from refined queries
        class_logits = self.class_head(queries)  # (B, num_queries, num_classes)
        class_probs = F.softmax(class_logits, dim=-1)
        
        # Final mask from queries
        final_mask = self._compute_coarse_mask(queries, encoder_tokens)  # (B, Q, N)
        
        # =====================================================================
        # 4. Combine masks with class predictions
        # =====================================================================
        # Reshape encoder tokens back to spatial
        B = x.shape[0]
        spatial_shape = p5.shape[2:]  # (D/16, H/16, W/16)
        
        # Combine: sum over queries weighted by class probabilities
        # final_mask: (B, Q, N), class_probs: (B, Q, C)
        # Result: (B, N, C) = (B, Q, N)^T @ (B, Q, C)
        combined = torch.bmm(final_mask.transpose(1, 2), class_probs)  # (B, N, C)
        
        # Reshape to spatial: (B, C, D, H, W)
        combined = combined.transpose(1, 2)  # (B, C, N)
        combined = combined.view(B, self.num_classes, *spatial_shape)
        
        # Project to decoder dimension
        decoder_feat = F.conv3d(combined, 
                                weight=torch.ones(self.embed_dim, self.num_classes, 1, 1, 1, device=x.device) / self.num_classes,
                                bias=None)
        
        # Use encoder tokens reshaped as decoder feature instead
        decoder_feat = encoder_tokens.transpose(1, 2).view(B, self.embed_dim, *spatial_shape)
        
        # =====================================================================
        # 5. U-Net Style Upsampling
        # =====================================================================
        d4 = self.up4(decoder_feat)
        d4 = torch.cat([d4, p4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, p3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, p2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, p1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out_conv(d1)


class RegionDiceLoss(nn.Module):
    """Region-based Dice loss for BraTS (WT, TC, ET)."""
    
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        total_loss = 0.0
        for i in range(pred.shape[1]):  # WT, TC, ET
            p = pred[:, i].reshape(pred.shape[0], -1)
            t = target[:, i].reshape(target.shape[0], -1)
            
            intersection = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            total_loss += (1 - dice).mean()
        
        return total_loss / pred.shape[1]


def dice_coefficient_regions(pred, target, smooth=1e-5):
    """Calculate Dice for each region."""
    pred = torch.sigmoid(pred)
    pred_binary = (pred > 0.5).float()
    
    region_names = ['WT', 'TC', 'ET']
    scores = {}
    
    for i, name in enumerate(region_names):
        p = pred_binary[:, i].reshape(-1)
        t = target[:, i].reshape(-1)
        
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        
        scores[name] = ((2. * intersection + smooth) / (union + smooth)).item()
    
    scores['Mean'] = np.mean([scores['WT'], scores['TC'], scores['ET']])
    return scores


def train_pytorch_model():
    """Main training function."""
    
    print("=" * 70)
    print("BraTS21 Training with PyTorch TransUNet-style Model")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Performance settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 1. Find data
    print("\n[Step 1] Loading data...")
    patients = find_brats_files(BRATS_DATA_PATH)
    print(f"Found {len(patients)} patients")
    
    if len(patients) == 0:
        print("[ERROR] No patients found!")
        return
    
    # 2. Split train/val
    train_patients, val_patients = train_test_split(
        patients, test_size=VAL_SPLIT, random_state=42
    )
    print(f"Train: {len(train_patients)}, Validation: {len(val_patients)}")
    
    # 3. Create datasets and loaders
    train_ds = BraTSDataset(train_patients, INPUT_SHAPE, augment=True)
    val_ds = BraTSDataset(val_patients, INPUT_SHAPE, augment=False)
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # 4. Create model
    print("\n[Step 2] Creating model...")
    model = TransUNet3D(
        in_channels=4,
        num_classes=NUM_CLASSES,
        encoder_depth=4,
        encoder_filters=(24, 48, 96, 192, 384),   # lighter encoder
        embed_dim=128,                             # smaller ViT embedding
        num_vit_layers=3,                          # fewer ViT blocks
        num_heads=4,                               # fewer attention heads
        mlp_dim=512,                               # smaller MLP
        num_queries=16,                            # fewer queries
        dropout_rate=0.1,
        decoder_filters=(192, 96, 48, 24),         # lighter decoder
        use_p1_attention=False,                    # P1 attention is very memory heavy
        p2_attn_pool=4,                            # more downsampling for attention
        p1_attn_pool=4,                            # Only used if use_p1_attention=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 5. Loss, optimizer, scheduler
    criterion = RegionDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    
    # Mixed precision
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler() if amp_dtype == torch.float16 else None
    print(f"Precision: {'BF16' if amp_dtype == torch.bfloat16 else 'FP16'}")
    
    # 6. Training loop
    print(f"\n[Step 3] Training for {EPOCHS} epochs...")
    best_dice = 0.0

    train_loss_history = []
    val_epoch_history = []
    val_mean_dice_history = []
    val_wt_history = []
    val_tc_history = []
    val_et_history = []
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS} [Train]", leave=True)
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                preds = model(images)
                loss = criterion(preds, labels)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        train_loss_history.append(avg_train_loss)
        
        # Validation
        if (epoch + 1) % VAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            model.eval()
            val_scores = {'WT': 0.0, 'TC': 0.0, 'ET': 0.0, 'Mean': 0.0}
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Ep {epoch+1}/{EPOCHS} [Val]", leave=False):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        preds = model(images)
                    
                    scores = dice_coefficient_regions(preds, labels)
                    for k in val_scores:
                        val_scores[k] += scores[k]
            
            for k in val_scores:
                val_scores[k] /= len(val_loader)

            val_epoch_history.append(epoch + 1)
            val_mean_dice_history.append(val_scores['Mean'])
            val_wt_history.append(val_scores['WT'])
            val_tc_history.append(val_scores['TC'])
            val_et_history.append(val_scores['ET'])
            
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            
            print(f"\n--> Ep {epoch+1}: Loss {avg_train_loss:.4f} | "
                  f"Dice WT:{val_scores['WT']:.4f} TC:{val_scores['TC']:.4f} "
                  f"ET:{val_scores['ET']:.4f} Mean:{val_scores['Mean']:.4f} | "
                  f"VRAM {vram_gb:.1f}GB | Time {epoch_time:.1f}s")
            
            # Save best
            if val_scores['Mean'] > best_dice:
                best_dice = val_scores['Mean']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                }, OUTPUT_DIR / "best_model.pth")
                print(f"    [Saved] New best: {best_dice:.4f}")
        else:
            print(f"--> Ep {epoch+1}: Loss {avg_train_loss:.4f} | "
                  f"LR {scheduler.get_last_lr()[0]:.2e} | Time {epoch_time:.1f}s")
    
    print("\n" + "=" * 70)
    print(f"Training Complete! Best Mean Dice: {best_dice:.4f}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    # Plot loss and dice curves
    try:
        # Loss curve
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
        plt.close()

        # Dice curves (validation epochs only)
        if val_epoch_history:
            plt.figure(figsize=(8, 5))
            plt.plot(val_epoch_history, val_mean_dice_history, label="Mean Dice")
            plt.plot(val_epoch_history, val_wt_history, label="WT")
            plt.plot(val_epoch_history, val_tc_history, label="TC")
            plt.plot(val_epoch_history, val_et_history, label="ET")
            plt.xlabel("Epoch")
            plt.ylabel("Dice")
            plt.title("Validation Dice")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "dice_curve.png", dpi=150)
            plt.close()
    except Exception as e:
        print(f"[Warning] Failed to save plots: {e}")


if __name__ == "__main__":
    train_pytorch_model()
