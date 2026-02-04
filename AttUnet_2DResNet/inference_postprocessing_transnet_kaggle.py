import os
import sys
import glob
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from monai.inferers import sliding_window_inference
from monai import transforms as tr
from skimage import measure

MODE = "VERIFY"  

USE_TTA = True             # Test Time Augmentation (Flips)
REMOVE_SMALL_OBJECTS = True
MIN_PIXELS = 30            # Minimum size of a blob to keep

BRATS_DATA_PATH = "/kaggle/input/brain-tumor-segmentation-hackathon" # Path to data with labels (for Verify)
TEST_PATH = "/kaggle/input/instant-odc-ai-hackathon/test"            # Path to unlabelled data (for Submission)
MODEL_PATH = "/kaggle/input/transnet-brats-e20-fulldata/pytorch/default/1/best_checkpoint.pth"


def remove_small_blobs(mask, min_size=30):
    """Removes connected components smaller than min_size."""
    if mask.sum() == 0: return mask
    
    # Label connected components
    labeled_mask, num_features = measure.label(mask, connectivity=2, return_num=True)
    
    # Calculate properties of regions
    props = measure.regionprops(labeled_mask)
    
    new_mask = np.zeros_like(mask)
    for prop in props:
        if prop.area >= min_size:
            new_mask[labeled_mask == prop.label] = 1
            
    return new_mask

def keep_largest_blob(mask):
    """Keeps only the single largest connected component."""
    if mask.sum() == 0: return mask
    
    labeled_mask, num_features = measure.label(mask, connectivity=2, return_num=True)
    if num_features == 0: return mask
    
    props = measure.regionprops(labeled_mask)
    largest_region = max(props, key=lambda x: x.area)
    
    new_mask = np.zeros_like(mask)
    new_mask[labeled_mask == largest_region.label] = 1
    return new_mask


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


def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0: return ""
    return ' '.join(str(x) for x in runs)

def calculate_dice(pred_mask, gt_mask):
    intersection = (pred_mask * gt_mask).sum()
    sum_ = pred_mask.sum() + gt_mask.sum()
    if sum_ == 0: return 1.0
    return (2.0 * intersection) / (sum_ + 1e-6)

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


class BraTSInferenceDataset(Dataset):
    """Dataset for inference - loads full volume without cropping."""
    def __init__(self, patient_ids, base_path, has_labels=True):
        import nibabel as nib
        self.nib = nib
        self.base_path = Path(base_path)
        self.has_labels = has_labels
        self.modalities = ['flair', 't1', 't1ce', 't2']
        
        # Filter to valid patients only
        self.patients = []
        for pid in patient_ids:
            patient_dir = self.base_path / pid
            if not patient_dir.exists():
                continue
            
            # Check all modalities exist and are valid
            valid = True
            for mod in self.modalities:
                fpath = _find_file(patient_dir, pid, mod)
                if fpath is None or not is_valid_nifti(fpath):
                    valid = False
                    break
            
            # Check segmentation if needed
            if has_labels:
                seg_path = _find_file(patient_dir, pid, "seg")
                if seg_path is None or not is_valid_nifti(seg_path):
                    print(f"[Skip] {pid}: Invalid/empty segmentation file")
                    valid = False
            
            if valid:
                self.patients.append(pid)
        
        print(f"[BraTSInferenceDataset] {len(self.patients)}/{len(patient_ids)} patients valid")
    
    def __len__(self):
        return len(self.patients)
    
    def _normalize(self, image):
        """Z-score normalize each channel using brain mask."""
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
        """Convert BraTS labels to WT/TC/ET regions."""
        wt = ((label == 1) | (label == 2) | (label == 4)).astype(np.float32)
        tc = ((label == 1) | (label == 4)).astype(np.float32)
        et = (label == 4).astype(np.float32)
        return np.stack([wt, tc, et], axis=0)
    
    def __getitem__(self, idx):
        pid = self.patients[idx]
        patient_dir = self.base_path / pid
        
        # Load modalities
        images = []
        for mod in self.modalities:
            fpath = _find_file(patient_dir, pid, mod)
            img = self.nib.load(fpath).get_fdata().astype(np.float32)
            images.append(img)
        image = np.stack(images, axis=0)
        image = self._normalize(image)
        
        result = {
            "id": pid,
            "image": torch.from_numpy(image)
        }
        
        # Load label if available
        if self.has_labels:
            seg_path = _find_file(patient_dir, pid, "seg")
            label = self.nib.load(seg_path).get_fdata().astype(np.int32)
            label = self._to_regions(label)
            result["label"] = torch.from_numpy(label)
        
        return result



def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU_COUNT = torch.cuda.device_count()
    BATCH_SIZE = max(1, GPU_COUNT)
    ROI_SIZE = (96, 96, 96)
    SW_BATCH_SIZE = 8 * BATCH_SIZE
    SW_OVERLAP = 0.5
    
    print(f"========================================")
    print(f" MODE: {MODE}")
    print(f" TTA: {USE_TTA} | Remove Small Objects: {REMOVE_SMALL_OBJECTS}")
    print(f"========================================")

    # 1. SETUP DATASET
    if MODE == "VERIFY":
        # Scan training dir, take last 20 patients as a 'Local Test Set'
        all_dirs = sorted([d for d in os.listdir(BRATS_DATA_PATH) if os.path.isdir(os.path.join(BRATS_DATA_PATH, d))])
        patient_ids = all_dirs[-20:] # Last 20 for verification
        dataset = BraTSInferenceDataset(patient_ids, BRATS_DATA_PATH, has_labels=True)
        print(f"Verifying on {len(patient_ids)} patients from training data...")
    else:
        # Submission Mode
        TEST_PATH_ACTUAL = "/kaggle/input/instant-odc-ai-hackathon/test"
        SAMPLE_SUB = "/kaggle/input/instant-odc-ai-hackathon/sample_submission.csv"
        df = pd.read_csv(SAMPLE_SUB)
        patient_ids = df['id'].apply(lambda x: "_".join(x.split('_')[:-1])).unique().tolist()
        dataset = BraTSInferenceDataset(patient_ids, TEST_PATH_ACTUAL, has_labels=False)
        print(f"Generating submission for {len(patient_ids)} patients...")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. LOAD MODEL
    print(f"Loading model from {MODEL_PATH}...")
    model = TransUNet3D(num_classes=3).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        print("Model file not found!")
        return

    if GPU_COUNT > 1: model = nn.DataParallel(model)
    model.eval()
    
    # 3. INFERENCE LOOP
    metrics = {'WT': [], 'TC': [], 'ET': [], 'Mean': []}
    metrics_post = {'WT': [], 'TC': [], 'ET': [], 'Mean': []}
    submission_rows = []

    print("Starting Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch["image"].to(DEVICE)
            batch_ids = batch["id"]
            
            # --- PREDICTION (Standard) ---
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                preds = sliding_window_inference(images, ROI_SIZE, SW_BATCH_SIZE, model, SW_OVERLAP, mode="gaussian", device=DEVICE)
                preds = torch.sigmoid(preds)
            
            # --- TTA (Test Time Augmentation) ---
            if USE_TTA:
                # Flip Depth
                images_flip_d = torch.flip(images, [2])
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                    preds_d = sliding_window_inference(images_flip_d, ROI_SIZE, SW_BATCH_SIZE, model, SW_OVERLAP, mode="gaussian", device=DEVICE)
                    preds_d = torch.flip(torch.sigmoid(preds_d), [2])
                
                # Flip Height
                images_flip_h = torch.flip(images, [3])
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                    preds_h = sliding_window_inference(images_flip_h, ROI_SIZE, SW_BATCH_SIZE, model, SW_OVERLAP, mode="gaussian", device=DEVICE)
                    preds_h = torch.flip(torch.sigmoid(preds_h), [3])

                # Average
                preds = (preds + preds_d + preds_h) / 3.0

            preds_np = (preds > 0.5).float().cpu().numpy()

            # Process Batch
            for i, p_id in enumerate(batch_ids):
                # Raw Masks
                wt = preds_np[i][0]
                tc = preds_np[i][1]
                et = preds_np[i][2]
                
                # Copy for Post-Processing
                wt_post, tc_post, et_post = wt.copy(), tc.copy(), et.copy()

                # --- APPLY POST PROCESSING ---
                if REMOVE_SMALL_OBJECTS:
                    wt_post = keep_largest_blob(wt_post) # WT is usually one big blob
                    tc_post = remove_small_blobs(tc_post, min_size=MIN_PIXELS)
                    et_post = remove_small_blobs(et_post, min_size=MIN_PIXELS)
                
                # --- CONSISTENCY ENFORCEMENT ---
                # ET must be inside TC, TC must be inside WT
                tc_post[et_post == 1] = 1
                wt_post[tc_post == 1] = 1

                # --- VERIFICATION METRICS ---
                if MODE == "VERIFY":
                    gt = batch["label"][i].numpy()
                    
                    # Original Scores
                    dice_wt = calculate_dice(wt, gt[0])
                    dice_tc = calculate_dice(tc, gt[1])
                    dice_et = calculate_dice(et, gt[2])
                    metrics['WT'].append(dice_wt); metrics['TC'].append(dice_tc); metrics['ET'].append(dice_et)
                    metrics['Mean'].append(np.mean([dice_wt, dice_tc, dice_et]))
                    
                    # Post-Processed Scores
                    dice_wt_p = calculate_dice(wt_post, gt[0])
                    dice_tc_p = calculate_dice(tc_post, gt[1])
                    dice_et_p = calculate_dice(et_post, gt[2])
                    metrics_post['WT'].append(dice_wt_p); metrics_post['TC'].append(dice_tc_p); metrics_post['ET'].append(dice_et_p)
                    metrics_post['Mean'].append(np.mean([dice_wt_p, dice_tc_p, dice_et_p]))

                # --- SUBMISSION PREPARATION ---
                else:
                    # BraTS Submission Logic
                    # 4: ET
                    # 1: NCR = TC - ET
                    # 2: ED = WT - TC
                    l4 = et_post
                    l1 = np.logical_and(tc_post == 1, et_post == 0).astype(np.float32)
                    l2 = np.logical_and(wt_post == 1, tc_post == 0).astype(np.float32)
                    
                    submission_rows.append([f"{p_id}_1", rle_encode(l1)])
                    submission_rows.append([f"{p_id}_2", rle_encode(l2)])
                    submission_rows.append([f"{p_id}_4", rle_encode(l4)])

    # 4. REPORT OR SAVE
    if MODE == "VERIFY":
        print("\n=== VERIFICATION RESULTS ===")
        print(f"Baseline Mean Dice: {np.mean(metrics['Mean']):.4f}")
        print(f"Post-Proc Mean Dice: {np.mean(metrics_post['Mean']):.4f}")
        print("-" * 30)
        print(f"WT: {np.mean(metrics['WT']):.4f} -> {np.mean(metrics_post['WT']):.4f}")
        print(f"TC: {np.mean(metrics['TC']):.4f} -> {np.mean(metrics_post['TC']):.4f}")
        print(f"ET: {np.mean(metrics['ET']):.4f} -> {np.mean(metrics_post['ET']):.4f}")
        
        diff = np.mean(metrics_post['Mean']) - np.mean(metrics['Mean'])
        if diff > 0: print(f"\nSUCCESS: Post-processing improved score by +{diff:.4f}")
        else: print(f"\nWARNING: Post-processing hurt score by {diff:.4f}")
        
    else:
        df_sub = pd.DataFrame(submission_rows, columns=['id', 'rle'])
        
        # Merge with sample submission to ensure correct row order/count
        sample = pd.read_csv(SAMPLE_SUB)
        
        # Helper to map IDs robustly
        rle_map = dict(zip(df_sub['id'], df_sub['rle']))
        
        def map_rle(row_id):
            # Try exact match
            if row_id in rle_map: return rle_map[row_id]
            # Try suffix variants (1 vs 11)
            base = "_".join(row_id.split('_')[:-1])
            suffix = row_id.split('_')[-1]
            if suffix in ['1', '11', '01']: key = f"{base}_1"
            elif suffix in ['2', '21', '02']: key = f"{base}_2"
            elif suffix in ['4', '41', '04']: key = f"{base}_4"
            else: key = row_id
            return rle_map.get(key, "")

        sample['rle'] = sample['id'].apply(map_rle).fillna("")
        sample.to_csv(OUTPUT_CSV, index=False)
        print(f"Submission saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()