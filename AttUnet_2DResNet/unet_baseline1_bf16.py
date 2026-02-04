import os
import sys
import time
import random
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (for MONAI)
os.environ['MONAI_DATA_COLLATE'] = '0'  # Suppress MONAI collate debug output
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Better memory management
warnings.filterwarnings('ignore', category=UserWarning, module='cupy')
warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')

import logging
logging.getLogger('monai').setLevel(logging.WARNING)

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader  # Keep for reference, using ThreadDataLoader instead
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from monai.data import Dataset, ThreadDataLoader, list_data_collate
from monai import transforms as tr
from monai.losses import DiceLoss
from learning.optimizer import Ranger2020
from learning.lr_scheduler import GradualWarmupScheduler

def dice_coefficient_regions(pred, target, smooth=1e-5):
    """
    Calculates Dice Coefficient for BraTS region-based evaluation.
    
    Args:
        pred (torch.Tensor): Prediction tensor (N, 3, H, W, D) - sigmoid outputs.
        target (torch.Tensor): Ground truth tensor (N, 3, H, W, D) - binary masks.
        smooth (float): Small constant to avoid division by zero.
        
    Returns:
        dict: Dice scores for WT, TC, ET and mean.
        
    Note: MONAI's ConvertToMultiChannelBasedOnBratsClasses outputs [TC, WT, ET] order!
        - Channel 0: TC (Tumor Core) = labels 1 + 4
        - Channel 1: WT (Whole Tumor) = labels 1 + 2 + 4
        - Channel 2: ET (Enhancing Tumor) = label 4
    """
    # Threshold predictions at 0.5
    pred_binary = (pred > 0.5).float()
    
    # MONAI channel order: [TC, WT, ET] - map to correct names
    region_names = ['TC', 'WT', 'ET']  # Channel 0=TC, Channel 1=WT, Channel 2=ET
    dice_scores = {}
    
    for i, name in enumerate(region_names):
        p = pred_binary[:, i].reshape(-1)
        t = target[:, i].reshape(-1)
        
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        
        score = (2. * intersection + smooth) / (union + smooth)
        dice_scores[name] = score.item()
    
    dice_scores['Mean'] = np.mean([dice_scores['WT'], dice_scores['TC'], dice_scores['ET']])
    return dice_scores

def get_norm_layer(norm_type, num_groups=8):
    """Factory for normalization layers."""
    if norm_type == 'batch': return nn.BatchNorm3d
    elif norm_type == 'instance': return nn.InstanceNorm3d
    elif norm_type == 'group': return lambda channels: nn.GroupNorm(num_groups, channels)
    raise NotImplementedError(f"Norm type {norm_type} not found")

def get_act(act_type):
    """Factory for activation functions."""
    if act_type == 'relu': return nn.ReLU(inplace=True)
    elif act_type == 'leaky_relu': return nn.LeakyReLU(0.01, inplace=True)
    raise NotImplementedError(f"Activation type {act_type} not found")

def init_weights(net, init_type='kaiming', gain=0.02):
    """Initializes weights of the network."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'kaiming': init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'xavier': init.xavier_normal_(m.weight.data, gain=gain)
            if hasattr(m, 'bias') and m.bias is not None: init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def create_brats_file_list(base_path, patient_ids):
    """
    Creates file list in MONAI format for PersistentDataset.
    
    Args:
        base_path: Root directory of dataset
        patient_ids: List of patient folder names
        
    Returns:
        List of dicts with 'image' and 'label' keys containing file paths
    """
    def find_nii_anywhere(path):
        if os.path.isfile(path) and (path.endswith(".nii") or path.endswith(".nii.gz")):
            return path
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".nii") or f.endswith(".nii.gz"):
                    return os.path.join(root, f)
        return None

    data_list = []
    missing = []
    modalities = ['flair', 't1', 't1ce', 't2']
    
    for p_id in patient_ids:
        patient_dir = os.path.join(base_path, p_id)
        
        # Find all modality files (robust to nested folders)
        img_files = []
        for mod in modalities:
            candidates = [
                os.path.join(patient_dir, f"{p_id}_{mod}.nii.gz"),
                os.path.join(patient_dir, f"{p_id}_{mod}.nii"),
                os.path.join(patient_dir, f"{p_id}_{mod}.nii"),
                os.path.join(patient_dir, f"{p_id}_{mod}.nii.gz"),
            ]
            found = None
            for c in candidates:
                found = find_nii_anywhere(c)
                if found:
                    break
            if found:
                img_files.append(found)
        
        # Find segmentation file (robust to nested folders)
        seg_candidates = [
            os.path.join(patient_dir, f"{p_id}_seg.nii.gz"),
            os.path.join(patient_dir, f"{p_id}_seg.nii"),
        ]
        seg_file = None
        for c in seg_candidates:
            seg_file = find_nii_anywhere(c)
            if seg_file:
                break
        
        # Only add if we have all 4 modalities and segmentation
        if len(img_files) == 4 and seg_file:
            data_list.append({
                'image': img_files,  # List of 4 modality paths
                'label': seg_file
            })
        else:
            missing.append(p_id)

    if missing:
        print(f">>> Skipped {len(missing)} cases with missing files (showing up to 5): {missing[:5]}")

    return data_list


def preprocess_to_npy(data_list, output_dir, patch_size):
    """
    Pre-process BraTS data to NPY files for fast loading.
    Only needs to run once - subsequent runs load from cache.
    
    This converts ~5 seconds/sample loading to ~0.1 seconds/sample.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if already preprocessed
    manifest_file = os.path.join(output_dir, "manifest.npy")
    if os.path.exists(manifest_file):
        print(f">>> Loading preprocessed data from {output_dir}")
        manifest = np.load(manifest_file, allow_pickle=True).item()
        return manifest['files']
    
    print(f">>> Pre-processing {len(data_list)} samples to NPY (one-time operation)...")
    
    # Deterministic preprocessing pipeline
    preprocess = tr.Compose([
        tr.LoadImaged(keys=["image", "label"], image_only=False),
        tr.EnsureChannelFirstd(keys=["label"]),
        tr.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        tr.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        tr.CropForegroundd(keys=["image", "label"], source_key="image"),
        tr.SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
    ])
    
    npy_files = []
    for i, item in enumerate(tqdm(data_list, desc="Preprocessing")):
        out_dir_sample = os.path.join(output_dir, f"sample_{i:04d}")
        img_path = os.path.join(out_dir_sample, "image.npy")
        lbl_path = os.path.join(out_dir_sample, "label.npy")
        
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            npy_files.append(out_dir_sample)
            continue
        
        # Process and save as UNCOMPRESSED .npy (much faster loading)
        os.makedirs(out_dir_sample, exist_ok=True)
        data = preprocess(item)
        np.save(img_path, data["image"].numpy().astype(np.float16))
        np.save(lbl_path, data["label"].numpy().astype(np.uint8))
        npy_files.append(out_dir_sample)
    
    # Save manifest
    np.save(manifest_file, {'files': npy_files})
    print(f">>> Preprocessing complete. Saved to {output_dir}")
    
    return npy_files


class FastBraTSDataset(torch.utils.data.Dataset):
    """
    Fast dataset that loads pre-processed NPY files.
    ~50x faster than loading NIfTI files each time.
    """
    def __init__(self, npy_files, patch_size, augment=True):
        self.files = npy_files
        self.patch_size = patch_size
        self.augment = augment
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Memory-mapped loading - OS handles caching, nearly instant access
        sample_dir = self.files[idx]
        # mmap_mode='r' = read-only memory map, no copy into RAM
        image = np.load(os.path.join(sample_dir, "image.npy"), mmap_mode='r')
        label = np.load(os.path.join(sample_dir, "label.npy"), mmap_mode='r')
        
        # Random crop FIRST (on numpy, slices the mmap without full load)
        if self.augment:
            image, label = self._random_crop_numpy(image, label)
        else:
            image, label = self._center_crop_numpy(image, label)
        
        # Convert to tensor AFTER cropping (only 128^3 instead of full volume)
        image = torch.from_numpy(image.astype(np.float32).copy())
        label = torch.from_numpy(label.astype(np.float32).copy())
        
        # Fast tensor augmentations
        if self.augment:
            image, label = self._random_augment(image, label)
        
        return {"image": image, "label": label}
    
    def _random_crop_numpy(self, image, label):
        """Random 3D crop on numpy/mmap array - only reads the cropped region."""
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        d_start = random.randint(0, max(0, d - pd))
        h_start = random.randint(0, max(0, h - ph))
        w_start = random.randint(0, max(0, w - pw))
        
        image = image[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        label = label[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        return image, label
    
    def _center_crop_numpy(self, image, label):
        """Center crop on numpy/mmap array."""
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        d_start = max(0, (d - pd) // 2)
        h_start = max(0, (h - ph) // 2)
        w_start = max(0, (w - pw) // 2)
        
        image = image[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        label = label[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        return image, label
    
    def _random_augment(self, image, label):
        """Fast tensor-based augmentations."""
        # Random flips (very fast on GPU tensors)
        if random.random() > 0.5:
            image = torch.flip(image, [1])
            label = torch.flip(label, [1])
        if random.random() > 0.5:
            image = torch.flip(image, [2])
            label = torch.flip(label, [2])
        if random.random() > 0.5:
            image = torch.flip(image, [3])
            label = torch.flip(label, [3])
        
        # Random intensity shift
        if random.random() > 0.7:
            shift = (random.random() - 0.5) * 0.2
            image = image + shift
        
        return image, label


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer, act):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 3, 1, 1), norm_layer(ch_out), get_act(act),
            nn.Conv3d(ch_out, ch_out, 3, 1, 1), norm_layer(ch_out), get_act(act)
        )
    def forward(self, x): return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer, act):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(ch_in, ch_out, 3, 1, 1), norm_layer(ch_out), get_act(act)
        )
    def forward(self, x): return self.up(x)

class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int, act):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(f_g, f_int, 1, 1, 0), nn.BatchNorm3d(f_int))
        self.W_x = nn.Sequential(nn.Conv3d(f_l, f_int, 1, 1, 0), nn.BatchNorm3d(f_int))
        self.psi = nn.Sequential(nn.Conv3d(f_int, 1, 1, 1, 0), nn.BatchNorm3d(1), nn.Sigmoid())
        self.relu = get_act(act)
    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)

class AttUnet(nn.Module):
    def __init__(self, img_ch=4, output_ch=3, features=[32, 64, 128, 256]):
        """
        Attention U-Net Implementation.
        Args:
            img_ch: Number of input channels (4 for BraTS: FLAIR, T1, T1ce, T2).
            output_ch: Number of output regions (3 for BraTS: WT, TC, ET).
        """
        super().__init__()
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        norm = get_norm_layer('group')
        act = 'relu'

        self.Conv1 = ConvBlock(img_ch, features[0], norm, act)
        self.Conv2 = ConvBlock(features[0], features[1], norm, act)
        self.Conv3 = ConvBlock(features[1], features[2], norm, act)
        self.Conv4 = ConvBlock(features[2], features[3], norm, act)

        self.Up4 = UpConv(features[3], features[2], norm, act)
        self.Att4 = AttentionBlock(features[2], features[2], features[1], act)
        self.Up_conv4 = ConvBlock(features[3], features[2], norm, act)

        self.Up3 = UpConv(features[2], features[1], norm, act)
        self.Att3 = AttentionBlock(features[1], features[1], features[0], act)
        self.Up_conv3 = ConvBlock(features[2], features[1], norm, act)

        self.Up2 = UpConv(features[1], features[0], norm, act)
        self.Att2 = AttentionBlock(features[0], features[0], features[0]//2, act)
        self.Up_conv2 = ConvBlock(features[1], features[0], norm, act)

        self.Conv_1x1 = nn.Conv3d(features[0], output_ch, 1, 1, 0)
        init_weights(self)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4_up = self.Up_conv4(d4)

        d3 = self.Up3(d4_up)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3_up = self.Up_conv3(d3)

        d2 = self.Up2(d3_up)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2_up = self.Up_conv2(d2)

        return self.Conv_1x1(d2_up)


def train_brats_pipeline():
    """
    Optimized training pipeline for RTX 5060Ti 16GB.
    
    Features:
    - MONAI PersistentDataset for disk-cached data loading (eliminates 120GB re-parsing)
    - torch.compile() with reduce-overhead mode for kernel fusion
    - Ranger2020 optimizer + 5-epoch warmup + cosine decay
    - Region-based segmentation (WT/TC/ET) matching BraTS evaluation
    - Validation every 10 epochs to maximize training throughput
    - BF16 mixed precision for Blackwell architecture
    """
    # --- CONFIGURATION ---
    BASE_PATH = r'C:\Users\Compumarts\Desktop\Orange Hackathon\data'  # Update to your data path
    CACHE_DIR = r'C:\Users\Compumarts\Desktop\Orange Hackathon\brats_cache'  # SSD recommended for cache
    # Fast preset for iteration speed
    FAST_PRESET = False
    PATCH_SIZE = (96, 96, 96) # smaller patch = faster
    BATCH_SIZE = 4
    LR = 3e-4
    EPOCHS = 150
    WARMUP_EPOCHS = 5
    VAL_INTERVAL = 2
    TRAIN_STEPS_PER_EPOCH = 200 if FAST_PRESET else None
    VAL_STEPS_PER_EPOCH = 50 if FAST_PRESET else None
    SAVE_PATH = "AttUnet_BraTS_Best.pth"
    
    # Performance optimizations
    torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for matmul on Ampere+
    torch.backends.cudnn.allow_tf32 = True
    
    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # --- 1. DATA PREPARATION WITH PERSISTENT CACHE ---
    print(">>> Scanning dataset...")
    all_patients = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
    all_patients = [p for p in all_patients if 'BraTS' in p]
    print(f">>> Found {len(all_patients)} patients")
    
    # Split: 80% Train, 20% Val
    train_ids, val_ids = train_test_split(all_patients, test_size=0.2, random_state=42)
    print(f">>> Split: {len(train_ids)} Train, {len(val_ids)} Validation")
    
    # Create file lists in MONAI format
    train_files = create_brats_file_list(BASE_PATH, train_ids)
    val_files = create_brats_file_list(BASE_PATH, val_ids)
    print(f">>> Valid files: {len(train_files)} Train, {len(val_files)} Validation")
    
    # Pre-process to NPY for fast loading (one-time operation)
    train_npy = preprocess_to_npy(train_files, os.path.join(CACHE_DIR, "train_npy"), PATCH_SIZE)
    val_npy = preprocess_to_npy(val_files, os.path.join(CACHE_DIR, "val_npy"), PATCH_SIZE)
    
    # Fast datasets with tensor-based augmentations
    train_ds = FastBraTSDataset(train_npy, PATCH_SIZE, augment=True)
    val_ds = FastBraTSDataset(val_npy, PATCH_SIZE, augment=False)
    
    # DataLoader: num_workers=0 is faster on Windows (avoids spawn overhead)
    # With uncompressed .npy files, single-process loading is ~0.05s/sample
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Single process faster on Windows
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # --- 2. SETUP MODEL & HARDWARE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device} ({torch.cuda.get_device_name(0)})")
    
    # Model with 3 output channels for regions (WT, TC, ET)
    model = AttUnet(img_ch=4, output_ch=3,
                    features=[24, 48, 96, 192] if FAST_PRESET else [32, 64, 128, 256])
    model = model.to(device)
    
    # torch.compile for kernel fusion (10-20% speedup on modern GPUs)
    # Note: Triton backend requires Linux; on Windows, use eager or skip
    if sys.platform != 'win32':
        print(">>> Compiling model with torch.compile (reduce-overhead mode)...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    else:
        print(">>> Skipping torch.compile (Triton not available on Windows)")
        print(">>> For best performance, consider running on Linux with Triton installed")
    
    # Ranger2020: RAdam + Lookahead + Gradient Centralization
    optimizer = Ranger2020(
        model.parameters(),
        lr=LR,
        betas=(0.95, 0.999),
        eps=1e-5,
        weight_decay=0,
        use_gc=True,  # Gradient Centralization
    )
    
    # DiceLoss for segmentation (sigmoid output)
    criterion = DiceLoss(
        include_background=True,
        sigmoid=True,
        squared_pred=True,
        batch=True  # Sum over batch for better gradient signal
    )
    
    # Cosine decay with 5-epoch warmup
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS - WARMUP_EPOCHS,
        eta_min=1e-6
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=WARMUP_EPOCHS,
        after_scheduler=cosine_scheduler
    )
    
    # BF16 mixed precision (RTX 5060Ti = Blackwell, native BF16)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    #amp_dtype = torch.float16

    scaler = torch.amp.GradScaler('cuda') if amp_dtype == torch.float16 else None
    print(f">>> Precision: {'BF16' if amp_dtype == torch.bfloat16 else 'FP16'}")
    
    # --- 3. TRAINING LOOP ---
    best_dice = 0.0
    
    print(f"\n>>> Starting Training: {EPOCHS} epochs, validate every {VAL_INTERVAL} epochs")
    print(f">>> Batch size: {BATCH_SIZE}, Patch size: {PATCH_SIZE}")
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # -- TRAIN --
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS} [Train]", leave=True)
        
        for step_idx, batch_data in enumerate(loop, start=1):
            imgs = batch_data["image"].to(device, non_blocking=True)
            masks = batch_data["label"].to(device, non_blocking=True).float()
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                preds = model(imgs)
                loss = criterion(preds, masks)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            if TRAIN_STEPS_PER_EPOCH is not None and step_idx >= TRAIN_STEPS_PER_EPOCH:
                break
        
        # Step scheduler after each epoch
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        # -- VALIDATION (every VAL_INTERVAL epochs or last epoch) --
        if (epoch + 1) % VAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            model.eval()
            val_dice_scores = {'WT': 0.0, 'TC': 0.0, 'ET': 0.0, 'Mean': 0.0}
            
            with torch.no_grad():
                for step_idx, batch_data in enumerate(tqdm(val_loader, desc=f"Ep {epoch+1}/{EPOCHS} [Val]", leave=False), start=1):
                    imgs = batch_data["image"].to(device, non_blocking=True)
                    masks = batch_data["label"].to(device, non_blocking=True).float()
                    
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        preds = torch.sigmoid(model(imgs))
                    
                    # Calculate region-based Dice scores
                    scores = dice_coefficient_regions(preds, masks)
                    for k in val_dice_scores:
                        val_dice_scores[k] += scores[k]

                    if VAL_STEPS_PER_EPOCH is not None and step_idx >= VAL_STEPS_PER_EPOCH:
                        break
            
            # Average over validation set
            denom = min(len(val_loader), VAL_STEPS_PER_EPOCH) if VAL_STEPS_PER_EPOCH is not None else len(val_loader)
            for k in val_dice_scores:
                val_dice_scores[k] /= max(1, denom)
            
            avg_val_dice = val_dice_scores['Mean']
            
            # VRAM usage
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            
            print(f"\n--> Ep {epoch+1}: Loss {avg_train_loss:.4f} | "
                  f"Dice WT:{val_dice_scores['WT']:.4f} TC:{val_dice_scores['TC']:.4f} ET:{val_dice_scores['ET']:.4f} Mean:{avg_val_dice:.4f} | "
                  f"VRAM {vram_gb:.1f}GB | Time {epoch_time:.1f}s")
            
            # Save Best Model
            if avg_val_dice > best_dice:
                best_dice = avg_val_dice
                # Handle compiled model state dict
                state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_dice': best_dice,
                }, SAVE_PATH)
                print(f"    [Saved] New Best Dice: {best_dice:.4f}")
        else:
            # Non-validation epoch: just print training stats
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            print(f"--> Ep {epoch+1}: Loss {avg_train_loss:.4f} | LR {optimizer.param_groups[0]['lr']:.2e} | "
                  f"VRAM {vram_gb:.1f}GB | Time {epoch_time:.1f}s")

    print(f"\n>>> Training Complete. Best Mean Dice: {best_dice:.4f}")

if __name__ == "__main__":
    train_brats_pipeline()