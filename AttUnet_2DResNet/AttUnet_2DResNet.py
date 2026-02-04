"""
Baseline 4: Hybrid 3D U-Net + 2D ResNet-18 (T1ce) with Late Fusion
==================================================================
Optimized for Kaggle Dual T4 GPUs. Self-contained - no external learning module needed.

Features:
- 80/10/10 Data Split with Test Evaluation
- Resume Training Capability
- Best & Last Checkpoint Saving
- Training Plots (Loss & Dice)
- Inlined Ranger2020 Optimizer & GradualWarmupScheduler
- DataParallel for Multi-GPU

Architecture:
    INPUT (B, 4, D, H, W) --> 3D AttUNet Encoder + 2D ResNet-18 (T1ce) --> Fusion --> Decoder --> Output
"""

import os
import sys
import math
import json
import time
import random
import warnings
import itertools
from pathlib import Path
from functools import partial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MONAI_DATA_COLLATE'] = '0'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Kaggle
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, _LRScheduler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torchvision.models as models
import nibabel as nib
BRATS_DATA_PATH = Path("/kaggle/input/brain-tumor-segmentation-hackathon")
OUTPUT_DIR = Path("/kaggle/working/hybrid_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 2          # 1 per GPU with 2 GPUs
NUM_CLASSES = 3         # WT, TC, ET
NUM_WORKERS = 0         # Safer for CUDA init on Kaggle
USE_DATA_PARALLEL = False  # Set True to try multi-GPU; disable if CUDA errors

FEATURES_3D = [24, 48, 96, 192]
FEATURES_2D = 128
DROPOUT = 0.1
FUSION_MODE = 'concat'  # 'concat' or 'attention'
PRETRAINED_2D = True

EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
VAL_INTERVAL = 1

RESUME_FROM = None

def centralized_gradient(x, use_gc=True, gc_conv_only=False):
    """Gradient Centralization - https://github.com/Yonghongwei/Gradient-Centralization"""
    if use_gc:
        if gc_conv_only:
            if len(list(x.size())) > 3:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
        else:
            if len(list(x.size())) > 1:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x

class Ranger2020(Optimizer):
    """RAdam + Lookahead + Gradient Centralization combined optimizer."""
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5,
                 betas=(0.95, 0.999), eps=1e-5, weight_decay=0, use_gc=True, gc_conv_only=False):
        defaults = dict(lr=lr, alpha=alpha, k=k, betas=betas, N_sma_threshhold=N_sma_threshhold,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.N_sma_threshhold = N_sma_threshhold
        self.alpha, self.k = alpha, k
        self.radam_buffer = [[None, None, None] for _ in range(10)]
        self.use_gc, self.gc_conv_only, self.eps = use_gc, gc_conv_only, eps

    def step(self, closure=None):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data.float()
                if grad.is_sparse: raise RuntimeError('Ranger does not support sparse gradients')
                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                grad = centralized_gradient(grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)
                state['step'] += 1
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * 
                                              (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    G_grad = exp_avg / denom
                else:
                    G_grad = exp_avg
                if group['weight_decay'] != 0:
                    G_grad.add_(p_data_fp32, alpha=group['weight_decay'])
                p_data_fp32.add_(G_grad, alpha=-step_size * group['lr'])
                p.data.copy_(p_data_fp32)
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    p.data.copy_(slow_p)
        return loss


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up LR then hand off to after_scheduler."""
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.: raise ValueError('multiplier should be >= 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None: self.after_scheduler.step(None)
            else: self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


def dice_coefficient_regions(pred, target, smooth=1e-5):
    pred_binary = (pred > 0.5).float()
    region_names = ['TC', 'WT', 'ET']
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
    if norm_type == 'batch': return nn.BatchNorm3d
    elif norm_type == 'instance': return nn.InstanceNorm3d
    elif norm_type == 'group': return lambda channels: nn.GroupNorm(num_groups, channels)
    raise NotImplementedError(f"Norm type {norm_type} not found")

def get_act(act_type):
    if act_type == 'relu': return nn.ReLU(inplace=True)
    elif act_type == 'leaky_relu': return nn.LeakyReLU(0.01, inplace=True)
    raise NotImplementedError(f"Activation type {act_type} not found")

def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'kaiming': init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None: init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class TTAChain:
    def __init__(self, functions): self.functions = functions or []
    def __call__(self, x):
        for f in self.functions: x = f(x)
        return x

class TTATransformer:
    def __init__(self, image_pipeline, mask_pipeline):
        self.image_pipeline, self.mask_pipeline = image_pipeline, mask_pipeline
    def augment_image(self, image): return self.image_pipeline(image)
    def deaugment_mask(self, mask): return self.mask_pipeline(mask)

class TTAHorizontalFlip:
    def __init__(self): self.pname, self.params = "apply", [False, True]
    def apply_aug_image(self, image, apply=False): return image.flip(3) if apply else image
    def apply_deaug_mask(self, mask, apply=False): return mask.flip(3) if apply else mask

class TTAVerticalFlip:
    def __init__(self): self.pname, self.params = "apply", [False, True]
    def apply_aug_image(self, image, apply=False): return image.flip(2) if apply else image
    def apply_deaug_mask(self, mask, apply=False): return mask.flip(2) if apply else mask

class TTACompose:
    def __init__(self, transforms):
        self.aug_transforms = transforms
        self.aug_transform_parameters = list(itertools.product(*[t.params for t in transforms]))
        self.deaug_transforms = transforms[::-1]
        self.deaug_transform_parameters = [p[::-1] for p in self.aug_transform_parameters]
    def __iter__(self):
        for aug_params, deaug_params in zip(self.aug_transform_parameters, self.deaug_transform_parameters):
            image_aug_chain = TTAChain([partial(t.apply_aug_image, **{t.pname: p}) for t, p in zip(self.aug_transforms, aug_params)])
            mask_deaug_chain = TTAChain([partial(t.apply_deaug_mask, **{t.pname: p}) for t, p in zip(self.deaug_transforms, deaug_params)])
            yield TTATransformer(image_pipeline=image_aug_chain, mask_pipeline=mask_deaug_chain)
    def __len__(self): return len(self.aug_transform_parameters)


class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal Loss for handling class imbalance."""
    def __init__(self, gamma=2.0, alpha=0.25, lambda_dice=1.0, lambda_focal=0.5, smooth=1e-5):
        super().__init__()
        self.gamma, self.alpha, self.smooth = gamma, alpha, smooth
        self.lambda_dice, self.lambda_focal = lambda_dice, lambda_focal

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        # Dice Loss (per-channel, then average)
        dice_loss = 0.0
        for i in range(pred.shape[1]):
            p, t = pred_sig[:, i].flatten(1), target[:, i].flatten(1)
            inter = (p * t).sum(1)
            union = (p * p).sum(1) + (t * t).sum(1)
            dice_loss += (1 - (2 * inter + self.smooth) / (union + self.smooth)).mean()
        dice_loss /= pred.shape[1]
        # Focal Loss
        pt = pred_sig * target + (1 - pred_sig) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        focal_loss = (self.alpha * focal_weight * bce).mean()
        return self.lambda_dice * dice_loss + self.lambda_focal * focal_loss


def is_valid_nifti(path: Path) -> bool:
    """Check if path is a valid NIfTI file."""
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= 100
    except Exception:
        return False

def _find_file(patient_dir: Path, patient_id: str, suffix: str) -> Path | None:
    """Find a file matching pattern {patient_id}_{suffix}.nii(.gz) with flexible structure handling."""
    # Priority 1: .nii.gz directly in patient_dir
    for ext in ['.nii.gz', '.nii']:
        candidate = patient_dir / f"{patient_id}_{suffix}{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate
    # Priority 2: Recursive search
    for pattern in [f"{patient_id}_{suffix}.nii.gz", f"{patient_id}_{suffix}.nii",
                    f"*_{suffix}.nii.gz", f"*_{suffix}.nii", f"*_brain_{suffix}.nii*"]:
        matches = [m for m in patient_dir.rglob(pattern) if m.is_file()]
        if matches: return matches[0]
    # Priority 3: Fake-file folders
    fake_folder = patient_dir / f"{patient_id}_{suffix}.nii"
    if fake_folder.exists() and fake_folder.is_dir():
        inner = list(fake_folder.glob("*.nii*"))
        if inner: return inner[0]
    # Priority 4: seg variations
    if suffix == "seg":
        for pat in ["*_final_seg.nii*", "*_seg.nii*"]:
            matches = [m for m in patient_dir.rglob(pat) if m.is_file()]
            if matches: return matches[0]
    return None

def find_brats_files(base_path: Path):
    """Find all valid BraTS patients with all 4 modalities + segmentation."""
    patients, modalities = [], ['flair', 't1', 't1ce', 't2']
    if not base_path.exists():
        print(f"[Error] Path {base_path} does not exist.")
        return []
    patient_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    print(f"[Data] Scanning {len(patient_dirs)} directories...")
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        img_files = []
        for mod in modalities:
            fp = _find_file(patient_dir, patient_id, mod)
            if fp and is_valid_nifti(fp): img_files.append(str(fp))
        seg_path = _find_file(patient_dir, patient_id, "seg")
        if len(img_files) == 4 and seg_path and is_valid_nifti(seg_path):
            patients.append({'id': patient_id, 'images': img_files, 'label': str(seg_path)})
    print(f"[Data] Found {len(patients)} valid patients.")
    return patients

class BraTSDataset(torch.utils.data.Dataset):
    """BraTS dataset with on-the-fly NIfTI loading and smart cropping."""
    def __init__(self, patient_list, patch_size, augment=True):
        self.patients, self.patch_size, self.augment = patient_list, patch_size, augment

    def __len__(self): return len(self.patients)

    def _load_nifti(self, path):
        return nib.load(path).get_fdata().astype(np.float32)

    def _normalize(self, image):
        """Z-score normalization per channel, brain region only."""
        brain_mask = image[0] > 0
        for c in range(image.shape[0]):
            if brain_mask.sum() > 0:
                mean, std = image[c][brain_mask].mean(), image[c][brain_mask].std()
                if std > 0: image[c] = (image[c] - mean) / std
        return image

    def _to_regions(self, label):
        """Convert BraTS labels to 3 regions: WT, TC, ET."""
        wt = ((label == 1) | (label == 2) | (label == 4)).astype(np.float32)
        tc = ((label == 1) | (label == 4)).astype(np.float32)
        et = (label == 4).astype(np.float32)
        return np.stack([wt, tc, et], axis=0)

    def _smart_crop(self, image, label):
        """66% chance to center on tumor if it exists."""
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        if self.augment and random.random() < 0.66:
            tumor_coords = np.argwhere(label[0] > 0)
            if len(tumor_coords) > 0:
                center = tumor_coords[random.randint(0, len(tumor_coords) - 1)]
                ds = min(max(0, center[0] - pd // 2), max(0, d - pd))
                hs = min(max(0, center[1] - ph // 2), max(0, h - ph))
                ws = min(max(0, center[2] - pw // 2), max(0, w - pw))
                return image[:, ds:ds+pd, hs:hs+ph, ws:ws+pw], label[:, ds:ds+pd, hs:hs+ph, ws:ws+pw]
        # Random/center crop fallback
        if self.augment:
            ds, hs, ws = random.randint(0, max(0, d-pd)), random.randint(0, max(0, h-ph)), random.randint(0, max(0, w-pw))
        else:
            ds, hs, ws = max(0, (d-pd)//2), max(0, (h-ph)//2), max(0, (w-pw)//2)
        return image[:, ds:ds+pd, hs:hs+ph, ws:ws+pw], label[:, ds:ds+pd, hs:hs+ph, ws:ws+pw]

    def _pad_if_needed(self, image, label):
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        if d < pd or h < ph or w < pw:
            pad_d, pad_h, pad_w = max(0, pd-d), max(0, ph-h), max(0, pw-w)
            image = np.pad(image, ((0,0), (0,pad_d), (0,pad_h), (0,pad_w)))
            label = np.pad(label, ((0,0), (0,pad_d), (0,pad_h), (0,pad_w)))
        return image, label

    def _augment(self, image, label):
        """Apply random augmentations."""
        # Flips
        for axis in [1, 2, 3]:
            if random.random() > 0.5:
                image, label = np.flip(image, axis).copy(), np.flip(label, axis).copy()
        # Rotation
        if random.random() > 0.7:
            k = random.randint(1, 3)
            image, label = np.rot90(image, k, axes=(2, 3)).copy(), np.rot90(label, k, axes=(2, 3)).copy()
        # Intensity
        if random.random() > 0.5: image = image + (random.random() - 0.5) * 0.3
        if random.random() > 0.7: image = image * (0.9 + random.random() * 0.2)
        if random.random() > 0.8: image = image + np.random.randn(*image.shape).astype(np.float32) * 0.05
        return image, label

    def __getitem__(self, idx):
        try:
            patient = self.patients[idx]
            images = [self._load_nifti(p) for p in patient['images']]
            image = np.stack(images, axis=0)
            label = self._to_regions(self._load_nifti(patient['label']).astype(np.int32))
            image = self._normalize(image)
            image, label = self._pad_if_needed(image, label)
            image, label = self._smart_crop(image, label)
            image, label = self._pad_if_needed(image, label)  # Ensure exact size
            if self.augment: image, label = self._augment(image, label)
            return torch.from_numpy(image.copy()), torch.from_numpy(label.copy())
        except Exception as e:
            print(f"[Warning] Failed to load {self.patients[idx]['id']}: {e}")
            return torch.zeros(4, *self.patch_size), torch.zeros(NUM_CLASSES, *self.patch_size)


class ConvBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer, act, dropout=0.0):
        super().__init__()
        layers = [nn.Conv3d(ch_in, ch_out, 3, 1, 1), norm_layer(ch_out), get_act(act),
                  nn.Conv3d(ch_out, ch_out, 3, 1, 1), norm_layer(ch_out), get_act(act)]
        if dropout > 0: layers.insert(3, nn.Dropout3d(dropout))
        self.conv = nn.Sequential(*layers)
    def forward(self, x): return self.conv(x)

class UpConv3D(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer, act):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                                nn.Conv3d(ch_in, ch_out, 3, 1, 1), norm_layer(ch_out), get_act(act))
    def forward(self, x): return self.up(x)

class AttentionBlock3D(nn.Module):
    def __init__(self, f_g, f_l, f_int, act):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(f_g, f_int, 1), nn.BatchNorm3d(f_int))
        self.W_x = nn.Sequential(nn.Conv3d(f_l, f_int, 1), nn.BatchNorm3d(f_int))
        self.psi = nn.Sequential(nn.Conv3d(f_int, 1, 1), nn.BatchNorm3d(1), nn.Sigmoid())
        self.relu = get_act(act)
    def forward(self, g, x):
        g = g.contiguous()
        x = x.contiguous()
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class ResNet2DBranch(nn.Module):
    """2D ResNet-18 for T1ce slices with pretrained ImageNet weights."""
    def __init__(self, pretrained=True, out_features=128):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # Convert to 1-channel input
        if pretrained:
            new_weight = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
            resnet.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
            resnet.conv1.weight.data = new_weight
        else:
            resnet.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.reduce = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, out_features), nn.ReLU(), nn.Dropout(0.1))
        self.out_features = out_features
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        # CRITICAL: Make tensor contiguous after permute to avoid DataParallel CUDA errors
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * D, 1, H, W)
        # Process through ResNet layers
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Reduce and reshape back
        x = self.reduce(x)
        x = x.view(B, D, self.out_features)
        x = x.permute(0, 2, 1).contiguous()
        return x  # (B, out_features, D)


class LateFusionModule(nn.Module):
    """Fuses 3D features (B,C,D,H,W) with 2D features (B,C,D)."""
    def __init__(self, ch_3d, ch_2d, ch_out, mode='concat'):
        super().__init__()
        self.mode = mode
        if mode == 'concat':
            self.fuse = nn.Sequential(nn.Conv3d(ch_3d + ch_2d, ch_out, 1), nn.GroupNorm(8, ch_out), nn.ReLU(True))
        elif mode == 'attention':
            self.proj_2d = nn.Sequential(nn.Conv1d(ch_2d, ch_3d, 1), nn.Sigmoid())
            self.fuse = nn.Sequential(nn.Conv3d(ch_3d, ch_out, 1), nn.GroupNorm(8, ch_out), nn.ReLU(True))
    
    def forward(self, feat_3d, feat_2d):
        B, C_3d, D_3d, H, W = feat_3d.shape
        _, C_2d, D_2d = feat_2d.shape
        # Interpolate 2D features to match 3D depth if needed
        if D_2d != D_3d:
            feat_2d = F.interpolate(feat_2d.unsqueeze(-1), size=(D_3d, 1), mode='bilinear', align_corners=True).squeeze(-1)
        if self.mode == 'concat':
            # Expand 2D features spatially and ensure contiguity
            feat_2d_spatial = feat_2d.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W).contiguous()
            return self.fuse(torch.cat([feat_3d, feat_2d_spatial], dim=1))
        elif self.mode == 'attention':
            attn = self.proj_2d(feat_2d).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W).contiguous()
            return self.fuse(feat_3d * attn)


class HybridAttUNet(nn.Module):
    """3D AttUNet + 2D ResNet-18 (T1ce) with Late Fusion."""
    def __init__(self, img_ch=4, output_ch=3, features_3d=[24,48,96,192], features_2d=128, dropout=0.1, fusion_mode='concat', pretrained_2d=True):
        super().__init__()
        norm, act = get_norm_layer('group'), 'relu'
        self.Maxpool = nn.MaxPool3d(2, 2)
        # 3D Encoder
        self.Conv1 = ConvBlock3D(img_ch, features_3d[0], norm, act)
        self.Conv2 = ConvBlock3D(features_3d[0], features_3d[1], norm, act, dropout)
        self.Conv3 = ConvBlock3D(features_3d[1], features_3d[2], norm, act, dropout)
        self.Conv4 = ConvBlock3D(features_3d[2], features_3d[3], norm, act, dropout)
        # 2D Branch
        self.resnet_2d = ResNet2DBranch(pretrained=pretrained_2d, out_features=features_2d)
        # Fusion
        self.fusion = LateFusionModule(features_3d[3], features_2d, features_3d[3], fusion_mode)
        # Decoder
        self.Up4 = UpConv3D(features_3d[3], features_3d[2], norm, act)
        self.Att4 = AttentionBlock3D(features_3d[2], features_3d[2], features_3d[1], act)
        self.Up_conv4 = ConvBlock3D(features_3d[3], features_3d[2], norm, act)
        self.Up3 = UpConv3D(features_3d[2], features_3d[1], norm, act)
        self.Att3 = AttentionBlock3D(features_3d[1], features_3d[1], features_3d[0], act)
        self.Up_conv3 = ConvBlock3D(features_3d[2], features_3d[1], norm, act)
        self.Up2 = UpConv3D(features_3d[1], features_3d[0], norm, act)
        self.Att2 = AttentionBlock3D(features_3d[0], features_3d[0], features_3d[0]//2, act)
        self.Up_conv2 = ConvBlock3D(features_3d[1], features_3d[0], norm, act)
        self.Conv_1x1 = nn.Conv3d(features_3d[0], output_ch, 1)
        init_weights(self)
    
    def forward(self, x):
        t1ce = x[:, 2:3].contiguous()  # Extract T1ce (channel 2), ensure contiguous
        # 3D Encoder
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        # 2D Branch + Fusion
        feat_2d = self.resnet_2d(t1ce)
        x4 = self.fusion(x4, feat_2d)
        # Decoder
        d4 = self.Up4(x4); d4 = self.Up_conv4(torch.cat([self.Att4(d4, x3), d4], 1))
        d3 = self.Up3(d4); d3 = self.Up_conv3(torch.cat([self.Att3(d3, x2), d3], 1))
        d2 = self.Up2(d3); d2 = self.Up_conv2(torch.cat([self.Att2(d2, x1), d2], 1))
        return self.Conv_1x1(d2)


def evaluate_with_tta(model, dataloader, device, amp_dtype, use_tta=True):
    """Evaluate model with optional TTA."""
    model.eval()
    dice_scores = {'WT': 0.0, 'TC': 0.0, 'ET': 0.0, 'Mean': 0.0}
    tta_transforms = TTACompose([TTAHorizontalFlip(), TTAVerticalFlip()]) if use_tta else None
    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="Evaluating", leave=False):
            imgs, masks = imgs.to(device), masks.to(device).float()
            if use_tta:
                preds_sum = None
                for tf in tta_transforms:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        aug_preds = torch.sigmoid(model(tf.augment_image(imgs)))
                    deaug = tf.deaugment_mask(aug_preds)
                    preds_sum = deaug if preds_sum is None else preds_sum + deaug
                preds = preds_sum / len(tta_transforms)
            else:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    preds = torch.sigmoid(model(imgs))
            scores = dice_coefficient_regions(preds, masks)
            for k in dice_scores: dice_scores[k] += scores[k]
    for k in dice_scores: dice_scores[k] /= len(dataloader)
    return dice_scores


def plot_history(history, save_path):
    """Save training plots."""
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 6))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss Curve'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True, alpha=0.3)
    # Dice
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_dice_mean'], label='Mean Dice', linewidth=2)
    plt.plot(epochs, history['val_dice_wt'], label='WT', linestyle='--')
    plt.plot(epochs, history['val_dice_tc'], label='TC', linestyle='--')
    plt.plot(epochs, history['val_dice_et'], label='ET', linestyle='--')
    plt.title('Validation Dice'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_training():
    """Main training loop with dual-GPU, resume, and plotting."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Running on {device} with {num_gpus} GPU(s)")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- DATA SPLIT (80/10/10) ---
    all_patients = find_brats_files(BRATS_DATA_PATH)
    if not all_patients:
        print("[Error] No patients found. Check BRATS_DATA_PATH.")
        return None
    train_patients, rest = train_test_split(all_patients, test_size=0.2, random_state=42)
    val_patients, test_patients = train_test_split(rest, test_size=0.5, random_state=42)
    print(f"Split: Train {len(train_patients)} | Val {len(val_patients)} | Test {len(test_patients)}")

    # --- DATA LOADERS ---
    train_loader = DataLoader(
        BraTSDataset(train_patients, PATCH_SIZE, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False
    )
    val_loader = DataLoader(
        BraTSDataset(val_patients, PATCH_SIZE, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False
    )

    # --- MODEL (with DataParallel for multi-GPU) ---
    model = HybridAttUNet(
        features_3d=FEATURES_3D, features_2d=FEATURES_2D, dropout=DROPOUT,
        fusion_mode=FUSION_MODE, pretrained_2d=PRETRAINED_2D
    ).to(device)
    if USE_DATA_PARALLEL and num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel across {num_gpus} GPUs")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # --- OPTIMIZER & SCHEDULER ---
    optimizer = Ranger2020(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, use_gc=True)
    after_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=WARMUP_EPOCHS, after_scheduler=after_scheduler)
    criterion = DiceFocalLoss()
    scaler = torch.cuda.amp.GradScaler()

    # --- RESUME LOGIC ---
    start_epoch, best_dice = 0, 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_dice_mean': [], 'val_dice_wt': [], 'val_dice_tc': [], 'val_dice_et': []}
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Resuming from {RESUME_FROM}...")
        ckpt = torch.load(RESUME_FROM, map_location=device)
        state_dict = ckpt['model_state_dict']
        # Handle DataParallel prefix mismatch
        if isinstance(model, nn.DataParallel) and 'module.' not in list(state_dict.keys())[0]:
            model.module.load_state_dict(state_dict)
        elif not isinstance(model, nn.DataParallel) and 'module.' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_sd = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
            model.load_state_dict(new_sd)
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0.0)
        history = ckpt.get('history', history)
        print(f"Resumed at Epoch {start_epoch + 1}, Best Dice: {best_dice:.4f}")

    # --- TRAINING LOOP ---
    amp_dtype = torch.float16  # Use FP16 for T4 GPUs
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=True)
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(device), lbls.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
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
            val_scores = evaluate_with_tta(model, val_loader, device, amp_dtype, use_tta=True)
            # Compute val_loss for plotting
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device).float()
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        val_loss += criterion(model(imgs), lbls).item()
            val_loss /= len(val_loader)

            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_dice_mean'].append(val_scores['Mean'])
            history['val_dice_wt'].append(val_scores['WT'])
            history['val_dice_tc'].append(val_scores['TC'])
            history['val_dice_et'].append(val_scores['ET'])

            print(f"--> Val Loss: {val_loss:.4f} | Dice Mean: {val_scores['Mean']:.4f} (WT: {val_scores['WT']:.3f}, TC: {val_scores['TC']:.3f}, ET: {val_scores['ET']:.3f})")

            # Checkpointing helper
            def save_ckpt(name):
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice, 'history': history
                }, OUTPUT_DIR / name)

            save_ckpt("last_checkpoint.pth")
            if val_scores['Mean'] > best_dice:
                best_dice = val_scores['Mean']
                save_ckpt("best_checkpoint.pth")
                print(f"    [Saved] New Best: {best_dice:.4f}")

            plot_history(history, OUTPUT_DIR / "training_plots.png")

    print("Training Complete.")
    return test_patients

def evaluate_test_set(test_patients):
    """Final evaluation on held-out test set."""
    print("\n" + "="*50 + "\nTEST SET EVALUATION\n" + "="*50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridAttUNet(
        features_3d=FEATURES_3D, features_2d=FEATURES_2D, dropout=DROPOUT,
        fusion_mode=FUSION_MODE, pretrained_2d=PRETRAINED_2D
    ).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    ckpt_path = OUTPUT_DIR / "best_checkpoint.pth"
    if not ckpt_path.exists():
        print("No best checkpoint found. Skipping test evaluation.")
        return
    print(f"Loading best model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_loader = DataLoader(
        BraTSDataset(test_patients, PATCH_SIZE, augment=False),
        batch_size=1, shuffle=False, num_workers=NUM_WORKERS
    )

    final_scores = {'WT': [], 'TC': [], 'ET': [], 'Mean': []}
    print(f"Evaluating on {len(test_patients)} unseen patients...")
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="Testing"):
            imgs, lbls = imgs.to(device), lbls.to(device).float()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = torch.sigmoid(model(imgs))
            scores = dice_coefficient_regions(preds, lbls)
            for k in final_scores: final_scores[k].append(scores[k])

    print("\n=== FINAL TEST METRICS ===")
    results = {k: np.mean(v) for k, v in final_scores.items()}
    print(f"Mean Dice: {results['Mean']:.4f}")
    print(f"WT Dice:   {results['WT']:.4f}")
    print(f"TC Dice:   {results['TC']:.4f}")
    print(f"ET Dice:   {results['ET']:.4f}")
    print(f"Val-Test Gap: {ckpt['best_dice'] - results['Mean']:.4f}")

    with open(OUTPUT_DIR / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    test_patients_list = run_training()
    if test_patients_list:
        evaluate_test_set(test_patients_list)