import os
import random
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
DATASET_DIR = "/kaggle/input/brain-tumor-segmentation-hackathon"
cases = sorted([d for d in os.listdir(DATASET_DIR) if d.startswith("BraTS")])
print("Num cases:", len(cases))
print("First 5:", cases[:5])
def load_nii_first_file(folder_path):
    """
    Loads the first .nii found inside a folder (your dataset uses nested folders).
    """
    for f in os.listdir(folder_path):
        if f.endswith(".nii"):
            return nib.load(os.path.join(folder_path, f)).get_fdata()
    raise FileNotFoundError(f"No .nii file found in: {folder_path}")

def load_case(case_dir):
    """
    Returns volumes: (t1, t1ce, t2, flair, seg)
    each shape: (H, W, D)
    """
    t1_dir    = os.path.join(case_dir, os.path.basename(case_dir) + "_t1.nii")
    t1ce_dir  = os.path.join(case_dir, os.path.basename(case_dir) + "_t1ce.nii")
    t2_dir    = os.path.join(case_dir, os.path.basename(case_dir) + "_t2.nii")
    flair_dir = os.path.join(case_dir, os.path.basename(case_dir) + "_flair.nii")
    seg_dir   = os.path.join(case_dir, os.path.basename(case_dir) + "_seg.nii")

    t1    = load_nii_first_file(t1_dir)
    t1ce  = load_nii_first_file(t1ce_dir)
    t2    = load_nii_first_file(t2_dir)
    flair = load_nii_first_file(flair_dir)
    seg   = load_nii_first_file(seg_dir)

    return t1, t1ce, t2, flair, seg



def find_nii_anywhere(path):
    """
    Find the first .nii or .nii.gz file inside 'path'.
    Works if:
    - file is directly inside the folder
    - file is inside nested subfolders (any depth)
    """
    # 1) If path itself is a file
    if os.path.isfile(path) and (path.endswith(".nii") or path.endswith(".nii.gz")):
        return path

    # 2) Search recursively
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                return os.path.join(root, f)

    raise FileNotFoundError(f"No .nii/.nii.gz file found under: {path}")

def load_nii_flexible(path):
    """
    Loads NIfTI volume from a folder or a file path.
    Returns numpy array.
    """
    nii_path = find_nii_anywhere(path)
    return nib.load(nii_path).get_fdata()

def load_case_flexible(case_dir):
    """
    Loads one BraTS case even if the dataset structure varies.
    Expects the following directories or files exist (either direct file or nested):
    - {case}_t1.nii
    - {case}_t1ce.nii
    - {case}_t2.nii
    - {case}_flair.nii
    - {case}_seg.nii
    """
    case_id = os.path.basename(case_dir)

    t1    = load_nii_flexible(os.path.join(case_dir, f"{case_id}_t1.nii"))
    t1ce  = load_nii_flexible(os.path.join(case_dir, f"{case_id}_t1ce.nii"))
    t2    = load_nii_flexible(os.path.join(case_dir, f"{case_id}_t2.nii"))
    flair = load_nii_flexible(os.path.join(case_dir, f"{case_id}_flair.nii"))
    seg   = load_nii_flexible(os.path.join(case_dir, f"{case_id}_seg.nii"))

    return t1, t1ce, t2, flair, seg


def normalize_mri(x, eps=1e-8):
    """
    Percentile normalization (robust)
    """
    lo, hi = np.percentile(x, (1, 99))
    x = np.clip(x, lo, hi)
    x = (x - x.min()) / (x.max() - x.min() + eps)
    return x

def crop_to_nonzero(vol4c, seg=None, margin=5):
    """
    Crop using non-zero mask from modalities.
    vol4c: (4, H, W, D)
    seg:   (H, W, D) or None
    """
    mask = (vol4c.sum(axis=0) > 0)  # (H, W, D)
    coords = np.array(np.where(mask))
    if coords.size == 0:
        return vol4c, seg

    minc = coords.min(axis=1)
    maxc = coords.max(axis=1)

    minc = np.maximum(minc - margin, 0)
    maxc = maxc + margin + 1

    h0, w0, d0 = minc
    h1, w1, d1 = maxc

    vol4c = vol4c[:, h0:h1, w0:w1, d0:d1]
    if seg is not None:
        seg = seg[h0:h1, w0:w1, d0:d1]
    return vol4c, seg
def random_crop_3d(vol4c, seg, patch_size=(96,96,96), tumor_prob=0.7):
    """
    vol4c: (4,H,W,D), seg:(H,W,D)
    returns patch_vol:(4,ps,ps,ps) patch_seg:(ps,ps,ps)
    """
    _, H, W, D = vol4c.shape
    ph, pw, pd = patch_size

    # pad if needed
    pad_h = max(ph - H, 0)
    pad_w = max(pw - W, 0)
    pad_d = max(pd - D, 0)
    if pad_h or pad_w or pad_d:
        vol4c = np.pad(vol4c, ((0,0),(0,pad_h),(0,pad_w),(0,pad_d)), mode="constant")
        seg   = np.pad(seg,   ((0,pad_h),(0,pad_w),(0,pad_d)), mode="constant")
        _, H, W, D = vol4c.shape

    # choose center: tumor-focused with probability tumor_prob
    if random.random() < tumor_prob and (seg > 0).any():
        tumor_coords = np.array(np.where(seg > 0))  # (3, N)
        idx = random.randint(0, tumor_coords.shape[1]-1)
        ch, cw, cd = tumor_coords[:, idx]
    else:
        ch = random.randint(0, H-1)
        cw = random.randint(0, W-1)
        cd = random.randint(0, D-1)

    # convert center -> start indices
    sh = np.clip(ch - ph//2, 0, H - ph)
    sw = np.clip(cw - pw//2, 0, W - pw)
    sd = np.clip(cd - pd//2, 0, D - pd)

    patch_vol = vol4c[:, sh:sh+ph, sw:sw+pw, sd:sd+pd]
    patch_seg = seg[sh:sh+ph, sw:sw+pw, sd:sd+pd]

    return patch_vol, patch_seg
class BraTSPatchDataset(Dataset):
    def __init__(self, dataset_dir, case_ids, patch_size=(96,96,96), patches_per_case=8):
        self.dataset_dir = dataset_dir
        self.case_ids = case_ids
        self.patch_size = patch_size
        self.patches_per_case = patches_per_case  # how many random patches per case per epoch

    def __len__(self):
        return len(self.case_ids) * self.patches_per_case

    def __getitem__(self, idx):
        case_idx = idx // self.patches_per_case
        case_id = self.case_ids[case_idx]
        case_dir = os.path.join(self.dataset_dir, case_id)

        t1, t1ce, t2, flair, seg = load_case_flexible(case_dir)

        # normalize modalities
        t1    = normalize_mri(t1)
        t1ce  = normalize_mri(t1ce)
        t2    = normalize_mri(t2)
        flair = normalize_mri(flair)

        # stack -> (4,H,W,D)
        vol4c = np.stack([t1, t1ce, t2, flair], axis=0).astype(np.float32)
        seg   = seg.astype(np.int64)

        # crop to brain area
        vol4c, seg = crop_to_nonzero(vol4c, seg, margin=5)

        # binary target: tumor vs background
        seg_bin = (seg > 0).astype(np.int64)

        # random patch crop
        x, y = random_crop_3d(vol4c, seg_bin, patch_size=self.patch_size, tumor_prob=0.7)

        # to torch: x (C,H,W,D) -> (C,D,H,W) preferred by Conv3d? Actually torch Conv3d uses (N,C,D,H,W)
        # We'll store as (C,D,H,W)
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # (4,D,H,W)
        y = torch.from_numpy(y).permute(2, 0, 1)     # (D,H,W)

        return x, y
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
random.shuffle(cases)
split = int(0.8 * len(cases))
train_cases = cases[:split]
val_cases   = cases[split:]

print("Train:", len(train_cases), "Val:", len(val_cases))

PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 1  # usually 1 for 3D due to memory

train_ds = BraTSPatchDataset(DATASET_DIR, train_cases, patch_size=PATCH_SIZE, patches_per_case=6)
val_ds   = BraTSPatchDataset(DATASET_DIR, val_cases,   patch_size=PATCH_SIZE, patches_per_case=2)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=2, base=32):
        super().__init__()
        self.enc1 = DoubleConv3D(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(base, base*2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv3D(base*2, base*4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3D(base*4, base*8)

        self.up3 = nn.ConvTranspose3d(base*8, base*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base*8, base*4)

        self.up2 = nn.ConvTranspose3d(base*4, base*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base*4, base*2)

        self.up1 = nn.ConvTranspose3d(base*2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base*2, base)

        self.out = nn.Conv3d(base, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (N,C,D,H,W)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        return self.out(d1)
def dice_score_from_logits(logits, targets, eps=1e-6):
    """
    logits: (N,2,D,H,W)
    targets: (N,D,H,W) with {0,1}
    """
    probs = torch.softmax(logits, dim=1)[:, 1]  # tumor prob (N,D,H,W)
    preds = (probs > 0.5).float()
    targets = targets.float()

    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return (2*inter + eps) / (union + eps)

class DiceCELoss(nn.Module):
    def __init__(self, ce_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)[:, 1]  # (N,D,H,W)
        targets_f = targets.float()

        inter = (probs * targets_f).sum()
        union = probs.sum() + targets_f.sum()
        dice_loss = 1 - (2*inter + 1e-6) / (union + 1e-6)

        return self.ce_weight * ce_loss + (1 - self.ce_weight) * dice_loss
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)  # should print: cuda

model = UNet3D(in_channels=4, num_classes=2, base=32).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
criterion = DiceCELoss(ce_weight=0.5)

use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler("cuda") if use_amp else None

def run_epoch(loader, train=True, max_steps=None):
    """
    GPU training loop with AMP (no deprecated torch.cuda.amp usage).
    max_steps can limit steps per epoch for faster iteration.
    """
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    steps = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            dice = dice_score_from_logits(logits, y)

        total_loss += float(loss.item())
        total_dice += float(dice.item())
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    return total_loss / max(1, steps), total_dice / max(1, steps)

EPOCHS = 20
TRAIN_STEPS_PER_EPOCH = 30
VAL_STEPS_PER_EPOCH = 10

best_val = 0.0
for epoch in range(1, EPOCHS + 1):
    train_loss, train_dice = run_epoch(train_loader, train=True,  max_steps=TRAIN_STEPS_PER_EPOCH)
    val_loss,   val_dice   = run_epoch(val_loader,   train=False, max_steps=VAL_STEPS_PER_EPOCH)

    print(f"Epoch {epoch:02d} | "
          f"Train Loss {train_loss:.4f} Dice {train_dice:.4f} | "
          f"Val Loss {val_loss:.4f} Dice {val_dice:.4f}")

    if val_dice > best_val:
        best_val = val_dice
        torch.save(model.state_dict(), "best_unet3d.pth")
        print("Saved best model!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet3D(in_channels=4, num_classes=2, base=32).to(device)
model.load_state_dict(torch.load("best_unet3d.pth", map_location=device))
model.eval()

print("Loaded best_unet3d.pth")
import os, numpy as np, nibabel as nib
import torch

def normalize_mri(x, eps=1e-8):
    lo, hi = np.percentile(x, (1, 99))
    x = np.clip(x, lo, hi)
    return (x - x.min()) / (x.max() - x.min() + eps)

def crop_to_nonzero(vol4c, seg=None, margin=5):
    mask = (vol4c.sum(axis=0) > 0)
    coords = np.array(np.where(mask))
    if coords.size == 0:
        return vol4c, seg

    minc = coords.min(axis=1)
    maxc = coords.max(axis=1)

    minc = np.maximum(minc - margin, 0)
    maxc = maxc + margin + 1

    h0, w0, d0 = minc
    h1, w1, d1 = maxc

    vol4c = vol4c[:, h0:h1, w0:w1, d0:d1]
    if seg is not None:
        seg = seg[h0:h1, w0:w1, d0:d1]
    return vol4c, seg

def find_nii_anywhere(path):
    if os.path.isfile(path) and (path.endswith(".nii") or path.endswith(".nii.gz")):
        return path
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No .nii/.nii.gz found under: {path}")

def load_nii_flexible(path):
    p = find_nii_anywhere(path)
    return nib.load(p).get_fdata()

def load_case_flexible(case_dir):
    cid = os.path.basename(case_dir)
    t1    = load_nii_flexible(os.path.join(case_dir, f"{cid}_t1.nii"))
    t1ce  = load_nii_flexible(os.path.join(case_dir, f"{cid}_t1ce.nii"))
    t2    = load_nii_flexible(os.path.join(case_dir, f"{cid}_t2.nii"))
    flair = load_nii_flexible(os.path.join(case_dir, f"{cid}_flair.nii"))
    seg   = load_nii_flexible(os.path.join(case_dir, f"{cid}_seg.nii"))
    return t1, t1ce, t2, flair, seg

@torch.no_grad()
def sliding_window_inference_3d(model, vol4c, patch_size=(96,96,96), overlap=0.5, device="cuda"):
    """
    vol4c: (4,H,W,D) numpy
    returns prob_map: (H,W,D) tumor probability
    """
    model.eval()
    C, H, W, D = vol4c.shape
    ph, pw, pd = patch_size

    sh = max(int(ph * (1 - overlap)), 1)
    sw = max(int(pw * (1 - overlap)), 1)
    sd = max(int(pd * (1 - overlap)), 1)

    prob_acc = np.zeros((H, W, D), dtype=np.float32)
    cnt_acc  = np.zeros((H, W, D), dtype=np.float32)

    for h0 in range(0, H - ph + 1, sh):
        for w0 in range(0, W - pw + 1, sw):
            for d0 in range(0, D - pd + 1, sd):
                patch = vol4c[:, h0:h0+ph, w0:w0+pw, d0:d0+pd]  # (4,ph,pw,pd)

                x = torch.from_numpy(patch.astype(np.float32)).permute(0,3,1,2).unsqueeze(0)  # (1,4,D,H,W)
                x = x.to(device)

                logits = model(x)                          # (1,2,D,H,W)
                probs  = torch.softmax(logits, dim=1)[:, 1] # (1,D,H,W)
                probs  = probs.squeeze(0).cpu().numpy()     # (D,H,W)
                probs  = np.transpose(probs, (1,2,0))       # (H,W,D)

                prob_acc[h0:h0+ph, w0:w0+pw, d0:d0+pd] += probs
                cnt_acc[h0:h0+ph, w0:w0+pw, d0:d0+pd]  += 1.0

    prob_map = prob_acc / np.maximum(cnt_acc, 1e-6)
    return prob_map

CASE_ID = val_cases[0] if len(val_cases) > 0 else train_cases[0]
case_dir = os.path.join(DATASET_DIR, CASE_ID)
print("Demo case:", CASE_ID)

t1, t1ce, t2, flair, seg = load_case_flexible(case_dir)

t1n, t1cen, t2n, flairn = map(normalize_mri, [t1, t1ce, t2, flair])
vol4c = np.stack([t1n, t1cen, t2n, flairn], axis=0).astype(np.float32)

vol4c_c, seg_c = crop_to_nonzero(vol4c, seg, margin=5)
gt_bin = (seg_c > 0).astype(np.uint8)

prob_map = sliding_window_inference_3d(model, vol4c_c, patch_size=PATCH_SIZE, overlap=0.5, device=device)

print("Cropped volume shape:", vol4c_c.shape, "Prob map shape:", prob_map.shape)



def best_tumor_slice_fast(seg_bin):
    counts = seg_bin.sum(axis=(0,1))
    return int(np.argmax(counts))

def dice_np(pred, gt, eps=1e-6):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    return (2*inter + eps) / (union + eps)

k = best_tumor_slice_fast(gt_bin)
thr = 0.5

base = vol4c_c[1]  # 0:T1 1:T1ce 2:T2 3:FLAIR

gt2d   = gt_bin[:, :, k].T
heat2d = prob_map[:, :, k].T
pred2d = (prob_map[:, :, k] > thr).astype(np.uint8).T

pred3d = (prob_map > thr).astype(np.uint8)
d = dice_np(pred3d, gt_bin)
print("Dice (case, binary):", d, "Tumor voxels pred:", int(pred3d.sum()))

plt.figure(figsize=(16,4))

plt.subplot(1,4,1); plt.title("T1ce"); plt.axis("off")
plt.imshow(base[:, :, k].T, cmap="gray")

plt.subplot(1,4,2); plt.title("GT overlay"); plt.axis("off")
plt.imshow(base[:, :, k].T, cmap="gray"); plt.imshow(gt2d, alpha=0.35)

plt.subplot(1,4,3); plt.title(f"Pred overlay (thr={thr})"); plt.axis("off")
plt.imshow(base[:, :, k].T, cmap="gray"); plt.imshow(pred2d, alpha=0.35)

plt.subplot(1,4,4); plt.title("Confidence heatmap"); plt.axis("off")
plt.imshow(base[:, :, k].T, cmap="gray"); plt.imshow(heat2d, alpha=0.45)

plt.tight_layout()
plt.show()

os.makedirs("demo_outputs", exist_ok=True)
out_path = f"demo_outputs/{CASE_ID}_slice{k}_demo.png"
plt.figure(figsize=(16,4))
plt.subplot(1,4,1); plt.axis("off"); plt.imshow(base[:, :, k].T, cmap="gray")
plt.subplot(1,4,2); plt.axis("off"); plt.imshow(base[:, :, k].T, cmap="gray"); plt.imshow(gt2d, alpha=0.35)
plt.subplot(1,4,3); plt.axis("off"); plt.imshow(base[:, :, k].T, cmap="gray"); plt.imshow(pred2d, alpha=0.35)
plt.subplot(1,4,4); plt.axis("off"); plt.imshow(base[:, :, k].T, cmap="gray"); plt.imshow(heat2d, alpha=0.45)
plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", out_path)