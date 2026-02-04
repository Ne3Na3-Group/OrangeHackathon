"""
Baseline 4 Inference Script (Hybrid 3D AttUNet + 2D ResNet-18)
=============================================================
Inference for AttUnet_2DResNet.py

Features:
- TTA (Test Time Augmentation) with flips
- Post-processing (remove small blobs, consistency enforcement)
- VERIFY mode: Evaluate on training data with labels
- SUBMISSION mode: Generate submission.csv for Kaggle
"""

import os
import sys
import glob
import types
import itertools
from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from monai.inferers import sliding_window_inference
from skimage import measure

MODE = "SUBMISSION"

USE_TTA = True              # Test Time Augmentation
REMOVE_SMALL_OBJECTS = False
MIN_PIXELS = 30             # Minimum size of a blob to keep

FEATURES_3D = [24, 48, 96, 192]
FEATURES_2D = 128
DROPOUT = 0.1
FUSION_MODE = 'concat'       # 'concat' or 'attention'
PRETRAINED_2D = True

BRATS_DATA_PATH = Path("/kaggle/input/brain-tumor-segmentation-hackathon")
TEST_PATH = Path("/kaggle/input/instant-odc-ai-hackathon/test")
MODEL_PATH = "/kaggle/input/your-model-dataset/HybridAttUnet_BraTS_Best.pth"  # Update this!
SAMPLE_SUB_PATH = "/kaggle/input/instant-odc-ai-hackathon/sample_submission.csv"
OUTPUT_CSV = "submission_hybrid.csv"


class TTAChain:
    def __init__(self, functions):
        self.functions = functions or []
    def __call__(self, x):
        for f in self.functions:
            x = f(x)
        return x

class TTATransformer:
    def __init__(self, image_pipeline, mask_pipeline):
        self.image_pipeline = image_pipeline
        self.mask_pipeline = mask_pipeline
    def augment_image(self, image):
        return self.image_pipeline(image)
    def deaugment_mask(self, mask):
        return self.mask_pipeline(mask)

class TTAHorizontalFlip:
    def __init__(self):
        self.pname = "apply"
        self.params = [False, True]
    def apply_aug_image(self, image, apply=False):
        return image.flip(3) if apply else image
    def apply_deaug_mask(self, mask, apply=False):
        return mask.flip(3) if apply else mask

class TTAVerticalFlip:
    def __init__(self):
        self.pname = "apply"
        self.params = [False, True]
    def apply_aug_image(self, image, apply=False):
        return image.flip(2) if apply else image
    def apply_deaug_mask(self, mask, apply=False):
        return mask.flip(2) if apply else mask

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

    def __len__(self):
        return len(self.aug_transform_parameters)


def remove_small_blobs(mask, min_size=30):
    if mask.sum() == 0: return mask
    labeled_mask, num_features = measure.label(mask, connectivity=2, return_num=True)
    props = measure.regionprops(labeled_mask)
    new_mask = np.zeros_like(mask)
    for prop in props:
        if prop.area >= min_size:
            new_mask[labeled_mask == prop.label] = 1
    return new_mask

def keep_largest_blob(mask):
    if mask.sum() == 0: return mask
    labeled_mask, num_features = measure.label(mask, connectivity=2, return_num=True)
    if num_features == 0: return mask
    props = measure.regionprops(labeled_mask)
    largest_region = max(props, key=lambda x: x.area)
    new_mask = np.zeros_like(mask)
    new_mask[labeled_mask == largest_region.label] = 1
    return new_mask


def is_valid_nifti(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= 100
    except Exception:
        return False

def _find_file(patient_dir: Path, patient_id: str, suffix: str) -> Path | None:
    # PRIORITY 1: .nii.gz files directly in patient_dir
    for ext in ['.nii.gz', '.nii']:
        candidate = patient_dir / f"{patient_id}_{suffix}{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate
    # PRIORITY 2: Recursive search
    for pattern in [f"{patient_id}_{suffix}.nii.gz", f"{patient_id}_{suffix}.nii",
                    f"*_{suffix}.nii.gz", f"*_{suffix}.nii", f"*_brain_{suffix}.nii*"]:
        matches = [m for m in patient_dir.rglob(pattern) if m.is_file()]
        if matches:
            return matches[0]
    # PRIORITY 3: Fake file folders
    fake_folder = patient_dir / f"{patient_id}_{suffix}.nii"
    if fake_folder.exists() and fake_folder.is_dir():
        inner_files = list(fake_folder.glob("*.nii")) + list(fake_folder.glob("*.nii.gz"))
        if inner_files:
            return inner_files[0]
    # PRIORITY 4: seg suffix variations
    if suffix == "seg":
        for pat in ["*_final_seg.nii*", "*_seg.nii*"]:
            matches = [m for m in patient_dir.rglob(pat) if m.is_file()]
            if matches:
                return matches[0]
    return None


def get_norm_layer(norm_type, num_groups=8):
    if norm_type == 'batch': return nn.BatchNorm3d
    elif norm_type == 'instance': return nn.InstanceNorm3d
    elif norm_type == 'group': return lambda channels: nn.GroupNorm(num_groups, channels)
    raise NotImplementedError(f"Norm type {norm_type} not found")

def get_act(act_type):
    if act_type == 'relu': return nn.ReLU(inplace=True)
    elif act_type == 'leaky_relu': return nn.LeakyReLU(0.01, inplace=True)
    raise NotImplementedError(f"Activation type {act_type} not found")

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
    def __init__(self, pretrained=True, out_features=128):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
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
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * D, 1, H, W)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.reduce(x)
        x = x.view(B, D, self.out_features)
        x = x.permute(0, 2, 1).contiguous()
        return x

class LateFusionModule(nn.Module):
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
        if D_2d != D_3d:
            feat_2d = F.interpolate(feat_2d.unsqueeze(-1), size=(D_3d, 1), mode='bilinear', align_corners=True).squeeze(-1)
        if self.mode == 'concat':
            feat_2d_spatial = feat_2d.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W).contiguous()
            return self.fuse(torch.cat([feat_3d, feat_2d_spatial], dim=1))
        elif self.mode == 'attention':
            attn = self.proj_2d(feat_2d).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W).contiguous()
            return self.fuse(feat_3d * attn)

class HybridAttUNet(nn.Module):
    def __init__(self, img_ch=4, output_ch=3, features_3d=[24,48,96,192], features_2d=128, dropout=0.1, fusion_mode='concat', pretrained_2d=True):
        super().__init__()
        norm, act = get_norm_layer('group'), 'relu'
        self.Maxpool = nn.MaxPool3d(2, 2)
        self.Conv1 = ConvBlock3D(img_ch, features_3d[0], norm, act)
        self.Conv2 = ConvBlock3D(features_3d[0], features_3d[1], norm, act, dropout)
        self.Conv3 = ConvBlock3D(features_3d[1], features_3d[2], norm, act, dropout)
        self.Conv4 = ConvBlock3D(features_3d[2], features_3d[3], norm, act, dropout)
        self.resnet_2d = ResNet2DBranch(pretrained=pretrained_2d, out_features=features_2d)
        self.fusion = LateFusionModule(features_3d[3], features_2d, features_3d[3], fusion_mode)
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

    def forward(self, x):
        t1ce = x[:, 2:3].contiguous()
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        feat_2d = self.resnet_2d(t1ce)
        x4 = self.fusion(x4, feat_2d)
        d4 = self.Up4(x4); d4 = self.Up_conv4(torch.cat([self.Att4(d4, x3), d4], 1))
        d3 = self.Up3(d4); d3 = self.Up_conv3(torch.cat([self.Att3(d3, x2), d3], 1))
        d2 = self.Up2(d3); d2 = self.Up_conv2(torch.cat([self.Att2(d2, x1), d2], 1))
        return self.Conv_1x1(d2)


def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
    return ' '.join(str(x) for x in runs)

def calculate_dice(pred_mask, gt_mask):
    intersection = (pred_mask * gt_mask).sum()
    sum_ = pred_mask.sum() + gt_mask.sum()
    if sum_ == 0: return 1.0
    return (2.0 * intersection) / (sum_ + 1e-6)


class BraTSInferenceDataset(Dataset):
    def __init__(self, patient_ids, base_path, has_labels=True):
        import nibabel as nib
        self.nib = nib
        self.base_path = Path(base_path)
        self.has_labels = has_labels
        self.modalities = ['flair', 't1', 't1ce', 't2']

        self.patients = []
        for pid in patient_ids:
            patient_dir = self.base_path / pid
            if not patient_dir.exists():
                continue
            valid = True
            for mod in self.modalities:
                fpath = _find_file(patient_dir, pid, mod)
                if fpath is None or not is_valid_nifti(fpath):
                    valid = False
                    break
            if has_labels:
                seg_path = _find_file(patient_dir, pid, "seg")
                if seg_path is None or not is_valid_nifti(seg_path):
                    valid = False
            if valid:
                self.patients.append(pid)

        print(f"[BraTSInferenceDataset] {len(self.patients)}/{len(patient_ids)} patients valid")

    def __len__(self):
        return len(self.patients)

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
        wt = ((label == 1) | (label == 2) | (label == 4)).astype(np.float32)
        tc = ((label == 1) | (label == 4)).astype(np.float32)
        et = (label == 4).astype(np.float32)
        return np.stack([wt, tc, et], axis=0)

    def __getitem__(self, idx):
        pid = self.patients[idx]
        patient_dir = self.base_path / pid
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

        if self.has_labels:
            seg_path = _find_file(patient_dir, pid, "seg")
            label = self.nib.load(seg_path).get_fdata().astype(np.int32)
            label = self._to_regions(label)
            result["label"] = torch.from_numpy(label)

        return result


def run_inference():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU_COUNT = torch.cuda.device_count()
    BATCH_SIZE = max(1, GPU_COUNT)
    ROI_SIZE = (96, 96, 96)
    SW_BATCH_SIZE = 8 * BATCH_SIZE
    SW_OVERLAP = 0.5

    print(f"========================================")
    print(f" HYBRID ATTUNET INFERENCE")
    print(f" MODE: {MODE}")
    print(f" TTA: {USE_TTA} | Remove Small Objects: {REMOVE_SMALL_OBJECTS}")
    print(f" Model: HybridAttUNet 3D={FEATURES_3D} 2D={FEATURES_2D} Dropout={DROPOUT} Fusion={FUSION_MODE}")
    print(f" GPUs: {GPU_COUNT} | Batch Size: {BATCH_SIZE}")
    print(f"========================================")

    if USE_TTA:
        tta_transforms = TTACompose([
            TTAHorizontalFlip(),
            TTAVerticalFlip(),
        ])
        print(f"[TTA] Using {len(tta_transforms)} augmentation combinations")

    if MODE == "VERIFY":
        all_dirs = sorted([d.name for d in BRATS_DATA_PATH.iterdir() if d.is_dir()])
        patient_ids = all_dirs[-100:]
        dataset = BraTSInferenceDataset(patient_ids, BRATS_DATA_PATH, has_labels=True)
        print(f"Verifying on {len(dataset)} patients from training data...")
    else:
        sample_df = pd.read_csv(SAMPLE_SUB_PATH)
        patient_ids = sample_df['id'].apply(lambda x: "_".join(x.split('_')[:-1])).unique().tolist()
        dataset = BraTSInferenceDataset(patient_ids, TEST_PATH, has_labels=False)
        print(f"Generating submission for {len(dataset)} patients...")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Loading model from {MODEL_PATH}...")
    model = HybridAttUNet(
        img_ch=4, output_ch=3,
        features_3d=FEATURES_3D, features_2d=FEATURES_2D,
        dropout=DROPOUT, fusion_mode=FUSION_MODE, pretrained_2d=PRETRAINED_2D
    )

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"[OK] Model loaded successfully")
    else:
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        return

    model.to(DEVICE)
    if GPU_COUNT > 1:
        model = nn.DataParallel(model)
    model.eval()

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    metrics = {'WT': [], 'TC': [], 'ET': [], 'Mean': []}
    metrics_post = {'WT': [], 'TC': [], 'ET': [], 'Mean': []}
    submission_rows = []

    print("Starting Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch["image"].to(DEVICE)
            batch_ids = batch["id"]

            if USE_TTA:
                preds_sum = None
                for transformer in tta_transforms:
                    aug_images = transformer.augment_image(images)
                    with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                        aug_preds = sliding_window_inference(
                            inputs=aug_images,
                            roi_size=ROI_SIZE,
                            sw_batch_size=SW_BATCH_SIZE,
                            predictor=model,
                            overlap=SW_OVERLAP,
                            mode="gaussian",
                            device=DEVICE
                        )
                    deaug_preds = transformer.deaugment_mask(torch.sigmoid(aug_preds))
                    if preds_sum is None:
                        preds_sum = deaug_preds
                    else:
                        preds_sum += deaug_preds
                preds = preds_sum / len(tta_transforms)
            else:
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                    preds = sliding_window_inference(
                        inputs=images,
                        roi_size=ROI_SIZE,
                        sw_batch_size=SW_BATCH_SIZE,
                        predictor=model,
                        overlap=SW_OVERLAP,
                        mode="gaussian",
                        device=DEVICE
                    )
                preds = torch.sigmoid(preds)

            preds_np = (preds > 0.5).float().cpu().numpy()

            for i, p_id in enumerate(batch_ids):
                # Model outputs: [WT, TC, ET]
                wt = preds_np[i][0]
                tc = preds_np[i][1]
                et = preds_np[i][2]

                wt_post, tc_post, et_post = wt.copy(), tc.copy(), et.copy()

                if REMOVE_SMALL_OBJECTS:
                    wt_post = keep_largest_blob(wt_post)
                    tc_post = remove_small_blobs(tc_post, min_size=MIN_PIXELS)
                    et_post = remove_small_blobs(et_post, min_size=MIN_PIXELS)

                # ET ⊆ TC ⊆ WT
                tc_post[et_post == 1] = 1
                wt_post[tc_post == 1] = 1

                if MODE == "VERIFY":
                    gt = batch["label"][i].numpy()
                    gt_wt, gt_tc, gt_et = gt[0], gt[1], gt[2]

                    dice_wt = calculate_dice(wt, gt_wt)
                    dice_tc = calculate_dice(tc, gt_tc)
                    dice_et = calculate_dice(et, gt_et)
                    metrics['WT'].append(dice_wt)
                    metrics['TC'].append(dice_tc)
                    metrics['ET'].append(dice_et)
                    metrics['Mean'].append(np.mean([dice_wt, dice_tc, dice_et]))

                    dice_wt_p = calculate_dice(wt_post, gt_wt)
                    dice_tc_p = calculate_dice(tc_post, gt_tc)
                    dice_et_p = calculate_dice(et_post, gt_et)
                    metrics_post['WT'].append(dice_wt_p)
                    metrics_post['TC'].append(dice_tc_p)
                    metrics_post['ET'].append(dice_et_p)
                    metrics_post['Mean'].append(np.mean([dice_wt_p, dice_tc_p, dice_et_p]))
                else:
                    l4 = et_post
                    l1 = np.logical_and(tc_post == 1, et_post == 0).astype(np.float32)
                    l2 = np.logical_and(wt_post == 1, tc_post == 0).astype(np.float32)

                    submission_rows.append([f"{p_id}_1", rle_encode(l1)])
                    submission_rows.append([f"{p_id}_2", rle_encode(l2)])
                    submission_rows.append([f"{p_id}_4", rle_encode(l4)])

    if MODE == "VERIFY":
        print("\n" + "=" * 50)
        print("VERIFICATION RESULTS")
        print("=" * 50)
        print(f"Baseline Mean Dice: {np.mean(metrics['Mean']):.4f}")
        print(f"Post-Proc Mean Dice: {np.mean(metrics_post['Mean']):.4f}")
        print("-" * 50)
        print(f"WT: {np.mean(metrics['WT']):.4f} -> {np.mean(metrics_post['WT']):.4f}")
        print(f"TC: {np.mean(metrics['TC']):.4f} -> {np.mean(metrics_post['TC']):.4f}")
        print(f"ET: {np.mean(metrics['ET']):.4f} -> {np.mean(metrics_post['ET']):.4f}")

        diff = np.mean(metrics_post['Mean']) - np.mean(metrics['Mean'])
        if diff > 0:
            print(f"\nSUCCESS: Post-processing improved score by +{diff:.4f}")
        else:
            print(f"\nWARNING: Post-processing hurt score by {diff:.4f}")
    else:
        df_sub = pd.DataFrame(submission_rows, columns=['id', 'rle'])
        sample = pd.read_csv(SAMPLE_SUB_PATH)
        rle_map = dict(zip(df_sub['id'], df_sub['rle']))

        def map_rle(row_id):
            if row_id in rle_map:
                return rle_map[row_id]
            base = "_".join(row_id.split('_')[:-1])
            suffix = row_id.split('_')[-1]
            if suffix in ['1', '11', '01']:
                key = f"{base}_1"
            elif suffix in ['2', '21', '02']:
                key = f"{base}_2"
            elif suffix in ['4', '41', '04']:
                key = f"{base}_4"
            else:
                key = row_id
            return rle_map.get(key, "")

        sample['rle'] = sample['id'].apply(map_rle).fillna("")
        sample.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSubmission saved to {OUTPUT_CSV}")

        print("\n>>> VERIFICATION CHECK:")
        check_df = pd.read_csv(OUTPUT_CSV, keep_default_na=False)
        print(f"    Rows: {len(check_df)}")
        print(f"    Empty Predictions: {(check_df['rle'] == '').sum()}")
        print(f"    Filled Predictions: {(check_df['rle'] != '').sum()}")


if __name__ == "__main__":
    run_inference()
```
</attachment>