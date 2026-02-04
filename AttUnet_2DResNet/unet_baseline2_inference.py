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
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from monai.inferers import sliding_window_inference
from monai import transforms as tr
from skimage import measure

MODE = "SUBMISSION"

USE_TTA = True              # Test Time Augmentation
REMOVE_SMALL_OBJECTS = False
MIN_PIXELS = 30             # Minimum size of a blob to keep

BRATS_DATA_PATH = Path("/kaggle/input/brain-tumor-segmentation-hackathon")
TEST_PATH = Path("/kaggle/input/instant-odc-ai-hackathon/test")
MODEL_PATH = "/kaggle/input/unet-brats-e150/pytorch/default/1/AttUnet_BraTS_Best.pth"
SAMPLE_SUB_PATH = "/kaggle/input/instant-odc-ai-hackathon/sample_submission.csv"
OUTPUT_CSV = "submission.csv"


class Chain:
    """Chain a series of calls together in a sequence."""
    def __init__(self, functions):
        self.functions = functions or []
    def __call__(self, x):
        for f in self.functions:
            x = f(x)
        return x

class Transformer:
    """A transform that processes data."""
    def __init__(self, image_pipeline, mask_pipeline):
        self.image_pipeline = image_pipeline
        self.mask_pipeline = mask_pipeline
    def augment_image(self, image):
        return self.image_pipeline(image)
    def deaugment_mask(self, mask):
        return self.mask_pipeline(mask)

class HorizontalFlip:
    """Flip images horizontally (left->right)."""
    def __init__(self):
        self.pname = "apply"
        self.params = [False, True]
    def apply_aug_image(self, image, apply=False):
        return image.flip(3) if apply else image
    def apply_deaug_mask(self, mask, apply=False):
        return mask.flip(3) if apply else mask

class VerticalFlip:
    """Flip images vertically (up->down)."""
    def __init__(self):
        self.pname = "apply"
        self.params = [False, True]
    def apply_aug_image(self, image, apply=False):
        return image.flip(2) if apply else image
    def apply_deaug_mask(self, mask, apply=False):
        return mask.flip(2) if apply else mask

class DepthFlip:
    """Flip images along depth axis."""
    def __init__(self):
        self.pname = "apply"
        self.params = [False, True]
    def apply_aug_image(self, image, apply=False):
        return image.flip(4) if apply else image
    def apply_deaug_mask(self, mask, apply=False):
        return mask.flip(4) if apply else mask

class TTACompose:
    """Compose TTA transforms - generates all combinations."""
    def __init__(self, transforms):
        self.aug_transforms = transforms
        self.aug_transform_parameters = list(itertools.product(*[t.params for t in self.aug_transforms]))
        self.deaug_transforms = transforms[::-1]
        self.deaug_transform_parameters = [p[::-1] for p in self.aug_transform_parameters]

    def __iter__(self):
        for aug_params, deaug_params in zip(self.aug_transform_parameters, self.deaug_transform_parameters):
            image_aug_chain = Chain([partial(t.apply_aug_image, **{t.pname: p})
                                     for t, p in zip(self.aug_transforms, aug_params)])
            mask_deaug_chain = Chain([partial(t.apply_deaug_mask, **{t.pname: p})
                                      for t, p in zip(self.deaug_transforms, deaug_params)])
            yield Transformer(image_pipeline=image_aug_chain, mask_pipeline=mask_deaug_chain)

    def __len__(self):
        return len(self.aug_transform_parameters)

def mock_missing_modules():
    learning = types.ModuleType("learning")
    learning_opt = types.ModuleType("learning.optimizer")
    learning_sch = types.ModuleType("learning.lr_scheduler")
    sys.modules["learning"] = learning
    sys.modules["learning.optimizer"] = learning_opt
    sys.modules["learning.lr_scheduler"] = learning_sch
    
    class DummyObject:
        def __init__(self, *args, **kwargs): pass
        def state_dict(self): return {}
        def load_state_dict(self, state): pass
    
    learning_opt.Ranger2020 = DummyObject
    learning_sch.GradualWarmupScheduler = DummyObject

mock_missing_modules()


def remove_small_blobs(mask, min_size=30):
    """Removes connected components smaller than min_size."""
    if mask.sum() == 0: return mask
    labeled_mask, num_features = measure.label(mask, connectivity=2, return_num=True)
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


def is_valid_nifti(path: Path) -> bool:
    """Check if path is a valid NIfTI file."""
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= 100
    except Exception:
        return False

def _find_file(patient_dir: Path, patient_id: str, suffix: str) -> Path | None:
    """
    Find a file matching pattern {patient_id}_{suffix}.nii(.gz) in patient_dir.
    Handles multiple folder structures.
    """
    # PRIORITY 1: .nii.gz files
    gz_pattern = f"{patient_id}_{suffix}.nii.gz"
    candidate = patient_dir / gz_pattern
    if candidate.exists() and candidate.is_file():
        return candidate
    matches = [m for m in patient_dir.rglob(gz_pattern) if m.is_file()]
    if matches:
        return matches[0]
    
    # PRIORITY 2: .nii files 
    nii_pattern = f"{patient_id}_{suffix}.nii"
    candidate = patient_dir / nii_pattern
    if candidate.exists() and candidate.is_file():
        return candidate
    matches = [m for m in patient_dir.rglob(nii_pattern) if m.is_file()]
    if matches:
        return matches[0]
    
    # PRIORITY 3: Fake file folders
    fake_folder = patient_dir / nii_pattern
    if fake_folder.exists() and fake_folder.is_dir():
        inner_files = list(fake_folder.glob("*.nii")) + list(fake_folder.glob("*.nii.gz"))
        if inner_files:
            return inner_files[0]
    
    # PRIORITY 4: Suffix pattern matching
    suffix_patterns = [f"*_{suffix}.nii.gz", f"*_{suffix}.nii", f"*_brain_{suffix}.nii", f"*_brain_{suffix}.nii.gz"]
    for pattern in suffix_patterns:
        matches = [m for m in patient_dir.rglob(pattern) if m.is_file()]
        if matches:
            return matches[0]
    
    # Handle seg suffix variations
    if suffix == "seg":
        seg_patterns = ["*_final_seg.nii", "*_final_seg.nii.gz", "*_seg.nii", "*_seg.nii.gz"]
        for pattern in seg_patterns:
            matches = [m for m in patient_dir.rglob(pattern) if m.is_file()]
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


def rle_encode(mask):
    """Encodes mask to RLE. Returns empty string for empty mask."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
    return ' '.join(str(x) for x in runs)

def calculate_dice(pred_mask, gt_mask):
    """Calculate Dice score between prediction and ground truth."""
    intersection = (pred_mask * gt_mask).sum()
    sum_ = pred_mask.sum() + gt_mask.sum()
    if sum_ == 0: return 1.0
    return (2.0 * intersection) / (sum_ + 1e-6)

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


class BraTSTestDataset(Dataset):
    """Legacy dataset for submission mode using MONAI transforms."""
    def __init__(self, patient_ids, base_path):
        self.patient_ids = patient_ids
        self.base_path = base_path
        self.modalities = ['flair', 't1', 't1ce', 't2']
        
        self.transforms = tr.Compose([
            tr.LoadImaged(keys=["image"], image_only=False),
            tr.EnsureChannelFirstd(keys=["image"]),
            tr.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        p_id = self.patient_ids[idx]
        patient_dir = os.path.join(self.base_path, p_id)
        is_flat = not os.path.isdir(patient_dir)
        search_dir = self.base_path if is_flat else patient_dir
        
        img_paths = []
        for mod in self.modalities:
            candidates = [
                os.path.join(search_dir, f"{p_id}_{mod}.nii.gz"),
                os.path.join(search_dir, f"{p_id}_{mod}.nii")
            ]
            found_path = None
            for c in candidates:
                if os.path.exists(c):
                    found_path = c
                    break
            
            if not found_path:
                pattern = os.path.join(search_dir, f"*{p_id}*{mod}*.nii*")
                matches = glob.glob(pattern)
                if matches:
                    found_path = matches[0]
            
            if found_path:
                img_paths.append(found_path)
            else:
                raise FileNotFoundError(f"Missing {mod} for {p_id}")

        data = {"image": img_paths, "id": p_id}
        data = self.transforms(data)
        return data


def run_inference():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU_COUNT = torch.cuda.device_count()
    BATCH_SIZE = max(1, GPU_COUNT)
    ROI_SIZE = (96, 96, 96)
    SW_BATCH_SIZE = 8 * BATCH_SIZE
    SW_OVERLAP = 0.5
    
    print(f"========================================")
    print(f" MODE: {MODE}")
    print(f" TTA: {USE_TTA} | Remove Small Objects: {REMOVE_SMALL_OBJECTS}")
    print(f" GPUs: {GPU_COUNT} | Batch Size: {BATCH_SIZE}")
    print(f"========================================")
    
    # Setup TTA transforms
    if USE_TTA:
        tta_transforms = TTACompose([
            HorizontalFlip(),
            VerticalFlip(),
        ])
        print(f"[TTA] Using {len(tta_transforms)} augmentation combinations")
    
    # --- SETUP DATASET ---
    if MODE == "VERIFY":
        # Scan training dir, take last 20 patients as a 'Local Test Set'
        all_dirs = sorted([d.name for d in BRATS_DATA_PATH.iterdir() if d.is_dir()])
        patient_ids = all_dirs[-100:]  # Last 20 for verification
        dataset = BraTSInferenceDataset(patient_ids, BRATS_DATA_PATH, has_labels=True)
        print(f"Verifying on {len(dataset)} patients from training data...")
    else:
        # Submission Mode
        sample_df = pd.read_csv(SAMPLE_SUB_PATH)
        patient_ids = sample_df['id'].apply(lambda x: "_".join(x.split('_')[:-1])).unique().tolist()
        dataset = BraTSInferenceDataset(patient_ids, TEST_PATH, has_labels=False)
        print(f"Generating submission for {len(dataset)} patients...")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- LOAD MODEL ---
    print(f"Loading model from {MODEL_PATH}...")
    model = AttUnet(img_ch=4, output_ch=3, features=[32, 64, 128, 256])
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        print("[ERROR] Model file not found!")
        return

    model.to(DEVICE)
    if GPU_COUNT > 1:
        model = nn.DataParallel(model)
    model.eval()
    
    # --- INFERENCE LOOP ---
    metrics = {'WT': [], 'TC': [], 'ET': [], 'Mean': []}
    metrics_post = {'WT': [], 'TC': [], 'ET': [], 'Mean': []}
    submission_rows = []

    print("Starting Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch["image"].to(DEVICE)
            batch_ids = batch["id"]
            
            # --- PREDICTION ---
            if USE_TTA:
                # TTA: Average predictions across all augmentations
                preds_sum = None
                for transformer in tta_transforms:
                    # Augment image
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
                    
                    # De-augment predictions
                    deaug_preds = transformer.deaugment_mask(torch.sigmoid(aug_preds))
                    
                    if preds_sum is None:
                        preds_sum = deaug_preds
                    else:
                        preds_sum += deaug_preds
                
                preds = preds_sum / len(tta_transforms)
            else:
                # Standard inference without TTA
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

            # Process Batch
            for i, p_id in enumerate(batch_ids):
                # Raw Masks (Note: AttUnet outputs TC, WT, ET order)
                tc = preds_np[i][0]
                wt = preds_np[i][1]
                et = preds_np[i][2]
                
                # Copy for Post-Processing
                wt_post, tc_post, et_post = wt.copy(), tc.copy(), et.copy()

                # --- APPLY POST PROCESSING ---
                if REMOVE_SMALL_OBJECTS:
                    wt_post = keep_largest_blob(wt_post)
                    tc_post = remove_small_blobs(tc_post, min_size=MIN_PIXELS)
                    et_post = remove_small_blobs(et_post, min_size=MIN_PIXELS)
                
                # --- CONSISTENCY ENFORCEMENT ---
                tc_post[et_post == 1] = 1
                wt_post[tc_post == 1] = 1

                # --- VERIFICATION METRICS ---
                if MODE == "VERIFY":
                    gt = batch["label"][i].numpy()
                    
                    # Original Scores
                    dice_wt = calculate_dice(wt, gt[0])
                    dice_tc = calculate_dice(tc, gt[1])
                    dice_et = calculate_dice(et, gt[2])
                    metrics['WT'].append(dice_wt)
                    metrics['TC'].append(dice_tc)
                    metrics['ET'].append(dice_et)
                    metrics['Mean'].append(np.mean([dice_wt, dice_tc, dice_et]))
                    
                    # Post-Processed Scores
                    dice_wt_p = calculate_dice(wt_post, gt[0])
                    dice_tc_p = calculate_dice(tc_post, gt[1])
                    dice_et_p = calculate_dice(et_post, gt[2])
                    metrics_post['WT'].append(dice_wt_p)
                    metrics_post['TC'].append(dice_tc_p)
                    metrics_post['ET'].append(dice_et_p)
                    metrics_post['Mean'].append(np.mean([dice_wt_p, dice_tc_p, dice_et_p]))

                # --- SUBMISSION PREPARATION ---
                else:
                    # BraTS Submission Logic
                    # 4: ET, 1: NCR = TC - ET, 2: ED = WT - TC
                    l4 = et_post
                    l1 = np.logical_and(tc_post == 1, et_post == 0).astype(np.float32)
                    l2 = np.logical_and(wt_post == 1, tc_post == 0).astype(np.float32)
                    
                    submission_rows.append([f"{p_id}_1", rle_encode(l1)])
                    submission_rows.append([f"{p_id}_2", rle_encode(l2)])
                    submission_rows.append([f"{p_id}_4", rle_encode(l4)])

    # --- REPORT OR SAVE ---
    if MODE == "VERIFY":
        print("\n=== VERIFICATION RESULTS ===")
        print(f"Baseline Mean Dice: {np.mean(metrics['Mean']):.4f}")
        print(f"Post-Proc Mean Dice: {np.mean(metrics_post['Mean']):.4f}")
        print("-" * 30)
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
        
        # Merge with sample submission to ensure correct row order
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
        print(f"Submission saved to {OUTPUT_CSV}")
        
        # Verification check
        print("\n>>> VERIFICATION CHECK:")
        check_df = pd.read_csv(OUTPUT_CSV, keep_default_na=False)
        print(f"    Rows: {len(check_df)}")
        print(f"    Empty Predictions: {(check_df['rle'] == '').sum()}")
        print(f"    Filled Predictions: {(check_df['rle'] != '').sum()}")


if __name__ == "__main__":
    run_inference()
import pandas as pd 

df = pd.read_csv("/kaggle/working/submission.csv", index_col=False)
print(df.head())
print(df[df['rle'].isnull()])
df2 = df.copy()
df2['rle'] = df2['rle'].fillna("")
df2.drop_duplicates(inplace=True)
print(df2.info())
output_file = "submission_AttUNet_E150_TTA.csv"
df2.to_csv(output_file, index=False)