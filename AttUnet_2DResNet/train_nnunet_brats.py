"""
BraTS21 Training Script using nnUNet v2
=========================================
Adapted from Kaggle nnUNet training template for local BraTS21 data.

This script:
1. Sets up nnUNet environment variables
2. Converts BraTS data to nnUNet format
3. Runs preprocessing and training using nnUNetv2 CLI

Requirements:
    pip install nnunetv2 nibabel tqdm requests

Usage:
    python train_nnunet_brats.py
"""

import os
import sys
import shutil
import subprocess
import json
import random
from pathlib import Path
from tqdm.auto import tqdm
import torch
import numpy as np
import nibabel as nib

BRATS_DATA_PATH = Path(r"E:\Orange Hackathon\data")
WORKING_DIR = Path(r"E:\Orange Hackathon\nnunet_workdir")
OUTPUT_DIR = Path(r"E:\Orange Hackathon\nnunet_results")

DATASET_ID = 21  # BraTS21 -> Dataset021
DATASET_NAME = f"Dataset{DATASET_ID:03d}_BraTS21"

CONFIGURATION = "3d_fullres"  # Options: 2d, 3d_lowres, 3d_fullres, 3d_cascade_fullres
FOLD = 0  # 0-4 for cross-validation, or "all" for training on all data
TRAINER = "nnUNetTrainer"  # Default trainer runs for 1000 epochs
PLANS = "nnUNetPlans"  # Default plans

NUM_WORKERS = 1  # Must be >= 1 for torch.set_num_threads (nnUNet requirement)
COMMAND_TIMEOUT = 3600 * 16  # 16 hours max

SKIP_CONVERSION = False      # Skip BraTS -> nnUNet format conversion
SKIP_PREPROCESSING = False  # Must complete preprocessing - some files are missing


def setup_environment():
    """Set up nnUNet environment variables."""
    # Create directories
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    nnunet_raw = WORKING_DIR / "nnUNet_raw"
    nnunet_preprocessed = WORKING_DIR / "nnUNet_preprocessed"
    nnunet_results = OUTPUT_DIR / "nnUNet_results"
    
    nnunet_raw.mkdir(parents=True, exist_ok=True)
    nnunet_preprocessed.mkdir(parents=True, exist_ok=True)
    nnunet_results.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    os.environ["nnUNet_raw"] = str(nnunet_raw)
    os.environ["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    os.environ["nnUNet_results"] = str(nnunet_results)
    os.environ["nnUNet_compile"] = "false"  # Disable torch.compile on Windows
    os.environ["nnUNet_n_proc_DA"] = str(NUM_WORKERS)
    os.environ["MASTER_PORT"] = str(random.randint(12000, 19000))
    
    print(f"[Setup] nnUNet_raw: {nnunet_raw}")
    print(f"[Setup] nnUNet_preprocessed: {nnunet_preprocessed}")
    print(f"[Setup] nnUNet_results: {nnunet_results}")
    
    return nnunet_raw, nnunet_preprocessed, nnunet_results


def _link_or_copy(src: Path, dst: Path) -> bool:
    """Create a hardlink on Windows when possible, otherwise copy the file."""
    try:
        if dst.exists():
            return True
        if sys.platform == 'win32':
            try:
                os.link(src, dst)
                return True
            except Exception:
                # fallback to copy if hardlink fails
                shutil.copy2(src, dst)
                return True
        else:
            dst.symlink_to(src.resolve())
            return True
    except PermissionError:
        print(f"[PermissionError] Cannot access: {src}")
        return False
    except Exception as e:
        print(f"[Error] Failed to copy/link {src} -> {dst}: {e}")
        return False


def _copy_and_remap_segmentation(src: Path, dst: Path) -> bool:
    """
    Copy segmentation file with label remapping: 4 -> 3.
    BraTS uses labels {0, 1, 2, 4}, but nnUNet requires consecutive {0, 1, 2, 3}.
    """
    try:
        if dst.exists():
            return True
        
        # Load the segmentation
        seg_nii = nib.load(src)
        seg_data = seg_nii.get_fdata().astype(np.uint8)
        
        # Remap label 4 -> 3
        seg_data[seg_data == 4] = 3
        
        # Create new NIfTI with same header/affine
        new_seg = nib.Nifti1Image(seg_data, seg_nii.affine, seg_nii.header)
        
        # Save as .nii.gz
        nib.save(new_seg, dst)
        return True
    except Exception as e:
        print(f"[Error] Failed to remap segmentation {src} -> {dst}: {e}")
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


def convert_brats_to_nnunet(brats_path: Path, nnunet_raw: Path):
    """
    Convert BraTS2021 data to nnUNet format.
    
    BraTS format:
        BraTS2021_00000/
            BraTS2021_00000_flair.nii(.gz)
            BraTS2021_00000_t1.nii(.gz)
            BraTS2021_00000_t1ce.nii(.gz)
            BraTS2021_00000_t2.nii(.gz)
            BraTS2021_00000_seg.nii(.gz)
    
    nnUNet format:
        Dataset021_BraTS21/
            imagesTr/
                BraTS2021_00000_0000.nii.gz  (FLAIR)
                BraTS2021_00000_0001.nii.gz  (T1)
                BraTS2021_00000_0002.nii.gz  (T1CE)
                BraTS2021_00000_0003.nii.gz  (T2)
            labelsTr/
                BraTS2021_00000.nii.gz
            dataset.json
    """
    dataset_dir = nnunet_raw / DATASET_NAME
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    
    # Check if already converted
    dataset_json = dataset_dir / "dataset.json"
    if dataset_json.exists():
        try:
            with open(dataset_json, "r") as f:
                existing = json.load(f)
            expected = int(existing.get("numTraining", 0))
            label_cases = {p.stem for p in labels_dir.glob("*.nii.gz")} if labels_dir.exists() else set()
            image_cases = set()
            if images_dir.exists():
                for p in images_dir.glob("*.nii.gz"):
                    base, _ = p.stem.rsplit("_", 1)
                    image_cases.add(base)
            if expected == len(label_cases) and label_cases and label_cases == image_cases:
                print(f"[Convert] Dataset already exists at {dataset_dir}")
                return dataset_dir
            else:
                # Resume mode: don't delete, just continue where we left off
                print(f"[Convert] Partial dataset found (labels: {len(label_cases)}, images: {len(image_cases)}). Resuming conversion...")
        except Exception as e:
            print(f"[Convert] Failed to validate existing dataset ({e}). Rebuilding...")
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir, ignore_errors=True)
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Convert] Converting BraTS data to nnUNet format...")
    print(f"[Convert] Source: {brats_path}")
    print(f"[Convert] Target: {dataset_dir}")
    
    # Find all patient folders
    patient_dirs = sorted([d for d in brats_path.iterdir() if d.is_dir() and "BraTS" in d.name])
    print(f"[Convert] Found {len(patient_dirs)} patients")
    
    modality_map = {
        'flair': '0000',
        't1': '0001',
        't1ce': '0002',
        't2': '0003'
    }
    
    training_cases = []
    
    for patient_dir in tqdm(patient_dirs, desc="Converting"):
        patient_id = patient_dir.name
        ok = True
        copied_files = []
        
        # Copy/link modality files
        for modality, nnunet_suffix in modality_map.items():
            # Find the file (handles flat, nested, or mixed structures)
            src_path = _find_file(patient_dir, patient_id, modality)
            dst_path = images_dir / f"{patient_id}_{nnunet_suffix}.nii.gz"
            
            if src_path is not None:
                if not _link_or_copy(src_path, dst_path):
                    ok = False
                else:
                    copied_files.append(dst_path)
            else:
                print(f"[Missing] {patient_id}_{modality}.nii(.gz) not found")
                ok = False
            
            if not ok:
                break
        
        # Find and copy segmentation (with label remapping: 4 -> 3)
        seg_path = _find_file(patient_dir, patient_id, "seg")
        dst_seg = labels_dir / f"{patient_id}.nii.gz"
        
        if seg_path is not None:
            if ok and _copy_and_remap_segmentation(seg_path, dst_seg):
                training_cases.append(patient_id)
            else:
                ok = False
        else:
            # Some test cases may not have segmentation - skip them
            print(f"[Missing] {patient_id}_seg.nii(.gz) not found (test case?)")
            ok = False

        if not ok:
            for f in copied_files:
                try:
                    if f.exists():
                        f.unlink()
                except Exception:
                    pass
            try:
                if dst_seg.exists():
                    dst_seg.unlink()
            except Exception:
                pass
    
    # Create dataset.json
    # Note: We remap BraTS label 4 -> 3 for nnUNet compatibility (consecutive labels required)
    dataset_json_content = {
        "channel_names": {
            "0": "FLAIR",
            "1": "T1",
            "2": "T1CE",
            "3": "T2"
        },
        "labels": {
            "background": 0,
            "necrotic_tumor_core": 1,
            "peritumoral_edema": 2,
            "enhancing_tumor": 3  # Remapped from BraTS label 4 -> 3
        },
        "regions_class_order": [1, 2, 3],
        "numTraining": len(training_cases),
        "file_ending": ".nii.gz",
        "name": "BraTS21"
    }
    
    with open(dataset_json, 'w') as f:
        json.dump(dataset_json_content, f, indent=2)
    
    print(f"[Convert] Conversion complete! {len(training_cases)} cases ready.")
    return dataset_dir


def run_command(cmd: str, desc: str, timeout: int = COMMAND_TIMEOUT):
    """Run a shell command with timeout."""
    print(f"\n>>> {desc}")
    print(f"    Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            timeout=timeout,
            env=os.environ.copy()
        )
        return True
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] Reached limit of {timeout}s. Training may still be in progress.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def check_gpu():
    """Check GPU availability."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[GPU] Found {num_gpus} GPU(s): {gpu_name}")
        return num_gpus
    else:
        print("[GPU] No CUDA GPU available. Training will be slow on CPU.")
        return 0


def run_nnunet_training():
    """Main training pipeline using nnUNet v2."""
    
    print("=" * 70)
    print("BraTS21 Training with nnUNet v2")
    print("=" * 70)
    
    # 1. Setup environment
    nnunet_raw, nnunet_preprocessed, nnunet_results = setup_environment()
    
    # 2. Check GPU
    num_gpus = check_gpu()
    gpu_flag = f"-num_gpus {num_gpus}" if num_gpus > 1 else ""
    
    # 3. Convert data to nnUNet format
    print("\n" + "=" * 70)
    print("Step 1: Data Conversion")
    print("=" * 70)
    if SKIP_CONVERSION:
        print("[Skip] SKIP_CONVERSION=True, using existing converted data.")
        print(f"[Skip] Expected location: {nnunet_raw / DATASET_NAME}")
    else:
        convert_brats_to_nnunet(BRATS_DATA_PATH, nnunet_raw)
    
    # 4. Run preprocessing (fingerprint extraction + planning + preprocessing)
    print("\n" + "=" * 70)
    print("Step 2: Preprocessing")
    print("=" * 70)
    
    if SKIP_PREPROCESSING:
        print("[Skip] SKIP_PREPROCESSING=True, using existing preprocessed data.")
        print(f"[Skip] Expected location: {nnunet_preprocessed / DATASET_NAME}")
    else:
        preprocess_cmd = (
            f"nnUNetv2_plan_and_preprocess -d {DATASET_ID} "
            f"--verify_dataset_integrity "
            f"-c {CONFIGURATION} "
            f"-np 1"  # Use 1 process on Windows
        )
        
        if not run_command(preprocess_cmd, "Running nnUNet preprocessing..."):
            print("[ERROR] Preprocessing failed!")
            return
    
    # 5. Run training
    print("\n" + "=" * 70)
    print("Step 3: Training")
    print("=" * 70)
    
    train_cmd = (
        f"nnUNetv2_train {DATASET_ID} {CONFIGURATION} {FOLD} "
        f"-tr {TRAINER} "
        f"-p {PLANS} "
        f"{gpu_flag}"
    )
    
    if not run_command(train_cmd, f"Training fold {FOLD}..."):
        print("[ERROR] Training failed!")
        return
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Results saved to: {nnunet_results}")
    print(f"Model: {DATASET_NAME}/{TRAINER}__{PLANS}__{CONFIGURATION}/fold_{FOLD}")


def run_nnunet_inference(input_folder: Path, output_folder: Path):
    """Run inference on new data."""
    
    print("=" * 70)
    print("Running nnUNet Inference")
    print("=" * 70)
    
    inference_cmd = (
        f"nnUNetv2_predict "
        f"-i {input_folder} "
        f"-o {output_folder} "
        f"-d {DATASET_ID} "
        f"-c {CONFIGURATION} "
        f"-f {FOLD} "
        f"-tr {TRAINER} "
        f"-p {PLANS}"
    )
    
    run_command(inference_cmd, "Running inference...")
    print(f"Predictions saved to: {output_folder}")


if __name__ == "__main__":
    try:
        import nnunetv2
        try:
            from importlib import metadata as importlib_metadata
            version = getattr(nnunetv2, "__version__", None) or importlib_metadata.version("nnunetv2")
        except Exception:
            version = "unknown"
        print(f"[Setup] nnUNet v2 version: {version}")
    except ImportError:
        print("[ERROR] nnunetv2 is not installed in this environment. Activate your venv and install it first.")
        raise
    
    # Run training
    run_nnunet_training()
    
    # Optionally run inference on test data
    # test_input = Path(r"E:\Orange Hackathon\test_data")
    # test_output = Path(r"E:\Orange Hackathon\predictions")
    # run_nnunet_inference(test_input, test_output)
