"""
Benchmark Models: U-Net, Att U-Net, TransNet, nnUNet, Hybrid AttUnet+2DResNet
=============================================================================
Per-patch FLOPs at 96^3, params, model size, inference time, and training time
for BF16 vs FP16.

Usage (examples):
  python benchmark_models.py --models unet,attunet,transnet,nnunet,hybrid --dtypes bf16,fp16
  python benchmark_models.py --models unet,attunet,transnet,hybrid --steps 50 --warmup 10

Notes:
- FLOPs are computed per-patch with input shape (1, 4, 96, 96, 96).
- nnUNet requires nnunetv2 + plans.json (optional; will skip if unavailable).
"""

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except Exception:
    FVCORE_AVAILABLE = False


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
    def __init__(self, in_channels=4, num_classes=3, base=32):
        super().__init__()
        self.enc1 = DoubleConv3D(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3D(base * 4, base * 8)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base * 2, base)
        self.out = nn.Conv3d(base, num_classes, kernel_size=1)

    def forward(self, x):
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



def build_unet() -> nn.Module:
    return UNet3D(in_channels=4, num_classes=3, base=32)


def build_attunet() -> nn.Module:
    from unet_baseline1_bf16 import AttUnet
    return AttUnet(img_ch=4, output_ch=3, features=[32, 64, 128, 256])


def build_transnet() -> nn.Module:
    from train_transnet_brats import TransUNet3D
    return TransUNet3D(in_channels=4, num_classes=3)


def build_hybrid() -> nn.Module:
    from AttUnet_2DResNet import HybridAttUNet
    return HybridAttUNet(img_ch=4, output_ch=3, features_3d=[24, 48, 96, 192], features_2d=128, dropout=0.1)


def build_nnunet(plans_path: str, dataset_json: str, configuration: str) -> Optional[nn.Module]:
    """Best-effort nnUNet model builder. Returns None if nnunetv2 is unavailable."""
    try:
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.utilities.network_initialization import get_network_from_plans
        import json

        plans_manager = PlansManager(plans_path)
        with open(dataset_json, "r") as f:
            dataset_info = json.load(f)

        # Build network from plans
        network = get_network_from_plans(
            plans_manager,
            dataset_info,
            configuration,
            num_input_channels=4,
            num_classes=3,
            deep_supervision=False,
        )
        return network
    except Exception:
        return None


MODEL_BUILDERS = {
    "unet": build_unet,
    "attunet": build_attunet,
    "transnet": build_transnet,
    "hybrid": build_hybrid,
}


@dataclass
class BenchResult:
    name: str
    dtype: str
    params_m: float
    model_size_mb: float
    flops_g: Optional[float]
    inference_ms: float
    train_ms: float



def get_dtype(dtype_name: str):
    if dtype_name.lower() == "bf16":
        return torch.bfloat16
    if dtype_name.lower() == "fp16":
        return torch.float16
    return torch.float32


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(params: int, dtype: torch.dtype) -> float:
    bytes_per_param = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    return params * bytes_per_param / (1024 ** 2)


def save_single_epoch_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str, save_dir: str, name: str, steps: int = 1) -> float:
    """Run a tiny 1-epoch training (steps) and save state_dict; return file size in MB."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"{name}_epoch1_{timestamp}.pth")

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, save_path)

    size_mb = os.path.getsize(save_path) / (1024 ** 2)
    return size_mb


def compute_flops(model: nn.Module, x: torch.Tensor) -> Optional[float]:
    if not FVCORE_AVAILABLE:
        return None
    try:
        model.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(model, x)
            # Silence unsupported-op and uncalled-submodule warnings
            try:
                flops.unsupported_ops_warnings(False)
                flops.uncalled_modules_warnings(False)
            except Exception:
                pass
            return flops.total() / 1e9
    except Exception:
        return None


def measure_inference(model: nn.Module, x: torch.Tensor, dtype: torch.dtype, warmup: int, iters: int, device: str) -> float:
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype, enabled=device.startswith("cuda")):
                _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype, enabled=device.startswith("cuda")):
                _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) * 1000 / iters


def measure_train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype, warmup: int, iters: int, device: str) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    # warmup
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype, enabled=device.startswith("cuda")):
            pred = model(x)
            loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype, enabled=device.startswith("cuda")):
            pred = model(x)
            loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000 / iters



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="unet,attunet,transnet,nnunet,hybrid")
    parser.add_argument("--dtypes", default="bf16,fp16")
    parser.add_argument("--patch", type=int, default=96)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations per model")
    parser.add_argument("--steps", type=int, default=None, help="Alias for --iters")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--nnunet-plans", default=r"e:\Orange Hackathon\nnunet_results\nnUNet_results\Dataset021_BraTS21\nnUNetTrainer__nnUNetPlans__3d_fullres\plans.json")
    parser.add_argument("--nnunet-dataset-json", default=r"e:\Orange Hackathon\nnunet_results\nnUNet_results\Dataset021_BraTS21\nnUNetTrainer__nnUNetPlans__3d_fullres\dataset.json")
    parser.add_argument("--nnunet-config", default="3d_fullres")
    parser.add_argument("--save-dir", default="benchmark_saved_models", help="Directory for saving single-epoch model files")
    parser.add_argument("--save-epoch-steps", type=int, default=1, help="Steps to simulate a single epoch before saving")
    args = parser.parse_args()

    device = args.device
    patch = args.patch
    batch = args.batch
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.steps is not None:
        args.iters = args.steps

    x = torch.randn(batch, 4, patch, patch, patch, device=device)
    y = torch.randn(batch, 3, patch, patch, patch, device=device)

    results: List[BenchResult] = []

    for name in model_names:
        if name == "nnunet":
            model = build_nnunet(args.nnunet_plans, args.nnunet_dataset_json, args.nnunet_config)
            if model is None:
                print("[nnUNet] nnunetv2 not available or plans load failed. Skipping.")
                continue
        else:
            builder = MODEL_BUILDERS.get(name)
            if builder is None:
                print(f"[Skip] Unknown model: {name}")
                continue
            model = builder()

        model = model.to(device)
        params = count_params(model)

        saved_size_mb = save_single_epoch_model(
            model=model,
            x=x,
            y=y,
            device=device,
            save_dir=args.save_dir,
            name=name,
            steps=args.save_epoch_steps,
        )

        # FLOPs on FP32 for stability
        flops_g = compute_flops(model, x) if FVCORE_AVAILABLE else None

        for dt in dtypes:
            dtype = get_dtype(dt)
            inf_ms = measure_inference(model, x, dtype, args.warmup, args.iters, device)
            train_ms = measure_train(model, x, y, dtype, args.warmup, args.iters, device)
            results.append(BenchResult(
                name=name,
                dtype=dt,
                params_m=params / 1e6,
                model_size_mb=saved_size_mb,
                flops_g=flops_g,
                inference_ms=inf_ms,
                train_ms=train_ms,
            ))

    # Print table
    print("\n=== Benchmark Results (per-patch 96^3) ===")
    header = f"{'Model':<12} {'Dtype':<6} {'Params(M)':<10} {'Size(MB)':<9} {'FLOPs(G)':<9} {'Infer(ms)':<10} {'Train(ms)':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        flops_str = f"{r.flops_g:.2f}" if r.flops_g is not None else "N/A"
        print(f"{r.name:<12} {r.dtype:<6} {r.params_m:<10.2f} {r.model_size_mb:<9.2f} {flops_str:<9} {r.inference_ms:<10.2f} {r.train_ms:<10.2f}")

    if not FVCORE_AVAILABLE:
        print("\n[Note] fvcore not available; install with: pip install fvcore")


if __name__ == "__main__":
    main()
