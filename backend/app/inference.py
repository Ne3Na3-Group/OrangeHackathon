"""
Ne3Na3 Inference Engine
Implements MONAI Sliding Window Inference with TTA and Anatomical Consistency
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from monai.inferers import sliding_window_inference
import logging

from .models import AttUnet, get_model_loader

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Ne3Na3 Inference Engine
    
    Features:
    - MONAI Sliding Window Inference (roi_size=(96,96,96), overlap=0.5)
    - Test-Time Augmentation (TTA) with flips
    - Anatomical Consistency Enforcement (TC ⊂ WT, ET ⊂ TC)
    - Attention Map Extraction for Explainability
    """
    
    def __init__(
        self,
        model: Optional[AttUnet] = None,
        roi_size: Tuple[int, int, int] = (96, 96, 96),
        overlap: float = 0.5,
        device: Optional[torch.device] = None
    ):
        self.roi_size = roi_size
        self.overlap = overlap
        
        if model is None:
            loader = get_model_loader()
            self.model = loader.load_model()
            self.device = loader.get_device()
        else:
            self.model = model
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()
        logger.info(f"InferenceEngine initialized with ROI: {roi_size}, overlap: {overlap}")
    
    def _sliding_window_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform sliding window inference using MONAI"""
        return sliding_window_inference(
            inputs=inputs,
            roi_size=self.roi_size,
            sw_batch_size=1,
            predictor=self.model,
            overlap=self.overlap,
            mode="gaussian"
        )
    
    def _apply_tta(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply Test-Time Augmentation with axis flips
        
        Augmentations:
        - Original
        - Flip along x-axis
        - Flip along y-axis
        - Flip along z-axis
        """
        outputs = []
        
        # Original
        with torch.no_grad():
            out = self._sliding_window_inference(inputs)
            outputs.append(out)
        
        # Flip augmentations
        flip_dims = [2, 3, 4]  # x, y, z axes
        for dim in flip_dims:
            with torch.no_grad():
                flipped_input = torch.flip(inputs, dims=[dim])
                flipped_output = self._sliding_window_inference(flipped_input)
                # Flip back
                outputs.append(torch.flip(flipped_output, dims=[dim]))
        
        # Average predictions
        return torch.stack(outputs).mean(dim=0)
    
    def _enforce_anatomical_consistency(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Enforce anatomical consistency constraints:
        - TC (Tumor Core) ⊂ WT (Whole Tumor)
        - ET (Enhancing Tumor) ⊂ TC
        
        BraTS Labels:
        - 0: Background
        - 1: NCR (Necrotic Core) - part of TC
        - 2: ED (Edema) - part of WT only
        - 3: ET (Enhancing Tumor) - part of TC
        
        Regions:
        - WT = NCR + ED + ET (labels 1, 2, 3)
        - TC = NCR + ET (labels 1, 3)
        - ET = ET only (label 3)
        """
        seg = segmentation.clone()
        
        # Get predictions (argmax over channels)
        if seg.dim() == 5:  # [B, C, D, H, W]
            pred = torch.argmax(seg, dim=1)  # [B, D, H, W]
        else:
            pred = seg
        
        # Enforce: Any ET (3) that's outside WT should become WT
        # Enforce: Any NCR (1) that's outside WT should become WT
        # Practically, ensure tumor regions are consistent
        
        # ET can only exist where there's tumor (not background)
        et_mask = (pred == 3)
        ncr_mask = (pred == 1)
        ed_mask = (pred == 2)
        
        # WT is any tumor region
        wt_mask = et_mask | ncr_mask | ed_mask
        
        # TC is NCR + ET
        tc_mask = ncr_mask | et_mask
        
        # Enforce ET ⊂ TC (ET should only be where TC is)
        # This is already satisfied by definition
        
        # Remove isolated small components (optional post-processing)
        # For now, just ensure consistency
        
        return pred
    
    def predict(
        self,
        inputs: torch.Tensor,
        use_tta: bool = True,
        enforce_consistency: bool = True
    ) -> Dict[str, Any]:
        """
        Run full inference pipeline
        
        Args:
            inputs: Input tensor of shape [B, 4, D, H, W] with 4 MRI modalities
            use_tta: Whether to apply Test-Time Augmentation
            enforce_consistency: Whether to enforce anatomical consistency
            
        Returns:
            Dictionary containing:
            - segmentation: Final segmentation mask
            - probabilities: Softmax probabilities for each class
            - attention_maps: Attention maps from decoder blocks
        """
        inputs = inputs.to(self.device)
        
        logger.info(f"Running inference on input shape: {inputs.shape}")
        
        # Run inference
        if use_tta:
            logits = self._apply_tta(inputs)
        else:
            with torch.no_grad():
                logits = self._sliding_window_inference(inputs)
        
        # Compute probabilities
        probabilities = F.softmax(logits, dim=1)
        
        # Get segmentation
        segmentation = torch.argmax(probabilities, dim=1)
        
        # Enforce anatomical consistency
        if enforce_consistency:
            segmentation = self._enforce_anatomical_consistency(probabilities)
        
        # Get attention maps
        attention_maps = self.model.get_attention_maps()
        
        return {
            "segmentation": segmentation.cpu().numpy(),
            "probabilities": probabilities.cpu().numpy(),
            "attention_maps": [am.cpu().numpy() for am in attention_maps] if attention_maps else [],
            "logits": logits.cpu().numpy()
        }
    
    def predict_from_nifti(
        self,
        t1: np.ndarray,
        t1ce: np.ndarray,
        t2: np.ndarray,
        flair: np.ndarray,
        use_tta: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference from NIfTI arrays
        
        Args:
            t1: T1-weighted MRI
            t1ce: T1-contrast enhanced MRI
            t2: T2-weighted MRI
            flair: FLAIR MRI
            use_tta: Whether to apply TTA
            
        Returns:
            Inference results dictionary
        """
        # Stack modalities: [D, H, W] -> [4, D, H, W]
        stacked = np.stack([t1, t1ce, t2, flair], axis=0)
        
        # Normalize each modality
        stacked = self._normalize(stacked)
        
        # Add batch dimension: [1, 4, D, H, W]
        inputs = torch.from_numpy(stacked).unsqueeze(0).float()
        
        return self.predict(inputs, use_tta=use_tta)
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalization per modality with brain mask"""
        normalized = np.zeros_like(data, dtype=np.float32)
        
        for i in range(data.shape[0]):
            modality = data[i]
            mask = modality > 0  # Brain mask
            
            if mask.sum() > 0:
                mean = modality[mask].mean()
                std = modality[mask].std()
                if std > 0:
                    normalized[i] = (modality - mean) / std
                else:
                    normalized[i] = modality - mean
            else:
                normalized[i] = modality
        
        return normalized


# Singleton inference engine
_inference_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get or create the global inference engine"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine()
    return _inference_engine
