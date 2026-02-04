"""
Ne3Na3 Pydantic Schemas
API request/response models
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime


# ==================== Health Check ====================

class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "Ne3Na3 Medical AI"
    version: str = "1.0.0"
    model_loaded: bool = False
    device: str = "cpu"


class ModelInfoResponse(BaseModel):
    model_dir: str
    model_filename: str
    model_path: str
    weights_available: bool
    device: str
    is_loaded: bool
    architecture: Optional[str] = None
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    num_parameters: Optional[int] = None
    trainable_parameters: Optional[int] = None


# ==================== Segmentation ====================

class SegmentationRequest(BaseModel):
    use_tta: bool = Field(default=True, description="Apply Test-Time Augmentation")
    enforce_consistency: bool = Field(default=True, description="Enforce anatomical consistency")


class VolumeInfo(BaseModel):
    voxels: int
    volume_mm3: float
    volume_cm3: float


class BoundingBox(BaseModel):
    min: List[int]
    max: List[int]
    size: List[int]
    size_mm: List[float]


class RegionInfo(BaseModel):
    volume_mm3: float
    volume_cm3: float
    centroid_mm: Optional[List[float]] = None
    bounding_box: Optional[BoundingBox] = None
    surface_area_mm2: float


class SummaryInfo(BaseModel):
    total_tumor_volume_cm3: float
    tumor_core_volume_cm3: float
    enhancing_tumor_volume_cm3: float
    edema_volume_cm3: float
    tumor_detected: bool
    max_asymmetry_region: Optional[str] = None


class InsightsResponse(BaseModel):
    timestamp: str
    scan_dimensions: List[int]
    voxel_spacing_mm: List[float]
    volumes: Dict[str, VolumeInfo]
    regions: Dict[str, RegionInfo]
    asymmetry: Dict[str, float]
    modality_importance: Dict[str, float]
    summary: SummaryInfo


class SegmentationResponse(BaseModel):
    success: bool
    message: str
    processing_time_seconds: float
    insights: Optional[InsightsResponse] = None
    segmentation_shape: Optional[List[int]] = None
    segmentation_values: Optional[List[int]] = None
    mask_downsample: Optional[int] = None


# ==================== Chatbot ====================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")


class ChatResponse(BaseModel):
    message: str
    type: str
    grounded_in_insights: bool
    timestamp: str


class ConversationHistory(BaseModel):
    history: List[Dict[str, str]]


# ==================== File Upload ====================

class FileUploadResponse(BaseModel):
    success: bool
    filename: str
    message: str


class MultiFileUploadResponse(BaseModel):
    success: bool
    files: Dict[str, str]
    message: str


# ==================== Attention Maps ====================

class AttentionMapResponse(BaseModel):
    success: bool
    num_layers: int
    shapes: List[List[int]]
    message: str


# ==================== Demo ====================

class DemoResponse(BaseModel):
    success: bool
    message: str
    insights: Optional[InsightsResponse] = None
