"""
Ne3Na3 FastAPI Application
Main application with CORS, routes, and middleware
"""

import os
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import nibabel as nib
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from .models import get_model_loader
from .inference import InferenceEngine, get_inference_engine
from .insights import InsightEngine, get_insight_engine
from .chatbot import SafeBot, get_safe_bot
from .schemas import (
    HealthResponse, ModelInfoResponse, 
    SegmentationRequest, SegmentationResponse,
    ChatRequest, ChatResponse, ConversationHistory,
    InsightsResponse, DemoResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
_current_insights: Optional[Dict[str, Any]] = None
_current_segmentation: Optional[np.ndarray] = None
_current_mri: Optional[Dict[str, np.ndarray]] = None
_current_voxel_spacing: Optional[tuple] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("🌿 Ne3Na3 Medical AI Server starting...")
    
    # Get model directory from environment or use default
    model_dir = os.environ.get("MODEL_DIR", "./model_weights")
    
    # Initialize model loader
    loader = get_model_loader(model_dir)
    
    try:
        # Pre-load model
        loader.load_model()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Model loading deferred: {e}")
    
    yield
    
    # Shutdown
    logger.info("🌿 Ne3Na3 Server shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Ne3Na3 Medical AI",
    description="""
    🌿 **Ne3Na3**: Senior Medical AI for Brain Tumor Segmentation
    
    Multi-modal BraTS segmentation using AttUnet with:
    - MONAI Sliding Window Inference
    - Test-Time Augmentation
    - Anatomical Consistency Enforcement
    
    **Features:**
    - 🧠 4-modality MRI processing (T1, T1ce, T2, FLAIR)
    - 📊 Comprehensive tumor metrics
    - 💬 Safe-Bot for result interpretation
    - 🎯 Attention-based explainability
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # React dev server
        "http://localhost:5173",      # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"                           # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health & Info Routes ====================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    loader = get_model_loader()
    return HealthResponse(
        status="healthy",
        service="Ne3Na3 Medical AI",
        version="1.0.0",
        model_loaded=loader.is_loaded(),
        device=str(loader.get_device())
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    loader = get_model_loader()
    return HealthResponse(
        status="healthy",
        service="Ne3Na3 Medical AI",
        version="1.0.0",
        model_loaded=loader.is_loaded(),
        device=str(loader.get_device())
    )


@app.get("/api/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    loader = get_model_loader()
    return loader.get_model_info()


# ==================== Segmentation Routes ====================

@app.post("/api/segment", response_model=SegmentationResponse)
async def segment_brain_mri(
    t1: UploadFile = File(..., description="T1-weighted MRI (NIfTI)"),
    t1ce: UploadFile = File(..., description="T1-contrast enhanced MRI (NIfTI)"),
    t2: UploadFile = File(..., description="T2-weighted MRI (NIfTI)"),
    flair: UploadFile = File(..., description="FLAIR MRI (NIfTI)"),
    use_tta: bool = Form(default=True),
    enforce_consistency: bool = Form(default=True),
    include_mask: bool = Form(default=False),
    mask_downsample: int = Form(default=1)
):
    """
    Process 4 NIfTI MRI files and return multi-class segmentation
    
    Returns segmentation with classes:
    - 0: Background
    - 1: NCR (Necrotic Core)
    - 2: ED (Peritumoral Edema)
    - 4: ET (Enhancing Tumor)
    """
    global _current_insights, _current_segmentation
    
    start_time = time.time()
    
    try:
        # Load NIfTI files
        modalities = {}
        voxel_spacing = None
        
        for name, file in [("t1", t1), ("t1ce", t1ce), ("t2", t2), ("flair", flair)]:
            # Determine file extension from original filename
            original_filename = file.filename.lower() if file.filename else ""
            if original_filename.endswith('.nii.gz'):
                suffix = ".nii.gz"
            elif original_filename.endswith('.nii'):
                suffix = ".nii"
            else:
                # Default: try to detect from content
                suffix = ".nii.gz"
            
            # Save to temp file
            content = await file.read()
            
            # Check if content is gzipped (gzip magic number: 1f 8b)
            is_gzipped = len(content) >= 2 and content[0] == 0x1f and content[1] == 0x8b
            
            # Adjust suffix based on actual content
            if not is_gzipped and suffix == ".nii.gz":
                suffix = ".nii"
            elif is_gzipped and suffix == ".nii":
                suffix = ".nii.gz"
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                # Load NIfTI
                nii = nib.load(tmp_path)
                modalities[name] = nii.get_fdata().astype(np.float32)
                
                # Get voxel spacing from first modality
                if voxel_spacing is None:
                    voxel_spacing = tuple(nii.header.get_zooms()[:3])
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
        
        logger.info(f"Loaded modalities with shape: {modalities['t1'].shape}")
        
        # Store MRI data for visualization
        global _current_mri, _current_voxel_spacing
        _current_mri = modalities
        _current_voxel_spacing = voxel_spacing
        
        # Run inference
        inference_engine = get_inference_engine()
        results = inference_engine.predict_from_nifti(
            t1=modalities["t1"],
            t1ce=modalities["t1ce"],
            t2=modalities["t2"],
            flair=modalities["flair"],
            use_tta=use_tta,
            enforce_consistency=enforce_consistency
        )
        
        segmentation = results["segmentation"]
        # Remove batch dimension if present (inference returns [1, D, H, W])
        if segmentation.ndim == 4 and segmentation.shape[0] == 1:
            segmentation = segmentation[0]
        _current_segmentation = segmentation
        
        # Generate insights
        insight_engine = get_insight_engine()
        insights = insight_engine.generate_insights(
            segmentation=segmentation,
            attention_maps=results.get("attention_maps"),
            voxel_spacing=voxel_spacing
        )
        _current_insights = insights
        
        # Update chatbot with insights
        safe_bot = get_safe_bot()
        safe_bot.set_insights(insights)
        
        processing_time = time.time() - start_time
        
        mask_values = None
        mask_shape = list(segmentation.shape)
        downsample_factor = max(1, int(mask_downsample))
        if include_mask:
            mask_volume = segmentation
            if downsample_factor > 1:
                mask_volume = mask_volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
                mask_shape = list(mask_volume.shape)
            mask_values = mask_volume.astype(np.int32).flatten().tolist()

        return SegmentationResponse(
            success=True,
            message="Segmentation completed successfully",
            processing_time_seconds=round(processing_time, 2),
            insights=insights,
            segmentation_shape=mask_shape,
            segmentation_values=mask_values,
            mask_downsample=downsample_factor if include_mask else None
        )
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/volume/mri")
async def get_mri_volume(
    modality: str = "t1ce",
    downsample: int = 2
):
    """
    Get downsampled MRI volume data for 3D visualization
    
    Args:
        modality: Which MRI modality (t1, t1ce, t2, flair)
        downsample: Downsampling factor (2 = half resolution)
    """
    if _current_mri is None:
        raise HTTPException(
            status_code=404,
            detail="No MRI data available. Please upload and process scans first."
        )
    
    if modality not in _current_mri:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid modality. Available: {list(_current_mri.keys())}"
        )
    
    # Get the volume and downsample for transfer
    volume = _current_mri[modality].copy()
    
    # Downsample to reduce data size
    if downsample > 1:
        volume = volume[::downsample, ::downsample, ::downsample]
    
    # Normalize to 0-1 range
    vmin, vmax = float(volume.min()), float(volume.max())
    if vmax > vmin:
        volume = ((volume - vmin) / (vmax - vmin)).astype(np.float32)
    else:
        volume = np.zeros_like(volume, dtype=np.float32)
    
    # Convert to list for JSON serialization (use tolist() for speed)
    # Ensure all values are native Python types, not numpy types
    shape = [int(s) for s in volume.shape]
    voxel_spacing = [float(v) for v in _current_voxel_spacing] if _current_voxel_spacing else [1.0, 1.0, 1.0]
    
    # Use numpy's built-in tolist() which converts to native Python types
    values = volume.flatten().tolist()
    
    return {
        "modality": modality,
        "shape": shape,
        "voxel_spacing": voxel_spacing,
        "downsample_factor": int(downsample),
        "values": values
    }


@app.get("/api/volume/segmentation")
async def get_segmentation_volume(downsample: int = 2):
    """
    Get downsampled segmentation volume for 3D visualization
    
    Args:
        downsample: Downsampling factor (2 = half resolution)
    """
    if _current_segmentation is None:
        raise HTTPException(
            status_code=404,
            detail="No segmentation available. Please run segmentation first."
        )
    
    # Get volume and ensure it's 3D (remove batch dim if present)
    volume = _current_segmentation.copy()
    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume[0]
    
    # Downsample to reduce data size
    if downsample > 1:
        volume = volume[::downsample, ::downsample, ::downsample]
    
    # Convert to native Python types using tolist() for speed
    # Ensure all values are native Python types, not numpy types
    shape = [int(s) for s in volume.shape]
    voxel_spacing = [float(v) for v in _current_voxel_spacing] if _current_voxel_spacing else [1.0, 1.0, 1.0]
    values = volume.flatten().astype(np.int32).tolist()
    
    return {
        "shape": shape,
        "voxel_spacing": voxel_spacing,
        "downsample_factor": int(downsample),
        "classes": {
            "0": "Background",
            "1": "NCR (Necrotic Core)",
            "2": "ED (Edema)",
            "4": "ET (Enhancing Tumor)"
        },
        "values": values
    }


@app.get("/api/volume/slice")
async def get_volume_slice(
    axis: str = "axial",
    slice_idx: int = 0,
    modality: str = "t1ce",
    include_segmentation: bool = True
):
    """
    Get a single 2D slice for efficient visualization
    
    Args:
        axis: Slice orientation (axial, sagittal, coronal)
        slice_idx: Index of the slice
        modality: MRI modality
        include_segmentation: Whether to include segmentation overlay
    """
    if _current_mri is None:
        raise HTTPException(
            status_code=404,
            detail="No MRI data available. Please upload and process scans first."
        )
    
    volume = _current_mri.get(modality)
    if volume is None:
        raise HTTPException(status_code=400, detail=f"Invalid modality: {modality}")
    
    # Extract slice based on axis
    if axis == "axial":
        slice_idx = min(max(0, slice_idx), volume.shape[0] - 1)
        mri_slice = volume[slice_idx, :, :]
        seg_slice = _current_segmentation[slice_idx, :, :] if _current_segmentation is not None else None
    elif axis == "sagittal":
        slice_idx = min(max(0, slice_idx), volume.shape[2] - 1)
        mri_slice = volume[:, :, slice_idx]
        seg_slice = _current_segmentation[:, :, slice_idx] if _current_segmentation is not None else None
    elif axis == "coronal":
        slice_idx = min(max(0, slice_idx), volume.shape[1] - 1)
        mri_slice = volume[:, slice_idx, :]
        seg_slice = _current_segmentation[:, slice_idx, :] if _current_segmentation is not None else None
    else:
        raise HTTPException(status_code=400, detail=f"Invalid axis: {axis}")
    
    # Normalize MRI slice
    vmin, vmax = float(mri_slice.min()), float(mri_slice.max())
    if vmax > vmin:
        mri_slice = (mri_slice - vmin) / (vmax - vmin)
    
    # Convert to native Python types using fast tolist()
    # Ensure all values are native Python types, not numpy types
    response = {
        "axis": axis,
        "slice_idx": int(slice_idx),
        "shape": [int(s) for s in mri_slice.shape],
        "max_slices": {
            "axial": int(volume.shape[0]),
            "sagittal": int(volume.shape[2]),
            "coronal": int(volume.shape[1])
        },
        "mri_values": mri_slice.flatten().astype(np.float32).tolist()
    }
    
    if include_segmentation and seg_slice is not None:
        response["segmentation_values"] = seg_slice.flatten().astype(np.int32).tolist()
    
    return response


@app.get("/api/insights", response_model=InsightsResponse)
async def get_current_insights():
    """Get the current analysis insights"""
    if _current_insights is None:
        raise HTTPException(
            status_code=404, 
            detail="No analysis available. Please run segmentation first."
        )
    return _current_insights


@app.post("/api/demo", response_model=DemoResponse)
async def run_demo_analysis():
    """
    Run a demo analysis with synthetic data
    Useful for testing the system without real MRI data
    """
    global _current_insights, _current_segmentation, _current_mri, _current_voxel_spacing
    
    # Create synthetic segmentation with smaller size for demo
    shape = (128, 128, 128)  # D, H, W - smaller for faster generation
    segmentation = np.zeros(shape, dtype=np.int64)
    mri_data = np.zeros(shape, dtype=np.float32)
    
    # Use numpy vectorized operations for speed
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # Brain center
    brain_center = (64, 64, 64)
    brain_dist = np.sqrt(
        ((z - brain_center[0]) / 50)**2 + 
        ((y - brain_center[1]) / 55)**2 + 
        ((x - brain_center[2]) / 55)**2
    )
    brain_mask = brain_dist < 1.0
    
    # Create brain MRI data
    mri_data[brain_mask] = 0.3 + 0.3 * (1 - brain_dist[brain_mask])
    mri_data += np.random.random(shape).astype(np.float32) * 0.1
    mri_data = np.clip(mri_data, 0, 1)
    
    # Tumor center (offset from brain center)
    tumor_center = (70, 50, 75)
    tumor_dist = np.sqrt(
        (z - tumor_center[0])**2 + 
        (y - tumor_center[1])**2 + 
        (x - tumor_center[2])**2
    )
    
    # Create tumor regions (vectorized)
    et_mask = (tumor_dist < 10) & brain_mask  # Enhancing tumor
    ncr_mask = (tumor_dist >= 10) & (tumor_dist < 16) & brain_mask  # Necrotic
    ed_mask = (tumor_dist >= 16) & (tumor_dist < 28) & brain_mask  # Edema
    
    # BraTS labels: 1=NCR, 2=ED, 4=ET
    segmentation[et_mask] = 4
    segmentation[ncr_mask] = 1
    segmentation[ed_mask] = 2
    
    # Update MRI intensities for tumor regions
    mri_data[et_mask] = 0.9 + np.random.random(np.sum(et_mask)) * 0.1
    mri_data[ncr_mask] = 0.15 + np.random.random(np.sum(ncr_mask)) * 0.1
    mri_data[ed_mask] = 0.65 + np.random.random(np.sum(ed_mask)) * 0.15
    
    _current_segmentation = segmentation
    _current_voxel_spacing = (1.0, 1.0, 1.0)
    
    # Create all modalities (quick variations)
    _current_mri = {
        "t1": (mri_data * 0.8).astype(np.float32),
        "t1ce": mri_data.astype(np.float32),
        "t2": (mri_data * 0.9).astype(np.float32),
        "flair": (mri_data * 0.95).astype(np.float32),
    }
    
    # Generate insights
    insight_engine = get_insight_engine()
    insights = insight_engine.generate_insights(
        segmentation=segmentation,
        voxel_spacing=(1.0, 1.0, 1.0)
    )
    _current_insights = insights
    
    # Update chatbot
    safe_bot = get_safe_bot()
    safe_bot.set_insights(insights)
    
    logger.info(f"Demo created - WT: {insights['summary']['total_tumor_volume_cm3']} cm³")
    
    return DemoResponse(
        success=True,
        message="Demo analysis completed with synthetic tumor data",
        insights=insights
    )


# ==================== Chatbot Routes ====================

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_safebot(request: ChatRequest):
    """
    Chat with Ne3Na3 Safe-Bot
    
    The bot provides information grounded in the current analysis.
    It refuses to provide diagnoses or treatment advice.
    """
    safe_bot = get_safe_bot()
    response = safe_bot.get_response(request.message)
    return ChatResponse(**response)


@app.get("/api/chat/history", response_model=ConversationHistory)
async def get_chat_history():
    """Get the conversation history"""
    safe_bot = get_safe_bot()
    return ConversationHistory(history=safe_bot.get_conversation_history())


@app.delete("/api/chat/history")
async def clear_chat_history():
    """Clear the conversation history"""
    safe_bot = get_safe_bot()
    safe_bot.clear_history()
    return {"message": "Chat history cleared"}


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


@app.post("/api/load-test")
async def load_test_data(test_folder: str = "Test1", run_inference: bool = True):
    """
    Load test data from Test1 or Test2 folder and optionally run inference
    
    Args:
        test_folder: Which test folder to load (Test1 or Test2)
        run_inference: Whether to run model inference (True) or just load ground truth (False)
    """
    global _current_insights, _current_segmentation, _current_mri, _current_voxel_spacing
    
    # Determine the test data path
    base_path = Path(__file__).parent.parent.parent  # Go up to ne3na3 folder
    test_path = base_path / test_folder
    
    if not test_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Test folder not found: {test_path}"
        )
    
    logger.info(f"Loading test data from: {test_path}")
    
    try:
        # Find the NIfTI files
        nii_files = list(test_path.glob("*.nii")) + list(test_path.glob("*.nii.gz"))
        
        if len(nii_files) < 4:
            raise HTTPException(
                status_code=400,
                detail=f"Expected at least 4 NIfTI files, found {len(nii_files)}"
            )
        
        # Load each modality
        modalities = {}
        ground_truth = None
        voxel_spacing = None
        
        for nii_file in nii_files:
            filename = nii_file.name.lower()
            nii = nib.load(str(nii_file))
            data = nii.get_fdata().astype(np.float32)
            
            if voxel_spacing is None:
                voxel_spacing = tuple(nii.header.get_zooms()[:3])
            
            if "flair" in filename:
                modalities["flair"] = data
                logger.info(f"Loaded FLAIR: {data.shape}")
            elif "t1ce" in filename:
                modalities["t1ce"] = data
                logger.info(f"Loaded T1ce: {data.shape}")
            elif "t1" in filename and "t1ce" not in filename:
                modalities["t1"] = data
                logger.info(f"Loaded T1: {data.shape}")
            elif "t2" in filename:
                modalities["t2"] = data
                logger.info(f"Loaded T2: {data.shape}")
            elif "seg" in filename or "label" in filename:
                ground_truth = data.astype(np.int64)
                logger.info(f"Loaded Ground Truth: {data.shape}, unique labels: {np.unique(ground_truth)}")
        
        # Check we have all modalities
        required = ["t1", "t1ce", "t2", "flair"]
        missing = [m for m in required if m not in modalities]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing modalities: {missing}"
            )
        
        # Store MRI data
        _current_mri = modalities
        _current_voxel_spacing = voxel_spacing
        
        if run_inference:
            # Run model inference
            logger.info("Running model inference...")
            inference_engine = get_inference_engine()
            results = inference_engine.predict_from_nifti(
                t1=modalities["t1"],
                t1ce=modalities["t1ce"],
                t2=modalities["t2"],
                flair=modalities["flair"],
                use_tta=False  # Faster for testing
            )
            
            segmentation = results["segmentation"]
            # Remove batch dimension if present
            if segmentation.ndim == 4 and segmentation.shape[0] == 1:
                segmentation = segmentation[0]
            _current_segmentation = segmentation
            
            logger.info(f"Inference complete. Segmentation shape: {segmentation.shape}")
            logger.info(f"Unique labels in prediction: {np.unique(segmentation)}")
        
        elif ground_truth is not None:
            # Use ground truth segmentation
            _current_segmentation = ground_truth
            logger.info(f"Using ground truth segmentation. Shape: {ground_truth.shape}")
            logger.info(f"Unique labels: {np.unique(ground_truth)}")
        else:
            raise HTTPException(
                status_code=400,
                detail="No ground truth found and inference disabled"
            )
        
        # Generate insights
        insight_engine = get_insight_engine()
        insights = insight_engine.generate_insights(
            segmentation=_current_segmentation,
            voxel_spacing=voxel_spacing or (1.0, 1.0, 1.0)
        )
        _current_insights = insights
        
        # Update chatbot
        safe_bot = get_safe_bot()
        safe_bot.set_insights(insights)
        
        # Convert all numpy types to native Python types for JSON serialization
        response = convert_numpy_types({
            "success": True,
            "message": f"Loaded {test_folder} data successfully",
            "used_inference": run_inference,
            "mri_shape": list(modalities["t1"].shape),
            "segmentation_shape": list(_current_segmentation.shape),
            "unique_labels": [int(x) for x in np.unique(_current_segmentation)],
            "voxel_spacing": [float(v) for v in voxel_spacing] if voxel_spacing else [1.0, 1.0, 1.0],
            "insights": insights
        })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/system-prompt")
async def get_system_prompt():
    """Get the chatbot system prompt (for reference)"""
    safe_bot = get_safe_bot()
    return {"system_prompt": safe_bot.get_system_prompt()}


# ==================== Attention Maps ====================

@app.get("/api/attention")
async def get_attention_maps():
    """Get attention maps from the last inference (for explainability)"""
    try:
        inference_engine = get_inference_engine()
        model = inference_engine.model
        
        attention_maps = model.get_attention_maps()
        
        if not attention_maps:
            return {
                "success": False,
                "message": "No attention maps available. Run segmentation first."
            }
        
        return {
            "success": True,
            "num_layers": len(attention_maps),
            "shapes": [list(am.shape) for am in attention_maps],
            "message": "Attention maps retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )


# Run with: uvicorn app.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
