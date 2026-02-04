import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Box,
  Layers,
  Eye,
  EyeOff,
  Play,
  Pause,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  ChevronLeft,
  ChevronRight,
  Maximize2,
  Grid3X3,
  Brain,
  Activity,
  Sparkles,
  SlidersHorizontal,
  Download,
  RefreshCw,
  AlertCircle,
  Loader2
} from 'lucide-react';
import { getMRIVolume, getSegmentationVolume, checkVolumeAvailability } from '../services/api';

// Helper functions for slice extraction (must be defined before SliceViewer)
// Data is stored as flat array in [depth, height, width] order (z, y, x)
// Index = z * (height * width) + y * width + x

const getAxialSlice = (data, z) => {
  if (!data.values || !data.width || !data.height || !data.depth) return [];
  const safeZ = Math.min(Math.max(0, z), data.depth - 1);
  const slice = [];
  // For axial view: iterate y (rows) then x (columns)
  // Flip vertically for correct anatomical orientation
  for (let y = data.height - 1; y >= 0; y--) {
    for (let x = 0; x < data.width; x++) {
      const idx = safeZ * data.width * data.height + y * data.width + x;
      slice.push(data.values[idx] ?? 0);
    }
  }
  return slice;
};

const getSagittalSlice = (data, x) => {
  if (!data.values || !data.width || !data.height || !data.depth) return [];
  const safeX = Math.min(Math.max(0, x), data.width - 1);
  const slice = [];
  // For sagittal view: iterate z (depth/rows) then y (height/columns)
  // Flip vertically for correct anatomical orientation
  for (let z = data.depth - 1; z >= 0; z--) {
    for (let y = 0; y < data.height; y++) {
      const idx = z * data.width * data.height + y * data.width + safeX;
      slice.push(data.values[idx] ?? 0);
    }
  }
  return slice;
};

const getCoronalSlice = (data, y) => {
  if (!data.values || !data.width || !data.height || !data.depth) return [];
  const safeY = Math.min(Math.max(0, y), data.height - 1);
  const slice = [];
  // For coronal view: iterate z (depth/rows) then x (width/columns)
  // Flip vertically for correct anatomical orientation
  for (let z = data.depth - 1; z >= 0; z--) {
    for (let x = 0; x < data.width; x++) {
      const idx = z * data.width * data.height + safeY * data.width + x;
      slice.push(data.values[idx] ?? 0);
    }
  }
  return slice;
};

const getSegmentationColor = (classId) => {
  // Bright, distinct colors for tumor regions (BraTS uses label 4 for ET)
  const colors = {
    1: { r: 255, g: 50, b: 50 },    // NCR (Necrotic Core) - Bright Red
    2: { r: 50, g: 255, b: 50 },    // ED (Edema) - Bright Green  
    3: { r: 50, g: 150, b: 255 },   // ET (Enhancing Tumor) - Legacy label 3 support
    4: { r: 50, g: 150, b: 255 },   // ET (Enhancing Tumor) - Bright Blue (BraTS label 4)
  };
  return colors[classId] || null;
};

// Slice viewer component for 2D cross-sections
const SliceViewer = ({ data, slice, axis, colorMap, overlay, overlayOpacity, title }) => {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    if (!canvasRef.current || !data || !data.values) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Get slice data based on axis
    let sliceData;
    let width, height;
    
    if (axis === 'axial') {
      // Axial: looking from top, shows width x height
      width = data.width;
      height = data.height;
      sliceData = getAxialSlice(data, slice);
    } else if (axis === 'sagittal') {
      // Sagittal: looking from side, shows height x depth
      width = data.height;
      height = data.depth;
      sliceData = getSagittalSlice(data, slice);
    } else { // coronal
      // Coronal: looking from front, shows width x depth
      width = data.width;
      height = data.depth;
      sliceData = getCoronalSlice(data, slice);
    }
    
    if (!sliceData || sliceData.length === 0) return;
    
    canvas.width = width;
    canvas.height = height;
    
    // Create image data
    const imageData = ctx.createImageData(width, height);
    
    for (let i = 0; i < sliceData.length; i++) {
      const value = Math.floor(Math.min(1, Math.max(0, sliceData[i])) * 255);
      const idx = i * 4;
      imageData.data[idx] = value;     // R
      imageData.data[idx + 1] = value; // G
      imageData.data[idx + 2] = value; // B
      imageData.data[idx + 3] = 255;   // A
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    // Draw overlay if exists - handle dimension differences
    if (overlay && overlay.values) {
      // Check if overlay has same dimensions as MRI data
      const dimsMatch = overlay.width === data.width && 
                        overlay.height === data.height && 
                        overlay.depth === data.depth;
      
      // Calculate the corresponding slice index for overlay if dimensions differ
      let overlaySliceIdx = slice;
      if (!dimsMatch) {
        // Scale the slice index proportionally
        if (axis === 'axial') {
          overlaySliceIdx = Math.floor(slice * (overlay.depth / data.depth));
        } else if (axis === 'sagittal') {
          overlaySliceIdx = Math.floor(slice * (overlay.width / data.width));
        } else {
          overlaySliceIdx = Math.floor(slice * (overlay.height / data.height));
        }
      }
      
      let overlaySlice;
      let overlayWidth, overlayHeight;
      
      if (axis === 'axial') {
        overlayWidth = overlay.width;
        overlayHeight = overlay.height;
        overlaySlice = getAxialSlice(overlay, overlaySliceIdx);
      } else if (axis === 'sagittal') {
        overlayWidth = overlay.height;
        overlayHeight = overlay.depth;
        overlaySlice = getSagittalSlice(overlay, overlaySliceIdx);
      } else {
        overlayWidth = overlay.width;
        overlayHeight = overlay.depth;
        overlaySlice = getCoronalSlice(overlay, overlaySliceIdx);
      }
      
      if (overlaySlice && overlaySlice.length > 0) {
        // Get current image to blend with
        const currentImageData = ctx.getImageData(0, 0, width, height);
        
        // If dimensions match, direct pixel mapping
        if (dimsMatch || (overlayWidth === width && overlayHeight === height)) {
          for (let i = 0; i < overlaySlice.length && i < sliceData.length; i++) {
            const segClass = Math.round(overlaySlice[i]);
            const color = getSegmentationColor(segClass);
            if (color) {  // Only draw if valid tumor class (1, 2, or 4)
              const idx = i * 4;
              // Blend overlay with MRI - use higher opacity for visibility
              const alpha = overlayOpacity;
              currentImageData.data[idx] = Math.round(color.r * alpha + currentImageData.data[idx] * (1 - alpha));
              currentImageData.data[idx + 1] = Math.round(color.g * alpha + currentImageData.data[idx + 1] * (1 - alpha));
              currentImageData.data[idx + 2] = Math.round(color.b * alpha + currentImageData.data[idx + 2] * (1 - alpha));
            }
          }
        } else {
          // Dimensions differ - need to resample/scale the overlay
          for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
              // Map current pixel to overlay coordinates
              const overlayX = Math.floor(x * (overlayWidth / width));
              const overlayY = Math.floor(y * (overlayHeight / height));
              const overlayIdx = overlayY * overlayWidth + overlayX;
              
              if (overlayIdx < overlaySlice.length) {
                const segClass = Math.round(overlaySlice[overlayIdx]);
                const color = getSegmentationColor(segClass);
                if (color) {  // Only draw if valid tumor class (1, 2, or 4)
                  const idx = (y * width + x) * 4;
                  const alpha = overlayOpacity;
                  currentImageData.data[idx] = Math.round(color.r * alpha + currentImageData.data[idx] * (1 - alpha));
                  currentImageData.data[idx + 1] = Math.round(color.g * alpha + currentImageData.data[idx + 1] * (1 - alpha));
                  currentImageData.data[idx + 2] = Math.round(color.b * alpha + currentImageData.data[idx + 2] * (1 - alpha));
                }
              }
            }
          }
        }
        
        ctx.putImageData(currentImageData, 0, 0);
      }
    }
  }, [data, slice, axis, overlay, overlayOpacity]);
  
  return (
    <div className="relative w-full h-full">
      <div className="absolute top-2 left-2 px-2 py-1 bg-gray-900/80 rounded-lg text-xs text-gray-300 z-10">
        {title}
      </div>
      <canvas 
        ref={canvasRef} 
        className="w-full h-full object-contain rounded-xl bg-gray-900"
        style={{ imageRendering: 'pixelated' }}
      />
    </div>
  );
};

// Main Visualization Component
const Visualization3D = ({ mriData, segmentationData, insights, hasProcessedData = false }) => {
  const [activeView, setActiveView] = useState('orthogonal'); // orthogonal, 3d, comparison
  const [axialSlice, setAxialSlice] = useState(50);
  const [sagittalSlice, setSagittalSlice] = useState(50);
  const [coronalSlice, setCoronalSlice] = useState(50);
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.6);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playAxis, setPlayAxis] = useState('axial');
  const [selectedModality, setSelectedModality] = useState('t1ce');
  const [rotation, setRotation] = useState({ x: -20, y: 30 });
  const [zoom, setZoom] = useState(1);
  const containerRef = useRef(null);
  
  // API data fetching states
  const [isLoading, setIsLoading] = useState(false);
  const [fetchError, setFetchError] = useState(null);
  const [fetchedMRIData, setFetchedMRIData] = useState(null);
  const [fetchedSegData, setFetchedSegData] = useState(null);
  const [dataAvailable, setDataAvailable] = useState(false);
  
  // Check if real data is available when component mounts or hasProcessedData changes
  useEffect(() => {
    const checkData = async () => {
      const available = await checkVolumeAvailability();
      setDataAvailable(available);
      if (available && hasProcessedData) {
        // Auto-fetch data if processed data is available
        fetchRealData();
      }
    };
    checkData();
  }, [hasProcessedData]);
  
  // Function to fetch real MRI and segmentation data from API
  const fetchRealData = useCallback(async () => {
    setIsLoading(true);
    setFetchError(null);
    
    try {
      // Fetch both MRI and segmentation data in parallel
      // Use downsample=4 for faster loading (32x32x32 volume instead of 128x128x128)
      const [mriResponse, segResponse] = await Promise.all([
        getMRIVolume(selectedModality, 4),
        getSegmentationVolume(4)
      ]);
      
      // Transform API response to our data format
      if (mriResponse && mriResponse.shape) {
        setFetchedMRIData({
          width: mriResponse.shape[2],
          height: mriResponse.shape[1],
          depth: mriResponse.shape[0],
          values: mriResponse.values
        });
      }
      
      if (segResponse && segResponse.shape) {
        setFetchedSegData({
          width: segResponse.shape[2],
          height: segResponse.shape[1],
          depth: segResponse.shape[0],
          values: segResponse.values
        });
      }
      
      setDataAvailable(true);
    } catch (error) {
      console.error('Error fetching volume data:', error);
      setFetchError(error.message || 'Failed to load volume data');
    } finally {
      setIsLoading(false);
    }
  }, [selectedModality]);
  
  // Refetch when modality changes (if we have fetched data)
  useEffect(() => {
    if (fetchedMRIData && dataAvailable) {
      fetchRealData();
    }
  }, [selectedModality]);
  
  // Demo data for when no real data is loaded
  const demoData = useMemo(() => ({
    width: 128,
    height: 128, 
    depth: 128,
    values: new Array(128 * 128 * 128).fill(0).map((_, i) => {
      const x = i % 128;
      const y = Math.floor(i / 128) % 128;
      const z = Math.floor(i / (128 * 128));
      // Create a brain-like ellipsoid
      const cx = 64, cy = 64, cz = 64;
      const dx = (x - cx) / 50;
      const dy = (y - cy) / 60;
      const dz = (z - cz) / 45;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < 1) {
        return Math.max(0, 1 - dist) * 0.8 + Math.random() * 0.2;
      }
      return 0;
    })
  }), []);

  const demoSegmentation = useMemo(() => ({
    width: 128,
    height: 128,
    depth: 128,
    values: new Array(128 * 128 * 128).fill(0).map((_, i) => {
      const x = i % 128;
      const y = Math.floor(i / 128) % 128;
      const z = Math.floor(i / (128 * 128));
      // Create tumor regions
      const tumorX = 75, tumorY = 70, tumorZ = 64;
      const dx = (x - tumorX) / 15;
      const dy = (y - tumorY) / 18;
      const dz = (z - tumorZ) / 12;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      // BraTS labels: 1=NCR, 2=ED, 4=ET
      if (dist < 0.3) return 4; // ET (core) - BraTS label 4
      if (dist < 0.6) return 1; // NCR
      if (dist < 1) return 2;   // ED (edema)
      return 0;
    })
  }), []);

  // Use fetched data > passed props > demo data
  const currentData = fetchedMRIData || mriData || demoData;
  const currentSegmentation = fetchedSegData || segmentationData || demoSegmentation;
  const isUsingRealData = !!(fetchedMRIData || mriData);
  
  const maxSlices = {
    axial: currentData.depth - 1,
    sagittal: currentData.width - 1,
    coronal: currentData.height - 1
  };

  // Center slices when data changes
  useEffect(() => {
    if (currentData) {
      // Set slices to center of each dimension
      setAxialSlice(Math.floor(currentData.depth / 2));
      setSagittalSlice(Math.floor(currentData.width / 2));
      setCoronalSlice(Math.floor(currentData.height / 2));
    }
  }, [currentData.depth, currentData.width, currentData.height, isUsingRealData]);

  // Animation loop for slice playback
  useEffect(() => {
    if (!isPlaying) return;
    
    const interval = setInterval(() => {
      if (playAxis === 'axial') {
        setAxialSlice(s => (s + 1) % (maxSlices.axial + 1));
      } else if (playAxis === 'sagittal') {
        setSagittalSlice(s => (s + 1) % (maxSlices.sagittal + 1));
      } else {
        setCoronalSlice(s => (s + 1) % (maxSlices.coronal + 1));
      }
    }, 100);
    
    return () => clearInterval(interval);
  }, [isPlaying, playAxis, maxSlices]);

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === ' ') {
        e.preventDefault();
        setIsPlaying(p => !p);
      } else if (e.key === 'ArrowUp') {
        if (playAxis === 'axial') setAxialSlice(s => Math.min(s + 1, maxSlices.axial));
        else if (playAxis === 'sagittal') setSagittalSlice(s => Math.min(s + 1, maxSlices.sagittal));
        else setCoronalSlice(s => Math.min(s + 1, maxSlices.coronal));
      } else if (e.key === 'ArrowDown') {
        if (playAxis === 'axial') setAxialSlice(s => Math.max(s - 1, 0));
        else if (playAxis === 'sagittal') setSagittalSlice(s => Math.max(s - 1, 0));
        else setCoronalSlice(s => Math.max(s - 1, 0));
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [playAxis, maxSlices]);

  const viewModes = [
    { id: 'orthogonal', label: 'Orthogonal', icon: Grid3X3 },
    { id: '3d', label: '3D View', icon: Box },
    { id: 'comparison', label: 'Before/After', icon: Layers },
  ];

  const modalities = [
    { id: 't1', label: 'T1' },
    { id: 't1ce', label: 'T1ce' },
    { id: 't2', label: 'T2' },
    { id: 'flair', label: 'FLAIR' },
  ];

  // 3D Volume Renderer Component
  const VolumeRenderer3D = () => {
    const canvasRef = useRef(null);
    const isDragging = useRef(false);
    const lastMouse = useRef({ x: 0, y: 0 });
    
    useEffect(() => {
      if (!canvasRef.current) return;
      
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      // Set canvas size
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      
      // Render 3D projection
      render3DVolume(ctx, canvas.offsetWidth, canvas.offsetHeight);
    }, [rotation, zoom, showOverlay, overlayOpacity]);
    
    const render3DVolume = (ctx, width, height) => {
      ctx.fillStyle = '#0a0a0f';
      ctx.fillRect(0, 0, width, height);
      
      const centerX = width / 2;
      const centerY = height / 2;
      const size = Math.min(width, height) * 0.35 * zoom;
      
      // Rotation matrices
      const cosX = Math.cos(rotation.x * Math.PI / 180);
      const sinX = Math.sin(rotation.x * Math.PI / 180);
      const cosY = Math.cos(rotation.y * Math.PI / 180);
      const sinY = Math.sin(rotation.y * Math.PI / 180);
      
      // Project 3D point to 2D
      const project = (x, y, z) => {
        // Rotate around Y
        const x1 = x * cosY - z * sinY;
        const z1 = x * sinY + z * cosY;
        // Rotate around X
        const y1 = y * cosX - z1 * sinX;
        const z2 = y * sinX + z1 * cosX;
        
        return {
          x: centerX + x1 * size,
          y: centerY + y1 * size,
          z: z2
        };
      };
      
      // Draw brain volume using ray marching approximation
      const slices = [];
      const numSlices = 64;
      
      for (let i = 0; i < numSlices; i++) {
        const z = (i / numSlices - 0.5) * 2;
        const sliceZ = Math.floor((i / numSlices) * currentData.depth);
        
        // Sample points on this slice
        for (let yi = 0; yi < 32; yi++) {
          for (let xi = 0; xi < 32; xi++) {
            const x = (xi / 32 - 0.5) * 2;
            const y = (yi / 32 - 0.5) * 2;
            
            const dataX = Math.floor((xi / 32) * currentData.width);
            const dataY = Math.floor((yi / 32) * currentData.height);
            const idx = sliceZ * currentData.width * currentData.height + dataY * currentData.width + dataX;
            
            const value = currentData.values[idx] || 0;
            const segValue = currentSegmentation.values[idx] || 0;
            
            if (value > 0.1) {
              const p = project(x, y, z);
              slices.push({
                x: p.x,
                y: p.y,
                z: p.z,
                value,
                segValue
              });
            }
          }
        }
      }
      
      // Sort by z-depth (painter's algorithm)
      slices.sort((a, b) => a.z - b.z);
      
      // Draw points
      for (const point of slices) {
        const alpha = 0.15 + point.value * 0.4;
        
        if (showOverlay && point.segValue > 0) {
          const color = getSegmentationColor(point.segValue);
          ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha * overlayOpacity})`;
        } else {
          const gray = Math.floor(point.value * 200 + 55);
          ctx.fillStyle = `rgba(${gray}, ${gray}, ${gray}, ${alpha})`;
        }
        
        const pointSize = 3 * zoom;
        ctx.beginPath();
        ctx.arc(point.x, point.y, pointSize, 0, Math.PI * 2);
        ctx.fill();
      }
      
      // Draw coordinate axes
      ctx.strokeStyle = 'rgba(0, 166, 118, 0.5)';
      ctx.lineWidth = 2;
      
      const axisLength = 0.6;
      const axes = [
        { from: [0, 0, 0], to: [axisLength, 0, 0], color: '#ff6b6b', label: 'R' },
        { from: [0, 0, 0], to: [0, axisLength, 0], color: '#4ecdc4', label: 'A' },
        { from: [0, 0, 0], to: [0, 0, axisLength], color: '#45b7d1', label: 'S' },
      ];
      
      for (const axis of axes) {
        const from = project(...axis.from);
        const to = project(...axis.to);
        
        ctx.strokeStyle = axis.color;
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
        
        ctx.fillStyle = axis.color;
        ctx.font = '12px Inter';
        ctx.fillText(axis.label, to.x + 5, to.y);
      }
    };
    
    const handleMouseDown = (e) => {
      isDragging.current = true;
      lastMouse.current = { x: e.clientX, y: e.clientY };
    };
    
    const handleMouseMove = (e) => {
      if (!isDragging.current) return;
      
      const dx = e.clientX - lastMouse.current.x;
      const dy = e.clientY - lastMouse.current.y;
      
      setRotation(r => ({
        x: Math.max(-90, Math.min(90, r.x + dy * 0.5)),
        y: r.y + dx * 0.5
      }));
      
      lastMouse.current = { x: e.clientX, y: e.clientY };
    };
    
    const handleMouseUp = () => {
      isDragging.current = false;
    };
    
    const handleWheel = (e) => {
      e.preventDefault();
      setZoom(z => Math.max(0.5, Math.min(3, z - e.deltaY * 0.001)));
    };
    
    return (
      <canvas
        ref={canvasRef}
        className="w-full h-full rounded-2xl cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />
    );
  };

  return (
    <div className="space-y-6 p-2 overflow-x-hidden">
      {/* Header */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        <div className="flex items-center gap-4 min-w-0">
          <div className="p-3 bg-gradient-to-br from-ne3na3-primary/20 to-ne3na3-neon/10 rounded-2xl border border-ne3na3-primary/20 shadow-lg shadow-ne3na3-primary/10 flex-shrink-0">
            <Box className="w-6 h-6 text-ne3na3-neon" />
          </div>
          <div className="min-w-0">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              3D Brain Visualization
              <Sparkles className="w-4 h-4 text-ne3na3-neon animate-pulse flex-shrink-0" />
            </h2>
            <p className="text-gray-400 text-sm mt-1 flex flex-wrap items-center gap-2">
              <span className="truncate">Interactive brain volume exploration</span>
              {isUsingRealData ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full border border-green-500/30 flex-shrink-0">
                  <Activity className="w-3 h-3" />
                  Real Data
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-yellow-500/20 text-yellow-400 text-xs rounded-full border border-yellow-500/30 flex-shrink-0">
                  <Brain className="w-3 h-3" />
                  Demo Mode
                </span>
              )}
            </p>
          </div>
        </div>
        
        {/* Load Data Button & View Mode Selector */}
        <div className="flex flex-wrap items-center gap-3 w-full lg:w-auto">
          {/* Load Real Data Button */}
          <motion.button
            onClick={fetchRealData}
            disabled={isLoading}
            className={`
              flex items-center justify-center gap-2 px-4 py-2 rounded-xl font-medium text-sm transition-all flex-shrink-0
              ${isLoading 
                ? 'bg-gray-700/50 text-gray-400 cursor-not-allowed' 
                : dataAvailable || hasProcessedData
                  ? 'bg-gradient-to-r from-ne3na3-primary to-ne3na3-secondary text-white hover:shadow-lg hover:shadow-ne3na3-primary/30'
                  : 'bg-gray-700/50 text-gray-400 cursor-not-allowed'
              }
            `}
            whileHover={!isLoading && (dataAvailable || hasProcessedData) ? { scale: 1.02 } : {}}
            whileTap={!isLoading && (dataAvailable || hasProcessedData) ? { scale: 0.98 } : {}}
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Loading...
              </>
            ) : (
              <>
                <RefreshCw className="w-4 h-4" />
                {isUsingRealData ? 'Refresh' : 'Load Data'}
              </>
            )}
          </motion.button>
          
          {/* View Mode Selector */}
          <div className="flex items-center gap-1 p-1 bg-gray-800/50 rounded-xl border border-gray-700/50 overflow-x-auto">
            {viewModes.map((mode) => (
              <motion.button
                key={mode.id}
                onClick={() => setActiveView(mode.id)}
                className={`
                  flex items-center gap-1.5 px-3 py-2 rounded-lg transition-all whitespace-nowrap flex-shrink-0
                  ${activeView === mode.id 
                    ? 'bg-ne3na3-primary text-white shadow-lg shadow-ne3na3-primary/30' 
                    : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                  }
                `}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <mode.icon className="w-4 h-4" />
                <span className="text-xs font-medium">{mode.label}</span>
              </motion.button>
            ))}
          </div>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="flex flex-wrap items-center gap-3 p-4 bg-gradient-to-br from-gray-800/40 to-gray-900/20 rounded-2xl border border-gray-700/50 overflow-x-auto">
        {/* Modality Selector */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-xs font-medium text-gray-400 hidden sm:inline">Modality:</span>
          <div className="flex gap-1 p-1 bg-gray-800/50 rounded-lg">
            {modalities.map((mod) => (
              <button
                key={mod.id}
                onClick={() => setSelectedModality(mod.id)}
                className={`
                  px-3 py-1.5 rounded-md text-xs font-semibold transition-all
                  ${selectedModality === mod.id 
                    ? 'bg-ne3na3-primary text-white shadow-md' 
                    : 'bg-transparent text-gray-400 hover:text-white hover:bg-gray-700/50'
                  }
                `}
              >
                {mod.label}
              </button>
            ))}
          </div>
        </div>

        <div className="w-px h-6 bg-gray-700/50 hidden sm:block flex-shrink-0" />

        {/* Overlay Toggle */}
        <button
          onClick={() => setShowOverlay(!showOverlay)}
          className={`
            flex items-center gap-1.5 px-3 py-2 rounded-lg transition-all font-medium flex-shrink-0
            ${showOverlay 
              ? 'bg-ne3na3-primary/20 text-ne3na3-neon border border-ne3na3-primary/30' 
              : 'bg-gray-700/50 text-gray-400 border border-transparent hover:border-gray-600'
            }
          `}
        >
          {showOverlay ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          <span className="text-xs hidden sm:inline">Overlay</span>
        </button>

        {/* Opacity Slider */}
        {showOverlay && (
          <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800/30 rounded-lg flex-shrink-0">
            <SlidersHorizontal className="w-4 h-4 text-gray-500" />
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={overlayOpacity}
              onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
              className="w-16 sm:w-20 accent-ne3na3-primary cursor-pointer"
            />
            <span className="text-xs text-gray-400 font-mono">{Math.round(overlayOpacity * 100)}%</span>
          </div>
        )}

        <div className="flex-1" />

        {/* Playback Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`
              p-2 rounded-lg transition-all
              ${isPlaying ? 'bg-ne3na3-primary text-white' : 'bg-gray-700/50 text-gray-400 hover:text-white'}
            `}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <select
            value={playAxis}
            onChange={(e) => setPlayAxis(e.target.value)}
            className="bg-gray-700/50 text-gray-300 text-xs rounded-lg px-2 py-1.5 border border-gray-600/50"
          >
            <option value="axial">Axial</option>
            <option value="sagittal">Sagittal</option>
            <option value="coronal">Coronal</option>
          </select>
        </div>

        {activeView === '3d' && (
          <>
            <div className="w-px h-6 bg-gray-700" />
            <div className="flex items-center gap-1">
              <button
                onClick={() => setZoom(z => Math.min(z + 0.2, 3))}
                className="p-2 bg-gray-700/50 text-gray-400 hover:text-white rounded-lg"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={() => setZoom(z => Math.max(z - 0.2, 0.5))}
                className="p-2 bg-gray-700/50 text-gray-400 hover:text-white rounded-lg"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <button
                onClick={() => { setRotation({ x: -20, y: 30 }); setZoom(1); }}
                className="p-2 bg-gray-700/50 text-gray-400 hover:text-white rounded-lg"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </>
        )}
      </div>

      {/* Error Banner */}
      <AnimatePresence>
        {fetchError && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400"
          >
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium">Failed to load volume data</p>
              <p className="text-xs text-red-400/70">{fetchError}</p>
            </div>
            <button
              onClick={() => setFetchError(null)}
              className="px-3 py-1 text-xs bg-red-500/20 hover:bg-red-500/30 rounded-lg transition-colors"
            >
              Dismiss
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading Overlay */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-gray-900/80 backdrop-blur-sm"
          >
            <div className="flex flex-col items-center gap-4 p-8 bg-gray-800/90 rounded-2xl border border-gray-700/50">
              <div className="relative">
                <Brain className="w-16 h-16 text-ne3na3-primary animate-pulse" />
                <Loader2 className="absolute -bottom-1 -right-1 w-6 h-6 text-ne3na3-neon animate-spin" />
              </div>
              <div className="text-center">
                <p className="text-white font-medium">Loading Volume Data</p>
                <p className="text-gray-400 text-sm">Fetching MRI and segmentation data...</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Visualization Area */}
      <AnimatePresence mode="wait">
        {activeView === 'orthogonal' && (
          <motion.div
            key="orthogonal"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-4"
          >
            {/* Top Row - Main Axial View (Larger) */}
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
              {/* Main Axial View - Takes 2 columns on XL */}
              <div className="xl:col-span-2 bg-gradient-to-br from-gray-800/50 to-gray-900/30 rounded-2xl border border-gray-700/50 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 bg-ne3na3-primary/20 rounded-lg">
                      <Layers className="w-4 h-4 text-ne3na3-neon" />
                    </div>
                    <div>
                      <span className="text-sm font-semibold text-white">Axial View</span>
                      <p className="text-xs text-gray-500">Top-Down</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 px-2 py-1 bg-gray-800/50 rounded-lg">
                    <span className="text-xs font-mono text-ne3na3-neon">{axialSlice}</span>
                    <span className="text-xs text-gray-500">/ {maxSlices.axial}</span>
                  </div>
                </div>
                <div className="relative aspect-square max-h-[350px] lg:max-h-[400px] mx-auto bg-gray-900 rounded-xl overflow-hidden shadow-xl shadow-black/50">
                  <SliceViewer
                    data={currentData}
                    slice={axialSlice}
                    axis="axial"
                    overlay={showOverlay ? currentSegmentation : null}
                    overlayOpacity={overlayOpacity}
                    title="Axial"
                  />
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setAxialSlice(Math.max(0, axialSlice - 1))}
                    className="p-1.5 bg-gray-700/50 hover:bg-gray-600/50 text-gray-400 hover:text-white rounded-lg transition-all"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <input
                    type="range"
                    min="0"
                    max={maxSlices.axial}
                    value={axialSlice}
                    onChange={(e) => setAxialSlice(parseInt(e.target.value))}
                    className="flex-1 h-1.5 accent-ne3na3-primary cursor-pointer"
                  />
                  <button
                    onClick={() => setAxialSlice(Math.min(maxSlices.axial, axialSlice + 1))}
                    className="p-1.5 bg-gray-700/50 hover:bg-gray-600/50 text-gray-400 hover:text-white rounded-lg transition-all"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Legend Panel - Side column on XL */}
              <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/30 rounded-2xl border border-gray-700/50 p-4 space-y-4">
                <div className="flex items-center gap-2">
                  <div className="p-1.5 bg-ne3na3-primary/20 rounded-lg">
                    <Sparkles className="w-4 h-4 text-ne3na3-neon" />
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold text-white">Segmentation Legend</h4>
                    <p className="text-xs text-gray-500">Tumor regions</p>
                  </div>
                </div>
                
                <div className="space-y-2">
                  {[
                    { label: 'NCR', fullName: 'Necrotic Core', color: 'bg-red-500', borderColor: 'border-red-500/30', desc: 'Dead tumor tissue' },
                    { label: 'ED', fullName: 'Edema', color: 'bg-green-500', borderColor: 'border-green-500/30', desc: 'Swelling around tumor' },
                    { label: 'ET', fullName: 'Enhancing Tumor', color: 'bg-blue-500', borderColor: 'border-blue-500/30', desc: 'Active tumor region' },
                  ].map((item) => (
                    <div key={item.label} className={`flex items-center gap-3 p-3 bg-gray-800/30 rounded-xl border ${item.borderColor}`}>
                      <div className={`w-5 h-5 ${item.color} rounded-md flex-shrink-0 shadow-lg`} />
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-semibold text-white">{item.label} - {item.fullName}</p>
                        <p className="text-xs text-gray-500">{item.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="pt-3 border-t border-gray-700/50 space-y-2">
                  <p className="text-xs font-medium text-gray-400">Keyboard Shortcuts</p>
                  <div className="flex flex-wrap gap-2">
                    <div className="flex items-center gap-1">
                      <kbd className="px-1.5 py-0.5 bg-gray-700/80 text-gray-300 text-xs rounded font-mono">Space</kbd>
                      <span className="text-xs text-gray-500">Play</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <kbd className="px-1.5 py-0.5 bg-gray-700/80 text-gray-300 text-xs rounded font-mono">↑↓</kbd>
                      <span className="text-xs text-gray-500">Navigate</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Bottom Row - Sagittal and Coronal Views */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Sagittal View */}
              <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/30 rounded-2xl border border-gray-700/50 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 bg-blue-500/20 rounded-lg">
                      <Layers className="w-4 h-4 text-blue-400" />
                    </div>
                    <div>
                      <span className="text-sm font-semibold text-white">Sagittal View</span>
                      <p className="text-xs text-gray-500">Side</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 px-2 py-1 bg-gray-800/50 rounded-lg">
                    <span className="text-xs font-mono text-blue-400">{sagittalSlice}</span>
                    <span className="text-xs text-gray-500">/ {maxSlices.sagittal}</span>
                  </div>
                </div>
                <div className="relative aspect-square max-h-[280px] mx-auto bg-gray-900 rounded-xl overflow-hidden shadow-lg shadow-black/30">
                  <SliceViewer
                    data={currentData}
                    slice={sagittalSlice}
                    axis="sagittal"
                    overlay={showOverlay ? currentSegmentation : null}
                    overlayOpacity={overlayOpacity}
                    title="Sagittal"
                  />
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setSagittalSlice(Math.max(0, sagittalSlice - 1))}
                    className="p-1.5 bg-gray-700/50 hover:bg-gray-600/50 text-gray-400 hover:text-white rounded-lg transition-all"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <input
                    type="range"
                    min="0"
                    max={maxSlices.sagittal}
                    value={sagittalSlice}
                    onChange={(e) => setSagittalSlice(parseInt(e.target.value))}
                    className="flex-1 h-1.5 accent-blue-500 cursor-pointer"
                  />
                  <button
                    onClick={() => setSagittalSlice(Math.min(maxSlices.sagittal, sagittalSlice + 1))}
                    className="p-1.5 bg-gray-700/50 hover:bg-gray-600/50 text-gray-400 hover:text-white rounded-lg transition-all"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Coronal View */}
              <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/30 rounded-2xl border border-gray-700/50 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 bg-purple-500/20 rounded-lg">
                      <Layers className="w-4 h-4 text-purple-400" />
                    </div>
                    <div>
                      <span className="text-sm font-semibold text-white">Coronal View</span>
                      <p className="text-xs text-gray-500">Front</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 px-2 py-1 bg-gray-800/50 rounded-lg">
                    <span className="text-xs font-mono text-purple-400">{coronalSlice}</span>
                    <span className="text-xs text-gray-500">/ {maxSlices.coronal}</span>
                  </div>
                </div>
                <div className="relative aspect-square max-h-[280px] mx-auto bg-gray-900 rounded-xl overflow-hidden shadow-lg shadow-black/30">
                  <SliceViewer
                    data={currentData}
                    slice={coronalSlice}
                    axis="coronal"
                    overlay={showOverlay ? currentSegmentation : null}
                    overlayOpacity={overlayOpacity}
                    title="Coronal"
                  />
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setCoronalSlice(Math.max(0, coronalSlice - 1))}
                    className="p-1.5 bg-gray-700/50 hover:bg-gray-600/50 text-gray-400 hover:text-white rounded-lg transition-all"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <input
                    type="range"
                    min="0"
                    max={maxSlices.coronal}
                    value={coronalSlice}
                    onChange={(e) => setCoronalSlice(parseInt(e.target.value))}
                    className="flex-1 h-1.5 accent-purple-500 cursor-pointer"
                  />
                  <button
                    onClick={() => setCoronalSlice(Math.min(maxSlices.coronal, coronalSlice + 1))}
                    className="p-1.5 bg-gray-700/50 hover:bg-gray-600/50 text-gray-400 hover:text-white rounded-lg transition-all"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {activeView === '3d' && (
          <motion.div
            key="3d"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-gradient-to-br from-gray-800/50 to-gray-900/30 rounded-2xl border border-gray-700/50 p-4 overflow-hidden"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-ne3na3-primary/20 rounded-xl">
                <Box className="w-5 h-5 text-ne3na3-neon" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">3D Volume Rendering</h3>
                <p className="text-xs text-gray-500">Interactive brain volume exploration</p>
              </div>
            </div>
            <div className="aspect-video min-h-[350px] lg:min-h-[450px] max-h-[500px] rounded-2xl overflow-hidden bg-gray-900 shadow-xl shadow-black/50">
              <VolumeRenderer3D />
            </div>
            <div className="flex flex-wrap items-center justify-center gap-4 mt-4 pt-3 border-t border-gray-700/50">
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <div className="p-1.5 bg-gray-700/50 rounded-lg">
                  <RotateCcw className="w-3.5 h-3.5" />
                </div>
                <span>Drag to rotate</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <div className="p-1.5 bg-gray-700/50 rounded-lg">
                  <ZoomIn className="w-3.5 h-3.5" />
                </div>
                <span>Scroll to zoom</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <div className="p-1.5 bg-gray-700/50 rounded-lg">
                  <Maximize2 className="w-3.5 h-3.5" />
                </div>
                <span>Reset to restore</span>
              </div>
            </div>
          </motion.div>
        )}

        {activeView === 'comparison' && (
          <motion.div
            key="comparison"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-4 overflow-hidden"
          >
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Before (Original MRI) */}
              <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/30 rounded-2xl border border-gray-700/50 p-4 space-y-3">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-gray-600/30 rounded-xl">
                    <Brain className="w-5 h-5 text-gray-400" />
                  </div>
                  <div>
                    <span className="text-base font-semibold text-white">Original MRI</span>
                    <p className="text-xs text-gray-500">Before segmentation</p>
                  </div>
                </div>
                <div className="relative aspect-square max-h-[350px] mx-auto bg-gray-900 rounded-xl overflow-hidden shadow-xl shadow-black/30">
                  <SliceViewer
                    data={currentData}
                    slice={axialSlice}
                    axis="axial"
                    overlay={null}
                    overlayOpacity={0}
                    title="Before"
                  />
                </div>
              </div>

              {/* After (With Segmentation) */}
              <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/30 rounded-2xl border border-ne3na3-primary/30 p-4 space-y-3">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-ne3na3-primary/20 rounded-xl">
                    <Activity className="w-5 h-5 text-ne3na3-neon" />
                  </div>
                  <div>
                    <span className="text-base font-semibold text-white">With Segmentation</span>
                    <p className="text-xs text-gray-500">AI-detected tumor regions</p>
                  </div>
                </div>
                <div className="relative aspect-square max-h-[350px] mx-auto bg-gray-900 rounded-xl overflow-hidden shadow-xl shadow-black/30 ring-1 ring-ne3na3-primary/20">
                  <SliceViewer
                    data={currentData}
                    slice={axialSlice}
                    axis="axial"
                    overlay={currentSegmentation}
                    overlayOpacity={overlayOpacity}
                    title="After"
                  />
                </div>
              </div>
            </div>

            {/* Shared Slider */}
            <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/30 rounded-2xl p-4 border border-gray-700/50">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-gray-700/50 rounded-xl">
                    <Layers className="w-4 h-4 text-gray-400" />
                  </div>
                  <span className="text-sm font-medium text-white">Navigate Slices</span>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800/50 rounded-xl">
                  <span className="text-sm font-mono text-ne3na3-neon">{axialSlice}</span>
                  <span className="text-xs text-gray-500">/ {maxSlices.axial}</span>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setAxialSlice(Math.max(0, axialSlice - 1))}
                  className="p-2 bg-gray-700/50 hover:bg-gray-600/50 text-gray-400 hover:text-white rounded-xl transition-all"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                <input
                  type="range"
                  min="0"
                  max={maxSlices.axial}
                  value={axialSlice}
                  onChange={(e) => setAxialSlice(parseInt(e.target.value))}
                  className="flex-1 h-2 accent-ne3na3-primary cursor-pointer"
                />
                <button
                  onClick={() => setAxialSlice(Math.min(maxSlices.axial, axialSlice + 1))}
                  className="p-2 bg-gray-700/50 hover:bg-gray-600/50 text-gray-400 hover:text-white rounded-xl transition-all"
                >
                  <ChevronRight className="w-5 h-5" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Info Box */}
      <motion.div 
        className="flex flex-col sm:flex-row items-start gap-4 p-4 bg-gradient-to-br from-ne3na3-primary/10 to-ne3na3-dark/5 
                   border border-ne3na3-primary/20 rounded-2xl overflow-hidden"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <div className="p-2 bg-ne3na3-primary/20 rounded-xl flex-shrink-0">
          <Eye className="w-5 h-5 text-ne3na3-neon" />
        </div>
        <div className="min-w-0">
          <p className="text-ne3na3-neon font-semibold text-base mb-1">Interactive Multi-Planar Visualization</p>
          <p className="text-gray-400 text-sm leading-relaxed">
            Explore brain MRI data across anatomical planes. Navigate slices with the sliders,
            toggle segmentation overlay, and switch between orthogonal, 3D volume, and comparison views.
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default Visualization3D;
