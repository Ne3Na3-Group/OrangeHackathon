/**
 * Ne3Na3 API Service
 * Handles all communication with the FastAPI backend
 */

import axios from 'axios';

// API base URL - use environment variable or default to localhost
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes timeout for large file processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`[Ne3Na3 API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('[Ne3Na3 API] Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('[Ne3Na3 API] Response error:', error);
    const message = error.response?.data?.detail || error.message || 'An error occurred';
    return Promise.reject(new Error(message));
  }
);

/**
 * Health check
 */
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

/**
 * Get model information
 */
export const getModelInfo = async () => {
  const response = await api.get('/api/model/info');
  return response.data;
};

/**
 * Run brain tumor segmentation on 4 MRI modalities
 * @param {File} t1 - T1-weighted MRI file
 * @param {File} t1ce - T1-contrast enhanced MRI file
 * @param {File} t2 - T2-weighted MRI file
 * @param {File} flair - FLAIR MRI file
 * @param {File|null} groundTruth - Optional ground truth segmentation file
 * @param {boolean} useTTA - Apply Test-Time Augmentation
 * @param {boolean} enforceConsistency - Enforce anatomical consistency
 * @param {function} onProgress - Progress callback
 */
export const runSegmentation = async (
  t1,
  t1ce,
  t2,
  flair,
  groundTruth = null,
  useTTA = true,
  enforceConsistency = true,
  onProgress = null
) => {
  const formData = new FormData();
  formData.append('t1', t1);
  formData.append('t1ce', t1ce);
  formData.append('t2', t2);
  formData.append('flair', flair);
  if (groundTruth) {
    formData.append('ground_truth', groundTruth);
  }
  formData.append('use_tta', useTTA);
  formData.append('enforce_consistency', enforceConsistency);

  const response = await api.post('/api/segment', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress({ phase: 'upload', percent });
      }
    },
  });

  return response.data;
};

/**
 * Get current analysis insights
 */
export const getInsights = async () => {
  const response = await api.get('/api/insights');
  return response.data;
};

/**
 * Run demo analysis with synthetic data
 */
export const runDemoAnalysis = async () => {
  const response = await api.post('/api/demo');
  return response.data;
};

/**
 * Send a message to the Safe-Bot
 * @param {string} message - User message
 */
export const sendChatMessage = async (message) => {
  const response = await api.post('/api/chat', { message });
  return response.data;
};

/**
 * Get chat history
 */
export const getChatHistory = async () => {
  const response = await api.get('/api/chat/history');
  return response.data;
};

/**
 * Clear chat history
 */
export const clearChatHistory = async () => {
  const response = await api.delete('/api/chat/history');
  return response.data;
};

/**
 * Get attention maps for explainability
 */
export const getAttentionMaps = async () => {
  const response = await api.get('/api/attention');
  return response.data;
};

/**
 * Get system prompt (for reference)
 */
export const getSystemPrompt = async () => {
  const response = await api.get('/api/chat/system-prompt');
  return response.data;
};

// ============================================
// 3D Volume Visualization API
// ============================================

/**
 * Get MRI volume data for 3D visualization
 * @param {string} modality - MRI modality (t1, t1ce, t2, flair)
 * @param {number} downsample - Downsampling factor (1=original, 2=half, etc.)
 */
export const getMRIVolume = async (modality = 't1ce', downsample = 4) => {
  const response = await api.get('/api/volume/mri', {
    params: { modality, downsample },
    timeout: 120000, // 120 second timeout for large volumes
  });
  return response.data;
};

/**
 * Get segmentation volume data for 3D visualization
 * @param {number} downsample - Downsampling factor
 */
export const getSegmentationVolume = async (downsample = 4) => {
  const response = await api.get('/api/volume/segmentation', {
    params: { downsample },
    timeout: 120000,
  });
  return response.data;
};

/**
 * Get a single 2D slice from the volume (more efficient for slice viewers)
 * @param {string} axis - Slice axis (axial, sagittal, coronal)
 * @param {number} sliceIdx - Slice index
 * @param {string} modality - MRI modality (for MRI slices)
 * @param {boolean} isSegmentation - Whether to get segmentation slice
 */
export const getVolumeSlice = async (
  axis = 'axial',
  sliceIdx = 0,
  modality = 't1ce',
  isSegmentation = false
) => {
  const response = await api.get('/api/volume/slice', {
    params: {
      axis,
      slice_idx: sliceIdx,
      modality,
      is_segmentation: isSegmentation,
    },
  });
  return response.data;
};

/**
 * Check if volume data is available (has been uploaded and processed)
 */
export const checkVolumeAvailability = async () => {
  try {
    const response = await api.get('/api/volume/mri', {
      params: { modality: 't1ce', downsample: 4 },
      timeout: 5000,
    });
    return response.data && response.data.shape;
  } catch (error) {
    return false;
  }
};

export default api;
