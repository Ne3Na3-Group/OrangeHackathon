import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileText, Check, X, Brain, AlertCircle, Sparkles, Layers } from 'lucide-react';

const MODALITIES = [
  { id: 't1', name: 'T1', description: 'T1-weighted', color: '#00A676', gradient: 'from-emerald-500/20 to-teal-500/20' },
  { id: 't1ce', name: 'T1ce', description: 'T1-contrast enhanced', color: '#00FFB3', gradient: 'from-cyan-500/20 to-emerald-500/20' },
  { id: 't2', name: 'T2', description: 'T2-weighted', color: '#10B981', gradient: 'from-green-500/20 to-emerald-500/20' },
  { id: 'flair', name: 'FLAIR', description: 'Fluid-attenuated', color: '#34D399', gradient: 'from-teal-500/20 to-green-500/20' },
];

const FileUploadZone = ({ files, setFiles, onAnalyze, isProcessing }) => {
  const [activeModality, setActiveModality] = useState(null);

  const onDrop = useCallback((acceptedFiles, rejectedFiles, event) => {
    if (acceptedFiles.length > 0 && activeModality) {
      const file = acceptedFiles[0];
      setFiles(prev => ({
        ...prev,
        [activeModality]: file
      }));
      setActiveModality(null);
    }
  }, [activeModality, setFiles]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/gzip': ['.gz'],
      'application/x-nifti': ['.nii', '.nii.gz'],
    },
    maxFiles: 1,
    disabled: !activeModality || isProcessing,
  });

  const removeFile = (modalityId) => {
    setFiles(prev => {
      const newFiles = { ...prev };
      delete newFiles[modalityId];
      return newFiles;
    });
  };

  const allFilesUploaded = MODALITIES.every(m => files[m.id]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-gradient-to-br from-ne3na3-primary/20 to-ne3na3-neon/10 rounded-2xl border border-ne3na3-primary/20">
            <Layers className="w-6 h-6 text-ne3na3-neon" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              Upload MRI Scans
              <Sparkles className="w-4 h-4 text-ne3na3-neon animate-pulse" />
            </h2>
            <p className="text-gray-400 text-sm mt-0.5">
              Upload all 4 modalities in NIfTI format
            </p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <div className="flex items-center gap-2 text-sm">
            <span className="text-gray-400 font-medium">
              {Object.keys(files).length}/4
            </span>
          </div>
          <div className="w-28 h-2 bg-gray-800/50 rounded-full overflow-hidden border border-gray-700/50">
            <motion.div 
              className="h-full bg-gradient-to-r from-ne3na3-primary via-ne3na3-neon to-ne3na3-primary"
              initial={{ width: 0 }}
              animate={{ width: `${(Object.keys(files).length / 4) * 100}%` }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              style={{ backgroundSize: '200% 100%' }}
            />
          </div>
        </div>
      </div>

      {/* Modality Cards */}
      <div className="grid grid-cols-2 gap-4">
        {MODALITIES.map((modality, index) => {
          const file = files[modality.id];
          const isActive = activeModality === modality.id;

          return (
            <motion.div
              key={modality.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`
                relative rounded-2xl border-2 transition-all duration-300 overflow-hidden group
                ${file 
                  ? 'border-ne3na3-primary/50 bg-gradient-to-br ' + modality.gradient
                  : isActive
                    ? 'border-ne3na3-neon bg-ne3na3-primary/5 shadow-neon'
                    : 'border-gray-700/50 bg-gray-800/30 hover:border-ne3na3-primary/30 hover:bg-gray-800/50'
                }
              `}
              whileHover={{ scale: 1.02, y: -2 }}
              whileTap={{ scale: 0.98 }}
            >
              {/* Glow effect on hover */}
              <div className="absolute inset-0 bg-gradient-to-br from-ne3na3-primary/0 to-ne3na3-neon/0 
                              group-hover:from-ne3na3-primary/5 group-hover:to-ne3na3-neon/5 transition-all duration-500" />
              
              {file ? (
                // File uploaded state
                <div className="relative p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <motion.div 
                        className="w-12 h-12 rounded-xl flex items-center justify-center bg-gradient-to-br from-ne3na3-primary/20 to-ne3na3-neon/10 border border-ne3na3-primary/30"
                        initial={{ scale: 0, rotate: -180 }}
                        animate={{ scale: 1, rotate: 0 }}
                        transition={{ type: "spring", bounce: 0.5 }}
                      >
                        <Check className="w-6 h-6 text-ne3na3-neon" />
                      </motion.div>
                      <div>
                        <p className="font-semibold text-white">{modality.name}</p>
                        <p className="text-xs text-gray-400 truncate max-w-[120px]">
                          {file.name}
                        </p>
                      </div>
                    </div>
                    <motion.button
                      onClick={() => removeFile(modality.id)}
                      className="p-2 rounded-xl hover:bg-red-500/20 text-gray-400 hover:text-red-400 transition-all"
                      disabled={isProcessing}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <X className="w-4 h-4" />
                    </motion.button>
                  </div>
                  <div className="mt-3 flex items-center justify-between">
                    <span className="text-xs text-gray-500 font-mono">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </span>
                    <span className="text-xs text-ne3na3-primary">Ready ✓</span>
                  </div>
                </div>
              ) : (
                // Upload state
                <button
                  className="relative w-full p-4 text-left"
                  onClick={() => setActiveModality(modality.id)}
                  disabled={isProcessing}
                >
                  <div className="flex items-center gap-3">
                    <div 
                      className="w-12 h-12 rounded-xl flex items-center justify-center border border-gray-600/50 bg-gray-800/50 
                                 group-hover:border-ne3na3-primary/30 group-hover:bg-ne3na3-primary/10 transition-all"
                    >
                      <FileText className="w-5 h-5 text-gray-400 group-hover:text-ne3na3-neon transition-colors" />
                    </div>
                    <div>
                      <p className="font-semibold text-white group-hover:text-ne3na3-neon transition-colors">
                        {modality.name}
                      </p>
                      <p className="text-xs text-gray-500">{modality.description}</p>
                    </div>
                  </div>
                  <p className="mt-3 text-xs text-gray-600 group-hover:text-gray-400 transition-colors flex items-center gap-1">
                    <Upload className="w-3 h-3" />
                    Click to upload
                  </p>
                </button>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Dropzone (when modality is selected) */}
      <AnimatePresence>
        {activeModality && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div
              {...getRootProps()}
              className={`
                relative p-8 border-2 border-dashed rounded-2xl text-center cursor-pointer
                transition-all duration-300 overflow-hidden
                ${isDragActive 
                  ? 'border-ne3na3-neon bg-ne3na3-primary/10 shadow-neon' 
                  : 'border-gray-600 bg-gray-800/30 hover:border-ne3na3-primary/50 hover:bg-gray-800/50'
                }
              `}
            >
              {/* Animated background */}
              <div className={`absolute inset-0 transition-opacity duration-300 ${isDragActive ? 'opacity-100' : 'opacity-0'}`}>
                <div className="absolute inset-0 bg-gradient-to-br from-ne3na3-primary/10 via-transparent to-ne3na3-neon/10" />
              </div>
              
              <input {...getInputProps()} />
              
              <motion.div
                animate={{ y: isDragActive ? -5 : 0 }}
                transition={{ type: "spring", bounce: 0.5 }}
              >
                <div className={`
                  w-16 h-16 mx-auto mb-4 rounded-2xl flex items-center justify-center
                  ${isDragActive 
                    ? 'bg-ne3na3-neon/20 border border-ne3na3-neon/50' 
                    : 'bg-gray-700/50 border border-gray-600/50'
                  }
                `}>
                  <Upload className={`w-8 h-8 transition-colors ${isDragActive ? 'text-ne3na3-neon' : 'text-gray-500'}`} />
                </div>
              </motion.div>
              
              <p className={`text-lg font-medium mb-2 transition-colors ${isDragActive ? 'text-ne3na3-neon' : 'text-white'}`}>
                Drop <span className="font-bold">{MODALITIES.find(m => m.id === activeModality)?.name}</span> file here
              </p>
              <p className="text-sm text-gray-400 mb-4">
                Accepts .nii and .nii.gz files
              </p>
              <button
                className="text-sm text-gray-500 hover:text-ne3na3-primary transition-colors px-4 py-2 rounded-xl hover:bg-gray-700/50"
                onClick={(e) => {
                  e.stopPropagation();
                  setActiveModality(null);
                }}
              >
                Cancel
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Analyze Button */}
      <motion.button
        className={`
          relative w-full py-4 rounded-2xl font-semibold text-lg flex items-center justify-center gap-3
          overflow-hidden transition-all duration-300
          ${allFilesUploaded && !isProcessing
            ? 'bg-gradient-to-r from-ne3na3-primary to-ne3na3-dark text-white shadow-neon hover:shadow-neon-lg'
            : 'bg-gray-800/50 text-gray-500 cursor-not-allowed border border-gray-700/50'
          }
        `}
        onClick={onAnalyze}
        disabled={!allFilesUploaded || isProcessing}
        whileHover={allFilesUploaded && !isProcessing ? { scale: 1.02, y: -2 } : {}}
        whileTap={allFilesUploaded && !isProcessing ? { scale: 0.98 } : {}}
      >
        {/* Animated shine effect */}
        {allFilesUploaded && !isProcessing && (
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"
            initial={{ x: '-100%' }}
            animate={{ x: '100%' }}
            transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
          />
        )}
        
        <span className="relative z-10 flex items-center gap-3">
          {isProcessing ? (
            <>
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Processing...</span>
            </>
          ) : (
            <>
              <Brain className="w-6 h-6" />
              <span>Run Segmentation</span>
              {allFilesUploaded && <Sparkles className="w-4 h-4 animate-pulse" />}
            </>
          )}
        </span>
      </motion.button>

      {/* Info Box */}
      <motion.div 
        className="flex items-start gap-4 p-5 bg-gradient-to-br from-ne3na3-primary/10 to-ne3na3-dark/5 
                   border border-ne3na3-primary/20 rounded-2xl backdrop-blur-xl"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <div className="p-2 bg-ne3na3-primary/20 rounded-xl">
          <AlertCircle className="w-5 h-5 text-ne3na3-neon" />
        </div>
        <div className="text-sm">
          <p className="text-ne3na3-neon font-semibold mb-1">Multi-Modal Analysis</p>
          <p className="text-gray-400 leading-relaxed">
            Ne3Na3 uses all 4 MRI sequences to accurately segment tumor regions: 
            <span className="text-white"> NCR</span> (Necrotic Core), 
            <span className="text-white"> ED</span> (Edema), and 
            <span className="text-white"> ET</span> (Enhancing Tumor).
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default FileUploadZone;
