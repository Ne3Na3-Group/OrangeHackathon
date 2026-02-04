import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, 
  Activity, 
  Sparkles,
  Menu,
  X,
  Zap,
  Shield,
  CheckCircle,
  AlertCircle,
  Play,
  RefreshCw,
  ChevronRight,
  Cpu,
  Hexagon,
  Box
} from 'lucide-react';

import {
  FileUploadZone,
  InsightsPanel,
  SafeBot,
  ProcessingOverlay,
  Visualization3D
} from './components';

import { 
  checkHealth, 
  runSegmentation, 
  runDemoAnalysis,
  getModelInfo
} from './services/api';

// Navigation tabs with enhanced styling
const TABS = [
  { id: 'upload', label: 'Upload', icon: Brain, description: 'Upload MRI scans' },
  { id: 'visualize', label: '3D View', icon: Box, description: '3D visualization' },
  { id: 'insights', label: 'Insights', icon: Activity, description: 'View analysis' },
];

// Floating Orb Component
const FloatingOrbs = () => (
  <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
    <div className="floating-orb orb-1" />
    <div className="floating-orb orb-2" />
    <div className="floating-orb orb-3" />
  </div>
);

// Logo Component
const Logo = () => (
  <motion.div 
    className="flex items-center gap-4"
    initial={{ opacity: 0, x: -20 }}
    animate={{ opacity: 1, x: 0 }}
    transition={{ duration: 0.5 }}
  >
    <div className="relative">
      <motion.div 
        className="w-12 h-12 bg-gradient-to-br from-ne3na3-primary via-ne3na3-neon to-ne3na3-dark rounded-2xl 
                    flex items-center justify-center shadow-lg"
        whileHover={{ scale: 1.05, rotate: 5 }}
        transition={{ type: "spring", stiffness: 400 }}
      >
        <Brain className="w-7 h-7 text-white" />
      </motion.div>
      <motion.div 
        className="absolute -inset-1 bg-gradient-to-br from-ne3na3-neon/30 to-ne3na3-primary/30 rounded-2xl blur-lg"
        animate={{ opacity: [0.5, 0.8, 0.5] }}
        transition={{ duration: 2, repeat: Infinity }}
      />
    </div>
    <div>
      <h1 className="text-2xl font-bold gradient-text-animated">Ne3Na3</h1>
      <p className="text-xs text-gray-500 flex items-center gap-1">
        <Sparkles className="w-3 h-3" />
        Medical AI Brain Segmentation
      </p>
    </div>
  </motion.div>
);

// Server Status Badge
const ServerStatusBadge = ({ status }) => (
  <motion.div 
    className={`
      flex items-center gap-2 px-4 py-2 rounded-2xl text-sm font-medium
      backdrop-blur-xl border transition-all duration-300
      ${status === 'connected' 
        ? 'bg-ne3na3-primary/10 border-ne3na3-primary/30 text-ne3na3-neon' 
        : status === 'error'
          ? 'bg-red-500/10 border-red-500/30 text-red-400'
          : 'bg-gray-800/50 border-gray-700 text-gray-400'
      }
    `}
    initial={{ opacity: 0, scale: 0.9 }}
    animate={{ opacity: 1, scale: 1 }}
    whileHover={{ scale: 1.02 }}
  >
    <div className={`status-dot ${status === 'connected' ? 'status-online' : status === 'error' ? 'status-offline' : 'status-pending'}`} />
    {status === 'connected' ? (
      <span>Connected</span>
    ) : status === 'error' ? (
      <span>Offline</span>
    ) : (
      <span className="flex items-center gap-2">
        <RefreshCw className="w-3 h-3 animate-spin" />
        Connecting
      </span>
    )}
  </motion.div>
);

function App() {
  // State
  const [activeTab, setActiveTab] = useState('upload');
  const [files, setFiles] = useState({});
  const [insights, setInsights] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingPhase, setProcessingPhase] = useState('upload');
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [serverStatus, setServerStatus] = useState('checking');
  const [modelInfo, setModelInfo] = useState(null);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [error, setError] = useState(null);

  // Check server status on mount
  useEffect(() => {
    const checkServer = async () => {
      try {
        const health = await checkHealth();
        setServerStatus(health.status === 'healthy' ? 'connected' : 'error');
        
        const info = await getModelInfo();
        setModelInfo(info);
      } catch (err) {
        setServerStatus('error');
        console.error('Server check failed:', err);
      }
    };

    checkServer();
    // Check every 30 seconds
    const interval = setInterval(checkServer, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handle segmentation
  const handleAnalyze = async () => {
    if (!files.t1 || !files.t1ce || !files.t2 || !files.flair) {
      setError('Please upload all 4 MRI modalities');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setProcessingPhase('upload');

    try {
      // Simulate processing phases
      setTimeout(() => setProcessingPhase('preprocess'), 1000);
      setTimeout(() => setProcessingPhase('inference'), 3000);
      setTimeout(() => setProcessingPhase('tta'), 5000);
      setTimeout(() => setProcessingPhase('consistency'), 7000);
      setTimeout(() => setProcessingPhase('insights'), 9000);

      const result = await runSegmentation(
        files.t1,
        files.t1ce,
        files.t2,
        files.flair,
        true, // useTTA
        true  // enforceConsistency
      );

      if (result.success) {
        setInsights(result.insights);
        setActiveTab('insights');

      } else {
        setError(result.message || 'Segmentation failed');
      }
    } catch (err) {
      setError(err.message || 'An error occurred during processing');
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle demo analysis
  const handleDemo = async () => {
    setIsProcessing(true);
    setError(null);
    setProcessingPhase('inference');

    try {
      const result = await runDemoAnalysis();
      
      if (result.success) {
        setInsights(result.insights);
        setActiveTab('insights');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen text-white relative">
      {/* Animated Background */}
      <div className="bg-mesh" />
      <FloatingOrbs />

      {/* Processing Overlay */}
      <AnimatePresence>
        {isProcessing && (
          <ProcessingOverlay 
            isProcessing={isProcessing}
            phase={processingPhase}
          />
        )}
      </AnimatePresence>

      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-white/5">
        <div className="absolute inset-0 bg-gray-950/60 backdrop-blur-2xl" />
        <div className="relative max-w-7xl mx-auto px-4 md:px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <Logo />

            {/* Desktop Nav */}
            <nav className="hidden lg:flex items-center">
              <div className="flex items-center gap-1 p-1.5 rounded-2xl bg-gray-900/50 border border-white/5">
                {TABS.map((tab, index) => (
                  <motion.button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`
                      relative flex items-center gap-2 px-5 py-2.5 rounded-xl transition-all duration-300
                      ${activeTab === tab.id 
                        ? 'text-white' 
                        : 'text-gray-400 hover:text-white hover:bg-white/5'
                      }
                    `}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {activeTab === tab.id && (
                      <motion.div
                        className="absolute inset-0 bg-gradient-to-r from-ne3na3-primary/80 to-ne3na3-dark/80 rounded-xl"
                        layoutId="activeTab"
                        transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                      />
                    )}
                    <span className="relative z-10 flex items-center gap-2">
                      <tab.icon className="w-4 h-4" />
                      <span className="text-sm font-medium">{tab.label}</span>
                    </span>
                  </motion.button>
                ))}
              </div>
            </nav>

            {/* Actions */}
            <div className="hidden md:flex items-center gap-3">
              <motion.button
                onClick={handleDemo}
                className="btn-pill btn-ghost flex items-center gap-2"
                disabled={isProcessing}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Play className="w-4 h-4" />
                <span>Demo</span>
              </motion.button>
              
              <ServerStatusBadge status={serverStatus} />
            </div>

            {/* Mobile Menu Button */}
            <motion.button
              className="lg:hidden p-2.5 rounded-xl bg-gray-800/50 border border-white/10 text-gray-400 hover:text-white"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              whileTap={{ scale: 0.95 }}
            >
              {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </motion.button>
          </div>

          {/* Mobile Menu */}
          <AnimatePresence>
            {isMobileMenuOpen && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="lg:hidden overflow-hidden"
              >
                <nav className="flex flex-col gap-2 pt-4 pb-2">
                  {TABS.map((tab) => (
                    <motion.button
                      key={tab.id}
                      onClick={() => {
                        setActiveTab(tab.id);
                        setIsMobileMenuOpen(false);
                      }}
                      className={`
                        flex items-center gap-3 px-4 py-3.5 rounded-2xl transition-all
                        ${activeTab === tab.id 
                          ? 'bg-gradient-to-r from-ne3na3-primary/20 to-ne3na3-dark/20 border border-ne3na3-primary/30 text-white' 
                          : 'text-gray-400 hover:bg-gray-800/50'
                        }
                      `}
                      whileTap={{ scale: 0.98 }}
                    >
                      <tab.icon className="w-5 h-5" />
                      <div className="text-left">
                        <span className="block font-medium">{tab.label}</span>
                        <span className="text-xs text-gray-500">{tab.description}</span>
                      </div>
                    </motion.button>
                  ))}
                  <motion.button
                    onClick={handleDemo}
                    className="flex items-center gap-3 px-4 py-3.5 bg-ne3na3-primary/10 border border-ne3na3-primary/30 rounded-2xl text-ne3na3-primary"
                    whileTap={{ scale: 0.98 }}
                  >
                    <Play className="w-5 h-5" />
                    <span className="font-medium">Run Demo Analysis</span>
                  </motion.button>
                </nav>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative max-w-7xl mx-auto px-4 md:px-6 py-8">
        {/* Error Alert */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20, scale: 0.95 }}
              className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-2xl 
                         flex items-center gap-4 backdrop-blur-xl"
            >
              <div className="p-2 bg-red-500/20 rounded-xl">
                <AlertCircle className="w-5 h-5 text-red-400" />
              </div>
              <p className="text-red-400 flex-1">{error}</p>
              <motion.button
                onClick={() => setError(null)}
                className="p-2 hover:bg-red-500/20 rounded-xl text-red-400 transition-colors"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <X className="w-4 h-4" />
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">
          {/* Main Panel */}
          <div className="lg:col-span-8">
            <motion.div 
              className="card-glass shine"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <AnimatePresence mode="wait">
                {activeTab === 'upload' && (
                  <motion.div
                    key="upload"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <FileUploadZone
                      files={files}
                      setFiles={setFiles}
                      onAnalyze={handleAnalyze}
                      isProcessing={isProcessing}
                    />
                  </motion.div>
                )}

                {activeTab === 'insights' && (
                  <motion.div
                    key="insights"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <InsightsPanel insights={insights} />
                  </motion.div>
                )}

                {activeTab === 'visualize' && (
                  <motion.div
                    key="visualize"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Visualization3D 
                      mriData={null}
                      segmentationData={null}
                      insights={insights}
                      hasProcessedData={!!insights}
                    />
                  </motion.div>
                )}

              </AnimatePresence>
            </motion.div>
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-4 space-y-6">
            {/* Model Info Card */}
            <motion.div 
              className="card-glass"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <div className="flex items-center gap-3 mb-5">
                <div className="p-2.5 bg-ne3na3-primary/10 rounded-xl border border-ne3na3-primary/20">
                  <Cpu className="w-5 h-5 text-ne3na3-primary" />
                </div>
                <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
                  Model Info
                </h3>
              </div>
              {modelInfo ? (
                <div className="space-y-4">
                  {[
                    { label: 'Architecture', value: modelInfo.architecture || 'AttUnet', highlight: true },
                    { label: 'Device', value: modelInfo.device || 'CPU', isDevice: true },
                    { label: 'Weights', value: modelInfo.weights_available ? 'Loaded' : 'Demo Mode', isStatus: true, loaded: modelInfo.weights_available },
                    ...(modelInfo.num_parameters ? [{ label: 'Parameters', value: `${(modelInfo.num_parameters / 1e6).toFixed(1)}M` }] : [])
                  ].map((item, index) => (
                    <motion.div 
                      key={item.label}
                      className="flex items-center justify-between"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <span className="text-sm text-gray-400">{item.label}</span>
                      <span className={`
                        text-sm font-mono px-3 py-1 rounded-lg
                        ${item.highlight ? 'bg-ne3na3-primary/10 text-ne3na3-neon border border-ne3na3-primary/20' : ''}
                        ${item.isDevice ? (item.value.includes('cuda') ? 'bg-green-500/10 text-green-400' : 'bg-gray-800 text-gray-400') : ''}
                        ${item.isStatus ? (item.loaded ? 'text-ne3na3-neon' : 'text-yellow-400') : ''}
                        ${!item.highlight && !item.isDevice && !item.isStatus ? 'text-white' : ''}
                      `}>
                        {item.isStatus && (item.loaded ? '✓ ' : '⚠ ')}{item.value}
                      </span>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="flex items-center gap-3 text-gray-500">
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Loading model info...</span>
                </div>
              )}
            </motion.div>

            {/* Quick Stats */}
            {insights && (
              <motion.div 
                className="card-glass"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <div className="flex items-center gap-3 mb-5">
                  <div className="p-2.5 bg-ne3na3-primary/10 rounded-xl border border-ne3na3-primary/20">
                    <Activity className="w-5 h-5 text-ne3na3-primary" />
                  </div>
                  <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
                    Quick Stats
                  </h3>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <motion.div 
                    className="card-metric text-center"
                    whileHover={{ scale: 1.02 }}
                  >
                    <motion.p 
                      className="text-3xl font-bold gradient-text"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", bounce: 0.4 }}
                    >
                      {insights.summary.total_tumor_volume_cm3.toFixed(1)}
                    </motion.p>
                    <p className="text-xs text-gray-500 mt-1">Total Vol (cm³)</p>
                  </motion.div>
                </div>
              </motion.div>
            )}

            {/* Safety Notice */}
            <motion.div 
              className="card-glass border-ne3na3-primary/20 animated-border"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <div className="flex items-start gap-4">
                <div className="p-2.5 bg-ne3na3-primary/10 rounded-xl border border-ne3na3-primary/20">
                  <Shield className="w-5 h-5 text-ne3na3-primary" />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-white mb-2">
                    Research Use Only
                  </h4>
                  <p className="text-xs text-gray-400 leading-relaxed">
                    Ne3Na3 is for educational and research purposes. 
                    Always consult qualified healthcare professionals 
                    for medical decisions.
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Powered By */}
            <motion.div 
              className="flex flex-wrap items-center justify-center gap-3 px-4 py-3"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              {['AttUnet', 'MONAI', 'PyTorch'].map((tech, i) => (
                <span key={tech} className="px-3 py-1.5 bg-gray-800/50 border border-gray-700/50 rounded-lg text-xs text-gray-400">
                  {tech}
                </span>
              ))}
            </motion.div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative border-t border-white/5 pt-12 pb-8 mt-12">
        <div className="absolute inset-0 bg-gradient-to-b from-gray-950/60 to-gray-950/90 backdrop-blur-xl" />
        <div className="relative max-w-7xl mx-auto px-4 md:px-6">
          {/* Main Footer Content */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-10">
            {/* Brand Section */}
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-ne3na3-primary/20 rounded-xl border border-ne3na3-primary/30">
                  <Hexagon className="w-6 h-6 text-ne3na3-neon" />
                </div>
                <div>
                  <h3 className="text-lg font-bold gradient-text">Ne3Na3</h3>
                  <p className="text-xs text-gray-500">Medical AI Platform</p>
                </div>
              </div>
              <p className="text-sm text-gray-400 leading-relaxed">
                Senior Project for Brain Tumor Segmentation using Deep Learning and Medical Imaging.
              </p>
              <div className="flex flex-wrap gap-2">
                {['AttUnet', 'MONAI', 'PyTorch', 'FastAPI', 'React'].map((tech) => (
                  <span key={tech} className="px-2 py-1 bg-gray-800/50 border border-gray-700/50 rounded-md text-xs text-gray-500">
                    {tech}
                  </span>
                ))}
              </div>
            </div>

            {/* Team Section */}
            <div className="lg:col-span-2">
              <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                <span className="w-8 h-px bg-gradient-to-r from-ne3na3-primary to-transparent"></span>
                Development Team
              </h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                {[
                  { name: 'Ahmed Samir', email: 'ahmedsamir1598@gmail.com' },
                  { name: 'Mohammad Emad', email: 'loaypre2510@gmail.com' },
                  { name: 'Loay Medhat', email: 'memad.20003@gmail.com' },
                  { name: 'Youssef Yasser', email: 'ywsfyasr68@gmail.com' },
                  { name: 'Yosry Ramadan', email: 'ramadanyousre@gmail.com' },
                ].map((member, index) => (
                  <motion.a
                    key={member.name}
                    href={`mailto:${member.email}`}
                    className="group p-3 bg-gray-800/30 hover:bg-gray-800/50 border border-gray-700/50 
                               hover:border-ne3na3-primary/30 rounded-xl transition-all duration-300"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ scale: 1.02, y: -2 }}
                  >
                    <div className="w-8 h-8 bg-gradient-to-br from-ne3na3-primary/30 to-ne3na3-dark/30 
                                    rounded-lg flex items-center justify-center mb-2 
                                    group-hover:from-ne3na3-primary/50 group-hover:to-ne3na3-neon/30 transition-all">
                      <span className="text-sm font-bold text-ne3na3-neon">
                        {member.name.split(' ').map(n => n[0]).join('')}
                      </span>
                    </div>
                    <p className="text-xs font-medium text-white group-hover:text-ne3na3-neon transition-colors truncate">
                      {member.name}
                    </p>
                    <p className="text-[10px] text-gray-500 group-hover:text-gray-400 transition-colors truncate">
                      {member.email}
                    </p>
                  </motion.a>
                ))}
              </div>
            </div>
          </div>

          {/* Bottom Bar */}
          <div className="pt-6 border-t border-gray-800/50">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
              <p className="text-xs text-gray-500 flex items-center gap-2">
                <span>🌿</span>
                <span>© 2026 Ne3Na3 — Senior Medical AI for Brain Tumor Segmentation</span>
              </p>
              <p className="text-xs text-gray-600">
                For Research & Educational Purposes Only
              </p>
            </div>
          </div>
        </div>
      </footer>

      {/* Safe-Bot Chatbot */}
      <SafeBot 
        isOpen={isChatOpen}
        onToggle={() => setIsChatOpen(!isChatOpen)}
        insights={insights}
      />
    </div>
  );
}

export default App;
