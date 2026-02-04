import React from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, 
  Box, 
  Target, 
  Layers,
  Maximize2,
  AlertTriangle,
  CheckCircle,
  Sparkles,
  Clock,
  Scan
} from 'lucide-react';

const InsightsPanel = ({ insights }) => {
  if (!insights) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500 py-16">
        <div className="p-6 bg-gray-800/30 rounded-3xl border border-gray-700/50 mb-6">
          <Activity className="w-16 h-16 opacity-40" />
        </div>
        <p className="text-lg font-semibold text-gray-300">No Analysis Yet</p>
        <p className="text-sm mt-2 text-gray-500">Upload MRI scans to see AI insights</p>
      </div>
    );
  }

  const { summary, volumes, regions, modality_importance } = insights;

  // Format volume for display
  const formatVolume = (vol) => {
    if (vol >= 1) return `${vol.toFixed(2)} cm³`;
    return `${(vol * 1000).toFixed(1)} mm³`;
  };

  // Volume cards data
  const volumeCards = [
    { 
      label: 'Whole Tumor', 
      key: 'WT',
      value: summary.total_tumor_volume_cm3,
      gradient: 'from-emerald-500 to-teal-500',
      bgGradient: 'from-emerald-500/10 to-teal-500/10',
      icon: Box
    },
    { 
      label: 'Tumor Core', 
      key: 'TC',
      value: summary.tumor_core_volume_cm3,
      gradient: 'from-amber-500 to-orange-500',
      bgGradient: 'from-amber-500/10 to-orange-500/10',
      icon: Target
    },
    { 
      label: 'Enhancing', 
      key: 'ET',
      value: summary.enhancing_tumor_volume_cm3,
      gradient: 'from-rose-500 to-red-500',
      bgGradient: 'from-rose-500/10 to-red-500/10',
      icon: Activity
    },
    { 
      label: 'Edema', 
      key: 'ED',
      value: summary.edema_volume_cm3,
      gradient: 'from-blue-500 to-cyan-500',
      bgGradient: 'from-blue-500/10 to-cyan-500/10',
      icon: Layers
    },
  ];

  return (
    <div className="space-y-6">
      {/* Detection Status */}
      <motion.div 
        className={`
          p-5 rounded-2xl border backdrop-blur-xl
          ${summary.tumor_detected 
            ? 'bg-gradient-to-r from-amber-500/10 to-orange-500/5 border-amber-500/30' 
            : 'bg-gradient-to-r from-ne3na3-primary/10 to-ne3na3-neon/5 border-ne3na3-primary/30'
          }
        `}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-4">
          <div className={`
            p-3 rounded-xl
            ${summary.tumor_detected 
              ? 'bg-amber-500/20 border border-amber-500/30' 
              : 'bg-ne3na3-primary/20 border border-ne3na3-primary/30'
            }
          `}>
            {summary.tumor_detected 
              ? <AlertTriangle className="w-6 h-6 text-amber-400" />
              : <CheckCircle className="w-6 h-6 text-ne3na3-neon" />
            }
          </div>
          <div>
            <span className={`
              font-semibold text-lg
              ${summary.tumor_detected ? 'text-amber-400' : 'text-ne3na3-neon'}
            `}>
              {summary.tumor_detected ? 'Tumor Regions Detected' : 'No Tumor Detected'}
            </span>
            <p className="text-sm text-gray-400 mt-0.5">
              {summary.tumor_detected 
                ? 'Analysis identified potential abnormal regions' 
                : 'No abnormal regions identified in scan'
              }
            </p>
          </div>
        </div>
      </motion.div>

      {/* Volume Cards Grid */}
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-4 uppercase tracking-wider flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-ne3na3-neon" />
          Volume Measurements
        </h3>
        <div className="grid grid-cols-2 gap-4">
          {volumeCards.map((card, index) => (
            <motion.div
              key={card.key}
              className={`
                relative p-4 rounded-2xl border border-gray-700/50 overflow-hidden
                bg-gradient-to-br ${card.bgGradient} backdrop-blur-xl
                hover:border-gray-600/50 transition-all duration-300 group
              `}
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.02, y: -2 }}
            >
              <div className="flex items-center gap-2 mb-3">
                <div className={`p-2 rounded-lg bg-gradient-to-br ${card.gradient}`}>
                  <card.icon className="w-4 h-4 text-white" />
                </div>
                <span className="text-xs text-gray-400 font-medium">{card.label}</span>
              </div>
              <motion.p 
                className="text-2xl font-bold text-white"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", bounce: 0.4, delay: index * 0.1 + 0.2 }}
              >
                {formatVolume(card.value)}
              </motion.p>
              {/* Glow effect */}
              <div className={`
                absolute -bottom-4 -right-4 w-24 h-24 rounded-full blur-2xl opacity-0 
                group-hover:opacity-50 transition-opacity bg-gradient-to-br ${card.gradient}
              `} />
            </motion.div>
          ))}
        </div>
      </div>

      {/* Bounding Box */}
      {regions.WT?.bounding_box && (
        <motion.div 
          className="p-5 rounded-2xl bg-gray-800/30 border border-gray-700/50 backdrop-blur-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-ne3na3-primary/20 rounded-xl border border-ne3na3-primary/30">
              <Maximize2 className="w-5 h-5 text-ne3na3-neon" />
            </div>
            <span className="text-sm font-semibold text-white">Tumor Extent (3D)</span>
          </div>
          <div className="grid grid-cols-3 gap-3 text-center">
            {['X', 'Y', 'Z'].map((axis, i) => (
              <motion.div 
                key={axis} 
                className="bg-gray-900/50 rounded-xl p-3 border border-gray-700/50"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.1 }}
              >
                <p className="text-xs text-gray-500 mb-1">{axis}-Axis</p>
                <p className="text-lg font-mono font-bold text-white">
                  {regions.WT.bounding_box.size_mm[i].toFixed(1)}
                  <span className="text-xs text-gray-400 ml-1">mm</span>
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Modality Importance */}
      <motion.div 
        className="p-5 rounded-2xl bg-gray-800/30 border border-gray-700/50 backdrop-blur-xl"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
          <Scan className="w-4 h-4 text-ne3na3-neon" />
          Modality Contribution
        </h3>
        <div className="space-y-3">
          {Object.entries(modality_importance)
            .sort((a, b) => b[1] - a[1])
            .map(([modality, percent], index) => (
              <motion.div 
                key={modality} 
                className="flex items-center gap-3"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <span className="text-sm text-gray-400 w-14 font-mono">{modality}</span>
                <div className="flex-1 h-2.5 bg-gray-900/50 rounded-full overflow-hidden">
                  <motion.div 
                    className="h-full bg-gradient-to-r from-ne3na3-primary to-ne3na3-neon rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${percent}%` }}
                    transition={{ duration: 0.7, delay: index * 0.1 }}
                  />
                </div>
                <span className="text-sm text-white font-mono font-bold w-12 text-right">
                  {percent}%
                </span>
              </motion.div>
            ))}
        </div>
      </motion.div>

      {/* Scan Info */}
      <motion.div 
        className="flex items-start gap-3 p-4 rounded-xl bg-gray-900/30 border border-gray-800/50"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <Clock className="w-4 h-4 text-gray-500 mt-0.5" />
        <div className="text-xs text-gray-500 space-y-1">
          <p>Scan: <span className="text-gray-400 font-mono">{insights.scan_dimensions.join(' × ')}</span> voxels</p>
          <p>Spacing: <span className="text-gray-400 font-mono">{insights.voxel_spacing_mm.map(v => v.toFixed(2)).join(' × ')}</span> mm</p>
          <p>Analyzed: <span className="text-gray-400">{new Date(insights.timestamp).toLocaleString()}</span></p>
        </div>
      </motion.div>
    </div>
  );
};

export default InsightsPanel;

/** */