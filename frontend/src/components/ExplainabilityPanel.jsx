import React from 'react';
import { motion } from 'framer-motion';
import { Eye, Layers, Info } from 'lucide-react';

const ExplainabilityPanel = ({ attentionData, modalityImportance }) => {
  // Sample visualization - in production, this would render actual attention heatmaps
  const layers = attentionData?.shapes || [];

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
          <Eye className="w-5 h-5 text-ne3na3-primary" />
          Attention Visualization
        </h3>
        <p className="text-sm text-gray-400 mb-4">
          Attention maps show where the model focused during segmentation
        </p>
      </div>

      {/* Attention Layers */}
      {layers.length > 0 ? (
        <div className="space-y-4">
          {layers.map((shape, index) => (
            <motion.div
              key={index}
              className="metric-card"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-ne3na3-primary" />
                  <span className="text-sm font-medium text-white">
                    Decoder Layer {index + 1}
                  </span>
                </div>
                <span className="text-xs text-gray-500 font-mono">
                  {shape.join(' × ')}
                </span>
              </div>
              
              {/* Placeholder visualization */}
              <div className="grid grid-cols-8 gap-1">
                {Array.from({ length: 32 }).map((_, i) => (
                  <motion.div
                    key={i}
                    className="aspect-square rounded-sm"
                    style={{
                      backgroundColor: `rgba(0, 166, 118, ${Math.random() * 0.8 + 0.2})`
                    }}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: i * 0.01 }}
                  />
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <Eye className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>Run segmentation to view attention maps</p>
        </div>
      )}

      {/* Modality Importance Breakdown */}
      {modalityImportance && Object.keys(modalityImportance).length > 0 && (
        <motion.div 
          className="metric-card"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h4 className="text-sm font-medium text-white mb-4 flex items-center gap-2">
            <Info className="w-4 h-4 text-ne3na3-primary" />
            Sequence Contribution Analysis
          </h4>
          
          <div className="space-y-4">
            {Object.entries(modalityImportance)
              .sort((a, b) => b[1] - a[1])
              .map(([modality, percent], index) => {
                const descriptions = {
                  'FLAIR': 'Primary for edema detection',
                  'T1ce': 'Primary for enhancing tumor',
                  'T2': 'Supports tissue differentiation',
                  'T1': 'Baseline anatomical reference',
                };
                
                const colors = {
                  'FLAIR': '#9F7AEA',
                  'T1ce': '#48BB78',
                  'T2': '#ED8936',
                  'T1': '#4299E1',
                };

                return (
                  <motion.div
                    key={modality}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div>
                        <span 
                          className="text-sm font-medium"
                          style={{ color: colors[modality] }}
                        >
                          {modality}
                        </span>
                        <span className="text-xs text-gray-500 ml-2">
                          {descriptions[modality]}
                        </span>
                      </div>
                      <span className="text-sm font-mono text-white">
                        {percent}%
                      </span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full rounded-full"
                        style={{ backgroundColor: colors[modality] }}
                        initial={{ width: 0 }}
                        animate={{ width: `${percent}%` }}
                        transition={{ duration: 0.5, delay: index * 0.1 }}
                      />
                    </div>
                  </motion.div>
                );
              })}
          </div>

          <div className="mt-4 p-3 bg-gray-800/50 rounded-xl text-xs text-gray-400">
            <p>
              <strong className="text-ne3na3-primary">FLAIR</strong> is typically most informative for 
              detecting peritumoral edema, while <strong className="text-ne3na3-primary">T1ce</strong> is 
              crucial for identifying enhancing tumor regions.
            </p>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default ExplainabilityPanel;
