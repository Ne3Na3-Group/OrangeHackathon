import React from 'react';
import { motion } from 'framer-motion';
import { 
  Brain, 
  Zap, 
  Shield, 
  Activity,
  Cpu,
  CheckCircle2,
  Sparkles
} from 'lucide-react';

const ProcessingOverlay = ({ isProcessing, phase, progress }) => {
  if (!isProcessing) return null;

  const steps = [
    { id: 'upload', label: 'Uploading Files', icon: Brain, color: 'from-emerald-500 to-teal-500' },
    { id: 'preprocess', label: 'Preprocessing MRI', icon: Cpu, color: 'from-teal-500 to-cyan-500' },
    { id: 'inference', label: 'Running AttUnet Inference', icon: Zap, color: 'from-cyan-500 to-green-500' },
    { id: 'tta', label: 'Test-Time Augmentation', icon: Activity, color: 'from-green-500 to-emerald-500' },
    { id: 'consistency', label: 'Anatomical Consistency', icon: Shield, color: 'from-emerald-500 to-teal-500' },
    { id: 'insights', label: 'Generating Insights', icon: CheckCircle2, color: 'from-teal-500 to-cyan-500' },
  ];

  const currentStepIndex = steps.findIndex(s => s.id === phase);

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      {/* Background */}
      <div className="absolute inset-0 bg-gray-950/95 backdrop-blur-2xl" />
      
      {/* Animated gradient mesh */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 -left-20 w-96 h-96 bg-ne3na3-primary/20 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-1/4 -right-20 w-96 h-96 bg-ne3na3-neon/15 rounded-full blur-3xl animate-float-delayed" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] 
                        bg-gradient-radial from-ne3na3-primary/10 via-transparent to-transparent rounded-full" />
      </div>

      <div className="relative max-w-lg w-full mx-4">
        {/* Brain Animation */}
        <motion.div 
          className="flex justify-center mb-10"
          animate={{ 
            scale: [1, 1.05, 1],
          }}
          transition={{ 
            duration: 3,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        >
          <div className="relative">
            {/* Outer glow rings */}
            <motion.div 
              className="absolute inset-0 -m-4 rounded-full border border-ne3na3-primary/30"
              animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0, 0.3] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <motion.div 
              className="absolute inset-0 -m-8 rounded-full border border-ne3na3-neon/20"
              animate={{ scale: [1, 1.3, 1], opacity: [0.2, 0, 0.2] }}
              transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
            />
            
            {/* Main brain container */}
            <div className="w-36 h-36 rounded-full bg-gradient-to-br from-ne3na3-primary/30 to-ne3na3-neon/20 
                            flex items-center justify-center border border-ne3na3-primary/30 shadow-neon">
              <div className="w-28 h-28 rounded-full bg-gray-900/80 flex items-center justify-center 
                              border border-ne3na3-primary/20">
                <Brain className="w-14 h-14 text-ne3na3-neon" />
              </div>
            </div>
            
            {/* Orbiting particles */}
            {[0, 1, 2, 3].map((i) => (
              <motion.div
                key={i}
                className="absolute w-2 h-2 bg-ne3na3-neon rounded-full shadow-neon"
                style={{
                  top: '50%',
                  left: '50%',
                  marginTop: -4,
                  marginLeft: -4,
                }}
                animate={{
                  x: [0, Math.cos(i * Math.PI / 2) * 80, 0],
                  y: [0, Math.sin(i * Math.PI / 2) * 80, 0],
                  scale: [1, 1.5, 1],
                  opacity: [0.5, 1, 0.5],
                }}
                transition={{
                  duration: 3,
                  delay: i * 0.4,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              />
            ))}
          </div>
        </motion.div>

        {/* Title */}
        <div className="text-center mb-10">
          <motion.h2 
            className="text-3xl font-bold text-white mb-3 flex items-center justify-center gap-3"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            Processing Your Scan
            <Sparkles className="w-6 h-6 text-ne3na3-neon animate-pulse" />
          </motion.h2>
          <motion.p 
            className="text-gray-400"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            Ne3Na3 is analyzing your brain MRI with AI
          </motion.p>
        </div>

        {/* Progress Steps */}
        <div className="space-y-3">
          {steps.map((step, index) => {
            const isActive = index === currentStepIndex;
            const isComplete = index < currentStepIndex;
            const isPending = index > currentStepIndex;

            return (
              <motion.div
                key={step.id}
                className={`
                  flex items-center gap-4 p-4 rounded-2xl transition-all duration-500 backdrop-blur-xl
                  ${isActive 
                    ? 'bg-gradient-to-r from-ne3na3-primary/20 to-ne3na3-dark/10 border border-ne3na3-primary/50 shadow-neon' 
                    : isComplete
                      ? 'bg-gray-800/30 border border-ne3na3-primary/20'
                      : 'bg-gray-900/30 border border-gray-800/50'
                  }
                `}
                initial={{ opacity: 0, x: -30 }}
                animate={{ 
                  opacity: isPending ? 0.5 : 1, 
                  x: 0,
                  scale: isActive ? 1.02 : 1
                }}
                transition={{ delay: index * 0.08 }}
              >
                <motion.div 
                  className={`
                    w-11 h-11 rounded-xl flex items-center justify-center transition-all duration-300
                    ${isActive 
                      ? `bg-gradient-to-br ${step.color} text-white shadow-lg` 
                      : isComplete
                        ? 'bg-ne3na3-primary/20 text-ne3na3-neon'
                        : 'bg-gray-800/50 text-gray-600'
                    }
                  `}
                  animate={isActive ? { scale: [1, 1.1, 1] } : {}}
                  transition={{ duration: 1, repeat: isActive ? Infinity : 0 }}
                >
                  {isComplete ? (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", bounce: 0.5 }}
                    >
                      <CheckCircle2 className="w-5 h-5" />
                    </motion.div>
                  ) : (
                    <step.icon className={`w-5 h-5 ${isActive ? 'animate-pulse' : ''}`} />
                  )}
                </motion.div>
                <div className="flex-1">
                  <p className={`
                    font-medium transition-colors
                    ${isActive ? 'text-white' : isComplete ? 'text-gray-400' : 'text-gray-600'}
                  `}>
                    {step.label}
                  </p>
                  {isActive && (
                    <motion.div 
                      className="mt-2.5 h-1.5 bg-gray-800/50 rounded-full overflow-hidden"
                      initial={{ opacity: 0, scaleX: 0 }}
                      animate={{ opacity: 1, scaleX: 1 }}
                    >
                      <motion.div 
                        className={`h-full bg-gradient-to-r ${step.color} rounded-full`}
                        animate={{ 
                          width: ['0%', '100%'],
                        }}
                        transition={{
                          duration: 2.5,
                          repeat: Infinity,
                          ease: "easeInOut"
                        }}
                      />
                    </motion.div>
                  )}
                </div>
                {isComplete && (
                  <motion.span 
                    className="text-xs text-ne3na3-primary font-medium"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    Done
                  </motion.span>
                )}
              </motion.div>
            );
          })}
        </div>

        {/* Footer */}
        <motion.div 
          className="text-center mt-10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <p className="text-gray-500 text-sm flex items-center justify-center gap-2">
            <Cpu className="w-4 h-4" />
            Using MONAI Sliding Window Inference with TTA
          </p>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default ProcessingOverlay;
