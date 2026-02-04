import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  MessageCircle, 
  Send, 
  Trash2, 
  Shield, 
  Leaf,
  X,
  Maximize2,
  Minimize2,
  Bot,
  Sparkles
} from 'lucide-react';
import { sendChatMessage, clearChatHistory } from '../services/api';

const SafeBot = ({ isOpen, onToggle, insights }) => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: `🌿 Hello! I'm **Ne3Na3 Safe-Bot**, your brain tumor imaging assistant.

I can help you understand:
- 📊 **Tumor volumes** - Sizes and measurements
- 📍 **Tumor regions** - NCR, ED, ET explained
- ⚖️ **Asymmetry** - Left vs right analysis
- 🔬 **MRI modalities** - T1, T1ce, T2, FLAIR

What would you like to know?

💚 *I provide educational information only.*`,
      timestamp: new Date().toISOString()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await sendChatMessage(input.trim());
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.message,
        type: response.type,
        timestamp: response.timestamp
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `🌿 I apologize, but I encountered an error. Please try again.

*Error: ${error.message}*`,
        type: 'error',
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = async () => {
    try {
      await clearChatHistory();
      setMessages([{
        role: 'assistant',
        content: `🌿 Conversation cleared. How can I help you with your brain MRI analysis?`,
        timestamp: new Date().toISOString()
      }]);
    } catch (error) {
      console.error('Failed to clear history:', error);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Quick action buttons
  const quickActions = [
    { label: 'Volumes', query: 'What are the tumor volumes?' },
    { label: 'Regions', query: 'Explain the tumor regions' },
    { label: 'Asymmetry', query: 'Tell me about asymmetry' },
    { label: 'Modalities', query: 'Which MRI was most important?' },
  ];

  // Render markdown-like content
  const renderContent = (content) => {
    // Simple markdown parsing
    return content.split('\n').map((line, i) => {
      // Bold
      line = line.replace(/\*\*(.+?)\*\*/g, '<strong class="text-white">$1</strong>');
      // Italic
      line = line.replace(/\*(.+?)\*/g, '<em class="text-gray-400">$1</em>');
      // Emoji preserved
      
      if (line.startsWith('- ')) {
        return (
          <div key={i} className="flex gap-2 my-1">
            <span className="text-ne3na3-primary">•</span>
            <span dangerouslySetInnerHTML={{ __html: line.slice(2) }} />
          </div>
        );
      }
      
      return (
        <p key={i} className="my-1" dangerouslySetInnerHTML={{ __html: line || '&nbsp;' }} />
      );
    });
  };

  if (!isOpen) {
    return (
      <motion.button
        className="fixed bottom-6 right-6 w-16 h-16 rounded-2xl 
                   flex items-center justify-center z-50 group
                   bg-gradient-to-br from-ne3na3-primary to-ne3na3-dark
                   shadow-neon hover:shadow-neon-lg transition-all duration-300"
        onClick={onToggle}
        whileHover={{ scale: 1.1, rotate: 5 }}
        whileTap={{ scale: 0.95 }}
        initial={{ scale: 0, rotate: -180 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={{ type: "spring", bounce: 0.5 }}
      >
        {/* Ripple effect */}
        <motion.div 
          className="absolute inset-0 rounded-2xl bg-ne3na3-neon/30"
          animate={{ scale: [1, 1.5], opacity: [0.5, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
        <Bot className="w-7 h-7 text-white relative z-10" />
        {/* Notification dot */}
        <span className="absolute top-1 right-1 w-3 h-3 bg-ne3na3-neon rounded-full animate-pulse" />
      </motion.button>
    );
  }

  return (
    <motion.div
      className={`
        fixed z-50 rounded-3xl overflow-hidden
        flex flex-col backdrop-blur-2xl
        ${isExpanded 
          ? 'inset-4' 
          : 'bottom-6 right-6 w-[400px] h-[650px]'
        }
      `}
      style={{
        background: 'linear-gradient(135deg, rgba(17,24,39,0.95) 0%, rgba(0,77,64,0.15) 100%)',
        border: '1px solid rgba(0,166,118,0.2)',
        boxShadow: '0 25px 50px -12px rgba(0,0,0,0.5), 0 0 40px -10px rgba(0,166,118,0.3)'
      }}
      initial={{ opacity: 0, y: 100, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 100, scale: 0.9 }}
    >
      {/* Header */}
      <div className="relative p-5 flex items-center justify-between border-b border-white/5">
        <div className="absolute inset-0 bg-gradient-to-r from-ne3na3-primary/20 to-ne3na3-dark/10" />
        <div className="relative flex items-center gap-4">
          <motion.div 
            className="w-12 h-12 bg-gradient-to-br from-ne3na3-primary to-ne3na3-dark rounded-2xl 
                       flex items-center justify-center shadow-lg border border-ne3na3-primary/30"
            animate={{ rotate: [0, 5, -5, 0] }}
            transition={{ duration: 4, repeat: Infinity }}
          >
            <Leaf className="w-6 h-6 text-ne3na3-neon" />
          </motion.div>
          <div>
            <h3 className="font-bold text-white flex items-center gap-2">
              Ne3Na3 Safe-Bot
              <Sparkles className="w-4 h-4 text-ne3na3-neon animate-pulse" />
            </h3>
            <div className="flex items-center gap-1.5 text-xs text-gray-400">
              <Shield className="w-3 h-3 text-ne3na3-primary" />
              <span>Safety-First Medical Assistant</span>
            </div>
          </div>
        </div>
        <div className="relative flex items-center gap-1">
          <motion.button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-2.5 hover:bg-white/10 rounded-xl transition-colors text-gray-400 hover:text-white"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
          >
            {isExpanded ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
          </motion.button>
          <motion.button
            onClick={onToggle}
            className="p-2.5 hover:bg-red-500/20 rounded-xl transition-colors text-gray-400 hover:text-red-400"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
          >
            <X className="w-4 h-4" />
          </motion.button>
        </div>
      </div>

      {/* Analysis Status */}
      {insights && (
        <motion.div 
          className="px-5 py-3 bg-gradient-to-r from-ne3na3-primary/10 to-transparent 
                     border-b border-ne3na3-primary/10 text-xs flex items-center gap-2"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
        >
          <span className="w-2 h-2 bg-ne3na3-neon rounded-full animate-pulse" />
          <span className="text-ne3na3-neon font-medium">Analysis loaded</span>
          <span className="text-gray-500 ml-2 font-mono">
            WT: {insights.summary?.total_tumor_volume_cm3?.toFixed(2)} cm³
          </span>
        </motion.div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        <AnimatePresence>
          {messages.map((message, index) => (
            <motion.div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.3 }}
            >
              <div
                className={`
                  max-w-[85%] text-sm rounded-2xl p-4
                  ${message.role === 'user' 
                    ? 'bg-gradient-to-br from-ne3na3-primary to-ne3na3-dark text-white rounded-br-md' 
                    : 'bg-gray-800/50 text-gray-300 border border-gray-700/50 rounded-bl-md backdrop-blur-xl'
                  }
                `}
              >
                {renderContent(message.content)}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        
        {isLoading && (
          <motion.div 
            className="flex justify-start"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <div className="bg-gray-800/50 border border-gray-700/50 rounded-2xl rounded-bl-md p-4 backdrop-blur-xl">
              <div className="flex items-center gap-2">
                <motion.span 
                  className="w-2 h-2 bg-ne3na3-neon rounded-full"
                  animate={{ scale: [1, 1.3, 1] }}
                  transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
                />
                <motion.span 
                  className="w-2 h-2 bg-ne3na3-neon rounded-full"
                  animate={{ scale: [1, 1.3, 1] }}
                  transition={{ duration: 0.6, repeat: Infinity, delay: 0.15 }}
                />
                <motion.span 
                  className="w-2 h-2 bg-ne3na3-neon rounded-full"
                  animate={{ scale: [1, 1.3, 1] }}
                  transition={{ duration: 0.6, repeat: Infinity, delay: 0.3 }}
                />
              </div>
            </div>
          </motion.div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Quick Actions */}
      <div className="px-5 py-3 border-t border-gray-800/50 flex gap-2 overflow-x-auto">
        {quickActions.map((action, index) => (
          <motion.button
            key={action.label}
            onClick={() => {
              setInput(action.query);
              handleSend();
            }}
            className="px-4 py-2 bg-gray-800/50 hover:bg-ne3na3-primary/20 
                       border border-gray-700/50 hover:border-ne3na3-primary/30
                       text-gray-400 hover:text-ne3na3-neon
                       text-xs rounded-xl whitespace-nowrap transition-all duration-300"
            disabled={isLoading}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            {action.label}
          </motion.button>
        ))}
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-800/50 bg-gray-900/50 backdrop-blur-xl">
        <div className="flex gap-3">
          <motion.button
            onClick={handleClear}
            className="p-3 text-gray-500 hover:text-red-400 hover:bg-red-400/10 
                       rounded-xl transition-all border border-transparent hover:border-red-500/20"
            title="Clear history"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
          >
            <Trash2 className="w-5 h-5" />
          </motion.button>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about your results..."
            className="flex-1 px-4 py-3 bg-gray-800/50 border border-gray-700/50 rounded-xl
                       text-white placeholder-gray-500 text-sm
                       focus:outline-none focus:border-ne3na3-primary/50 focus:ring-1 focus:ring-ne3na3-primary/20
                       transition-all duration-300"
            disabled={isLoading}
          />
          <motion.button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className={`
              p-3 rounded-xl transition-all duration-300
              ${input.trim() && !isLoading
                ? 'bg-gradient-to-br from-ne3na3-primary to-ne3na3-dark text-white shadow-neon'
                : 'bg-gray-800/50 text-gray-500 cursor-not-allowed border border-gray-700/50'
              }
            `}
            whileHover={input.trim() && !isLoading ? { scale: 1.1 } : {}}
            whileTap={input.trim() && !isLoading ? { scale: 0.95 } : {}}
          >
            <Send className="w-5 h-5" />
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};

export default SafeBot;
