/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Ne3Na3 Brand Colors
        'ne3na3': {
          primary: '#00A676',      // Healing Green
          secondary: '#E6F4F1',    // Mint Frost
          dark: '#004D40',         // Deep Green
          neon: '#00FFB3',         // Neon Mint (for hotspots)
          light: '#B2DFDB',        // Light Mint
        },
        // Extended palette
        'healing': {
          50: '#E6F4F1',
          100: '#B2DFDB',
          200: '#80CBC4',
          300: '#4DB6AC',
          400: '#26A69A',
          500: '#00A676',
          600: '#00897B',
          700: '#00796B',
          800: '#00695C',
          900: '#004D40',
        }
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      borderRadius: {
        'pill': '9999px',
        '4xl': '2rem',
      },
      backdropBlur: {
        'glass': '20px',
      },
      boxShadow: {
        'glass': '0 8px 32px 0 rgba(0, 166, 118, 0.15)',
        'glass-dark': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        'neon': '0 0 20px rgba(0, 255, 179, 0.5)',
        'neon-lg': '0 0 40px rgba(0, 255, 179, 0.6), 0 0 80px rgba(0, 166, 118, 0.3)',
        'neon-xl': '0 0 60px rgba(0, 255, 179, 0.7), 0 0 120px rgba(0, 166, 118, 0.4)',
      },
      animation: {
        'pulse-soft': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'float': 'float 6s ease-in-out infinite',
        'float-delayed': 'float 6s ease-in-out infinite 3s',
        'shimmer': 'shimmer 2s linear infinite',
        'gradient': 'gradient 8s linear infinite',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(0, 166, 118, 0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(0, 255, 179, 0.8)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0) rotate(0deg)' },
          '50%': { transform: 'translateY(-20px) rotate(5deg)' },
        },
        shimmer: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
        gradient: {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        }
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
      }
    },
  },
  plugins: [],
}
