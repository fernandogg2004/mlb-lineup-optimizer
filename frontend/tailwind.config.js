/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        green: {
          DEFAULT: '#00FF87',
          400: '#00FF87',
          500: '#00E87A',
          950: '#001a0d',
        },
        red: {
          DEFAULT: '#FF4560',
          400: '#FF4560',
          500: '#FF4560',
          950: '#1a000a',
        },
        amber: {
          DEFAULT: '#FFB800',
          400: '#FFB800',
          500: '#E6A600',
          950: '#1a1100',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'live': 'pulse-live 2s ease-in-out infinite',
        'flash': 'flash-green 400ms ease-out forwards',
        'card': 'fade-in-up 0.3s ease-out both',
      },
    },
  },
  plugins: [],
};
