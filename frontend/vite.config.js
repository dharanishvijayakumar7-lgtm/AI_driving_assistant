import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy WebSocket + health requests to the FastAPI backend, for both
// `npm run dev` and `npm run preview` (the production build).
const proxy = {
  '/ws': {
    target: 'ws://localhost:8000',
    ws: true,
    changeOrigin: true,
  },
  '/health': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  },
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy,
  },
  preview: {
    port: 4173,
    proxy,
  },
})
