import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies to the FastAPI backend so the browser sees one
// origin. That matters for the microphone: getUserMedia only works on a secure
// context, and http://localhost counts as one — a LAN IP does not.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:8010', changeOrigin: true },
      '/ws': { target: 'ws://localhost:8010', ws: true },
    },
  },
})
