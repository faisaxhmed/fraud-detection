import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Matches backend CORS allowlist (ALLOWED_ORIGINS=http://localhost:3000 in backend/.env)
  server: { port: 3000 },
})
