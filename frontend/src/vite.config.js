// This file is intentionally left without a proxy configuration.
// Vite loads its configuration from the project root:
//   frontend/vite.config.js   ← the active config (with /api proxy)
// This src/ copy is not read by Vite and is kept only for reference.

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
})
