// Build settings for the website (Vite): React for the components, Tailwind for the styles.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In development the Python backend runs on :8000 (python -m demo); in production it serves dist/ itself.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // During `pnpm dev`, requests to /api are forwarded to the Python backend.
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
