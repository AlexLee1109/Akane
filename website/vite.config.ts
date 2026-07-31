import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    base: env.VITE_BASE_PATH || "/",
    plugins: [react()],
    server: {
      proxy: {
        "/api": {
          target: env.AKANE_BACKEND_URL || "http://raspberrypi.local:8000",
          changeOrigin: true,
        },
      },
    },
  };
});
