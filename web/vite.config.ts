/// <reference types="vitest/config" />
import { defineConfig } from "vite";

export default defineConfig({
  esbuild: { jsx: "automatic", jsxImportSource: "preact" },
  build: {
    outDir: "dist",
    assetsDir: "assets",
    sourcemap: true,
    target: "es2020",
    modulePreload: { polyfill: false },
  },
  server: {
    // `npm run dev` proxies the API to a locally running `petperception serve`.
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  test: {
    environment: "happy-dom",
    include: ["tests/unit/**/*.test.ts"],
  },
});
