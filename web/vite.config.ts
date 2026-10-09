/// <reference types="vitest/config" />
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const here = path.dirname(fileURLToPath(import.meta.url));
const LITERT_WASM = path.join(here, "node_modules/@litertjs/core/wasm");

// Ships the LiteRT.js WebAssembly runtime at /litert/ (same origin, cached by
// the service worker) so on-device inference works offline.
function litertWasm(): Plugin {
  return {
    name: "litert-wasm",
    configureServer(server) {
      server.middlewares.use("/litert/", (req, res, next) => {
        const f = path.join(LITERT_WASM, path.basename((req.url ?? "").split("?")[0]));
        if (!fs.existsSync(f)) return next();
        res.setHeader("Content-Type", f.endsWith(".wasm") ? "application/wasm" : "text/javascript");
        fs.createReadStream(f).pipe(res);
      });
    },
    generateBundle() {
      for (const name of fs.readdirSync(LITERT_WASM)) {
        this.emitFile({ type: "asset", fileName: `litert/${name}`, source: fs.readFileSync(path.join(LITERT_WASM, name)) });
      }
    },
  };
}

export default defineConfig({
  esbuild: { jsx: "automatic", jsxImportSource: "preact" },
  plugins: [litertWasm()],
  build: {
    outDir: "dist",
    assetsDir: "assets",
    sourcemap: true,
    target: "es2022",
    modulePreload: { polyfill: false },
  },
  server: {
    // `npm run dev` proxies the API to a locally running `petperception serve`.
    proxy: { "/api": "http://127.0.0.1:8000", "/ondevice": "http://127.0.0.1:8000" },
    headers: { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" },
  },
  test: {
    environment: "happy-dom",
    include: ["tests/unit/**/*.test.ts"],
  },
});
