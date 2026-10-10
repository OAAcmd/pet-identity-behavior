// Builds the phone test page script (src/devicetest) as one ES module, dist-devicetest/app.js.
// `petperception export-device-test` assembles it with the page, the LiteRT.js runtime and the
// model bundle into a static HTTPS-hostable site (docs/DEVICE_TEST_PROTOCOL.md).
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const here = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  esbuild: { jsx: "automatic", jsxImportSource: "preact" },
  publicDir: false,
  build: {
    outDir: "dist-devicetest",
    emptyOutDir: true,
    target: "es2022",
    sourcemap: false,
    lib: { entry: path.join(here, "src/devicetest/main.tsx"), formats: ["es"], fileName: () => "app.js" },
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
});
