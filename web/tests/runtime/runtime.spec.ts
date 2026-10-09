// Browser runtime parity: the same .tflite files run in headless Chromium via
// LiteRT.js (WASM / XNNPACK) and are compared with Python LiteRT outputs on
// identical deterministic inputs. Desktop Chromium only: NOT a phone result.
import { expect, test } from "@playwright/test";
import { build } from "esbuild";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../../..");
const parityDir = path.join(root, "var/ondevice/parity");
const wasmDir = path.join(here, "../../node_modules/@litertjs/core/wasm");
const modelDirs = [path.join(root, "var/ondevice/models"), ...fs.readdirSync(path.join(root, "var/models")).map((d) => path.join(root, "var/models", d))];
const outDir = path.join(root, "results/ondevice");

let bundle = "";
test.beforeAll(async () => {
  const r = await build({ entryPoints: [path.join(here, "entry.ts")], bundle: true, format: "esm", write: false, platform: "browser", target: "es2022" });
  bundle = r.outputFiles[0].text;
});

function findModel(name: string): string | null {
  for (const d of modelDirs) {
    const p = path.join(d, name);
    if (fs.existsSync(p)) return p;
  }
  return null;
}

const ONLY = process.env.PARITY_ONLY ? process.env.PARITY_ONLY.split(",") : undefined;
const CONFIGS = [{ accel: "wasm", threads: false }, { accel: "wasm", threads: true }] as const;

for (const cfg of CONFIGS) {
  test(`LiteRT.js ${cfg.accel} threads=${cfg.threads}: parity with Python LiteRT`, async ({ page }) => {
    test.skip(!fs.existsSync(path.join(parityDir, "manifest.json")), "run: petperception export-parity");
    const isolate = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
    await page.route("https://parity.test/**", async (route) => {
      const u = new URL(route.request().url());
      const send = (body: Buffer | string, type: string) => route.fulfill({ status: 200, body, headers: { "content-type": type, ...isolate } });
      if (u.pathname === "/") return send(`<!doctype html><meta charset="utf-8"><script type="module">${bundle}</script>`, "text/html");
      if (u.pathname.startsWith("/wasm/")) {
        const f = path.join(wasmDir, path.basename(u.pathname));
        return send(fs.readFileSync(f), f.endsWith(".wasm") ? "application/wasm" : "text/javascript");
      }
      if (u.pathname.startsWith("/parity/")) return send(fs.readFileSync(path.join(parityDir, path.basename(u.pathname))), "application/octet-stream");
      if (u.pathname.startsWith("/models/")) {
        const f = findModel(path.basename(u.pathname));
        return f ? send(fs.readFileSync(f), "application/octet-stream") : route.fulfill({ status: 404, body: "missing" });
      }
      return route.fulfill({ status: 404, body: "not found" });
    });
    await page.goto("https://parity.test/");
    await page.waitForFunction(() => "runParity" in window);
    const report = await page.evaluate((o) => (window as unknown as { runParity: (x: unknown) => Promise<unknown> }).runParity(o), { ...cfg, repeats: 5, only: ONLY });
    fs.mkdirSync(outDir, { recursive: true });
    const tag = ONLY ? `_${ONLY.join("+")}` : "";
    fs.writeFileSync(path.join(outDir, `runtime_parity_chromium_${cfg.accel}_${cfg.threads ? "mt" : "st"}${tag}.json`), JSON.stringify(report, null, 1));
    const cases = (report as { cases: Record<string, { error?: string; inputBitExact?: boolean; comparisons?: { maxAbsDiff?: number; maxAbsRef?: number; matchedOutput: number }[] }> }).cases;
    for (const [name, c] of Object.entries(cases)) {
      expect(c.error, `${name} failed`).toBeUndefined();
      expect(c.inputBitExact, `${name} input generator`).toBe(true);
      for (const cmp of c.comparisons ?? []) {
        expect(cmp.matchedOutput, `${name} output matched`).toBeGreaterThanOrEqual(0);
      }
    }
  });
}
