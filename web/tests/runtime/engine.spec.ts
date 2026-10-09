// Engine parity: the on-device TypeScript engine (LiteRT.js WASM, desktop
// headless Chromium) analyses the same media as the Python reference engine.
// Writes var/ondevice/engine_parity/web/*.json; scoring is done in Python
// (petperception.ondevice.engine_parity.compare). Not a phone result.
import { expect, test } from "@playwright/test";
import { build } from "esbuild";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../../..");
const parity = path.join(root, "var/ondevice/engine_parity");
const bundleDir = path.join(root, "var/ondevice/bundle");
const wasmDir = path.join(here, "../../node_modules/@litertjs/core/wasm");

test("on-device engine analyses the parity media set", async ({ page }) => {
  test.skip(!fs.existsSync(path.join(parity, "cases.json")) || !fs.existsSync(path.join(bundleDir, "manifest.json")), "run export-ondevice and the reference first");
  test.setTimeout(1_800_000);
  const r = await build({ entryPoints: [path.join(here, "engine_entry.ts")], bundle: true, format: "esm", write: false, platform: "browser", target: "es2022" });
  const bundle = r.outputFiles[0].text;
  const cases = JSON.parse(fs.readFileSync(path.join(parity, "cases.json"), "utf8")).cases as { name: string; media: string }[];
  const byName = new Map(cases.map((c) => [c.name, path.resolve(root, c.media)]));
  const isolate = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
  await page.route("https://engine.test/**", async (route) => {
    const u = new URL(route.request().url());
    const send = (body: Buffer | string, type: string) => route.fulfill({ status: 200, body, headers: { "content-type": type, ...isolate } });
    if (u.pathname === "/") return send(`<!doctype html><meta charset="utf-8"><script type="module">${bundle}</script>`, "text/html");
    if (u.pathname.startsWith("/wasm/")) { const f = path.join(wasmDir, path.basename(u.pathname)); return send(fs.readFileSync(f), f.endsWith(".wasm") ? "application/wasm" : "text/javascript"); }
    if (u.pathname.startsWith("/bundle/")) return send(fs.readFileSync(path.join(bundleDir, decodeURIComponent(u.pathname.slice("/bundle/".length)))), "application/octet-stream");
    if (u.pathname.startsWith("/media/")) {
      const f = byName.get(decodeURIComponent(u.pathname.slice("/media/".length)));
      return f ? send(fs.readFileSync(f), "application/octet-stream") : route.fulfill({ status: 404, body: "" });
    }
    return route.fulfill({ status: 404, body: "" });
  });
  await page.goto("https://engine.test/");
  await page.waitForFunction(() => "runEngine" in window);
  const only = process.env.ENGINE_ONLY ? new Set(process.env.ENGINE_ONLY.split(",")) : null;
  const todo = cases.filter((c) => !only || only.has(c.name));
  const res = await page.evaluate((cs) => (window as unknown as { runEngine: (x: unknown) => Promise<{ loadMs: number; results: Record<string, unknown> }> }).runEngine(cs), todo);
  fs.mkdirSync(path.join(parity, "web"), { recursive: true });
  for (const [name, v] of Object.entries(res.results)) fs.writeFileSync(path.join(parity, "web", `${name}.json`), JSON.stringify(v));
  fs.writeFileSync(path.join(parity, "web_meta.json"), JSON.stringify({ loadMs: res.loadMs, cases: todo.length }));
  // Inputs the reference engine rejects (e.g. a truncated file) must be rejected here too.
  const refError = (name: string) => {
    const f = path.join(parity, "reference", `${name}.json`);
    return fs.existsSync(f) && "error" in JSON.parse(fs.readFileSync(f, "utf8"));
  };
  const errors = Object.entries(res.results).filter(([k, v]) => Boolean((v as { error?: string }).error) !== refError(k));
  expect(errors.map(([k, v]) => `${k}: ${(v as { error: string }).error.slice(0, 300)}`)).toEqual([]);
});
