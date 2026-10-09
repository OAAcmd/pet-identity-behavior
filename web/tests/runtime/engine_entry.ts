// Browser entry for the engine parity test: runs the on-device engine on each case.
import type { BundleManifest } from "../../src/engine/modelstore";
import { buildEngine } from "../../src/engine/local";

async function bytes(url: string): Promise<Uint8Array> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

(window as unknown as { runEngine: unknown }).runEngine = async (cases: { name: string; media: string; zones: unknown; modalities: ("audio" | "video")[] }[]) => {
  const manifest = JSON.parse(new TextDecoder().decode(await bytes("/bundle/manifest.json"))) as BundleManifest;
  const t0 = performance.now();
  const engine = await buildEngine(manifest, (p) => bytes(`/bundle/${p}`), { wasmBase: "/wasm/" });
  const loadMs = performance.now() - t0;
  const out: Record<string, unknown> = {};
  for (const c of cases) {
    try {
      const blob = new Blob([await bytes(`/media/${encodeURIComponent(c.name)}`)], { type: c.media.endsWith(".wav") ? "audio/wav" : c.media.endsWith(".jpg") ? "image/jpeg" : "video/mp4" });
      const t = performance.now();
      const r = await engine.analyze(blob, { name: c.media.split("/").pop()!, zones: c.zones as never, modalities: c.modalities, analysisId: c.name });
      out[c.name] = { ...r, wall_ms: performance.now() - t };
    } catch (e) {
      out[c.name] = { error: String(e instanceof Error ? e.stack ?? e.message : e) };
    }
  }
  return { loadMs, crossOriginIsolated: self.crossOriginIsolated, results: out };
};
