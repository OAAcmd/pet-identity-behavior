// Build the on-device engine from bundle files (installed bundle or a test fixture).
import type { EngineAssets } from "./assets";
import { LocalEngine, type LocalModels } from "./engine";
import type { BundleManifest } from "./modelstore";
import { compileModel, initRuntime, runtimeVariant, type Accel } from "./runtime";

export const LITERT_WASM_BASE = "/litert/";

export async function buildEngine(m: BundleManifest, read: (path: string) => Promise<Uint8Array>,
  opts: { wasmBase?: string; accel?: Accel } = {}): Promise<LocalEngine> {
  const threads = typeof self !== "undefined" && (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
  await initRuntime(opts.wasmBase ?? LITERT_WASM_BASE, { threads });
  const assets = JSON.parse(new TextDecoder().decode(await read("engine_assets.json"))) as EngineAssets;
  const accel = opts.accel ?? "wasm";
  const models: LocalModels = {};
  if (assets.audio) models.yamnet = await compileModel("yamnet", await read(assets.audio.model_file), accel);
  if (assets.detector) models.detector = await compileModel("detector", await read(assets.detector.model_file), accel);
  if (assets.backbone) models.backbone = await compileModel("backbone", await read(assets.backbone.model_file), accel);
  const info = {
    bundle_version: m.bundle_version, runtime: `litertjs-${m.runtime.litertjs_core}-${runtimeVariant() ?? "unknown"}`, accelerator: accel,
    models: m.files.filter((f) => f.kind === "model").map((f) => ({ model_id: f.resource_id ?? f.path, sha256: f.sha256, file: f.path })),
  };
  return new LocalEngine(assets, models, info);
}
