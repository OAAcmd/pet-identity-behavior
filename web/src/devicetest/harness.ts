// Phone test harness: the HTTPS test build for real-device acceptance
// (docs/DEVICE_TEST_PROTOCOL.md). It runs the same TypeScript engine, LiteRT.js
// runtime and model files as the app, from a static site (a claude.ai Artifact
// page or any HTTPS host). Recordings are analysed on the phone and never sent;
// only the JSON report leaves the device, and only when the tester sends it.
import { isWebGPUSupported } from "@litertjs/core";
import { evaluateBudgets, guessTier, TIERS, type Tier } from "../engine/budgets";
import type { LocalEngine } from "../engine/engine";
import { ENGINE_VERSION } from "../engine/engine";
import { buildEngine } from "../engine/local";
import type { BundleFile, BundleManifest } from "../engine/modelstore";
import { normalize, syntheticAudio, syntheticImage } from "../engine/parity";
import { runModel, runtimeVariant, type LoadedModel } from "../engine/runtime";

export const HARNESS_VERSION = "phone-test-1";
export const REPORT_SCHEMA = "petperception.device_check/3";
const CACHE = "pp-phone-test-v1";

export interface Part { path: string; sha256: string; bytes: number; encoding?: "base64" }
/** Written by `petperception export-device-test`: the app bundle manifest plus how large files were split for hosting. */
export interface SiteManifest { schema: string; harness_version: string; created_at: string; bundle: BundleManifest; parts: Record<string, Part[]> }

export async function sha256Hex(buf: ArrayBuffer | Uint8Array): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", buf as BufferSource);
  return Array.from(new Uint8Array(d)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Decodes Base64 text (the host serves model files as text). */
export function fromBase64(text: string): Uint8Array {
  const native = (Uint8Array as unknown as { fromBase64?: (s: string) => Uint8Array }).fromBase64;
  const clean = text.replace(/\s+/g, "");
  if (native) return native(clean);
  const bin = atob(clean);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Bytes actually transferred for the given URLs (Resource Timing; 0 where the browser does not report it). */
export function transferBytes(match: (url: string) => boolean): number {
  return performance.getEntriesByType("resource").filter((e) => match(e.name))
    .reduce((n, e) => n + ((e as PerformanceResourceTiming).transferSize || 0), 0);
}

export interface DownloadStats { files: number; total_bytes: number; network_bytes: number; from_cache: number; ms: number; cache_storage: boolean }

/** Verified file store: memory, then Cache Storage (re-verified), then the network (parts reassembled, SHA-256 checked). */
export class FileStore {
  private files = new Map<string, Uint8Array>();
  cacheStorage: boolean | null = null;
  constructor(readonly site: SiteManifest, readonly base: URL) {}

  private entry(path: string): BundleFile {
    const f = this.site.bundle.files.find((x) => x.path === path);
    if (!f) throw new Error(`${path} is not part of bundle ${this.site.bundle.bundle_version}`);
    return f;
  }

  private async cache(): Promise<Cache | null> {
    try {
      if (typeof caches === "undefined") { this.cacheStorage = false; return null; }
      const c = await caches.open(CACHE);
      this.cacheStorage = true;
      return c;
    } catch { this.cacheStorage = false; return null; }
  }

  private cacheKey(f: BundleFile): string { return new URL(`__phone_test_cache__/${f.sha256}`, this.base).href; }

  private async fetchBytes(path: string, sha: string, onBytes?: (n: number) => void, encoding?: "base64"): Promise<Uint8Array> {
    const r = await fetch(new URL(path, this.base).href, { cache: "no-store" });
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    const buf = encoding === "base64" ? fromBase64(await r.text()) : new Uint8Array(await r.arrayBuffer());
    onBytes?.(buf.byteLength);
    const got = await sha256Hex(buf);
    if (got !== sha) throw new Error(`${path}: checksum mismatch (expected ${sha.slice(0, 12)}…, got ${got.slice(0, 12)}…)`);
    return buf;
  }

  /** Returns [bytes, source]. */
  async load(path: string, onBytes?: (n: number) => void): Promise<[Uint8Array, "memory" | "cache" | "network"]> {
    const hit = this.files.get(path);
    if (hit) return [hit, "memory"];
    const f = this.entry(path);
    const c = await this.cache();
    if (c) {
      try {
        const r = await c.match(this.cacheKey(f));
        if (r) {
          const buf = new Uint8Array(await r.arrayBuffer());
          if ((await sha256Hex(buf)) === f.sha256) { this.files.set(path, buf); return [buf, "cache"]; }
          await c.delete(this.cacheKey(f));
        }
      } catch { /* fall through to the network */ }
    }
    let buf: Uint8Array;
    const parts = this.site.parts[path];
    if (parts) {
      const chunks: Uint8Array[] = [];
      for (const p of parts) chunks.push(await this.fetchBytes(p.path, p.sha256, onBytes, p.encoding));
      buf = new Uint8Array(chunks.reduce((n, x) => n + x.byteLength, 0));
      let o = 0;
      for (const x of chunks) { buf.set(x, o); o += x.byteLength; }
      const got = await sha256Hex(buf);
      if (got !== f.sha256) throw new Error(`${path}: reassembled file fails its checksum`);
    } else {
      buf = await this.fetchBytes(`bundle/${path}`, f.sha256, onBytes);
    }
    this.files.set(path, buf);
    if (c) { try { await c.put(this.cacheKey(f), new Response(buf as BodyInit)); } catch { /* storage full or refused: keep in memory */ } }
    return [buf, "network"];
  }

  read = async (path: string): Promise<Uint8Array> => (await this.load(path))[0];

  async loadAll(onProgress?: (done: number, total: number, file: string) => void): Promise<DownloadStats> {
    const t0 = performance.now();
    const total = this.site.bundle.total_bytes;
    let done = 0, net = 0, fromCache = 0;
    for (const f of this.site.bundle.files) {
      const [, src] = await this.load(f.path, (n) => { net += n; });
      if (src === "cache") fromCache++;
      done += f.bytes;
      onProgress?.(done, total, f.path);
    }
    return { files: this.site.bundle.files.length, total_bytes: total, network_bytes: net, from_cache: fromCache, ms: performance.now() - t0, cache_storage: this.cacheStorage === true };
  }

  async cachedCount(): Promise<number> {
    const c = await this.cache();
    if (!c) return 0;
    let n = 0;
    for (const f of this.site.bundle.files) { try { if (await c.match(this.cacheKey(f))) n++; } catch { /* ignore */ } }
    return n;
  }

  async clearCache(): Promise<void> {
    this.files.clear();
    try { if (typeof caches !== "undefined") await caches.delete(CACHE); } catch { /* ignore */ }
  }
}

export async function startEngine(store: FileStore): Promise<{ engine: LocalEngine; ms: number }> {
  const t0 = performance.now();
  const engine = await buildEngine(store.site.bundle, store.read, { wasmBase: new URL("litert/", store.base).href });
  return { engine, ms: performance.now() - t0 };
}

// ---------------------------------------------------------------- environment

export function environment(): Record<string, unknown> {
  const nav = navigator as Navigator & { deviceMemory?: number; userAgentData?: { platform?: string; mobile?: boolean; brands?: { brand: string; version: string }[] };
    connection?: { effectiveType?: string; saveData?: boolean } };
  const v = document.createElement("video");
  const rec = typeof MediaRecorder !== "undefined" ? ["video/webm;codecs=vp9,opus", "video/webm", "video/mp4", "audio/webm;codecs=opus", "audio/mp4"]
    .filter((t) => { try { return MediaRecorder.isTypeSupported(t); } catch { return false; } }) : [];
  return {
    user_agent: navigator.userAgent, platform: nav.userAgentData?.platform ?? null, mobile: nav.userAgentData?.mobile ?? null,
    brands: nav.userAgentData?.brands?.map((b) => `${b.brand} ${b.version}`) ?? null, ...parseUserAgent(navigator.userAgent),
    hardware_concurrency: navigator.hardwareConcurrency ?? null, device_memory_gb: nav.deviceMemory ?? null,
    cross_origin_isolated: (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated === true,
    webassembly: typeof WebAssembly !== "undefined", webgpu: safe(() => isWebGPUSupported(), false),
    in_frame: safe(() => window.top !== window.self, true), online: navigator.onLine, connection: nav.connection?.effectiveType ?? null,
    media_recorder_types: rec, can_play: { webm_vp9: v.canPlayType('video/webm; codecs="vp9"'), mp4_h264: v.canPlayType('video/mp4; codecs="avc1.42E01E"'),
      mp4_hevc: v.canPlayType('video/mp4; codecs="hvc1"'), mov: v.canPlayType("video/quicktime") },
    screen: `${screen.width}x${screen.height}@${devicePixelRatio}`, language: navigator.language,
  };
}

function safe<T>(f: () => T, fallback: T): T { try { return f(); } catch { return fallback; } }

/** Best-effort OS and browser from the user agent; the tester confirms both in the form. */
export function parseUserAgent(ua: string): { os_guess: string | null; browser_guess: string | null; in_app_webview: boolean } {
  let os: string | null = null;
  const ios = ua.match(/(?:iPhone|iPad|iPod).*? OS (\d+[_\d]*)/);
  const android = ua.match(/Android (\d+(?:\.\d+)*)/);
  if (ios) os = `iOS ${ios[1].replace(/_/g, ".")}`;
  else if (android) os = `Android ${android[1]}`;
  else if (/Mac OS X/.test(ua)) os = "macOS";
  else if (/Windows/.test(ua)) os = "Windows";
  else if (/Linux/.test(ua)) os = "Linux";
  const m = (re: RegExp) => ua.match(re)?.[1];
  let browser: string | null = null;
  if (m(/EdgA?\/([\d.]+)/)) browser = `Edge ${m(/EdgA?\/([\d.]+)/)}`;
  else if (m(/SamsungBrowser\/([\d.]+)/)) browser = `Samsung Internet ${m(/SamsungBrowser\/([\d.]+)/)}`;
  else if (m(/Firefox\/([\d.]+)/) || m(/FxiOS\/([\d.]+)/)) browser = `Firefox ${m(/Firefox\/([\d.]+)/) ?? m(/FxiOS\/([\d.]+)/)}`;
  else if (m(/CriOS\/([\d.]+)/)) browser = `Chrome (iOS) ${m(/CriOS\/([\d.]+)/)}`;
  else if (m(/Chrome\/([\d.]+)/)) browser = `Chrome ${m(/Chrome\/([\d.]+)/)}`;
  else if (ios && m(/Version\/([\d.]+)/)) browser = `Safari ${m(/Version\/([\d.]+)/)}`;
  else if (m(/Version\/([\d.]+).*Safari/)) browser = `Safari ${m(/Version\/([\d.]+)/)}`;
  // Embedded web views (the Claude app, mail and chat apps) are not the browser under test.
  const webview = /; wv\)/.test(ua) || (ios !== null && !/Safari\//.test(ua)) || /Claude|FBAN|FBAV|Instagram|Line\//.test(ua);
  return { os_guess: os, browser_guess: browser, in_app_webview: webview };
}

// ---------------------------------------------------------------- observation helpers

export interface PageEvent { t_ms: number; type: string; detail?: string }
export const pageEvents: PageEvent[] = [];
const log = (type: string, detail?: string) => pageEvents.push({ t_ms: Math.round(performance.now()), type, ...(detail ? { detail } : {}) });
if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => log(`visibility:${document.visibilityState}`));
  for (const ev of ["pagehide", "pageshow", "freeze", "resume", "online", "offline"]) window.addEventListener(ev, () => log(ev));
}

/** Records every network resource request the page makes while active (PerformanceObserver). */
export function watchNetwork(): { stop: () => { supported: boolean; requests: number; examples: string[] } } {
  const seen: string[] = [];
  let obs: PerformanceObserver | null = null;
  try {
    obs = new PerformanceObserver((l) => { for (const e of l.getEntries()) if (!/^(blob|data):/.test(e.name)) seen.push(e.name); });
    obs.observe({ type: "resource", buffered: false });
  } catch { obs = null; }
  return { stop: () => { obs?.disconnect(); return { supported: obs !== null, requests: seen.length, examples: seen.slice(0, 5) }; } };
}

export function jsHeap(): { used: number; limit: number } | null {
  const mem = (performance as unknown as { memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
  return mem ? { used: mem.usedJSHeapSize, limit: mem.jsHeapSizeLimit } : null;
}

interface BatteryLike { level: number; charging: boolean }
export async function battery(): Promise<BatteryLike | null> {
  const nav = navigator as Navigator & { getBattery?: () => Promise<BatteryLike> };
  try { const b = await nav.getBattery?.(); return b ? { level: b.level, charging: b.charging } : null; } catch { return null; }
}

const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : NaN; };

// ---------------------------------------------------------------- checks

interface RefCase { input: string; first: number[]; mean_abs: number; argmax: number; elements: number }
export interface Expected { file: string; mime: string; tracks: number; species: string[]; segment_kinds: string[]; decision: string; sound_labels: string[] }

/** Deterministic inputs, identical to src/petperception/ondevice/parity.py CASES. */
function parityInput(c: string): Float32Array | Uint8Array {
  if (c === "yamnet") return syntheticAudio();
  if (c === "detector_float") return normalize(syntheticImage(320), 127.5, 127.5);
  if (c === "detector_int8") return syntheticImage(320);
  if (c === "backbone") return normalize(syntheticImage(224), 127, 128);
  if (c === "backbone_int8") return syntheticImage(224);
  throw new Error(`unknown reference input ${c}`);
}

export async function modelCheck(engine: LocalEngine, store: FileStore): Promise<Record<string, unknown>> {
  const refs = JSON.parse(new TextDecoder().decode(await store.read("device_check/reference_outputs.json"))) as Record<string, RefCase>;
  const models: Record<string, LoadedModel | undefined> = { yamnet: engine.m.yamnet, detector_float: engine.m.detector, backbone: engine.m.backbone };
  const out: Record<string, unknown> = {};
  for (const [name, mdl] of Object.entries(models)) {
    if (!mdl || !refs[name]) continue;
    const input = parityInput(refs[name].input);
    const times: number[] = [];
    let outs: ArrayLike<number>[] = [];
    for (let k = 0; k < 6; k++) {
      const t = performance.now();
      outs = (await runModel(mdl, [input])) as ArrayLike<number>[];
      if (k > 0) times.push(performance.now() - t);
    }
    const ref = refs[name];
    const o = outs.find((x) => x.length === ref.elements);
    if (!o) { out[name] = { error: `no output with ${ref.elements} elements` }; continue; }
    let maxDiff = 0, meanAbs = 0, arg = 0;
    for (let i = 0; i < ref.first.length; i++) maxDiff = Math.max(maxDiff, Math.abs(o[i] - ref.first[i]));
    for (let i = 0; i < o.length; i++) { meanAbs += Math.abs(o[i]); if (o[i] > o[arg]) arg = i; }
    meanAbs /= o.length;
    out[name] = { median_ms: median(times), min_ms: Math.min(...times), max_abs_diff_first_values: maxDiff,
      mean_abs_relative_diff: Math.abs(meanAbs - ref.mean_abs) / Math.max(ref.mean_abs, 1e-12), argmax_equal: arg === ref.argmax };
  }
  return out;
}

export async function referenceClip(store: FileStore): Promise<{ clip: Blob; expected: Expected }> {
  const expected = JSON.parse(new TextDecoder().decode(await store.read("device_check/expected.json"))) as Expected[];
  const v = document.createElement("video");
  const pick = expected.find((e) => v.canPlayType(e.mime) !== "") ?? expected[0];
  return { clip: new Blob([await store.read(pick.file) as BlobPart], { type: pick.mime.split(";")[0] }), expected: pick };
}

type Result = { observations: { animal_tracks: { species: string; behavior_segments: { kind: string; related_object?: string | null }[]; posture?: unknown[] }[];
  sound_events: { label: string }[] }; decision: { state: string }; timings_ms: Record<string, number>; media: { duration_s: number | null; container?: string } ;
  quality_flags?: { code: string }[]; hypotheses?: { name: string; status: string }[] };

export function summarize(r: Result) {
  const tracks = r.observations.animal_tracks;
  return { tracks: tracks.length, species: tracks.map((x) => x.species),
    segment_kinds: [...new Set(tracks.flatMap((x) => x.behavior_segments.map((b) => b.kind)))].sort(),
    decision: r.decision.state, sound_labels: r.observations.sound_events.map((x) => x.label),
    quality_flags: (r.quality_flags ?? []).map((q) => q.code), media_duration_s: r.media.duration_s, timings_ms: r.timings_ms };
}

export async function analyzeOnce(engine: LocalEngine, file: Blob, name: string, id: string,
  progress?: (stage: string, frac: number) => void): Promise<{ wall_ms: number; summary: ReturnType<typeof summarize>; result: Result }> {
  const t = performance.now();
  const r = (await engine.analyze(file, { name, zones: null, modalities: ["audio", "video"], analysisId: id, progress })) as unknown as Result;
  return { wall_ms: performance.now() - t, summary: summarize(r), result: r };
}

export function matchesExpected(s: ReturnType<typeof summarize>, e: Expected): boolean {
  return s.tracks === e.tracks && s.decision === e.decision && JSON.stringify(s.segment_kinds) === JSON.stringify([...e.segment_kinds].sort());
}

export function runStats(walls: number[], durationS: number | null) {
  return { n_runs: walls.length, wall_ms_median: median(walls), wall_ms_max: Math.max(...walls),
    realtime_factor: durationS ? median(walls) / (1000 * durationS) : null,
    sustained_slowdown: walls.length >= 10 ? median(walls.slice(-3)) / Math.max(median(walls.slice(0, 3)), 1) : null };
}

// ---------------------------------------------------------------- report

export interface Tester { device_model: string; os: string; browser: string; tier: Tier; notes: string }

export function budgets(tier: Tier, rep: Record<string, unknown>) {
  const dl = rep.download as DownloadStats | undefined;
  const eng = rep.engine as { load_ms?: number; reload_from_cache_ms?: number } | undefined;
  const sus = rep.sustained as { realtime_factor?: number | null; sustained_slowdown?: number | null; battery_pct_per_media_minute?: number | null } | undefined;
  const ref = rep.reference as { realtime_factor?: number | null } | undefined;
  const mdl = rep.models as Record<string, { median_ms?: number }> | undefined;
  const heap = rep.js_heap_peak as { used: number } | null | undefined;
  return { tier, tier_description: TIERS[tier], note: "Limits are assumptions until measured on physical phones of each tier (docs/DEVICE_TIERS_AND_BUDGETS.md).",
    results: evaluateBudgets(tier, {
      download_mb: dl ? (dl.total_bytes + (rep.runtime_bytes as number ?? 0)) / 2 ** 20 : null,
      storage_mb: (rep.storage as { usage?: number } | undefined)?.usage != null ? (rep.storage as { usage: number }).usage / 2 ** 20 : null,
      engine_load_ms: eng?.reload_from_cache_ms ?? null,
      realtime_factor: sus?.realtime_factor ?? ref?.realtime_factor ?? null,
      detector_ms: mdl?.detector_float?.median_ms ?? null,
      // performance.memory is the JS heap only (Chromium); WebAssembly memory is not included, so this is a lower bound.
      memory_mb: heap ? heap.used / 2 ** 20 : null,
      sustained_slowdown: sus?.sustained_slowdown ?? null, battery_pct_per_media_min: sus?.battery_pct_per_media_minute ?? null,
    }) };
}

export function newReport(site: SiteManifest, tester: Tester): Record<string, unknown> {
  return {
    schema: REPORT_SCHEMA, harness: { version: HARNESS_VERSION, host: (window as unknown as { claude?: unknown }).claude ? "claude-artifact" : location.host },
    created_at: new Date().toISOString(), engine_version: ENGINE_VERSION, bundle_version: site.bundle.bundle_version, bundle_engine_format: site.bundle.engine_version,
    release_profile: site.bundle.release_profile, litertjs: site.bundle.runtime.litertjs_core,
    model_files: site.bundle.files.filter((f) => f.kind === "model").map((f) => ({ id: f.resource_id, file: f.path, sha256: f.sha256, bytes: f.bytes })),
    tester: { ...tester, tier_guess_from_browser: guessTier() }, environment: environment(),
  };
}

export function finalize(rep: Record<string, unknown>, tier: Tier): Record<string, unknown> {
  const steps = (rep.steps ?? {}) as Record<string, { status: string }>;
  rep.runtime_backend = { variant: runtimeVariant(), accelerator: "wasm (XNNPACK, CPU)" };
  rep.page_events = pageEvents.slice(-200);
  rep.budgets = budgets(tier, rep);
  rep.functional_pass = ["download", "engine", "models", "reference"].every((k) => steps[k]?.status === "passed");
  return rep;
}

/** Sends the report to the artifact's database (claude.ai viewer, signed in, Contributor or Editor access). */
export async function sendReport(rep: Record<string, unknown>): Promise<string> {
  const c = (window as unknown as { claude?: { use: (n: string) => Promise<unknown> } }).claude;
  if (!c?.use) throw new Error("Sending needs the claude.ai page; copy or save the report instead.");
  const [db, user] = (await Promise.all([c.use("db"), c.use("user")])) as [
    { doc: (p: string) => { set: (d: Record<string, unknown>) => Promise<void> } } | null, { id: () => Promise<string | null> } | null];
  if (!db) throw new Error("Report storage is not available in this view; copy or save the report instead.");
  const id = user ? await user.id() : null;
  if (!id) throw new Error("Sign in to claude.ai to send the report, or copy or save it instead.");
  const runId = `${new Date().toISOString().replace(/[:.]/g, "-")}-${Math.random().toString(36).slice(2, 8)}`;
  await db.doc(`reports/${id}/runs/${runId}`).set(JSON.parse(JSON.stringify(rep)) as Record<string, unknown>);
  return runId;
}

export async function saveReport(text: string): Promise<boolean> {
  const c = (window as unknown as { claude?: { use: (n: string) => Promise<unknown> } }).claude;
  const dl = c?.use ? ((await c.use("downloads")) as { save: (r: { filename: string; data: string }) => Promise<unknown> } | null) : null;
  if (!dl) return false;
  await dl.save({ filename: `phone-test-${Date.now()}.json`, data: text });
  return true;
}
