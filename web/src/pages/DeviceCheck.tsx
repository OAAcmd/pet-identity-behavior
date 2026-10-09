// Device check: the real-phone acceptance harness (docs/DEVICE_TEST_PROTOCOL.md).
// Runs entirely on this device. Nothing is uploaded unless "Send report" is tapped.
import { isWebGPUSupported } from "@litertjs/core";
import { useState } from "preact/hooks";
import { ErrorBanner, InfoBanner, ProgressBar } from "../components/ui";
import { evaluateBudgets, guessTier, TIERS, type Tier } from "../engine/budgets";
import { normalize, syntheticAudio, syntheticImage } from "../engine/parity";
import { readFile } from "../engine/modelstore";
import { runModel, type LoadedModel } from "../engine/runtime";
import { api } from "../lib/api";
import { fmtBytes } from "../lib/format";
import { useOnDevice } from "../lib/ondevice";
import { useApp } from "../state";

interface RefCase { input: string; output_index_by_size: number; first: number[]; mean_abs: number; argmax: number; elements: number }
interface Expected { file: string; mime: string; tracks: number; species: string[]; segment_kinds: string[]; decision: string; sound_labels: string[] }

const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const MB = 1024 * 1024;

interface BatteryLike { level: number; charging: boolean }
async function battery(): Promise<BatteryLike | null> {
  const nav = navigator as Navigator & { getBattery?: () => Promise<BatteryLike> };
  try { const b = await nav.getBattery?.(); return b ? { level: b.level, charging: b.charging } : null; } catch { return null; }
}

/** Page memory via the standard API (Chromium, cross-origin isolated only); null when unavailable or slow. */
async function pageMemoryBytes(): Promise<number | null> {
  const perf = performance as Performance & { measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }> };
  if (!perf.measureUserAgentSpecificMemory || !(self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated) return null;
  const timeout = new Promise<null>((r) => setTimeout(() => r(null), 30_000));
  try { return await Promise.race([perf.measureUserAgentSpecificMemory().then((m) => m.bytes), timeout]); } catch { return null; }
}

/** Bytes of the cached WebAssembly runtime (service worker cache). */
async function runtimeBytes(): Promise<number | null> {
  if (typeof caches === "undefined") return null;
  let total = 0;
  for (const k of await caches.keys()) {
    const c = await caches.open(k);
    for (const req of await c.keys()) {
      if (!new URL(req.url).pathname.startsWith("/litert/")) continue;
      const r = await c.match(req);
      if (r) total += (await r.blob()).size;
    }
  }
  return total || null;
}

function env(): Record<string, unknown> {
  const nav = navigator as Navigator & { deviceMemory?: number; userAgentData?: { platform?: string; mobile?: boolean } };
  const rec = typeof MediaRecorder !== "undefined" ? ["video/webm;codecs=vp9,opus", "video/webm", "video/mp4", "audio/webm;codecs=opus", "audio/mp4"]
    .filter((t) => MediaRecorder.isTypeSupported(t)) : [];
  const v = document.createElement("video");
  return {
    user_agent: navigator.userAgent, platform: nav.userAgentData?.platform ?? null, mobile: nav.userAgentData?.mobile ?? null,
    hardware_concurrency: navigator.hardwareConcurrency ?? null, device_memory_gb: nav.deviceMemory ?? null,
    cross_origin_isolated: (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated === true,
    webgpu: isWebGPUSupported(), online: navigator.onLine, standalone_pwa: matchMedia("(display-mode: standalone)").matches,
    media_recorder_types: rec, can_play: { webm_vp9: v.canPlayType('video/webm; codecs="vp9"'), mp4_h264: v.canPlayType('video/mp4; codecs="avc1.42E01E"') },
    screen: `${screen.width}x${screen.height}@${devicePixelRatio}`,
  };
}

export function DeviceCheckPage() {
  const { status, getEngine } = useOnDevice();
  const { toast } = useApp();
  const [running, setRunning] = useState<string | null>(null);
  const [report, setReport] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [tier, setTier] = useState<Tier>(guessTier() ?? "B");
  const [deviceModel, setDeviceModel] = useState("");
  const [sustained, setSustained] = useState(false);

  const run = async () => {
    setError(null);
    setReport(null);
    const m = status.installed;
    if (!m) { setError(new Error("Install the on-device models first (Settings).")); return; }
    const rep: Record<string, unknown> = { schema: "petperception.device_check/2", created_at: new Date().toISOString(), bundle_version: m.bundle_version,
      environment: env(), tester: { device_model: deviceModel.trim().slice(0, 120) || null, tier, tier_guess_from_browser: guessTier(), mode: sustained ? "sustained" : "quick" } };
    try {
      setRunning("verifying installed files");
      const integrity: Record<string, boolean> = {};
      for (const f of m.files) integrity[f.path] = await readFile(m, f.path).then(() => true, () => false);
      rep.integrity = integrity;
      setRunning("loading models");
      const t0 = performance.now();
      const engine = await getEngine();
      rep.engine_load_ms = performance.now() - t0;
      const refs = JSON.parse(new TextDecoder().decode(await readFile(m, "device_check/reference_outputs.json"))) as Record<string, RefCase>;
      const models: Record<string, LoadedModel | undefined> = { yamnet: engine.m.yamnet, detector_float: engine.m.detector, backbone: engine.m.backbone };
      const bench: Record<string, unknown> = {};
      for (const [name, mdl] of Object.entries(models)) {
        if (!mdl || !refs[name]) continue;
        setRunning(`benchmarking ${name}`);
        const input = name === "yamnet" ? syntheticAudio() : name === "detector_float" ? normalize(syntheticImage(320), 127.5, 127.5) : normalize(syntheticImage(224), 127, 128);
        const times: number[] = [];
        let outs: ArrayLike<number>[] = [];
        for (let k = 0; k < 6; k++) {
          const t = performance.now();
          outs = (await runModel(mdl, [input])) as ArrayLike<number>[];
          if (k > 0) times.push(performance.now() - t);
        }
        const ref = refs[name];
        const out = outs.find((o) => o.length === ref.elements)!;
        let maxDiff = 0, meanAbs = 0, arg = 0;
        for (let i = 0; i < ref.first.length; i++) maxDiff = Math.max(maxDiff, Math.abs(out[i] - ref.first[i]));
        for (let i = 0; i < out.length; i++) { meanAbs += Math.abs(out[i]); if (out[i] > out[arg]) arg = i; }
        meanAbs /= out.length;
        bench[name] = { median_ms: median(times), min_ms: Math.min(...times), max_abs_diff_first_values: maxDiff,
          mean_abs_relative_diff: Math.abs(meanAbs - ref.mean_abs) / Math.max(ref.mean_abs, 1e-12), argmax_equal: arg === ref.argmax };
      }
      rep.models = bench;
      const expected = JSON.parse(new TextDecoder().decode(await readFile(m, "device_check/expected.json"))) as Expected[];
      const v = document.createElement("video");
      const pick = expected.find((e) => v.canPlayType(e.mime) !== "") ?? expected[0];
      const clip = new Blob([await readFile(m, pick.file) as BlobPart], { type: pick.mime.split(";")[0] });
      const e2e: Record<string, unknown>[] = [];
      const nRuns = sustained ? 10 : 3;
      const bat0 = await battery();
      for (let k = 0; k < nRuns; k++) {
        setRunning(`end-to-end analysis ${k + 1}/${nRuns} (offline-capable)`);
        const t = performance.now();
        const r = (await engine.analyze(clip, { name: pick.file.split("/").pop()!, zones: null, modalities: ["audio", "video"], analysisId: `device-check-${k}` })) as {
          observations: { animal_tracks: { species: string; behavior_segments: { kind: string }[] }[]; sound_events: { label: string }[] }; decision: { state: string };
          timings_ms: Record<string, number>; media: { duration_s: number | null } };
        const tracks = r.observations.animal_tracks;
        const kinds = [...new Set(tracks.flatMap((x) => x.behavior_segments.map((b) => b.kind)))].sort();
        e2e.push({ run: k + 1, wall_ms: performance.now() - t, media_duration_s: r.media.duration_s, timings_ms: r.timings_ms, tracks: tracks.length, species: tracks.map((x) => x.species),
          segment_kinds: kinds, decision: r.decision.state, sound_labels: r.observations.sound_events.map((x) => x.label),
          matches_expected: tracks.length === pick.tracks && r.decision.state === pick.decision && JSON.stringify(kinds) === JSON.stringify([...pick.segment_kinds].sort()) });
      }
      const bat1 = await battery();
      const walls = e2e.map((x) => x.wall_ms as number);
      const dur = (e2e[0].media_duration_s as number | null) ?? null;
      // Sustained slowdown compares the last three runs with the first three (10-run sustained mode only).
      const slowdown = nRuns >= 10 ? median(walls.slice(-3)) / Math.max(median(walls.slice(0, 3)), 1) : null;
      const mediaMin = dur ? (nRuns * dur) / 60 : null;
      const batteryUsed = bat0 && bat1 && !bat0.charging && !bat1.charging && mediaMin ? (100 * (bat0.level - bat1.level)) / mediaMin : null;
      rep.end_to_end = { clip: pick.file, expected: pick, runs: e2e, online_during_test: navigator.onLine, wall_ms_median: median(walls),
        realtime_factor: dur ? median(walls) / (1000 * dur) : null, sustained_slowdown: slowdown,
        battery: { start: bat0, end: bat1, pct_per_media_minute: batteryUsed, note: "1 % resolution; only meaningful unplugged and in sustained mode" } };
      setRunning("measuring memory and storage");
      const mem = (performance as unknown as { memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
      if (mem) rep.js_heap = { used: mem.usedJSHeapSize, limit: mem.jsHeapSizeLimit };
      const pageMem = await pageMemoryBytes();
      rep.page_memory_bytes = pageMem;
      const est = await navigator.storage?.estimate?.();
      if (est) rep.storage = { usage: est.usage ?? null, quota: est.quota ?? null };
      const rt = await runtimeBytes();
      rep.download_bytes = { bundle: m.total_bytes, runtime: rt };
      rep.budgets = { tier, tier_description: TIERS[tier], status_note: "Budget limits are assumptions until measured on physical phones of this tier.",
        results: evaluateBudgets(tier, {
          download_mb: rt !== null ? (m.total_bytes + rt) / MB : null, storage_mb: est?.usage != null ? est.usage / MB : null,
          engine_load_ms: rep.engine_load_ms as number, realtime_factor: (rep.end_to_end as { realtime_factor: number | null }).realtime_factor,
          detector_ms: (bench.detector_float as { median_ms?: number } | undefined)?.median_ms ?? null,
          memory_mb: pageMem !== null ? pageMem / MB : null, sustained_slowdown: slowdown, battery_pct_per_media_min: batteryUsed,
        }) };
      // "passed" is functional correctness only; budgets are reported separately.
      rep.passed = Object.values(integrity).every(Boolean) && Object.values(bench).every((b) => (b as { argmax_equal: boolean }).argmax_equal)
        && e2e.every((x) => x.matches_expected);
      setReport(rep);
    } catch (e) {
      rep.error = String(e instanceof Error ? e.stack ?? e.message : e);
      setReport(rep);
      setError(e);
    } finally {
      setRunning(null);
    }
  };

  const text = report ? JSON.stringify(report, null, 1) : "";
  return (
    <section>
      <h1 tabIndex={-1}>Device check</h1>
      <p>This check runs the on-device models and a complete reference analysis on this phone, without the server.
        For the offline acceptance test, install the models, turn on airplane mode, then tap Run.</p>
      <div class="card">
        <div class="field">
          <label for="dc-model">Phone model (optional; stays on this device unless you send the report)</label>
          <input id="dc-model" type="text" value={deviceModel} maxLength={120} placeholder="for example Pixel 7a"
            onInput={(e) => setDeviceModel((e.target as HTMLInputElement).value)} />
        </div>
        <div class="field">
          <label for="dc-tier">Device tier</label>
          <select id="dc-tier" value={tier} onChange={(e) => setTier((e.target as HTMLSelectElement).value as Tier)}>
            {(Object.keys(TIERS) as Tier[]).map((t) => <option key={t} value={t}>{t}: {TIERS[t].name}</option>)}
          </select>
          <p class="hint">{TIERS[tier].examples}</p>
        </div>
        <label class="check"><input type="checkbox" checked={sustained} onChange={(e) => setSustained((e.target as HTMLInputElement).checked)} />
          Sustained check: 10 consecutive analyses (1-3 minutes) to measure slowdown and battery use. Unplug the phone first.</label>
      </div>
      {status.state !== "ready" && <InfoBanner tone="warn">On-device models are not installed ({status.state}). Install them in Settings first.</InfoBanner>}
      {status.installed && <p class="muted">Installed bundle {status.installed.bundle_version} ({fmtBytes(status.installed.total_bytes)}), runtime LiteRT.js {status.installed.runtime.litertjs_core}.</p>}
      <button type="button" class="btn btn-primary" onClick={run} disabled={running !== null || !status.installed}>Run device check</button>
      {running && <ProgressBar value={0.5} label={running} />}
      <ErrorBanner error={error} />
      {report && (
        <div class="card">
          <h2>{report.passed ? "Passed" : "Not passed"}</h2>
          <BudgetTable report={report} />
          <pre class="report" aria-label="Device check report">{text}</pre>
          <div class="row wrap">
            <button type="button" class="btn btn-secondary" onClick={() => navigator.clipboard?.writeText(text).then(() => toast("Report copied."))}>Copy report</button>
            <a class="btn btn-secondary" download={`device-check-${Date.now()}.json`} href={URL.createObjectURL(new Blob([text], { type: "application/json" }))}>Download report</a>
            <button type="button" class="btn btn-secondary" disabled={!navigator.onLine}
              onClick={() => api.post("/device-reports", { report }).then(() => toast("Report sent (no media included)."), (e) => setError(e))}>Send report to server</button>
          </div>
        </div>
      )}
    </section>
  );
}

function BudgetTable({ report }: { report: Record<string, unknown> }) {
  const b = report.budgets as { tier: Tier; results: { id: string; label: string; unit: string; value: number | null; limit: number; status: string }[] } | undefined;
  if (!b) return null;
  return (
    <table class="budget">
      <caption>Resource budgets, tier {b.tier} (limits are assumptions until measured on phones)</caption>
      <thead><tr><th scope="col">Budget</th><th scope="col">Measured</th><th scope="col">Limit</th><th scope="col">Result</th></tr></thead>
      <tbody>
        {b.results.map((r) => (
          <tr key={r.id}><td>{r.label}</td><td>{r.value === null ? "-" : `${r.value < 10 ? r.value.toFixed(2) : Math.round(r.value)} ${r.unit}`}</td>
            <td>{`${r.limit} ${r.unit}`}</td><td>{r.status === "not_measured" ? "not measured" : r.status}</td></tr>
        ))}
      </tbody>
    </table>
  );
}
