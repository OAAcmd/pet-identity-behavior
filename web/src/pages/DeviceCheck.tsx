// Device check: the real-phone acceptance harness (docs/DEVICE_TEST_PROTOCOL.md).
// Runs entirely on this device. Nothing is uploaded unless "Send report" is tapped.
import { isWebGPUSupported } from "@litertjs/core";
import { useState } from "preact/hooks";
import { ErrorBanner, InfoBanner, ProgressBar } from "../components/ui";
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

  const run = async () => {
    setError(null);
    setReport(null);
    const m = status.installed;
    if (!m) { setError(new Error("Install the on-device models first (Settings).")); return; }
    const rep: Record<string, unknown> = { schema: "petperception.device_check/1", created_at: new Date().toISOString(), bundle_version: m.bundle_version, environment: env() };
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
      for (let k = 0; k < 3; k++) {
        setRunning(`end-to-end analysis ${k + 1}/3 (offline-capable)`);
        const t = performance.now();
        const r = (await engine.analyze(clip, { name: pick.file.split("/").pop()!, zones: null, modalities: ["audio", "video"], analysisId: `device-check-${k}` })) as {
          observations: { animal_tracks: { species: string; behavior_segments: { kind: string }[] }[]; sound_events: { label: string }[] }; decision: { state: string };
          timings_ms: Record<string, number> };
        const tracks = r.observations.animal_tracks;
        const kinds = [...new Set(tracks.flatMap((x) => x.behavior_segments.map((b) => b.kind)))].sort();
        e2e.push({ run: k + 1, wall_ms: performance.now() - t, timings_ms: r.timings_ms, tracks: tracks.length, species: tracks.map((x) => x.species),
          segment_kinds: kinds, decision: r.decision.state, sound_labels: r.observations.sound_events.map((x) => x.label),
          matches_expected: tracks.length === pick.tracks && r.decision.state === pick.decision && JSON.stringify(kinds) === JSON.stringify([...pick.segment_kinds].sort()) });
      }
      rep.end_to_end = { clip: pick.file, expected: pick, runs: e2e, online_during_test: navigator.onLine,
        sustained_slowdown: (e2e[2].wall_ms as number) / Math.max(e2e[0].wall_ms as number, 1) };
      const mem = (performance as unknown as { memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
      if (mem) rep.js_heap = { used: mem.usedJSHeapSize, limit: mem.jsHeapSizeLimit };
      const est = await navigator.storage?.estimate?.();
      if (est) rep.storage = { usage: est.usage ?? null, quota: est.quota ?? null };
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
      {status.state !== "ready" && <InfoBanner tone="warn">On-device models are not installed ({status.state}). Install them in Settings first.</InfoBanner>}
      {status.installed && <p class="muted">Installed bundle {status.installed.bundle_version} ({fmtBytes(status.installed.total_bytes)}), runtime LiteRT.js {status.installed.runtime.litertjs_core}.</p>}
      <button type="button" class="btn btn-primary" onClick={run} disabled={running !== null || !status.installed}>Run device check</button>
      {running && <ProgressBar value={0.5} label={running} />}
      <ErrorBanner error={error} />
      {report && (
        <div class="card">
          <h2>{report.passed ? "Passed" : "Not passed"}</h2>
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
