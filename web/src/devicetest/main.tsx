// Phone test page (see harness.ts). Rendered into #app of the page built by
// `petperception export-device-test`.
import { render } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { guessTier, TIERS, type Tier } from "../engine/budgets";
import type { LocalEngine } from "../engine/engine";
import { runtimeVariant } from "../engine/runtime";
import { analyzeOnce, battery, FileStore, finalize, jsHeap, matchesExpected, modelCheck, newReport, parseUserAgent, referenceClip, runStats,
  saveReport, sendReport, startEngine, transferBytes, watchNetwork, pageEvents, type SiteManifest, type Tester } from "./harness";

type Status = "not run" | "running" | "passed" | "failed" | "done" | "skipped";
type Steps = Record<string, { status: Status; note?: string }>;
const KEY = "pp-phone-test-report-v1";
const BASE = new URL(".", location.href);

function load<T>(k: string): T | null { try { const s = localStorage.getItem(k); return s ? (JSON.parse(s) as T) : null; } catch { return null; } }
function store(k: string, v: unknown) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode or blocked storage */ } }
const fmt = (ms: number | null | undefined) => (ms == null || !Number.isFinite(ms) ? "–" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
const mb = (b: number) => `${(b / 2 ** 20).toFixed(1)} MB`;

function App() {
  const [site, setSite] = useState<SiteManifest | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const ua = parseUserAgent(navigator.userAgent);
  const saved = load<{ report: Record<string, unknown> }>(KEY);
  const rep = useRef<Record<string, unknown>>(saved?.report ?? {});
  const [, bump] = useState(0);
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState<string>("");
  const [cached, setCached] = useState(0);
  const [sent, setSent] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const t0 = (saved?.report.tester ?? {}) as Partial<Tester>;
  const [tester, setTester] = useState<Tester>({ device_model: t0.device_model ?? "", os: t0.os ?? ua.os_guess ?? "", browser: t0.browser ?? ua.browser_guess ?? "",
    tier: t0.tier ?? guessTier() ?? "B", notes: t0.notes ?? "" });
  const [recorded, setRecorded] = useState("");
  const storeRef = useRef<FileStore | null>(null);
  const engineRef = useRef<LocalEngine | null>(null);

  const steps = (rep.current.steps ?? {}) as Steps;
  const setStep = (k: string, status: Status, note?: string) => {
    rep.current.steps = { ...(rep.current.steps as Steps ?? {}), [k]: { status, ...(note ? { note } : {}) } };
    rep.current.tester = { ...tester, tier_guess_from_browser: guessTier() };
    store(KEY, { report: rep.current });
    bump((n) => n + 1);
  };

  useEffect(() => {
    fetch(new URL("bundle/site_manifest.json", BASE).href, { cache: "no-store" })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() as Promise<SiteManifest>; })
      .then(async (s) => {
        setSite(s);
        storeRef.current = new FileStore(s, BASE);
        setCached(await storeRef.current.cachedCount());
        if (!rep.current.schema || rep.current.bundle_version !== s.bundle.bundle_version) rep.current = newReport(s, tester);
        store(KEY, { report: rep.current });
        bump((n) => n + 1);
      })
      .catch((e) => setFatal(`The test files could not be loaded (${String(e)}). Check the connection and reload.`));
  }, []);

  const guard = async (name: string, f: () => Promise<void>) => {
    if (busy) return;
    setBusy(name); setMessage(null);
    try { await f(); } catch (e) {
      setStep(name, "failed", String(e instanceof Error ? e.message : e).slice(0, 400));
      rep.current[`${name}_error`] = String(e instanceof Error ? e.stack ?? e.message : e).slice(0, 2000);
    } finally { setBusy(null); setProgress(""); store(KEY, { report: rep.current }); bump((n) => n + 1); }
  };

  const peakHeap = () => {
    const h = jsHeap();
    const p = rep.current.js_heap_peak as { used: number } | null | undefined;
    if (h && (!p || h.used > p.used)) rep.current.js_heap_peak = h;
  };

  const ensureEngine = async (): Promise<LocalEngine> => {
    if (engineRef.current) return engineRef.current;
    const s = storeRef.current!;
    const t = performance.now();
    setProgress("Reading and verifying the saved models");
    const dl = await s.loadAll((d, tot) => setProgress(`Models ${mb(d)} of ${mb(tot)}`));
    setProgress("Starting the engine");
    const { engine } = await startEngine(s);
    engineRef.current = engine;
    // Every file came from on-phone storage (none from memory or the network): a cold start after a reload.
    if (dl.network_bytes === 0 && dl.from_cache === dl.files) {
      rep.current.engine = { ...(rep.current.engine as object ?? {}), reload_from_cache_ms: performance.now() - t, reload_variant: runtimeVariant() };
    }
    peakHeap();
    return engine;
  };

  const runDownload = () => guard("download", async () => {
    const s = storeRef.current!;
    setStep("download", "running");
    const dl = await s.loadAll((d, tot, f) => setProgress(`${f.split("/").pop()} · ${mb(d)} of ${mb(tot)}`));
    rep.current.download = { ...dl, transfer_bytes_reported: transferBytes((u) => u.includes("/bundle/")) || null };
    setCached(await s.cachedCount());
    const est = await navigator.storage?.estimate?.().catch(() => undefined);
    if (est) rep.current.storage = { usage: est.usage ?? null, quota: est.quota ?? null };
    setStep("download", "passed", `${mb(dl.total_bytes)} verified in ${fmt(dl.ms)} (${mb(dl.network_bytes)} downloaded, ${dl.from_cache} files from storage; saved for reload: ${dl.cache_storage ? "yes" : "no"})`);
  });

  const runEngine = () => guard("engine", async () => {
    setStep("engine", "running");
    setProgress("Compiling the models for this phone");
    const t = performance.now();
    const { engine } = await startEngine(storeRef.current!);
    engineRef.current = engine;
    const runtimeBytes = performance.getEntriesByType("resource").filter((e) => e.name.includes("/litert/"))
      .reduce((n, e) => n + ((e as PerformanceResourceTiming).decodedBodySize || 0), 0);
    rep.current.runtime_bytes = runtimeBytes || null;
    rep.current.engine = { ...(rep.current.engine as object ?? {}), load_ms: performance.now() - t, variant: runtimeVariant(), components: engine.components() };
    peakHeap();
    setStep("engine", "passed", `${runtimeVariant() ?? "unknown"} WebAssembly runtime started in ${fmt(performance.now() - t)}`);
  });

  const runModels = () => guard("models", async () => {
    setStep("models", "running");
    setProgress("Running each model on fixed inputs");
    const engine = await ensureEngine();
    const res = await modelCheck(engine, storeRef.current!);
    rep.current.models = res;
    peakHeap();
    const ok = Object.values(res).every((r) => (r as { argmax_equal?: boolean; max_abs_diff_first_values?: number }).argmax_equal
      && ((r as { max_abs_diff_first_values: number }).max_abs_diff_first_values ?? 1) <= 1e-3);
    const det = (res.detector_float as { median_ms?: number } | undefined)?.median_ms;
    setStep("models", ok ? "passed" : "failed", `${Object.keys(res).length} models match the reference: ${ok ? "yes" : "no"}; detector ${fmt(det)} per frame`);
  });

  const runReference = (n: number, key: "reference" | "sustained") => guard(key, async () => {
    setStep(key, "running");
    const engine = await ensureEngine();
    const { clip, expected } = await referenceClip(storeRef.current!);
    const evStart = pageEvents.length;
    const b0 = key === "sustained" ? await battery() : null;
    const runs: Record<string, unknown>[] = [];
    for (let k = 0; k < n; k++) {
      setProgress(`Analysis ${k + 1} of ${n}`);
      const r = await analyzeOnce(engine, clip, expected.file.split("/").pop()!, `${key}-${k}`);
      peakHeap();
      runs.push({ run: k + 1, wall_ms: r.wall_ms, ...r.summary, matches_expected: matchesExpected(r.summary, expected) });
    }
    const b1 = key === "sustained" ? await battery() : null;
    const dur = (runs[0].media_duration_s as number | null) ?? null;
    const stats = runStats(runs.map((x) => x.wall_ms as number), dur);
    const mediaMin = dur ? (n * dur) / 60 : null;
    const batt = b0 && b1 && !b0.charging && !b1.charging && mediaMin ? (100 * (b0.level - b1.level)) / mediaMin : null;
    const events = pageEvents.slice(evStart);
    rep.current[key] = { clip: expected.file, expected, runs, ...stats, online: navigator.onLine,
      ...(key === "sustained" ? { battery: { start: b0, end: b1 }, battery_pct_per_media_minute: batt, interruptions: events } : {}) };
    const ok = runs.every((x) => x.matches_expected);
    setStep(key, ok ? "passed" : "failed", `${runs.filter((x) => x.matches_expected).length}/${n} match the expected result; median ${fmt(stats.wall_ms_median)} for a ${dur?.toFixed(1)} s clip`
      + (stats.sustained_slowdown ? `; slowdown ${stats.sustained_slowdown.toFixed(2)}x` : "") + (events.length ? `; ${events.length} page events (app switches) during the runs` : ""));
  });

  const runOffline = () => guard("offline", async () => {
    setStep("offline", "running");
    const engine = await ensureEngine();
    const { clip, expected } = await referenceClip(storeRef.current!);
    const reportedOnline = navigator.onLine;
    const net = watchNetwork();
    setProgress("Analysing with the network off");
    const r = await analyzeOnce(engine, clip, expected.file.split("/").pop()!, "offline");
    const n = net.stop();
    const ok = matchesExpected(r.summary, expected) && n.requests === 0;
    rep.current.offline = { browser_reported_online: reportedOnline, network_requests_during_analysis: n, wall_ms: r.wall_ms, ...r.summary, matches_expected: matchesExpected(r.summary, expected) };
    setStep("offline", ok ? "passed" : "failed", `${reportedOnline ? "The browser still reported a connection. " : ""}Result ${matchesExpected(r.summary, expected) ? "matches" : "differs"}; ${n.requests} network requests during analysis`);
  });

  const runRecording = (file: File | undefined) => file && guard("recording", async () => {
    setStep("recording", "running");
    const engine = await ensureEngine();
    const net = watchNetwork();
    const evStart = pageEvents.length;
    const r = await analyzeOnce(engine, file, file.name || "recording", `recording-${Date.now()}`, (s, f) => setProgress(`${s} ${Math.round(100 * f)}%`));
    const n = net.stop();
    peakHeap();
    const list = (rep.current.recordings as unknown[] | undefined) ?? [];
    const sp = r.summary.species;
    const entry = { file_type: file.type || "unknown", bytes: file.size, wall_ms: r.wall_ms, realtime_factor: r.summary.media_duration_s ? r.wall_ms / (1000 * r.summary.media_duration_s) : null,
      tester_says_recorded: recorded || null, ...r.summary, network_requests_during_analysis: n.requests, page_events: pageEvents.slice(evStart) };
    rep.current.recordings = [...list, entry];
    setStep("recording", "done", `${(file.type || "file")}, ${r.summary.media_duration_s?.toFixed(1) ?? "?"} s analysed in ${fmt(r.wall_ms)}: `
      + `${sp.length ? sp.join(", ") : "no cat or dog found"}; ${r.summary.sound_labels.length} sounds; ${r.summary.segment_kinds.join(", ") || "no behavior segments"}`);
  });

  const reloadReady = cached > 0 && site && cached === site.bundle.files.length && !engineRef.current;
  const runReload = () => guard("reload", async () => {
    setStep("reload", "running");
    const t = performance.now();
    await ensureEngine();
    const e = rep.current.engine as { reload_from_cache_ms?: number } | undefined;
    setStep("reload", e?.reload_from_cache_ms ? "passed" : "failed", e?.reload_from_cache_ms ? `Started from saved models in ${fmt(e.reload_from_cache_ms)} without downloading`
      : `The models were not all available from storage on this phone (took ${fmt(performance.now() - t)})`);
  });

  const finalReport = () => finalize({ ...rep.current, tester: { ...tester, tier_guess_from_browser: guessTier() } }, tester.tier);
  const text = () => JSON.stringify(finalReport(), null, 1);
  const send = async () => {
    setMessage(null);
    try { const id = await sendReport(finalReport()); setSent(id); setMessage("Report sent. Thank you."); }
    catch (e) { setMessage(String(e instanceof Error ? e.message : e)); }
  };
  const copy = async () => {
    try { await navigator.clipboard.writeText(text()); setMessage("Report copied."); }
    catch { setMessage("Copy is blocked here: select the report text below and copy it."); }
  };
  const save = async () => { try { if (!(await saveReport(text()))) setMessage("Saving files is not available here; use Copy."); } catch (e) { setMessage(String(e)); } };
  const restart = async () => { try { localStorage.removeItem(KEY); } catch { /* ignore */ } await storeRef.current?.clearCache(); location.reload(); };

  if (fatal) return <main class="wrap"><h1>Phone test</h1><p class="bad">{fatal}</p></main>;
  if (!site) return <main class="wrap"><h1>Phone test</h1><p>Loading the test files…</p></main>;
  const ready = steps.download?.status === "passed" || engineRef.current !== null;
  const has = (k: string) => steps[k]?.status === "passed" || steps[k]?.status === "done";

  return (
    <main class="wrap">
      <header class="head">
        <p class="eyebrow">Pet Observer · on-device acceptance</p>
        <h1>Phone test</h1>
        <p class="lede">Runs the app's analysis engine on this phone: the same models (bundle <code>{site.bundle.bundle_version}</code>, {mb(site.bundle.total_bytes)}) and the same WebAssembly runtime.
          Recordings are analysed here and never leave the phone. Only the measurement report is sent, and only when you tap Send.</p>
        {ua.in_app_webview && <p class="warn">This looks like a browser built into another app. Open the link in Safari (iPhone) or Chrome (Android) so the test measures the real browser.</p>}
        {(window as unknown as { claude?: unknown }).claude === undefined && <p class="muted">Opened outside claude.ai: copy the report when you are done.</p>}
      </header>

      <section class="card" aria-labelledby="who">
        <h2 id="who">This phone</h2>
        <div class="grid">
          <label for="pt-model">Phone model<input id="pt-model" value={tester.device_model} placeholder="for example Pixel 7a or iPhone 12" maxLength={120}
            onInput={(e) => setTester({ ...tester, device_model: (e.target as HTMLInputElement).value })} /></label>
          <label for="pt-os">Operating system<input id="pt-os" value={tester.os} maxLength={60} onInput={(e) => setTester({ ...tester, os: (e.target as HTMLInputElement).value })} /></label>
          <label for="pt-browser">Browser<input id="pt-browser" value={tester.browser} maxLength={60} onInput={(e) => setTester({ ...tester, browser: (e.target as HTMLInputElement).value })} /></label>
          <label for="pt-tier">Device tier<select id="pt-tier" value={tester.tier} onChange={(e) => setTester({ ...tester, tier: (e.target as HTMLSelectElement).value as Tier })}>
            {(Object.keys(TIERS) as Tier[]).map((t) => <option key={t} value={t}>{t}: {TIERS[t].name}</option>)}</select></label>
        </div>
        <p class="hint">{TIERS[tester.tier].examples}</p>
      </section>

      <ol class="steps">
        <Step n="download" title="Download and verify the models" s={steps.download} busy={busy}
          help="Use Wi-Fi. Every file is checked against its SHA-256 checksum before use." action={!ready ? "Download" : "Download again"} onRun={runDownload} />
        <Step n="engine" title="Start the engine" s={steps.engine} busy={busy} disabled={!ready} help="Loads the WebAssembly runtime and compiles the three models." action="Start" onRun={runEngine} />
        <Step n="models" title="Check the models" s={steps.models} busy={busy} disabled={!ready} help="Runs each model on fixed inputs and compares the outputs with the reference computer." action="Check" onRun={runModels} />
        <Step n="reference" title="Analyse the reference clip three times" s={steps.reference} busy={busy} disabled={!ready}
          help="An 11-second test clip with a cat. Each run must give the expected result." action="Analyse" onRun={() => runReference(3, "reference")} />
        <Step n="sustained" title="Sustained run with an interruption" s={steps.sustained} busy={busy} disabled={!ready}
          help="Unplug the charger. Ten analyses in a row (about 1-3 minutes). During the run, switch to another app for about 10 seconds once, then come back."
          action="Start ten analyses" onRun={() => runReference(10, "sustained")} />
        <Step n="offline" title="Analyse with the network off" s={steps.offline} busy={busy} disabled={!ready}
          help="Turn on airplane mode (Wi-Fi off too), then tap Analyse. Turn the network back on afterwards to send the report." action="Analyse offline" onRun={runOffline} />
        <li class="step">
          <StepHead title="Record a clip and analyse it" s={steps.recording} />
          <p class="help">Record 10-30 seconds of your cat or dog (or anything else). The clip is analysed on this phone and is not uploaded or stored by this page.</p>
          <label for="pt-what" class="inline">What will you record?<select id="pt-what" value={recorded} onChange={(e) => setRecorded((e.target as HTMLSelectElement).value)}>
            <option value="">choose</option><option value="cat">a cat</option><option value="dog">a dog</option><option value="cat_and_dog">a cat and a dog</option>
            <option value="no_pet">no pet</option></select></label>
          <div class="row">
            <label class={`btn ${busy || !ready ? "off" : ""}`} for="pt-video">Record video<input id="pt-video" type="file" accept="video/*" capture="environment" disabled={!!busy || !ready}
              onChange={(e) => runRecording((e.target as HTMLInputElement).files?.[0])} /></label>
            <label class={`btn ghost ${busy || !ready ? "off" : ""}`} for="pt-file">Choose a saved video<input id="pt-file" type="file" accept="video/*,audio/*" disabled={!!busy || !ready}
              onChange={(e) => runRecording((e.target as HTMLInputElement).files?.[0])} /></label>
          </div>
          {steps.recording?.note && <p class="result">{steps.recording.note}</p>}
        </li>
        <li class="step">
          <StepHead title="Reload and start from saved models" s={steps.reload} />
          <p class="help">Pull down to reload this page (or use the browser's reload), then tap Start below. Your progress is kept on this phone.</p>
          <button type="button" class="btn" disabled={!!busy || !reloadReady} onClick={runReload}>Start from saved models</button>
          {!reloadReady && <p class="hint">{cached === site.bundle.files.length ? "Reload the page first." : "Available after the download step on this phone."}</p>}
          {steps.reload?.note && <p class="result">{steps.reload.note}</p>}
        </li>
      </ol>

      {busy && <p class="progress" role="status" aria-live="polite">{progress || "Working…"}</p>}

      <section class="card" aria-labelledby="send">
        <h2 id="send">Send the report</h2>
        <label for="pt-notes">Notes (optional: anything unusual, such as the phone getting hot or the browser reloading)
          <textarea id="pt-notes" rows={3} maxLength={1000} value={tester.notes} onInput={(e) => setTester({ ...tester, notes: (e.target as HTMLTextAreaElement).value })} /></label>
        <p class="hint">The report contains the phone and browser details above, timings, memory and battery readings, and the analysis summaries. It contains no video, audio or images.</p>
        <div class="row">
          <button type="button" class="btn" disabled={!!busy || !has("download")} onClick={send}>Send report</button>
          <button type="button" class="btn ghost" onClick={copy}>Copy report</button>
          <button type="button" class="btn ghost" onClick={save}>Save report</button>
          <button type="button" class="btn ghost" disabled={!!busy} onClick={restart}>Start a new test</button>
        </div>
        {message && <p class="result" role="status">{message}{sent ? ` (id ${sent})` : ""}</p>}
        <details><summary>Show report</summary><pre class="report">{text()}</pre></details>
      </section>

      {site.credits && site.credits.length > 0 && (
        <section class="card" aria-labelledby="credits">
          <h2 id="credits">Credits</h2>
          <ul class="credits">
            {site.credits.map((c) => <li key={c.item}><strong>{c.item}</strong>{c.credit ? ` - ${c.credit}` : ""}<br /><span class="hint">{c.terms}</span></li>)}
          </ul>
        </section>
      )}
    </main>
  );
}

function StepHead({ title, s }: { title: string; s?: { status: Status } }) {
  const st = s?.status ?? "not run";
  return <div class="stephead"><h3>{title}</h3><span class={`chip ${st.replace(" ", "-")}`}>{st}</span></div>;
}

function Step({ n, title, s, help, action, onRun, busy, disabled }: { n: string; title: string; s?: { status: Status; note?: string }; help: string; action: string;
  onRun: () => void; busy: string | null; disabled?: boolean }) {
  return (
    <li class="step" id={`step-${n}`}>
      <StepHead title={title} s={s} />
      <p class="help">{help}</p>
      <button type="button" class="btn" disabled={!!busy || disabled} onClick={onRun}>{busy === n ? "Running…" : action}</button>
      {s?.note && <p class={`result ${s.status === "failed" ? "bad" : ""}`}>{s.note}</p>}
    </li>
  );
}

render(<App />, document.getElementById("app")!);
