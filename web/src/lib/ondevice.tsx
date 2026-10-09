// On-device analysis state: model bundle status, install/update/remove and the
// lazily built local engine. Analysis runs on this device; media are not uploaded.
import { createContext, type ComponentChildren } from "preact";
import { useCallback, useContext, useEffect, useState } from "preact/hooks";
import type { LocalEngine } from "../engine/engine";
import { buildEngine } from "../engine/local";
import { runtimeVariant } from "../engine/runtime";
import { activeManifest, bundleCompatible, fetchRemoteManifest, install, readFile, storageAvailable, uninstall, type BundleManifest, type InstallProgress } from "../engine/modelstore";

export const BUNDLE_BASE = "/ondevice/";
/** Progress "file" value while the freshly installed engine is started once. */
export const WARMUP = "#warmup";

export type OnDeviceState = "unsupported" | "checking" | "not_installed" | "installing" | "ready" | "error";

export interface OnDeviceStatus {
  state: OnDeviceState;
  installed: BundleManifest | null;
  remote: BundleManifest | null;
  progress: InstallProgress | null;
  error: string | null;
  crossOriginIsolated: boolean;
  /** Runtime build actually loaded on this device (known after the first engine start). */
  runtime: "threads" | "single-thread" | null;
}

interface Ctx {
  status: OnDeviceStatus;
  installOrUpdate: () => Promise<void>;
  remove: () => Promise<void>;
  checkForUpdate: () => Promise<void>;
  getEngine: () => Promise<LocalEngine>;
}

const OnDeviceCtx = createContext<Ctx | null>(null);

let engineCache: { version: string; engine: Promise<LocalEngine> } | null = null;

export function unsupportedReason(): string | null {
  if (typeof WebAssembly === "undefined") return "This browser cannot run WebAssembly.";
  if (!storageAvailable()) return "This browser cannot store models (a secure https connection is required).";
  if (typeof OfflineAudioContext === "undefined") return "This browser cannot decode audio for analysis.";
  return null;
}

export function OnDeviceProvider({ children }: { children: ComponentChildren }) {
  const [status, setStatus] = useState<OnDeviceStatus>({ state: "checking", installed: null, remote: null, progress: null, error: null, runtime: runtimeVariant(),
    crossOriginIsolated: typeof self !== "undefined" && (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated === true });

  const checkForUpdate = useCallback(async () => {
    let note: string | null = null;
    const remote = await fetchRemoteManifest(BUNDLE_BASE).catch((e) => {
      // Offline or server unreachable: silently keep the installed version. A format the app
      // cannot use is reported, because only reloading the app (online) resolves it.
      if (e instanceof Error && e.message.includes("needs a newer app version")) note = `A model update is available but ${e.message}.`;
      return null;
    });
    setStatus((s) => ({ ...s, remote, error: note ?? s.error }));
  }, []);

  useEffect(() => {
    const why = unsupportedReason();
    if (why) { setStatus((s) => ({ ...s, state: "unsupported", error: why })); return; }
    activeManifest().then((installed) => {
      if (installed && !bundleCompatible(installed)) {
        setStatus((s) => ({ ...s, installed, state: "error", error: `The installed models (format ${installed.engine_version}) do not match this app version. Update the models in Settings.` }));
      } else setStatus((s) => ({ ...s, installed, state: installed ? "ready" : "not_installed" }));
      if (navigator.onLine !== false) checkForUpdate().catch(() => undefined);
    }).catch((e) => setStatus((s) => ({ ...s, state: "error", error: String(e) })));
  }, [checkForUpdate]);

  const installOrUpdate = useCallback(async () => {
    setStatus((s) => ({ ...s, state: "installing", error: null, progress: { done_bytes: 0, total_bytes: 1, file: "" } }));
    try {
      const remote = await fetchRemoteManifest(BUNDLE_BASE);
      // Before switching versions, start the engine once from the new files: proves the models
      // compile on this device and makes the service worker cache the WebAssembly runtime,
      // which offline analysis needs. If this fails, the previous version stays active.
      let engine: Promise<LocalEngine> | null = null;
      await install(BUNDLE_BASE, remote, (progress) => setStatus((s) => ({ ...s, progress })), undefined, async () => {
        setStatus((s) => ({ ...s, progress: { done_bytes: 1, total_bytes: 1, file: WARMUP } }));
        engine = buildEngine(remote, (p) => readFile(remote, p));
        try {
          await engine;
        } catch (e) {
          throw new Error(`the models could not be started on this device (${e instanceof Error ? e.message : String(e)})`);
        }
      });
      engineCache = engine ? { version: remote.bundle_version, engine } : null;
      setStatus((s) => ({ ...s, state: "ready", installed: remote, remote, progress: null, runtime: runtimeVariant() }));
    } catch (e) {
      // The previously installed version (if any) stays active.
      const installed = await activeManifest().catch(() => null);
      setStatus((s) => ({ ...s, state: installed ? "ready" : "error", installed, progress: null,
        error: `Model installation failed: ${e instanceof Error ? e.message : String(e)}. ${installed ? "The previous version is still in use." : "Try again when you have a stable connection."}` }));
    }
  }, []);

  const remove = useCallback(async () => {
    await uninstall();
    engineCache = null;
    setStatus((s) => ({ ...s, state: "not_installed", installed: null }));
  }, []);

  const getEngine = useCallback(async () => {
    const m = status.installed ?? (await activeManifest());
    if (!m) throw new Error("On-device models are not installed yet.");
    if (!engineCache || engineCache.version !== m.bundle_version) {
      engineCache = { version: m.bundle_version, engine: buildEngine(m, (p) => readFile(m, p)) };
      engineCache.engine.then(() => setStatus((s) => ({ ...s, runtime: runtimeVariant() })), () => { engineCache = null; });
    }
    return engineCache.engine;
  }, [status.installed]);

  return <OnDeviceCtx.Provider value={{ status, installOrUpdate, remove, checkForUpdate, getEngine }}>{children}</OnDeviceCtx.Provider>;
}

export function useOnDevice(): Ctx {
  const v = useContext(OnDeviceCtx);
  if (!v) throw new Error("OnDeviceProvider missing");
  return v;
}

// Device preference: send media to the server for analysis (off by default).
const SERVER_KEY = "pp.serverAnalysis";
export function serverAnalysisEnabled(): boolean {
  try { return localStorage.getItem(SERVER_KEY) === "on"; } catch { return false; }
}
export function setServerAnalysis(on: boolean): void {
  try { localStorage.setItem(SERVER_KEY, on ? "on" : "off"); } catch { /* private mode */ }
}
