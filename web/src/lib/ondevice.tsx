// On-device analysis state: model bundle status, install/update/remove and the
// lazily built local engine. Analysis runs on this device; media are not uploaded.
import { createContext, type ComponentChildren } from "preact";
import { useCallback, useContext, useEffect, useState } from "preact/hooks";
import type { LocalEngine } from "../engine/engine";
import { buildEngine } from "../engine/local";
import { activeManifest, fetchRemoteManifest, install, readFile, storageAvailable, uninstall, type BundleManifest, type InstallProgress } from "../engine/modelstore";

export const BUNDLE_BASE = "/ondevice/";

export type OnDeviceState = "unsupported" | "checking" | "not_installed" | "installing" | "ready" | "error";

export interface OnDeviceStatus {
  state: OnDeviceState;
  installed: BundleManifest | null;
  remote: BundleManifest | null;
  progress: InstallProgress | null;
  error: string | null;
  crossOriginIsolated: boolean;
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
  const [status, setStatus] = useState<OnDeviceStatus>({ state: "checking", installed: null, remote: null, progress: null, error: null,
    crossOriginIsolated: typeof self !== "undefined" && (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated === true });

  const checkForUpdate = useCallback(async () => {
    const remote = await fetchRemoteManifest(BUNDLE_BASE).catch(() => null);
    setStatus((s) => ({ ...s, remote }));
  }, []);

  useEffect(() => {
    const why = unsupportedReason();
    if (why) { setStatus((s) => ({ ...s, state: "unsupported", error: why })); return; }
    activeManifest().then((installed) => {
      setStatus((s) => ({ ...s, installed, state: installed ? "ready" : "not_installed" }));
      if (navigator.onLine !== false) checkForUpdate().catch(() => undefined);
    }).catch((e) => setStatus((s) => ({ ...s, state: "error", error: String(e) })));
  }, [checkForUpdate]);

  const installOrUpdate = useCallback(async () => {
    setStatus((s) => ({ ...s, state: "installing", error: null, progress: { done_bytes: 0, total_bytes: 1, file: "" } }));
    try {
      const remote = await fetchRemoteManifest(BUNDLE_BASE);
      await install(BUNDLE_BASE, remote, (progress) => setStatus((s) => ({ ...s, progress })));
      engineCache = null;
      setStatus((s) => ({ ...s, state: "ready", installed: remote, remote, progress: null }));
    } catch (e) {
      // The previously installed version (if any) stays active.
      const installed = await activeManifest().catch(() => null);
      setStatus((s) => ({ ...s, state: installed ? "ready" : "error", installed, progress: null,
        error: `Model download failed: ${e instanceof Error ? e.message : String(e)}. ${installed ? "The previous version is still in use." : "Try again when you have a stable connection."}` }));
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
      engineCache.engine.catch(() => { engineCache = null; });
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
