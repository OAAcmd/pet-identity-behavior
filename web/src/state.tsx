import { createContext, type ComponentChildren } from "preact";
import { useCallback, useContext, useEffect, useRef, useState } from "preact/hooks";
import { ApiError, api, setCsrf } from "./lib/api";
import { pendingStore, type PendingUpload } from "./lib/idb";
import type { Capabilities, Me, Pet, Place } from "./lib/types";
import { processPending, type UploadProgress } from "./lib/uploader";

export interface Toast {
  id: number;
  text: string;
  tone: "info" | "error";
}

interface AppState {
  me: Me | null;
  setMe: (m: Me | null) => void;
  pets: Pet[];
  places: Place[];
  caps: Capabilities | null;
  refreshPets: () => Promise<void>;
  refreshPlaces: () => Promise<void>;
  toast: (text: string, tone?: Toast["tone"]) => void;
  toasts: Toast[];
  pending: PendingUpload[];
  progress: Record<string, UploadProgress>;
  refreshPending: () => Promise<void>;
  startPending: (item: PendingUpload) => Promise<{ id: string } | { error: unknown }>;
  online: boolean;
}

const Ctx = createContext<AppState | null>(null);

/** Small, non-secret account data kept on the device for offline use (no tokens). */
export const offlineCache = {
  save(key: string, v: unknown): void {
    try { localStorage.setItem(`pp.cache.${key}`, JSON.stringify(v)); } catch { /* storage full or private mode */ }
  },
  load<T>(key: string): T | null {
    try { const s = localStorage.getItem(`pp.cache.${key}`); return s ? (JSON.parse(s) as T) : null; } catch { return null; }
  },
  clear(): void {
    try { for (const k of ["me", "pets", "places"]) localStorage.removeItem(`pp.cache.${k}`); } catch { /* ignore */ }
  },
};

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("AppProvider missing");
  return v;
}

export function AppProvider({ children }: { children: ComponentChildren }) {
  const [me, setMeState] = useState<Me | null>(null);
  const [pets, setPets] = useState<Pet[]>([]);
  const [places, setPlaces] = useState<Place[]>([]);
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [pending, setPending] = useState<PendingUpload[]>([]);
  const [progress, setProgress] = useState<Record<string, UploadProgress>>({});
  const [online, setOnline] = useState(typeof navigator === "undefined" ? true : navigator.onLine !== false);
  const active = useRef(new Set<string>());
  const nextId = useRef(1);

  const setMe = useCallback((m: Me | null) => {
    setCsrf(m?.csrf_token ?? null);
    setMeState(m);
  }, []);

  const toast = useCallback((text: string, tone: Toast["tone"] = "info") => {
    const id = nextId.current++;
    setToasts((t) => [...t, { id, text, tone }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 6000);
  }, []);

  // Pets and places are cached on the device so offline (on-device) analysis can use them.
  const refreshPets = useCallback(async () => {
    try {
      const v = await api.get<Pet[]>("/pets");
      setPets(v);
      offlineCache.save("pets", v);
    } catch (e) {
      if (e instanceof ApiError && e.isNetwork) setPets(offlineCache.load<Pet[]>("pets") ?? []);
      else throw e;
    }
  }, []);
  const refreshPlaces = useCallback(async () => {
    try {
      const v = await api.get<Place[]>("/places");
      setPlaces(v);
      offlineCache.save("places", v);
    } catch (e) {
      if (e instanceof ApiError && e.isNetwork) setPlaces(offlineCache.load<Place[]>("places") ?? []);
      else throw e;
    }
  }, []);
  const refreshPending = useCallback(async () => {
    if (!pendingStore.available || !me) return;
    const all = await pendingStore.all().catch(() => [] as PendingUpload[]);
    setPending(all.filter((p) => p.userId === me.id).sort((a, b) => a.createdAt - b.createdAt));
  }, [me]);

  const startPending = useCallback(async (item: PendingUpload): Promise<{ id: string } | { error: unknown }> => {
    if (active.current.has(item.localId)) return { error: new ApiError(409, "already_uploading", "This upload is already in progress.") };
    active.current.add(item.localId);
    try {
      const a = await processPending(item, (pr) => setProgress((p) => ({ ...p, [item.localId]: pr })));
      toast("Upload finished - analysis started.");
      return { id: a.id };
    } catch (e) {
      // Input problems cannot be fixed by retrying: drop the local copy so it is not resumed forever.
      if (e instanceof ApiError && !e.isNetwork && e.status >= 400 && e.status < 500 && ![401, 409, 429].includes(e.status)) {
        await pendingStore.remove(item.localId).catch(() => undefined);
      }
      return { error: e };
    } finally {
      active.current.delete(item.localId);
      await refreshPending();
    }
  }, [refreshPending, toast]);

  useEffect(() => {
    if (!me) return;
    refreshPets().catch(() => undefined);
    refreshPlaces().catch(() => undefined);
    api.get<Capabilities>("/capabilities").then(setCaps).catch(() => undefined);
    refreshPending().catch(() => undefined);
  }, [me, refreshPets, refreshPlaces, refreshPending]);

  // Resume paused uploads automatically when connectivity returns.
  useEffect(() => {
    const up = () => {
      setOnline(true);
      pending.filter((p) => p.state === "paused" || p.state === "uploading").forEach((p) => void startPending(p));
    };
    const down = () => setOnline(false);
    window.addEventListener("online", up);
    window.addEventListener("offline", down);
    return () => {
      window.removeEventListener("online", up);
      window.removeEventListener("offline", down);
    };
  }, [pending, startPending]);

  return (
    <Ctx.Provider value={{ me, setMe, pets, places, caps, refreshPets, refreshPlaces, toast, toasts, pending, progress, refreshPending, startPending, online }}>
      {children}
    </Ctx.Provider>
  );
}
