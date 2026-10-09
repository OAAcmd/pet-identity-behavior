// On-device storage for local analyses and their feedback (IndexedDB).
// Nothing here is sent to the server unless the user turns on result sync.
import type { AnalysisResult, Feedback } from "./types";

const DB = "pet-observer-local";
const VERSION = 1;

export interface LocalAnalysis {
  id: string;
  userId: string;
  createdAt: number;
  filename: string;
  kind: "video" | "audio" | "image";
  subject_mode: string;
  pet_ids: string[];
  place_id: string | null;
  note: string;
  bundle_version: string;
  result: AnalysisResult;
  media: Blob | null; // kept on the device for playback; deleted with the analysis
  synced_at: number | null;
}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains("analyses")) db.createObjectStore("analyses", { keyPath: "id" });
      if (!db.objectStoreNames.contains("feedback")) db.createObjectStore("feedback", { keyPath: "id" }).createIndex("analysis_id", "analysis_id");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function run<T>(store: string, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  return new Promise<T>((resolve, reject) => {
    const t = db.transaction(store, mode);
    const r = fn(t.objectStore(store));
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    t.oncomplete = () => db.close();
  });
}

export const localAnalyses = {
  available: typeof indexedDB !== "undefined",
  put: (a: LocalAnalysis) => run("analyses", "readwrite", (s) => s.put(a)),
  get: (id: string) => run<LocalAnalysis | undefined>("analyses", "readonly", (s) => s.get(id) as IDBRequest<LocalAnalysis | undefined>),
  all: async (userId: string) => (await run<LocalAnalysis[]>("analyses", "readonly", (s) => s.getAll() as IDBRequest<LocalAnalysis[]>))
    .filter((a) => a.userId === userId).sort((a, b) => b.createdAt - a.createdAt),
  async remove(id: string): Promise<void> {
    await run("analyses", "readwrite", (s) => s.delete(id));
    for (const f of await localFeedback.list(id)) await run("feedback", "readwrite", (s) => s.delete(f.id));
  },
  async removeMedia(id: string): Promise<void> {
    const a = await localAnalyses.get(id);
    if (a) await localAnalyses.put({ ...a, media: null });
  },
};

export const localFeedback = {
  list: (analysisId: string) => run<Feedback[]>("feedback", "readonly", (s) => s.index("analysis_id").getAll(analysisId) as IDBRequest<Feedback[]>),
  add: (f: Feedback) => run("feedback", "readwrite", (s) => s.put(f)),
};

export function newLocalAnalysisId(): string {
  const b = new Uint8Array(12);
  crypto.getRandomValues(b);
  return "dev" + Array.from(b).map((x) => x.toString(16).padStart(2, "0")).join("");
}
