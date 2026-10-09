// Minimal IndexedDB wrapper for durable pending uploads (survive reloads / offline).

const DB_NAME = "pet-observer";
const STORE = "pendingUploads";

export interface PendingUpload {
  localId: string;
  userId: string;
  blob: Blob;
  filename: string;
  sha256: string | null;
  uploadId: string | null;
  analysis: {
    subject_mode: string;
    pet_ids: string[];
    place_id: string | null;
    modalities: string[];
    note: string;
  };
  createdAt: number;
  state: "pending" | "uploading" | "paused" | "failed" | "done";
  lastError: string | null;
  analysisId: string | null;
}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE, { keyPath: "localId" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  return new Promise<T>((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const r = fn(t.objectStore(STORE));
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    t.oncomplete = () => db.close();
  });
}

export const pendingStore = {
  available: typeof indexedDB !== "undefined",
  put: (p: PendingUpload) => tx("readwrite", (s) => s.put(p)),
  get: (id: string) => tx<PendingUpload | undefined>("readonly", (s) => s.get(id) as IDBRequest<PendingUpload | undefined>),
  all: () => tx<PendingUpload[]>("readonly", (s) => s.getAll() as IDBRequest<PendingUpload[]>),
  remove: (id: string) => tx("readwrite", (s) => s.delete(id)),
};
