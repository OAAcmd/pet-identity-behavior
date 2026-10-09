// Resumable chunked uploads that survive network loss, tab reloads and app restarts.
//
// Protocol (server): POST /uploads {filename,size_bytes,sha256} -> {id, received_bytes}
//                    PUT  /uploads/{id}?offset=N  (raw chunk) -> {received_bytes, status}
//                    GET  /uploads/{id}           -> current offset (resume point)
// The blob and progress are persisted in IndexedDB, so an interrupted upload
// resumes from the server's offset instead of starting over.

import { ApiError, api, request } from "./api";
import { pendingStore, type PendingUpload } from "./idb";
import type { Analysis, Upload } from "./types";

export const CLIENT_CHUNK_BYTES = 2 * 1024 * 1024;

export interface UploadProgress {
  sent: number;
  total: number;
  phase: "hashing" | "uploading" | "starting" | "done" | "waiting_for_network" | "failed";
  message?: string;
}

export async function sha256Hex(blob: Blob): Promise<string | null> {
  if (!globalThis.crypto?.subtle) return null; // insecure context: skip optional checksum
  const buf = await blob.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

const sleep = (ms: number): Promise<void> => new Promise<void>((r) => setTimeout(r, ms));

export function backoffMs(attempt: number): number {
  return Math.min(30000, 1000 * 2 ** Math.min(attempt, 5));
}

export interface UploadDeps {
  getUpload: (id: string) => Promise<Upload>;
  createUpload: (body: { filename: string; size_bytes: number; content_type: string; sha256: string | null }) => Promise<Upload>;
  putChunk: (id: string, offset: number, chunk: Blob) => Promise<Upload>;
  online: () => boolean;
  wait: (ms: number) => Promise<void>;
}

export const defaultDeps: UploadDeps = {
  getUpload: (id) => api.get<Upload>(`/uploads/${id}`),
  createUpload: (body) => api.post<Upload>("/uploads", body),
  putChunk: (id, offset, chunk) => request<Upload>("PUT", `/uploads/${id}?offset=${offset}`, chunk),
  online: () => (typeof navigator === "undefined" ? true : navigator.onLine !== false),
  wait: sleep,
};

/** Send all bytes of `blob`, resuming from the server offset. Returns the completed upload. */
export async function sendResumable(
  p: Pick<PendingUpload, "blob" | "filename" | "sha256" | "uploadId">,
  onUploadId: (id: string) => Promise<void> | void,
  onProgress: (pr: UploadProgress) => void,
  deps: UploadDeps = defaultDeps,
  maxNetworkRetries = 8,
): Promise<Upload> {
  let up: Upload;
  if (p.uploadId) {
    up = await deps.getUpload(p.uploadId);
  } else {
    up = await deps.createUpload({ filename: p.filename, size_bytes: p.blob.size, content_type: p.blob.type || "application/octet-stream", sha256: p.sha256 });
    await onUploadId(up.id);
  }
  const chunk = Math.min(CLIENT_CHUNK_BYTES, up.max_chunk_bytes || CLIENT_CHUNK_BYTES);
  let offset = up.received_bytes;
  let failures = 0;
  while (up.status === "receiving") {
    onProgress({ sent: offset, total: p.blob.size, phase: "uploading" });
    if (!deps.online()) {
      onProgress({ sent: offset, total: p.blob.size, phase: "waiting_for_network", message: "Offline - the upload will continue when you are back online." });
      await deps.wait(2000);
      continue;
    }
    try {
      up = await deps.putChunk(up.id, offset, p.blob.slice(offset, offset + chunk));
      offset = up.received_bytes;
      failures = 0;
    } catch (e) {
      if (e instanceof ApiError && e.code === "upload_offset_mismatch") {
        // The server already has more (or fewer) bytes than we thought: resume from its offset.
        up = await deps.getUpload(up.id);
        offset = up.received_bytes;
        continue;
      }
      if (e instanceof ApiError && (e.isNetwork || e.status >= 500 || e.status === 429)) {
        failures += 1;
        if (failures > maxNetworkRetries) throw e;
        onProgress({ sent: offset, total: p.blob.size, phase: "waiting_for_network", message: `Connection problem - retrying (${failures}/${maxNetworkRetries})...` });
        await deps.wait(backoffMs(failures));
        try {
          up = await deps.getUpload(up.id);
          offset = up.received_bytes;
        } catch {
          /* still offline: loop and retry */
        }
        continue;
      }
      throw e;
    }
  }
  if (up.status !== "complete") {
    throw new ApiError(409, "upload_not_receiving", `The upload is ${up.status}.`, "Start the upload again.");
  }
  onProgress({ sent: p.blob.size, total: p.blob.size, phase: "starting" });
  return up;
}

/** Persist, upload and start the analysis for a pending item. */
export async function processPending(item: PendingUpload, onProgress: (pr: UploadProgress) => void): Promise<Analysis> {
  item.state = "uploading";
  item.lastError = null;
  await pendingStore.put(item).catch(() => undefined);
  if (item.sha256 === null && item.blob.size <= 150 * 1024 * 1024) {
    onProgress({ sent: 0, total: item.blob.size, phase: "hashing" });
    item.sha256 = await sha256Hex(item.blob);
    await pendingStore.put(item).catch(() => undefined);
  }
  try {
    const up = await sendResumable(item, async (id) => {
      item.uploadId = id;
      await pendingStore.put(item).catch(() => undefined);
    }, onProgress);
    const analysis = await api.post<Analysis>("/analyses", { upload_id: up.id, ...item.analysis });
    item.state = "done";
    item.analysisId = analysis.id;
    await pendingStore.remove(item.localId).catch(() => undefined);
    onProgress({ sent: item.blob.size, total: item.blob.size, phase: "done" });
    return analysis;
  } catch (e) {
    item.state = e instanceof ApiError && e.isNetwork ? "paused" : "failed";
    item.lastError = e instanceof Error ? e.message : String(e);
    if (e instanceof ApiError && ["upload_not_receiving", "checksum_mismatch"].includes(e.code)) item.uploadId = null; // restart cleanly next time
    await pendingStore.put(item).catch(() => undefined);
    onProgress({ sent: 0, total: item.blob.size, phase: "failed", message: item.lastError ?? undefined });
    throw e;
  }
}

export function newLocalId(): string {
  return `local-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
