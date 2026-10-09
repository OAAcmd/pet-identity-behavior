// Opt-in backup of on-device results to the user's account: result JSON and
// feedback only. Recordings never leave the device. Off by default.
import { api } from "./api";
import { localAnalyses, localFeedback, type LocalAnalysis } from "./localstore";
import type { AnalysisResult, Feedback } from "./types";

const KEY = "pp.syncResults";

export function syncEnabled(): boolean {
  try { return localStorage.getItem(KEY) === "1"; } catch { return false; }
}

export function setSyncEnabled(on: boolean): void {
  try { if (on) localStorage.setItem(KEY, "1"); else localStorage.removeItem(KEY); } catch { /* per-device preference only */ }
}

export interface SyncedRow { id: string; created_at: string; synced_at: string; filename: string; bundle_version: string | null; decision_state: string | null; bytes: number }

/** Upload one result (and its feedback); marks it synced on this device. */
export async function pushResult(a: LocalAnalysis): Promise<void> {
  const feedback = await localFeedback.list(a.id);
  await api.put(`/synced-results/${a.id}`, { filename: a.filename, created_at: new Date(a.createdAt).toISOString(), bundle_version: a.bundle_version,
    result: a.result, feedback });
  const cur = await localAnalyses.get(a.id);
  if (cur) await localAnalyses.put({ ...cur, synced_at: Date.now() });
}

/** Push every result of this user that is not synced yet. Returns how many were sent. */
export async function syncPending(userId: string): Promise<number> {
  let n = 0;
  for (const a of await localAnalyses.all(userId)) {
    if (a.synced_at) continue;
    await pushResult(a);
    n++;
  }
  return n;
}

/** Best-effort push after a change, when sync is on and the device is online. */
export function syncSoon(a: LocalAnalysis, done?: () => void): void {
  if (!syncEnabled() || navigator.onLine === false) return;
  pushResult(a).then(done, () => undefined); // on failure it stays unsynced; retried when the result is opened or by "Back up now"
}

export const listSynced = () => api.get<SyncedRow[]>("/synced-results");

/** Copy backed-up results that are missing on this device (without recordings). */
export async function restoreMissing(userId: string): Promise<number> {
  let n = 0;
  for (const row of await listSynced()) {
    if (await localAnalyses.get(row.id)) continue;
    const full = await api.get<{ result: AnalysisResult; feedback: Feedback[]; filename: string; bundle_version: string | null; created_at: string }>(`/synced-results/${row.id}`);
    const media = full.result.media as { video?: unknown; duration_s?: number | null } | undefined;
    await localAnalyses.put({ id: row.id, userId, createdAt: Date.parse(full.created_at) || Date.now(), filename: full.filename,
      kind: media?.video ? "video" : "audio", subject_mode: "unknown", pet_ids: [], place_id: null, note: "", bundle_version: full.bundle_version ?? "",
      result: full.result, media: null, synced_at: Date.now() });
    for (const f of full.feedback) await localFeedback.add(f);
    n++;
  }
  return n;
}

export async function deleteSynced(id: string): Promise<void> {
  await api.del(`/synced-results/${id}`).catch((e: { status?: number }) => { if (e?.status !== 404) throw e; });
}
