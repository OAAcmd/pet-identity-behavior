// On-device model bundle store.
// * install: download every file of a bundle manifest, verify SHA-256, keep in
//   Cache Storage under a content-addressed key; the new version becomes
//   active only after every file verified ("update recovery": an interrupted
//   or corrupt update leaves the previous version in use);
// * load: read the active version offline, re-verifying each file;
// * old versions are deleted after a successful switch.

export interface BundleFile { path: string; sha256: string; bytes: number; kind: string; resource_id: string | null }
export interface BundleManifest {
  schema: string; bundle_version: string; engine_version: string; release_profile: string; created_at: string;
  runtime: { litertjs_core: string; accelerator: string }; files: BundleFile[]; total_bytes: number; notes: string[];
}

const CACHE = "pp-ondevice-models-v1";
const ACTIVE_KEY = "/__ondevice__/active-manifest.json";

async function sha256(buf: ArrayBuffer): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(d)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

const keyFor = (f: BundleFile) => `/__ondevice__/files/${f.sha256}/${f.path}`;

export function storageAvailable(): boolean {
  return typeof caches !== "undefined" && typeof crypto !== "undefined" && !!crypto.subtle;
}

export async function activeManifest(): Promise<BundleManifest | null> {
  if (!storageAvailable()) return null;
  const c = await caches.open(CACHE);
  const r = await c.match(ACTIVE_KEY);
  return r ? ((await r.json()) as BundleManifest) : null;
}

export async function fetchRemoteManifest(base: string): Promise<BundleManifest> {
  const r = await fetch(`${base}manifest.json`, { cache: "no-store" });
  if (!r.ok) throw new Error(`model manifest unavailable (HTTP ${r.status})`);
  const m = (await r.json()) as BundleManifest;
  if (m.schema !== "petperception.ondevice_bundle/1") throw new Error(`unsupported bundle schema ${m.schema}`);
  return m;
}

export interface InstallProgress { done_bytes: number; total_bytes: number; file: string }

export async function install(base: string, m: BundleManifest, onProgress?: (p: InstallProgress) => void, signal?: AbortSignal): Promise<void> {
  if (!storageAvailable()) throw new Error("this browser cannot store models (Cache Storage / Web Crypto unavailable; a secure https context is required)");
  try { await navigator.storage?.persist?.(); } catch { /* best effort */ }
  const c = await caches.open(CACHE);
  let done = 0;
  for (const f of m.files) {
    if (signal?.aborted) throw new DOMException("cancelled", "AbortError");
    const have = await c.match(keyFor(f));
    if (have) {
      done += f.bytes;
      onProgress?.({ done_bytes: done, total_bytes: m.total_bytes, file: f.path });
      continue;
    }
    const r = await fetch(`${base}${f.path}`, { cache: "no-store", signal });
    if (!r.ok) throw new Error(`${f.path}: HTTP ${r.status}`);
    const buf = await r.arrayBuffer();
    const got = await sha256(buf);
    if (got !== f.sha256) throw new Error(`${f.path}: checksum mismatch (expected ${f.sha256.slice(0, 12)}…, got ${got.slice(0, 12)}…)`);
    await c.put(keyFor(f), new Response(buf, { headers: { "content-type": "application/octet-stream" } }));
    done += f.bytes;
    onProgress?.({ done_bytes: done, total_bytes: m.total_bytes, file: f.path });
  }
  // Switch only after every file verified, then drop files no longer referenced.
  await c.put(ACTIVE_KEY, new Response(JSON.stringify(m), { headers: { "content-type": "application/json" } }));
  const keep = new Set(m.files.map((f) => new URL(keyFor(f), location.href).href));
  for (const req of await c.keys()) {
    if (req.url.includes("/__ondevice__/files/") && !keep.has(req.url)) await c.delete(req);
  }
}

export async function readFile(m: BundleManifest, path: string): Promise<Uint8Array> {
  const f = m.files.find((x) => x.path === path);
  if (!f) throw new Error(`${path} is not part of bundle ${m.bundle_version}`);
  const c = await caches.open(CACHE);
  const r = await c.match(keyFor(f));
  if (!r) throw new Error(`${path} is missing from the device; reinstall the models`);
  const buf = await r.arrayBuffer();
  if ((await sha256(buf)) !== f.sha256) {
    await c.delete(keyFor(f));
    throw new Error(`${path} failed its integrity check on this device; reinstall the models`);
  }
  return new Uint8Array(buf);
}

export async function uninstall(): Promise<void> {
  if (storageAvailable()) await caches.delete(CACHE);
}
