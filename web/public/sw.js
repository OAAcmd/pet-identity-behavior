// Service worker: offline app shell and the on-device runtime (LiteRT.js WASM).
// API responses and media are never cached; model files are managed by the
// app's model store (Cache Storage, SHA-256 verified), not here.
const VERSION = "pet-observer-shell-v3";
const SHELL = ["/", "/manifest.webmanifest", "/icons/icon-192.png", "/icons/icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin || url.pathname.startsWith("/api/")) return; // never cache API or media
  if (url.pathname.startsWith("/assets/") || url.pathname.startsWith("/litert/")) {
    // Hashed build assets and the versioned WASM runtime: cache first (needed offline).
    event.respondWith(caches.open(VERSION).then(async (c) => (await c.match(req)) ?? fetch(req).then((r) => { if (r.ok) c.put(req, r.clone()); return r; })));
    return;
  }
  if (req.mode === "navigate") {
    // Network first so updates arrive; fall back to the cached shell when offline.
    event.respondWith(fetch(req).then((r) => { const copy = r.clone(); caches.open(VERSION).then((c) => c.put("/", copy)); return r; })
      .catch(() => caches.match("/")));
  }
});
