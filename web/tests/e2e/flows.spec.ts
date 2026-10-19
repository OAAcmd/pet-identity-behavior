// End-to-end tests: built PWA + REAL backend (real models, temp database).
// Media come from var/e2e (real public photo/audio composites; see docs/TEST_REPORT.md).
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const FIX = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../var/e2e");
const EX = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../var/examples");
const USER = "e2e_owner";
const PW = "e2e-password-123";

async function signIn(page: Page) {
  await page.goto("/");
  const status = await (await page.request.get("/api/v1/auth/status")).json();
  if (!status.signed_in) {
    await page.getByLabel("Username").fill(USER);
    await page.getByLabel("Password").fill(PW);
    await page.getByRole("button", { name: status.registration_open ? "Create account" : "Sign in" }).click();
  }
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Hello");
}

async function ensurePet(page: Page, name: string) {
  const pets = await (await page.request.get("/api/v1/pets")).json();
  if (pets.some((p: { name: string }) => p.name === name)) return;
  await page.goto("/#/pets");
  await page.getByRole("button", { name: "Add a pet" }).click();
  await page.getByLabel("Name").fill(name);
  await page.getByLabel("Cat").check();
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByText(name, { exact: true })).toBeVisible();
}

/** Opt in to server analysis (off by default) for tests of the upload path. */
async function useServerAnalysis(page: Page) {
  await page.addInitScript(() => localStorage.setItem("pp.serverAnalysis", "on"));
}

async function installModels(page: Page) {
  await page.goto("/#/settings");
  await page.getByRole("button", { name: "Install models" }).click();
  await expect(page.getByText(/Installed: model bundle/)).toBeVisible({ timeout: 120_000 });
}

async function waitForResult(page: Page) {
  await expect(page).toHaveURL(/#\/analyses\/[a-f0-9]+/, { timeout: 120_000 });
  await expect(page.getByRole("heading", { name: "What was observed" })).toBeVisible({ timeout: 150_000 });
}

test.describe.configure({ mode: "serial" });

test("first-run registration, accessible pages, no horizontal overflow @layout", async ({ page }) => {
  await signIn(page);
  for (const route of ["/", "/capture", "/history", "/pets", "/places", "/settings", "/about"]) {
    await page.goto(`/#${route}`);
    await expect(page.locator("main h1")).toBeVisible();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow, `horizontal overflow on ${route}`).toBeLessThanOrEqual(1);
    const axe = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
    const serious = axe.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(serious.map((v) => `${v.id}: ${v.help}`), `axe on ${route}`).toEqual([]);
  }
});

test("pet profile and place with a drawn food-bowl zone", async ({ page }) => {
  await signIn(page);
  await ensurePet(page, "Milo");
  await page.goto("/#/places");
  await page.getByRole("button", { name: "Add a place" }).click();
  await page.getByLabel("Name").fill("Kitchen camera");
  const canvas = page.getByRole("application", { name: "Zone drawing area" });
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.1, box.y + box.height * 0.1);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.9, box.y + box.height * 0.95, { steps: 8 });
  await page.mouse.up();
  await expect(page.locator(".zone-rect span", { hasText: "Food bowl" })).toBeVisible();
  await page.getByRole("button", { name: "Save place" }).click();
  await expect(page.getByText("Kitchen camera")).toBeVisible();
  const places = await (await page.request.get("/api/v1/places")).json();
  expect(places[0].zones.zones[0].semantic_tag).toBe("food_area");
});

test("[server analysis] upload an existing clip, see observations and clearly-marked hypotheses, give feedback, delete", async ({ page }) => {
  await useServerAnalysis(page);
  await signIn(page);
  await ensurePet(page, "Milo");
  await page.goto("/#/capture");
  await page.getByRole("tab", { name: "Upload file" }).click();
  await page.locator("#file").setInputFiles(path.join(FIX, "big_cat_clip.mp4"));
  await page.getByLabel("One of my pets").check();
  await page.getByLabel("Milo (cat)").check();
  await page.getByLabel("Place (optional)").selectOption({ label: "Kitchen camera" });
  const puts: string[] = [];
  page.on("request", (r) => { if (r.method() === "PUT" && r.url().includes("/api/v1/uploads/")) puts.push(r.url()); });
  await page.getByRole("button", { name: "Upload and analyze" }).click();
  await waitForResult(page);
  expect(puts.length, "a 6.6 MB file is sent in several resumable chunks").toBeGreaterThanOrEqual(3);
  await expect(page.getByText("You said: Milo")).toBeVisible();
  await expect(page.getByRole("heading", { name: /Cat seen/ }).first()).toBeVisible();
  const exp = page.locator(".exp-card");
  await expect(exp.getByText("Experimental", { exact: true })).toBeVisible();
  await expect(exp).toContainText("not probabilities");
  expect(await exp.innerText()).not.toMatch(/\d\s*%/);
  await expect(page.getByRole("heading", { name: "Timeline" })).toBeVisible();
  // Feedback is stored.
  await page.getByLabel("What was your pet actually doing?").selectOption("eating or waiting for food");
  await page.getByRole("button", { name: "Save feedback" }).click();
  await expect(page.getByText(/Saved feedback \(1\)/)).toBeVisible();
  // Deletion is complete and immediate.
  const id = page.url().split("/").pop()!;
  await page.getByRole("button", { name: "Delete analysis" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Delete" }).click();
  await expect(page).toHaveURL(/#\/history/);
  expect((await page.request.get(`/api/v1/analyses/${id}`)).status()).toBe(404);
});

test("record video with the (fake) camera and analyze it on the device", async ({ page }) => {
  await signIn(page);
  await installModels(page);
  await page.goto("/#/capture");
  await expect(page.getByRole("button", { name: "Start recording" })).toBeEnabled({ timeout: 20_000 });
  await page.getByRole("button", { name: "Start recording" }).click();
  await page.waitForTimeout(4000);
  await page.getByRole("button", { name: "Stop recording" }).click();
  await expect(page.getByRole("heading", { name: "Review" })).toBeVisible();
  await expect(page.locator("text=/recording-.*\\.(webm|mp4)/")).toBeVisible();
  await page.getByRole("button", { name: "Analyze on this device" }).click();
  await expect(page).toHaveURL(/#\/local\/dev[a-f0-9]+/, { timeout: 120_000 });
  await expect(page.getByRole("heading", { name: /Cat seen/ }).first()).toBeVisible();
});

test("record sound only and analyze it on the device", async ({ page }) => {
  await signIn(page);
  await installModels(page);
  await page.goto("/#/capture");
  await page.getByRole("tab", { name: "Sound only" }).click();
  await expect(page.getByRole("button", { name: "Start recording" })).toBeEnabled({ timeout: 20_000 });
  await page.getByRole("button", { name: "Start recording" }).click();
  await page.waitForTimeout(3500);
  await page.getByRole("button", { name: "Stop recording" }).click();
  await page.getByRole("button", { name: "Analyze on this device" }).click();
  await expect(page).toHaveURL(/#\/local\/dev[a-f0-9]+/, { timeout: 120_000 });
  await expect(page.locator(".decision")).toBeVisible();
});

test("[server analysis] an interrupted chunk upload resumes from the server offset", async ({ page }) => {
  await useServerAnalysis(page);
  await signIn(page);
  let aborted = 0;
  await page.route(/\/api\/v1\/uploads\/[a-f0-9]+\?offset=[1-9]/, async (route) => {
    if (aborted === 0) {
      aborted += 1;
      return route.abort("connectionreset");
    }
    return route.continue();
  });
  await page.goto("/#/capture");
  await page.getByRole("tab", { name: "Upload file" }).click();
  await page.locator("#file").setInputFiles(path.join(FIX, "big_cat_clip.mp4"));
  await page.getByRole("button", { name: "Upload and analyze" }).click();
  await waitForResult(page);
  expect(aborted).toBe(1);
});

test("[server analysis] going offline mid-upload pauses and resumes automatically", async ({ page, context }) => {
  await useServerAnalysis(page);
  await signIn(page);
  let tripped = false;
  await page.route(/\/api\/v1\/uploads\/[a-f0-9]+\?offset=[1-9]/, async (route) => {
    if (!tripped) {
      tripped = true;
      await context.setOffline(true);
      return route.abort("internetdisconnected");
    }
    return route.continue();
  });
  await page.goto("/#/capture");
  await page.getByRole("tab", { name: "Upload file" }).click();
  await page.locator("#file").setInputFiles(path.join(FIX, "big_cat_clip.mp4"));
  await page.getByRole("button", { name: "Upload and analyze" }).click();
  await expect(page.getByText(/You are offline/)).toBeVisible();
  await page.waitForTimeout(2500);
  await context.setOffline(false);
  await waitForResult(page);
});

test("[server analysis] a corrupt file produces an actionable error", async ({ page }) => {
  await useServerAnalysis(page);
  await signIn(page);
  await page.goto("/#/capture");
  await page.getByRole("tab", { name: "Upload file" }).click();
  await page.locator("#file").setInputFiles(path.join(FIX, "broken.mp4"));
  await page.getByRole("button", { name: "Upload and analyze" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "could not be read" })).toBeVisible();
  await expect(page.getByRole("alert").filter({ hasText: "different file" })).toBeVisible();
});

test("[server analysis] a long analysis can be cancelled", async ({ page }) => {
  await useServerAnalysis(page);
  await signIn(page);
  await page.goto("/#/capture");
  await page.getByRole("tab", { name: "Upload file" }).click();
  await page.locator("#file").setInputFiles(path.join(EX, "e6_long_64s.mp4"));
  await page.getByRole("button", { name: "Upload and analyze" }).click();
  await expect(page).toHaveURL(/#\/analyses\//, { timeout: 60_000 });
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(page.getByText("This analysis was cancelled.")).toBeVisible({ timeout: 60_000 });
});

// SIMULATED denial: headless Chromium with fake-device flags never shows a real
// permission prompt, so the browser's rejection is injected deterministically.
// Real-device permission behavior is listed as untested in docs/TEST_REPORT.md.
test("denied camera permission (simulated) shows guidance and the upload alternative", async ({ browser }) => {
  const context = await browser.newContext({ baseURL: test.info().project.use.baseURL });
  await context.addInitScript(() => {
    const deny = () => Promise.reject(new DOMException("Permission denied", "NotAllowedError"));
    if (navigator.mediaDevices) navigator.mediaDevices.getUserMedia = deny;
  });
  const page = await context.newPage();
  await signIn(page);
  await page.goto("/#/capture");
  const alert = page.getByRole("alert").filter({ hasText: /blocked/i });
  await expect(alert).toBeVisible();
  await expect(page.getByRole("button", { name: "Upload a file instead" })).toBeVisible();
  await page.getByRole("button", { name: "Upload a file instead" }).click();
  await expect(page.locator("#file")).toBeAttached();
  await context.close();
});

test("on-device analysis of an existing file uploads no media", async ({ page }) => {
  await signIn(page);
  await installModels(page);
  const sent: string[] = [];
  page.on("request", (r) => { if (/\/api\/v1\/(uploads|analyses)/.test(r.url()) && r.method() !== "GET") sent.push(`${r.method()} ${r.url()}`); });
  await page.goto("/#/capture");
  await page.getByRole("tab", { name: "Upload file" }).click();
  await page.locator("#file").setInputFiles(path.join(FIX, "big_cat_clip.webm"));
  await expect(page.getByText("Analysed on this device; the recording is not uploaded.")).toBeVisible();
  await page.getByRole("button", { name: "Analyze on this device" }).click();
  await expect(page).toHaveURL(/#\/local\/dev[a-f0-9]+/, { timeout: 120_000 });
  await expect(page.getByText(/Analysed on this device\. The recording and the result stay on this device/)).toBeVisible();
  await expect(page.getByRole("heading", { name: /Cat seen/ }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Timeline" })).toBeVisible();
  await page.getByLabel("What was your pet actually doing?").selectOption("resting or sleeping");
  await page.getByRole("button", { name: "Save feedback" }).click();
  await expect(page.getByText(/Saved feedback \(1\)/)).toBeVisible();
  expect(sent, "no upload or server-analysis request").toEqual([]);
  await page.goto("/#/history");
  await expect(page.getByRole("list", { name: "Analysed on this device" }).getByText("big_cat_clip.webm")).toBeVisible();
});

test("opt-in result backup sends result JSON only (never the recording) and restores on a wiped device", async ({ page }) => {
  await signIn(page);
  await installModels(page);
  await page.getByLabel("Back up on-device results to my account (off by default)").check();
  const sent: { url: string; method: string; bytes: number; type: string }[] = [];
  page.on("request", (r) => {
    if (/\/api\/v1\/(uploads|analyses|synced-results)/.test(r.url()) && r.method() !== "GET")
      sent.push({ url: r.url(), method: r.method(), bytes: r.postDataBuffer()?.length ?? 0, type: r.headers()["content-type"] ?? "" });
  });
  await page.goto("/#/capture");
  await page.getByRole("tab", { name: "Upload file" }).click();
  await page.locator("#file").setInputFiles(path.join(FIX, "big_cat_clip.webm"));
  await page.getByRole("button", { name: "Analyze on this device" }).click();
  await expect(page).toHaveURL(/#\/local\/dev[a-f0-9]+/, { timeout: 120_000 });
  const id = page.url().split("/local/")[1];
  await expect(page.getByText(/is backed up to your account/)).toBeVisible({ timeout: 20_000 });
  const clipBytes = fs.statSync(path.join(FIX, "big_cat_clip.webm")).size;
  expect(sent.filter((r) => /uploads|analyses/.test(r.url)), "no media upload or server analysis").toEqual([]);
  const puts = sent.filter((r) => r.url.endsWith(`/synced-results/${id}`) && r.method === "PUT");
  expect(puts.length).toBeGreaterThan(0);
  for (const r of puts) expect(r.type.startsWith("application/json") && r.bytes < clipBytes / 4, JSON.stringify(r)).toBe(true);
  const backed = await (await page.request.get("/api/v1/synced-results")).json();
  expect(backed.some((x: { id: string }) => x.id === id)).toBe(true);
  // A wiped device (or a new phone) restores the results, without recordings.
  await page.evaluate(() => new Promise<void>((res) => { const r = indexedDB.deleteDatabase("pet-observer-local"); r.onsuccess = r.onerror = r.onblocked = () => res(); }));
  await page.goto("/#/settings");
  await page.getByRole("button", { name: "Restore backed-up results here" }).click();
  await expect(page.getByText(/Restored [1-9]\d* result\(s\) to this device/)).toBeVisible();
  await page.goto(`/#/local/${id}`);
  await expect(page.getByText(/The recording has been removed; the result remains/)).toBeVisible();
  // Deleting the analysis deletes its backup too.
  await page.getByRole("button", { name: "Delete analysis" }).click();
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page).toHaveURL(/#\/history/);
  const after = await (await page.request.get("/api/v1/synced-results")).json();
  expect(after.some((x: { id: string }) => x.id === id)).toBe(false);
  await page.goto("/#/settings");
  await page.getByLabel("Back up on-device results to my account (off by default)").uncheck();
});

test("offline end-to-end: after installing models, analysis works with the network off (desktop Chromium)", async ({ page, context }) => {
  await signIn(page);
  await installModels(page);
  // Let the service worker take control and cache the app shell and runtime.
  await page.evaluate(async () => { await navigator.serviceWorker.ready; });
  await page.reload();
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await page.goto("/#/capture");
  await page.getByRole("tab", { name: "Upload file" }).click();
  // Playwright's offline switch is not guaranteed to cover service-worker fetches, so assert
  // directly that everything offline analysis needs is already in Cache Storage.
  const cached = await page.evaluate(async () => {
    const paths: string[] = [];
    for (const k of await caches.keys()) for (const r of await (await caches.open(k)).keys()) paths.push(new URL(r.url).pathname);
    return paths;
  });
  expect(cached.some((p) => /^\/litert\/litert_wasm.*\.wasm$/.test(p)), `runtime cached: ${cached.join(", ")}`).toBe(true);
  expect(cached.some((p) => /^\/litert\/litert_wasm.*\.js$/.test(p))).toBe(true);
  await context.setOffline(true);
  await page.reload();
  await expect(page.getByText(/You are offline/)).toBeVisible();
  await page.goto("/#/capture");
  await page.getByRole("tab", { name: "Upload file" }).click();
  await page.locator("#file").setInputFiles(path.join(FIX, "big_cat_clip.webm"));
  await page.getByRole("button", { name: "Analyze on this device" }).click();
  await expect(page).toHaveURL(/#\/local\/dev[a-f0-9]+/, { timeout: 120_000 });
  await expect(page.getByRole("heading", { name: /Cat seen/ }).first()).toBeVisible();
  await context.setOffline(false);
});

test("a corrupt model update is rejected and the installed version keeps working; incompatible bundles are refused", async ({ page }) => {
  await signIn(page);
  await installModels(page);
  const installed = (await page.getByText(/Installed: model bundle/).innerText()).match(/model bundle (\w+)/)![1];
  // Serve a "new" bundle whose detector bytes do not match the manifest checksum.
  const real = JSON.parse(fs.readFileSync(path.join(FIX, "../ondevice/bundle/manifest.json"), "utf8"));
  // Whichever detector build the bundle ships (MediaPipe or TF Hub raw decoding), tamper with it.
  const detector = real.files.find((f: { path: string }) => /^models\/efficientdet.*\.tflite$/.test(f.path));
  expect(detector, "bundle has a detector model").toBeTruthy();
  const bad = { ...real, bundle_version: "ffffffffffffffff", files: real.files.map((f: { path: string; sha256: string }) =>
    f.path === detector.path ? { ...f, path: "models/tampered.tflite" } : f) };
  await page.route("**/ondevice/manifest.json", (r) => r.fulfill({ json: bad }));
  await page.route("**/ondevice/models/tampered.tflite", (r) => r.fulfill({ body: Buffer.from("not a model") }));
  await page.reload();
  await page.getByRole("button", { name: "Update models" }).click();
  await expect(page.getByRole("alert").filter({ hasText: /checksum mismatch/ })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(/previous version is still in use/)).toBeVisible();
  await expect(page.getByText(/Installed: model bundle/)).toContainText(`model bundle ${installed} `);
  // A bundle in a format this app does not understand is refused before any download.
  await page.unroute("**/ondevice/manifest.json");
  await page.route("**/ondevice/manifest.json", (r) => r.fulfill({ json: { ...real, bundle_version: "eeeeeeeeeeeeeeee", engine_version: "9.0.0" } }));
  await page.reload();
  await expect(page.getByText(/needs a newer app version/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Update models" })).toHaveCount(0);
  await expect(page.getByText(/Installed: model bundle/)).toContainText(`model bundle ${installed} `);
  await page.unroute("**/ondevice/manifest.json");
  // The installed version still analyses.
  await page.goto("/#/capture");
  await page.getByRole("tab", { name: "Upload file" }).click();
  await page.locator("#file").setInputFiles(path.join(FIX, "big_cat_clip.webm"));
  await page.getByRole("button", { name: "Analyze on this device" }).click();
  await expect(page).toHaveURL(/#\/local\/dev[a-f0-9]+/, { timeout: 120_000 });
});

test("device check passes in desktop Chromium (not a phone)", async ({ page }) => {
  await signIn(page);
  await installModels(page);
  await page.goto("/#/device-check");
  await page.getByRole("button", { name: "Run device check" }).click();
  await expect(page.getByRole("heading", { name: /^(Passed|Not passed)$/ })).toBeVisible({ timeout: 240_000 });
  const report = JSON.parse(await page.getByLabel("Device check report").innerText());
  report.test_environment = "AUTOMATED: Playwright headless Chromium on a 4-core cloud VM with Pixel 7 emulation (user agent, viewport and touch only). NOT a physical phone.";
  fs.mkdirSync(path.join(FIX, "../../results/ondevice"), { recursive: true });
  fs.writeFileSync(path.join(FIX, "../../results/ondevice/device_check_desktop_chromium.json"), JSON.stringify(report, null, 1));
  expect(report.passed, JSON.stringify(report.end_to_end ?? report.error)).toBe(true);
});
