// Device tiers and resource budgets for on-device analysis
// (docs/DEVICE_TIERS_AND_BUDGETS.md). Every limit below is an ASSUMPTION
// until it has been measured on physical phones of that tier; the device
// check reports each one as pass / fail / not_measured, never silently.

export type Tier = "A" | "B" | "C";

export const TIERS: Record<Tier, { name: string; examples: string }> = {
  A: { name: "current flagship", examples: "2023 or newer flagship phones, at least 6 GB RAM (for example iPhone 15, Pixel 8, Galaxy S23)" },
  B: { name: "mid-range (reference tier)", examples: "2021-2023 mid-range phones, 4-8 GB RAM (for example Pixel 6a/7a, Galaxy A54, iPhone 12)" },
  C: { name: "minimum supported", examples: "budget phones with 3-4 GB RAM, 2020-2022 (for example Galaxy A14, Moto G 2022)" },
};

export interface Budget { id: string; label: string; unit: string; limits: Record<Tier, number>; higherIsWorse: true; note: string }

export const BUDGETS: Budget[] = [
  { id: "download_mb", label: "One-time model download (bundle + runtime)", unit: "MB", limits: { A: 50, B: 50, C: 50 }, higherIsWorse: true,
    note: "bundle files plus the LiteRT.js WebAssembly runtime" },
  { id: "storage_mb", label: "Storage used by the app (origin total)", unit: "MB", limits: { A: 120, B: 120, C: 120 }, higherIsWorse: true,
    note: "navigator.storage.estimate(); includes cached app shell and local results" },
  { id: "engine_load_ms", label: "Model load from cache (cold)", unit: "ms", limits: { A: 2000, B: 4000, C: 8000 }, higherIsWorse: true,
    note: "read, verify and compile all models" },
  { id: "realtime_factor", label: "Analysis time / recording duration (audio + video, 2 frames per second)", unit: "x", limits: { A: 0.5, B: 1.0, C: 2.0 },
    higherIsWorse: true, note: "median over the runs; 1.0 means a 60 s clip takes 60 s" },
  { id: "detector_ms", label: "Detector inference, one frame (median)", unit: "ms", limits: { A: 80, B: 160, C: 400 }, higherIsWorse: true,
    note: "EfficientDet-Lite0 320x320, WebAssembly CPU" },
  { id: "memory_mb", label: "Peak memory of the page", unit: "MB", limits: { A: 400, B: 300, C: 250 }, higherIsWorse: true,
    note: "performance.measureUserAgentSpecificMemory() (Chromium, cross-origin isolated); not available in Safari" },
  { id: "sustained_slowdown", label: "Sustained slowdown (last runs / first runs)", unit: "x", limits: { A: 1.25, B: 1.25, C: 1.25 }, higherIsWorse: true,
    note: "proxy for thermal throttling; only measured by the sustained check (10 consecutive analyses)" },
  { id: "battery_pct_per_media_min", label: "Battery used per minute of analysed recording", unit: "%", limits: { A: 1.0, B: 1.5, C: 2.0 }, higherIsWorse: true,
    note: "Battery Status API (Chromium on Android only), 1 % resolution, unplugged phone; sustained check only" },
];

export interface BudgetResult { id: string; label: string; unit: string; value: number | null; limit: number; status: "pass" | "fail" | "not_measured"; assumption: true; note: string }

export function evaluateBudgets(tier: Tier, measured: Record<string, number | null | undefined>): BudgetResult[] {
  return BUDGETS.map((b) => {
    const v = measured[b.id];
    const value = typeof v === "number" && Number.isFinite(v) ? v : null;
    const limit = b.limits[tier];
    return { id: b.id, label: b.label, unit: b.unit, value, limit, status: value === null ? "not_measured" : value <= limit ? "pass" : "fail", assumption: true, note: b.note };
  });
}

/** Tier guess from what the browser exposes; the tester confirms or overrides it. */
export function guessTier(): Tier | null {
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  if (mem === undefined) return null; // Safari exposes neither memory nor a model name
  if (mem >= 8 && navigator.hardwareConcurrency >= 8) return "A";
  if (mem >= 4) return "B";
  return "C";
}
