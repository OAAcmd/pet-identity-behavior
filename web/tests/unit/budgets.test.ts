import { describe, expect, it } from "vitest";
import { BUDGETS, evaluateBudgets } from "../../src/engine/budgets";

describe("device budgets", () => {
  it("reports every budget, marks missing measurements, and compares against the tier limit", () => {
    const r = evaluateBudgets("B", { realtime_factor: 0.8, detector_ms: 200, memory_mb: null, engine_load_ms: Number.NaN });
    expect(r.map((x) => x.id)).toEqual(BUDGETS.map((b) => b.id));
    const by = Object.fromEntries(r.map((x) => [x.id, x]));
    expect(by.realtime_factor.status).toBe("pass");
    expect(by.detector_ms.status).toBe("fail");
    expect(by.memory_mb.status).toBe("not_measured");
    expect(by.engine_load_ms.status).toBe("not_measured");
    expect(r.every((x) => x.assumption)).toBe(true);
  });

  it("is stricter on speed for higher tiers and stricter on memory for lower tiers", () => {
    for (const b of BUDGETS) {
      if (b.id === "memory_mb") expect(b.limits.C).toBeLessThanOrEqual(b.limits.A);
      else expect(b.limits.A).toBeLessThanOrEqual(b.limits.C);
    }
  });
});

describe("bundle compatibility", () => {
  it("accepts only the asset format major.minor this app supports", async () => {
    const { bundleCompatible, SUPPORTED_BUNDLE_ENGINE } = await import("../../src/engine/modelstore");
    expect(bundleCompatible({ engine_version: `${SUPPORTED_BUNDLE_ENGINE}.7` })).toBe(true);
    expect(bundleCompatible({ engine_version: "9.9.0" })).toBe(false);
  });
});
