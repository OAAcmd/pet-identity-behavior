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

describe("quantized input encoding", () => {
  it("matches the Python reference (tests/test_quantization.py) bit for bit", async () => {
    const { encodeImage, dequantize } = await import("../../src/engine/imageops");
    const img = { width: 3, height: 1, data: new Uint8ClampedArray([0, 1, 2, 127, 128, 129, 200, 254, 255]) };
    expect(Array.from(encodeImage(img as never, 127, 128, "uint8", [0.012566016986966133, 131]))).toEqual([52, 53, 53, 131, 132, 132, 176, 210, 211]);
    expect(Array.from(encodeImage(img as never, 127.5, 127.5, "int8", [0.0078125, -1]))).toEqual([-128, -128, -127, -2, 0, 1, 72, 126, 127]);
    expect(Array.from(dequantize(new Int8Array([-128, 0, 127]), [0.5, -128]))).toEqual([0, 64, 127.5]);
  });
});
