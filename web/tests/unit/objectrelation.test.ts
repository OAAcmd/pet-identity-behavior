import { describe, expect, it } from "vitest";
import { onFurnitureSegments } from "../../src/engine/behavior";
import { features, ObjectRelation } from "../../src/engine/objectrelation";

const b64 = (xs: number[]) => Buffer.from(new Float32Array(xs).buffer).toString("base64");

// Same expected values as tests/test_object_relation.py (Python reference).
describe("pet on furniture", () => {
  it("computes the same features as the Python reference", () => {
    const close = (a: number[], b: number[]) => { expect(a.length).toBe(b.length); a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 5)); };
    close(features([0.0, 0.3, 1.0, 1.0], [0.3, 0.35, 0.6, 0.6], 640, 480, "cat", "bed"),
      [0.107143, 1.0, 0.107143, -0.05, -0.25, -2.233592, 0.644357, 0.470004, 0.0, 1.0, 0.7, 0.075, 1, 0, 1, 0, 0, 0, 0]);
    close(features([0.0, 0.2, 0.5, 0.7], [0.6, 0.45, 0.95, 0.95], 640, 480, "dog", "couch"),
      [0.0, 0.0, 0.0, 1.05, 0.5, -0.356675, 0.287682, -0.068993, 0.2, 0.0, 0.25, 0.175, 0, 1, 0, 1, 0, 0, 0]);
  });

  it("splits segments on object changes and gaps, like the Python reference", () => {
    const segs = onFurnitureSegments([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
      [[0.9, "couch"], [0.8, "couch"], [0.9, "bed"], [0.95, "bed"], null, [0.9, "chair"], [0.2, "chair"], [0.9, "chair"]], 0.5, 2, 1.5, "t1");
    expect(segs.map((g) => [g.t_start_s, g.t_end_s, g.related_object, g.measurements.frames])).toEqual([[0, 0.5, "couch", 2], [1, 1.5, "bed", 2]]);
    expect(segs.every((g) => g.kind === "on_furniture_candidate" && g.validation === "per_frame_model_public_still_images")).toBe(true);
    expect(onFurnitureSegments([0, 5], [[0.9, "bed"], [0.9, "bed"]], 0.5, 2, 1.5, "t1")).toEqual([]);
    const bridged = onFurnitureSegments([0, 0.5, 1, 1.5, 2, 2.5], [[0.9, "couch"], null, [0.9, "couch"], [0.1, "couch"], [0.9, "couch"], null], 0.5, 2, 1.5, "t1");
    expect(bridged.map((g) => [g.t_start_s, g.t_end_s, g.measurements.frames])).toEqual([[0, 1, 2]]);
  });

  it("maps detector classes to object groups and keeps the best object per relation", () => {
    // A head that scores the "fraction of the pet inside the object" feature only.
    const coef = Array(19).fill(0); coef[1] = 10;
    const rel = new ObjectRelation({ version: "test", groups: ["bed", "couch", "chair", "table", "car"],
      detector_class: { bed: "bed", couch: "couch", chair: "chair", table: "dining table", car: "car" },
      heads: { on: { groups: ["bed", "couch", "chair", "table"], mean_b64: b64(Array(19).fill(0)), scale_b64: b64(Array(19).fill(1)), coef_b64: b64(coef), intercept: -5, threshold: 0.5 } },
      min_object_score: 0.35, min_frames: 2 });
    expect(rel.detectorClasses).toEqual(["bed", "chair", "couch", "dining table"]);
    const best = rel.best([{ label: "couch", box: [0.0, 0.2, 0.5, 0.7] }, { label: "dining table", box: [0.0, 0.3, 1.0, 1.0] }, { label: "tv", box: [0, 0, 1, 1] }],
      [0.3, 0.35, 0.6, 0.6], 640, 480, "cat");
    expect(best.get("on")![1]).toBe("table");
    expect(best.get("on")![0]).toBeCloseTo(1 / (1 + Math.exp(-5)), 6);
  });
});
