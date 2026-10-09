import { describe, expect, it } from "vitest";
import { interactionSegments } from "../../src/engine/behavior";
import { geometry } from "../../src/engine/interaction";

// Same expected values as tests/test_interaction.py (Python reference).
describe("person-pet interaction", () => {
  it("computes the same geometry as the Python reference", () => {
    const close = (a: number[], b: number[]) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 5));
    close(geometry([0.2, 0.1, 0.6, 0.9], [0.3, 0.4, 0.5, 0.7], 640, 480), [0.1875, 1.0, 0.1875, 0.0, 0.0625, -1.673976, -0.405465, -0.117783, 0.0, 1.0, 0.32, 0.06]);
    close(geometry([0.1, 0.1, 0.3, 0.5], [0.6, 0.5, 0.9, 0.9], 640, 480), [0.0, 0.0, 0.0, 2.75, 1.0, 0.405465, -0.405465, 0.0, 1.06066, 0.0, 0.08, 0.12]);
  });

  it("forms segments from consecutive frames with a person, like the Python reference", () => {
    const segs = interactionSegments([0, 0.5, 1, 1.5, 2, 2.5, 3], [0.9, 0.8, null, 0.9, 0.2, 0.9, 0.95], 0.7, 2, 1.5, "t1");
    expect(segs.map((g) => [g.t_start_s, g.t_end_s, g.measurements.frames])).toEqual([[0, 0.5, 2], [2.5, 3, 2]]);
    expect(interactionSegments([0, 5], [0.9, 0.9], 0.7, 2, 1.5, "t1")).toEqual([]);
  });
});
