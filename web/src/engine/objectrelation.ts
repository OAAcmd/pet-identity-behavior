// Pet-object relations from detection-box geometry (E15). Mirrors
// src/petperception/vision/object_relation.py; a still-image model scored per sampled frame.
// Only the relations that passed the E15 gate are in the bundle (currently "on": bed, couch, chair, table).
import { f32FromB64, type ObjectRelationAssets } from "./assets";
import { geometry } from "./interaction";

type Box = ArrayLike<number>; // normalized x1, y1, x2, y2

export const GROUPS = ["bed", "couch", "chair", "table", "car"];

export function features(obj: Box, pet: Box, width: number, height: number, species: string, group: string): number[] {
  return [...geometry(obj, pet, width, height), species === "cat" ? 1 : 0, species === "cat" ? 0 : 1, ...GROUPS.map((g) => (g === group ? 1 : 0))];
}

class Head {
  readonly mean: Float32Array;
  readonly scale: Float32Array;
  readonly coef: Float32Array;
  constructor(readonly groups: string[], meanB64: string, scaleB64: string, coefB64: string, readonly intercept: number, readonly threshold: number) {
    this.mean = f32FromB64(meanB64);
    this.scale = f32FromB64(scaleB64);
    this.coef = f32FromB64(coefB64);
  }

  score(x: number[]): number {
    let z = this.intercept;
    for (let i = 0; i < x.length; i++) z += ((x[i] - this.mean[i]) / this.scale[i]) * this.coef[i];
    return 1 / (1 + Math.exp(-z));
  }
}

export class ObjectRelation {
  readonly heads = new Map<string, Head>();
  readonly groupOfClass = new Map<string, string>();
  constructor(readonly a: ObjectRelationAssets) {
    for (const [r, h] of Object.entries(a.heads)) this.heads.set(r, new Head(h.groups, h.mean_b64, h.scale_b64, h.coef_b64, h.intercept, h.threshold));
    for (const [g, c] of Object.entries(a.detector_class)) this.groupOfClass.set(c, g);
  }

  get detectorClasses(): string[] {
    const s = new Set<string>();
    for (const h of this.heads.values()) for (const g of h.groups) s.add(this.a.detector_class[g]);
    return [...s].sort();
  }

  /** Highest score per relation over the detected objects: relation -> [score, object group]. */
  best(objects: { label: string; box: Box }[], pet: Box, width: number, height: number, species: string): Map<string, [number, string]> {
    const out = new Map<string, [number, string]>();
    for (const o of objects) {
      const g = this.groupOfClass.get(o.label);
      if (g === undefined) continue;
      const x = features(o.box, pet, width, height, species, g);
      for (const [r, h] of this.heads) {
        if (!h.groups.includes(g)) continue;
        const s = h.score(x);
        const cur = out.get(r);
        if (!cur || s > cur[0]) out.set(r, [s, g]);
      }
    }
    return out;
  }
}
