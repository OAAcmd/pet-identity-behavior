// EfficientNet-Lite0 backbone (patched to expose feature maps), the crop
// verifier (E7, "crop" feature set) and the coarse posture classifier (E10).
// Mirrors src/petperception/vision/verifier.py, keypoints.py and posture.py.
import { f32FromB64, type PostureAssets, type VerifierAssets } from "./assets";
import type { Detection } from "./detector";
import { cropBox, resizeArea, resizeBilinear, squareCrop, toFloat, type RGBImage } from "./imageops";
import { runModel, type LoadedModel } from "./runtime";

export interface BackboneOut {
  softmax: Float32Array; // 1000
  pooled: Float32Array; // 1280 (mean of the 7x7x1280 head)
}

export class Backbone {
  constructor(readonly model: LoadedModel, readonly mean: number, readonly std: number) {}

  async run(crop224: RGBImage): Promise<BackboneOut> {
    const outs = await runModel(this.model, [toFloat(crop224, this.mean, this.std)]);
    const bySize = (n: number) => outs.find((o) => o.length === n) as Float32Array;
    const head = bySize(7 * 7 * 1280);
    const pooled = new Float32Array(1280);
    for (let i = 0; i < 49; i++) for (let c = 0; c < 1280; c++) pooled[c] += head[i * 1280 + c];
    for (let c = 0; c < 1280; c++) pooled[c] /= 49;
    return { softmax: bySize(1000), pooled };
  }
}

export class Verifier {
  readonly masks: Record<string, Uint8Array> = {};
  constructor(readonly backbone: Backbone, readonly a: VerifierAssets) {
    const named = new Uint8Array(1000);
    for (const g of a.groups) this.masks[g] = new Uint8Array(1000);
    for (const [g, [lo, hi]] of Object.entries(a.ranges)) for (let i = lo; i <= hi; i++) { this.masks[g][i] = 1; named[i] = 1; }
    this.masks.teddy[a.teddy_index] = 1;
    for (let i = 0; i < 1000; i++) {
      const animal = i <= a.animal_end;
      this.masks.other_animal[i] = animal && !named[i] ? 1 : 0;
      this.masks.non_animal[i] = !animal && i !== a.teddy_index ? 1 : 0;
    }
  }

  /** Crop with context margin, resized like the Python reference (AREA when shrinking both sides, else LINEAR). */
  crop(img: RGBImage, box: ArrayLike<number>): RGBImage {
    const m = this.a.margin;
    const bw = Math.max(box[2] - box[0], 1e-3), bh = Math.max(box[3] - box[1], 1e-3);
    const c = cropBox(img, [Math.max(0, box[0] - m * bw), Math.max(0, box[1] - m * bh), Math.min(1, box[2] + m * bw), Math.min(1, box[3] + m * bh)]);
    return Math.min(c.width, c.height) > 224 ? resizeArea(c, 224, 224) : resizeBilinear(c, 224, 224);
  }

  async verify(img: RGBImage, det: Detection): Promise<{ score: number; accepted: boolean } | null> {
    const cv = this.a.params.classes[det.label];
    if (!cv) return null;
    if (cv.feature_set !== "crop") throw new Error(`on-device verifier supports the 'crop' feature set only (got ${cv.feature_set})`);
    const { softmax } = await this.backbone.run(this.crop(img, det.box));
    let tot = 0;
    for (const v of softmax) tot += v;
    const eps = this.a.eps;
    const group = (g: string) => {
      let s = 0;
      for (let i = 0; i < 1000; i++) if (this.masks[g][i]) s += softmax[i];
      return s / Math.max(tot, 1e-12);
    };
    let top1 = 0;
    for (const v of softmax) top1 = Math.max(top1, v / Math.max(tot, 1e-12));
    const s = Math.min(Math.max(det.score, eps), 1 - eps);
    const [h, w] = [img.height, img.width];
    const bw = Math.max(det.box[2] - det.box[0], 1e-4), bh = Math.max(det.box[3] - det.box[1], 1e-4);
    const x = [Math.log(s / (1 - s)), Math.log(bw * bh), Math.log((bw * w) / (bh * h)),
      ...this.a.groups.map((g) => Math.log(group(g) + eps)), Math.log(top1 + eps)];
    let z = cv.intercept;
    for (let i = 0; i < x.length; i++) z += ((x[i] - cv.mean[i]) / cv.scale[i]) * cv.coef[i];
    const score = 1 / (1 + Math.exp(-z));
    return { score, accepted: score >= cv.tau };
  }
}

export class Posture {
  readonly mean: Float32Array;
  readonly scale: Float32Array;
  readonly coef: Float32Array;
  constructor(readonly backbone: Backbone, readonly a: PostureAssets) {
    this.mean = f32FromB64(a.mean_b64);
    this.scale = f32FromB64(a.scale_b64);
    this.coef = f32FromB64(a.coef_b64);
  }

  async predict(img: RGBImage, box: ArrayLike<number>, species: string): Promise<{ label: string; score: number }> {
    const { pooled } = await this.backbone.run(squareCrop(img, box).crop);
    const bw = Math.max(box[2] - box[0], 1e-4), bh = Math.max(box[3] - box[1], 1e-4);
    const x = [...pooled, Math.log((bw * img.width) / (bh * img.height)), Math.log(bw * bh), species === "cat" ? 1 : 0, species === "dog" ? 1 : 0];
    const [k, d] = this.a.coef_shape;
    const z = new Array(k).fill(0).map((_, c) => {
      let v = this.a.intercept[c];
      for (let i = 0; i < d; i++) v += ((x[i] - this.mean[i]) / this.scale[i]) * this.coef[c * d + i];
      return v;
    });
    const mx = Math.max(...z);
    const e = z.map((v) => Math.exp(v - mx));
    const tot = e.reduce((p, q) => p + q, 0);
    const p = e.map((v) => v / tot);
    const best = p.indexOf(Math.max(...p));
    return { label: p[best] >= this.a.gate ? this.a.labels[best] : "uncertain", score: p[best] };
  }
}
