// EfficientDet-Lite (MediaPipe raw-anchor build) decoding + per-class NMS.
// Mirrors src/petperception/vision/detector.py (EfficientDetDetector).
import { f32FromB64, type DetectorAssets } from "./assets";
import { encodeImage, resizeBilinear, type RGBImage } from "./imageops";
import { runModel, type LoadedModel } from "./runtime";

export interface Detection {
  label: string;
  score: number;
  box: [number, number, number, number]; // normalized x1, y1, x2, y2
}

export function iou(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const ua = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]) + Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

export class Detector {
  readonly anchors: Float32Array;
  constructor(readonly model: LoadedModel, readonly a: DetectorAssets) {
    this.anchors = f32FromB64(a.decoding.anchors_b64);
  }

  async detect(img: RGBImage, classes: string[], scoreThreshold: number, iouThr: number, maxPerClass: number): Promise<Detection[]> {
    const [h, w] = this.a.input_hw;
    const x = encodeImage(resizeBilinear(img, w, h), this.a.norm_mean, this.a.norm_std, this.model.inputs[0].dtype, this.a.input_quant ?? null);
    const outs = await runModel(this.model, [x]);
    const scores = outs[this.a.output_order.indexOf("scores")] as Float32Array;
    const raw = outs[this.a.output_order.indexOf("boxes")] as Float32Array;
    const d = this.a.decoding;
    const n = d.num_boxes, nc = d.num_classes;
    const out: Detection[] = [];
    for (const cls of classes) {
      const ci = this.a.labels.indexOf(cls);
      if (ci < 0) throw new Error(`class '${cls}' not in detector label map`);
      const cand: number[] = [];
      for (let i = 0; i < n; i++) {
        let s = scores[i * nc + ci];
        if (d.sigmoid_score) s = 1 / (1 + Math.exp(-s));
        if (s >= scoreThreshold) cand.push(i);
      }
      if (!cand.length) continue;
      const boxOf = (i: number): [number, number, number, number] => {
        const A = this.anchors; // x_center, y_center, width, height
        const yc = (raw[i * 4] / d.y_scale) * A[i * 4 + 3] + A[i * 4 + 1];
        const xc = (raw[i * 4 + 1] / d.x_scale) * A[i * 4 + 2] + A[i * 4];
        const hh = d.apply_exponential_on_box_size ? Math.exp(raw[i * 4 + 2] / d.h_scale) * A[i * 4 + 3] : (raw[i * 4 + 2] / d.h_scale) * A[i * 4 + 3];
        const ww = d.apply_exponential_on_box_size ? Math.exp(raw[i * 4 + 3] / d.w_scale) * A[i * 4 + 2] : (raw[i * 4 + 3] / d.w_scale) * A[i * 4 + 2];
        const c = (v: number) => Math.min(1, Math.max(0, v));
        return [c(xc - ww / 2), c(yc - hh / 2), c(xc + ww / 2), c(yc + hh / 2)];
      };
      const sc = (i: number) => (d.sigmoid_score ? 1 / (1 + Math.exp(-scores[i * nc + ci])) : scores[i * nc + ci]);
      let order = cand.map((i) => ({ i, s: sc(i), b: boxOf(i) })).sort((p, q) => q.s - p.s || p.i - q.i);
      let kept = 0;
      while (order.length && kept < maxPerClass) {
        const top = order.shift()!;
        out.push({ label: cls, score: top.s, box: top.b });
        kept++;
        order = order.filter((o) => iou(top.b, o.b) <= iouThr);
      }
    }
    out.sort((p, q) => q.score - p.score);
    return out;
  }
}
