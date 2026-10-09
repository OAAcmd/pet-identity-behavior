// Person-pet interaction from detection-box geometry (E13). Mirrors
// src/petperception/vision/interaction.py; a still-image model scored per sampled frame.
import { f32FromB64, type InteractionAssets } from "./assets";

type Box = ArrayLike<number>; // normalized x1, y1, x2, y2

function iou(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}

export function geometry(person: Box, pet: Box, width: number, height: number): number[] {
  const pw = Math.max(person[2] - person[0], 1e-4), ph = Math.max(person[3] - person[1], 1e-4);
  const qw = Math.max(pet[2] - pet[0], 1e-4), qh = Math.max(pet[3] - pet[1], 1e-4);
  const ix = Math.max(0, Math.min(person[2], pet[2]) - Math.max(person[0], pet[0]));
  const iy = Math.max(0, Math.min(person[3], pet[3]) - Math.max(person[1], pet[1]));
  const inter = ix * iy;
  const pcx = (person[0] + person[2]) / 2, pcy = (person[1] + person[3]) / 2;
  const qcx = (pet[0] + pet[2]) / 2, qcy = (pet[1] + pet[3]) / 2;
  const gx = Math.max(0, Math.max(person[0], pet[0]) - Math.min(person[2], pet[2]));
  const gy = Math.max(0, Math.max(person[1], pet[1]) - Math.min(person[3], pet[3]));
  return [
    iou(person, pet), inter / (qw * qh), inter / (pw * ph), (qcx - pcx) / pw, (qcy - pcy) / ph, Math.log((qw * qh) / (pw * ph)),
    Math.log((pw * width) / (ph * height)), Math.log((qw * width) / (qh * height)), Math.hypot(gx, gy) / Math.sqrt(pw * ph),
    person[0] <= qcx && qcx <= person[2] && person[1] <= qcy && qcy <= person[3] ? 1 : 0, pw * ph, qw * qh,
  ];
}

export class Interaction {
  readonly mean: Float32Array;
  readonly scale: Float32Array;
  readonly coef: Float32Array;
  constructor(readonly a: InteractionAssets) {
    this.mean = f32FromB64(a.mean_b64);
    this.scale = f32FromB64(a.scale_b64);
    this.coef = f32FromB64(a.coef_b64);
  }

  score(person: Box, pet: Box, width: number, height: number, species: string): number {
    const x = [...geometry(person, pet, width, height), species === "cat" ? 1 : 0, species === "cat" ? 0 : 1];
    let z = this.a.intercept;
    for (let i = 0; i < x.length; i++) z += ((x[i] - this.mean[i]) / this.scale[i]) * this.coef[i];
    return 1 / (1 + Math.exp(-z));
  }
}
