// Species merging, detection flags, Kalman/IoU trackers and movement summary.
// Mirrors src/petperception/vision/tracking.py and quality.py. Camera motion
// differs: the browser uses coarse-to-fine block matching of background
// pixels (translation only) instead of OpenCV ORB + RANSAC.
import { iou, type Detection } from "./detector";

export interface AnimalDetection {
  t_s: number;
  frame_index: number;
  box: [number, number, number, number];
  species_scores: Record<string, number>;
  flags: string[];
  source: Detection;
  verify_score: number | null;
  posture: { label: string; score: number } | null;
  interaction: number | null; // max person-pet interaction score in this frame (E13); null without a person
}

export const species = (d: AnimalDetection) => Object.entries(d.species_scores).sort((a, b) => b[1] - a[1])[0][0];
export const score = (d: AnimalDetection) => Math.max(...Object.values(d.species_scores));
export const area = (b: ArrayLike<number>) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
export const center = (b: ArrayLike<number>): [number, number] => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];

export function mergeSpecies(dets: Detection[], t: number, frameIndex: number, iouThr = 0.7): AnimalDetection[] {
  const out: AnimalDetection[] = [];
  for (const d of [...dets].sort((a, b) => b.score - a.score)) {
    const hit = out.find((a) => iou(a.box, d.box) >= iouThr);
    if (hit) hit.species_scores[d.label] = Math.max(hit.species_scores[d.label] ?? 0, d.score);
    else out.push({ t_s: t, frame_index: frameIndex, box: [...d.box], species_scores: { [d.label]: d.score }, flags: [], source: d, verify_score: null, posture: null, interaction: null });
  }
  for (const a of out) {
    const s = Object.values(a.species_scores).sort((x, y) => y - x);
    if (s.length > 1 && s[1] >= 0.6 * s[0]) a.flags.push("species_ambiguous");
  }
  return out;
}

export function detectionFlags(dets: AnimalDetection[], tinyArea: number, border = 0.01): void {
  dets.forEach((d, i) => {
    if (area(d.box) < tinyArea) d.flags.push("tiny_subject");
    const b = d.box;
    if (b[0] <= border || b[1] <= border || b[2] >= 1 - border || b[3] >= 1 - border) d.flags.push("truncated_at_frame_edge");
    if (dets.length > 1 && dets.some((o, j) => j !== i && iou(d.box, o.box) > 0.05)) d.flags.push("overlaps_other_animal_possible_occlusion");
  });
}

export interface Track {
  track_id: string;
  detections: AnimalDetection[];
  missed_frames: number;
  verification: { checked: number; median: number; tau: number; accepted: boolean } | null;
}

export const firstT = (t: Track) => t.detections[0].t_s;
export const lastT = (t: Track) => t.detections[t.detections.length - 1].t_s;

export function speciesSummary(t: Track): Record<string, { mean: number; max: number; frames_top: number }> {
  const out: Record<string, { mean: number; max: number; frames_top: number }> = {};
  const sps = new Set(t.detections.flatMap((d) => Object.keys(d.species_scores)));
  for (const sp of sps) {
    const vals = t.detections.map((d) => d.species_scores[sp] ?? 0);
    out[sp] = { mean: vals.reduce((a, b) => a + b, 0) / vals.length, max: Math.max(...vals), frames_top: t.detections.filter((d) => species(d) === sp).length };
  }
  return out;
}

export function trackSpecies(t: Track): string {
  const s = speciesSummary(t);
  return Object.keys(s).sort((a, b) => s[b].mean - s[a].mean)[0];
}

/** Hungarian assignment (minimisation) for small rectangular matrices. */
export function hungarian(cost: number[][]): [number, number][] {
  const n = cost.length, m = n ? cost[0].length : 0;
  if (!n || !m) return [];
  const N = Math.max(n, m);
  const big = 1e9;
  const a: number[][] = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => (i < n && j < m ? cost[i][j] : big / 10)));
  const u = new Array(N + 1).fill(0), v = new Array(N + 1).fill(0), p = new Array(N + 1).fill(0), way = new Array(N + 1).fill(0);
  for (let i = 1; i <= N; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(N + 1).fill(Infinity), used = new Array(N + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity, j1 = 0;
      for (let j = 1; j <= N; j++) {
        if (used[j]) continue;
        const cur = a[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= N; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  const out: [number, number][] = [];
  for (let j = 1; j <= N; j++) if (p[j] && p[j] - 1 < n && j - 1 < m) out.push([p[j] - 1, j - 1]);
  return out.sort((x, y) => x[0] - y[0]);
}

interface Tracker {
  update(t: number, dets: AnimalDetection[], shift: [number, number]): void;
  tracks(minHits: number): Track[];
}

export class IouTracker implements Tracker {
  active: Track[] = [];
  finished: Track[] = [];
  next = 1;
  constructor(readonly iouThr = 0.3, readonly maxGap = 1.5) {}

  update(t: number, dets: AnimalDetection[], shift: [number, number] = [0, 0]): void {
    const still: Track[] = [];
    for (const tr of this.active) (t - lastT(tr) <= this.maxGap ? still : this.finished).push(tr);
    this.active = still;
    const usedD = new Set<number>();
    if (this.active.length && dets.length) {
      const prev = this.active.map((tr) => { const b = tr.detections[tr.detections.length - 1].box; return [b[0] + shift[0], b[1] + shift[1], b[2] + shift[0], b[3] + shift[1]]; });
      const cost = prev.map((p) => dets.map((d) => 1 - iou(p, d.box)));
      const usedT = new Set<number>();
      for (const [r, c] of hungarian(cost)) {
        if (1 - cost[r][c] >= this.iouThr) {
          this.active[r].detections.push(dets[c]);
          usedT.add(r);
          usedD.add(c);
        }
      }
      this.active.forEach((tr, r) => { if (!usedT.has(r)) tr.missed_frames++; });
    } else this.active.forEach((tr) => tr.missed_frames++);
    dets.forEach((d, i) => { if (!usedD.has(i)) this.active.push({ track_id: `track-${this.next++}`, detections: [d], missed_frames: 0, verification: null }); });
  }

  tracks(minHits: number): Track[] {
    return [...this.finished, ...this.active].filter((t) => t.detections.length >= minHits).sort((a, b) => firstT(a) - firstT(b));
  }
}

type Mat = number[][];
const matMul = (A: Mat, B: Mat): Mat => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const transpose = (A: Mat): Mat => A[0].map((_, j) => A.map((r) => r[j]));
const eye = (n: number): Mat => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

function inv4(m: Mat): Mat {
  const n = 4;
  const a = m.map((r, i) => [...r, ...eye(n)[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    [a[c], a[p]] = [a[p], a[c]];
    const d = a[c][c];
    for (let j = 0; j < 2 * n; j++) a[c][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = a[r][c];
      for (let j = 0; j < 2 * n; j++) a[r][j] -= f * a[c][j];
    }
  }
  return a.map((r) => r.slice(n));
}

/** SORT-style constant-velocity Kalman tracker (mirrors KalmanTracker in tracking.py). */
export class KalmanTracker implements Tracker {
  active: Track[] = [];
  finished: Track[] = [];
  next = 1;
  kf = new Map<string, { x: number[]; P: Mat; t: number }>();
  H: Mat = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]];
  constructor(readonly iouThr = 0.3, readonly maxGap = 1.5, readonly centerGate = 1.0, readonly accelStd = 0.5, readonly measStd = 0.02) {}

  static z(b: ArrayLike<number>): number[] { return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]]; }
  static box(x: number[]): [number, number, number, number] {
    const w = Math.max(x[2], 1e-4), h = Math.max(x[3], 1e-4);
    return [x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2];
  }

  predict(tid: string, t: number): { x: number[]; P: Mat } {
    const s = this.kf.get(tid)!;
    const dt = Math.max(t - s.t, 0);
    const F = eye(6);
    F[0][4] = F[1][5] = dt;
    const q = this.accelStd ** 2;
    const G: Mat = [[dt * dt / 2, 0], [0, dt * dt / 2], [0, 0], [0, 0], [dt, 0], [0, dt]];
    const Q = matMul(G, transpose(G)).map((r) => r.map((v) => v * q));
    Q[2][2] += (0.05 * dt) ** 2;
    Q[3][3] += (0.05 * dt) ** 2;
    const x = F.map((r) => r.reduce((a, v, k) => a + v * s.x[k], 0));
    const P = matMul(matMul(F, s.P), transpose(F)).map((r, i) => r.map((v, j) => v + Q[i][j]));
    return { x, P };
  }

  init(tid: string, d: AnimalDetection): void {
    const P = eye(6).map((r, i) => r.map((v) => v * (i < 4 ? this.measStd ** 2 : 0.25)));
    this.kf.set(tid, { x: [...KalmanTracker.z(d.box), 0, 0], P, t: d.t_s });
  }

  correct(tid: string, x: number[], P: Mat, d: AnimalDetection): void {
    const H = this.H;
    const zz = KalmanTracker.z(d.box);
    const y = zz.map((v, i) => v - H[i].reduce((a, h, k) => a + h * x[k], 0));
    const S = matMul(matMul(H, P), transpose(H)).map((r, i) => r.map((v, j) => v + (i === j ? this.measStd ** 2 : 0)));
    const K = matMul(matMul(P, transpose(H)), inv4(S));
    const xn = x.map((v, i) => v + K[i].reduce((a, k, j) => a + k * y[j], 0));
    const KH = matMul(K, H);
    const Pn = matMul(eye(6).map((r, i) => r.map((v, j) => v - KH[i][j])), P);
    this.kf.set(tid, { x: xn, P: Pn, t: d.t_s });
  }

  update(t: number, dets: AnimalDetection[], shift: [number, number] = [0, 0]): void {
    const still: Track[] = [];
    for (const tr of this.active) (t - lastT(tr) <= this.maxGap ? still : this.finished).push(tr);
    this.active = still;
    const preds = new Map<string, { x: number[]; P: Mat }>();
    for (const tr of this.active) {
      const p = this.predict(tr.track_id, t);
      p.x[0] += shift[0];
      p.x[1] += shift[1];
      preds.set(tr.track_id, p);
    }
    const usedD = new Set<number>();
    if (this.active.length && dets.length) {
      const pb = this.active.map((tr) => KalmanTracker.box(preds.get(tr.track_id)!.x));
      const cost: number[][] = [];
      const ok: boolean[][] = [];
      pb.forEach((b, r) => {
        const size = Math.sqrt(Math.max((b[2] - b[0]) * (b[3] - b[1]), 1e-6));
        const pc = center(b);
        cost.push([]);
        ok.push([]);
        dets.forEach((d) => {
          const v = iou(b, d.box);
          const dc = center(d.box);
          const dist = Math.hypot(pc[0] - dc[0], pc[1] - dc[1]) / size;
          const weak = v < this.iouThr && dist <= this.centerGate;
          const good = v >= this.iouThr || weak;
          ok[r].push(good);
          cost[r].push(!good ? 10 : weak ? 1 - this.iouThr + 0.01 + 0.01 * dist : 1 - v);
        });
      });
      const usedT = new Set<number>();
      for (const [r, c] of hungarian(cost)) {
        if (ok[r][c]) {
          const tr = this.active[r];
          tr.detections.push(dets[c]);
          const p = preds.get(tr.track_id)!;
          this.correct(tr.track_id, p.x, p.P, dets[c]);
          usedT.add(r);
          usedD.add(c);
        }
      }
      this.active.forEach((tr, r) => { if (!usedT.has(r)) tr.missed_frames++; });
    } else this.active.forEach((tr) => tr.missed_frames++);
    dets.forEach((d, i) => {
      if (usedD.has(i)) return;
      const tid = `track-${this.next++}`;
      this.active.push({ track_id: tid, detections: [d], missed_frames: 0, verification: null });
      this.init(tid, d);
    });
  }

  tracks(minHits: number): Track[] {
    return [...this.finished, ...this.active].filter((t) => t.detections.length >= minHits).sort((a, b) => firstT(a) - firstT(b));
  }
}

export function makeTracker(method: string, iouThr: number, maxGap: number): Tracker {
  if (method === "kalman") return new KalmanTracker(iouThr, maxGap);
  if (method === "iou") return new IouTracker(iouThr, maxGap);
  throw new Error(`unknown tracker method '${method}'`);
}

export function movementSummary(track: Track, fw: number, fh: number, camera: Map<number, [number, number]> | null, movingThreshold = 0.05): Record<string, unknown> {
  const dets = track.detections;
  const diag = Math.hypot(fw, fh);
  if (dets.length < 2) return { available: false, reason: "fewer than two detections", units: "image_diagonals_per_second" };
  const speeds: number[] = [], rel: number[] = [];
  let path = 0, compensated = 0;
  for (let i = 0; i + 1 < dets.length; i++) {
    const a = dets[i], b = dets[i + 1];
    const dt = b.t_s - a.t_s;
    if (dt <= 0) continue;
    const ca = center(a.box), cb = center(b.box);
    let dx = cb[0] - ca[0], dy = cb[1] - ca[1];
    if (camera && camera.has(b.t_s)) {
      const [cx, cy] = camera.get(b.t_s)!;
      dx -= cx;
      dy -= cy;
      compensated++;
    }
    const dist = Math.hypot(dx * fw, dy * fh) / diag;
    path += dist;
    speeds.push(dist / dt);
    const sizePx = Math.sqrt(Math.max(area(a.box), 1e-9) * fw * fh);
    rel.push(Math.hypot(dx * fw, dy * fh) / sizePx / dt);
  }
  if (!speeds.length) return { available: false, reason: "non-increasing timestamps", units: "image_diagonals_per_second" };
  const sorted = [...speeds].sort((x, y) => x - y);
  const med = (xs: number[]) => { const s = [...xs].sort((x, y) => x - y); const n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
  const pct = (p: number) => { const r = (p / 100) * (sorted.length - 1); const lo = Math.floor(r), hi = Math.ceil(r); return sorted[lo] + (sorted[hi] - sorted[lo]) * (r - lo); };
  return {
    available: true, units: "image_diagonals_per_second", median_speed: med(speeds), p90_speed: pct(90), path_length_diagonals: path,
    moving_fraction: speeds.filter((s) => s > movingThreshold).length / speeds.length, moving_threshold: movingThreshold,
    median_speed_box_units_per_second: med(rel), box_area_change_ratio: area(dets[dets.length - 1].box) / Math.max(area(dets[0].box), 1e-9),
    camera_compensated_steps: compensated, steps: speeds.length,
  };
}

/** Background translation prev->cur (normalized), coarse-to-fine block matching; null when unreliable. */
export function estimateCameraMotion(prev: Uint8Array, cur: Uint8Array, w: number, h: number, masks: ArrayLike<number>[]):
  { dx: number; dy: number; reliable: boolean } | null {
  const pyr = (g: Uint8Array, gw: number, gh: number, tw: number) => {
    const f = gw / tw;
    const th = Math.max(8, Math.round(gh / f));
    const out = new Float32Array(tw * th);
    for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) {
      let s = 0, n = 0;
      for (let yy = Math.floor(y * f); yy < Math.min(gh, Math.floor((y + 1) * f)); yy++)
        for (let xx = Math.floor(x * f); xx < Math.min(gw, Math.floor((x + 1) * f)); xx++) { s += g[yy * gw + xx]; n++; }
      out[y * tw + x] = n ? s / n : 0;
    }
    return { d: out, w: tw, h: th };
  };
  const maskOf = (mw: number, mh: number) => {
    const m = new Uint8Array(mw * mh);
    for (const b of masks) {
      for (let y = Math.max(0, Math.floor(b[1] * mh)); y < Math.min(mh, Math.ceil(b[3] * mh)); y++)
        for (let x = Math.max(0, Math.floor(b[0] * mw)); x < Math.min(mw, Math.ceil(b[2] * mw)); x++) m[y * mw + x] = 1;
    }
    return m;
  };
  const sad = (a: Float32Array, b: Float32Array, mw: number, mh: number, m: Uint8Array, dx: number, dy: number) => {
    let s = 0, n = 0;
    for (let y = Math.max(0, -dy); y < Math.min(mh, mh - dy); y++) for (let x = Math.max(0, -dx); x < Math.min(mw, mw - dx); x++) {
      if (m[y * mw + x] || m[(y + dy) * mw + x + dx]) continue;
      s += Math.abs(a[y * mw + x] - b[(y + dy) * mw + x + dx]);
      n++;
    }
    return n > (mw * mh) / 4 ? s / n : Infinity;
  };
  const c1 = pyr(prev, w, h, 80), d1 = pyr(cur, w, h, 80);
  const m1 = maskOf(c1.w, c1.h);
  let best = { dx: 0, dy: 0, s: Infinity };
  const all: number[] = [];
  for (let dy = -6; dy <= 6; dy++) for (let dx = -10; dx <= 10; dx++) {
    const s = sad(c1.d, d1.d, c1.w, c1.h, m1, dx, dy);
    if (Number.isFinite(s)) all.push(s);
    if (s < best.s) best = { dx, dy, s };
  }
  if (!Number.isFinite(best.s) || all.length < 10) return null;
  const sortedS = [...all].sort((a, b) => a - b);
  const medS = sortedS[Math.floor(sortedS.length / 2)];
  const c0 = pyr(prev, w, h, 160), d0 = pyr(cur, w, h, 160);
  const m0 = maskOf(c0.w, c0.h);
  let fine = { dx: best.dx * 2, dy: best.dy * 2, s: Infinity };
  for (let dy = best.dy * 2 - 2; dy <= best.dy * 2 + 2; dy++) for (let dx = best.dx * 2 - 2; dx <= best.dx * 2 + 2; dx++) {
    const s = sad(c0.d, d0.d, c0.w, c0.h, m0, dx, dy);
    if (s < fine.s) fine = { dx, dy, s };
  }
  // Texture check: a flat or dark scene gives no reliable minimum.
  const reliable = medS > 2 && best.s < 0.7 * medS;
  return { dx: fine.dx / c0.w, dy: fine.dy / c0.h, reliable };
}
