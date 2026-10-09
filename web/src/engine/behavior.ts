// Temporal behavior segments (mirrors src/petperception/vision/behavior.py).
import type { BehaviorParams } from "./assets";
import { anchorPoint, zoneDistance, type ZoneConfig } from "./zones";

export interface Segment {
  kind: string; t_start_s: number; t_end_s: number; reliability: string; detail: string;
  track_id: string | null; measurements: Record<string, number>; method: string;
}

export interface StepSeries { t0: number[]; t1: number[]; speed: number[]; raw: number[]; pos: [number, number][]; times: number[] }

const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); const n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const f = (v: number, d: number) => v.toFixed(d);

export function stepSeries(times: number[], boxes: ArrayLike<number>[], fw: number, fh: number, camera: Map<number, [number, number]> | null, maxGap: number): StepSeries {
  const diag = Math.hypot(fw, fh), sx = fw / diag, sy = fh / diag;
  const c = boxes.map((b) => [((b[0] + b[2]) / 2) * sx, ((b[1] + b[3]) / 2) * sy] as [number, number]);
  const pos = c.map((p) => [...p] as [number, number]);
  let ax = 0, ay = 0;
  for (let i = 1; i < times.length; i++) {
    const cm = camera?.get(times[i]);
    if (cm) { ax += cm[0] * sx; ay += cm[1] * sy; }
    pos[i] = [c[i][0] - ax, c[i][1] - ay];
  }
  const t0: number[] = [], t1: number[] = [], raw: number[] = [];
  for (let i = 0; i + 1 < times.length; i++) {
    const dt = times[i + 1] - times[i];
    if (dt <= 0 || dt > maxGap) continue;
    t0.push(times[i]); t1.push(times[i + 1]);
    raw.push(Math.hypot(pos[i + 1][0] - pos[i][0], pos[i + 1][1] - pos[i][1]) / dt);
  }
  const speed = raw.length >= 3 ? raw.map((_, k) => median(raw.slice(Math.max(0, k - 1), k + 2))) : [...raw];
  return { t0, t1, speed, raw, pos, times };
}

function runs(mask: boolean[], t0: number[], t1: number[]): [number, number, number[]][] {
  const out: [number, number, number[]][] = [];
  let cur: number[] = [];
  for (let k = 0; k < mask.length; k++) {
    if (mask[k] && cur.length && Math.abs(t0[k] - t1[cur[cur.length - 1]]) > 1e-6) { out.push([t0[cur[0]], t1[cur[cur.length - 1]], cur]); cur = []; }
    if (mask[k]) cur.push(k);
    else if (cur.length) { out.push([t0[cur[0]], t1[cur[cur.length - 1]], cur]); cur = []; }
  }
  if (cur.length) out.push([t0[cur[0]], t1[cur[cur.length - 1]], cur]);
  return out;
}

function mode3(s: number[]): number[] {
  if (s.length < 3) return s;
  const out = [...s];
  for (let k = 1; k < s.length - 1; k++) if (s[k - 1] === s[k + 1] && s[k + 1] !== s[k]) out[k] = s[k - 1];
  return out;
}

export function motionSegments(ss: StepSeries, p: BehaviorParams, tid: string, rel: string): Segment[] {
  if (!ss.speed.length) return [];
  const st = mode3(ss.speed.map((v) => (v >= p.active_speed ? 2 : v >= p.moving_speed ? 1 : 0)));
  const segs: Segment[] = [];
  for (const [a, b, ix] of runs(st.map((v) => v === 0), ss.t0, ss.t1)) {
    if (b - a >= p.min_stationary_s) {
      const m = median(ix.map((k) => ss.speed[k]));
      segs.push({ kind: "stationary", t_start_s: a, t_end_s: b, reliability: rel, detail: `median speed ${f(m, 3)} diag/s for ${f(b - a, 1)} s`, track_id: tid,
        measurements: { median_speed: m, duration_s: b - a }, method: "speed below moving threshold" });
    }
  }
  for (const [a, b, ix] of runs(st.map((v) => v >= 1), ss.t0, ss.t1)) {
    if (b - a >= p.min_moving_s) {
      const path = ix.reduce((s, k) => s + ss.raw[k] * (ss.t1[k] - ss.t0[k]), 0);
      segs.push({ kind: "moving", t_start_s: a, t_end_s: b, reliability: rel, detail: `moved ${f(path, 2)} image diagonals in ${f(b - a, 1)} s`, track_id: tid,
        measurements: { median_speed: median(ix.map((k) => ss.speed[k])), path_diagonals: path, duration_s: b - a }, method: "speed at or above moving threshold" });
    }
  }
  for (const [a, b, ix] of runs(st.map((v) => v === 2), ss.t0, ss.t1)) {
    if (b - a >= p.min_active_s) {
      const m = median(ix.map((k) => ss.speed[k]));
      segs.push({ kind: "active_bout", t_start_s: a, t_end_s: b, reliability: rel, detail: `fast movement (median ${f(m, 2)} diag/s) for ${f(b - a, 1)} s`, track_id: tid,
        measurements: { median_speed: m, duration_s: b - a }, method: "speed at or above active threshold" });
    }
  }
  return segs;
}

function zigzag(x: number[], t: number[], thr: number): [number, number][] {
  const n = x.length;
  if (n < 2) return [];
  const idx: number[] = [];
  let dir = 0, ext = 0, lo = 0, hi = 0;
  for (let i = 1; i < n; i++) {
    if (dir === 0) {
      if (x[i] < x[lo]) lo = i;
      if (x[i] > x[hi]) hi = i;
      if (x[i] - x[lo] >= thr) { idx.push(lo); dir = 1; ext = i; }
      else if (x[hi] - x[i] >= thr) { idx.push(hi); dir = -1; ext = i; }
    } else if (dir === 1) {
      if (x[i] > x[ext]) ext = i;
      else if (x[ext] - x[i] >= thr) { idx.push(ext); dir = -1; ext = i; }
    } else {
      if (x[i] < x[ext]) ext = i;
      else if (x[i] - x[ext] >= thr) { idx.push(ext); dir = 1; ext = i; }
    }
  }
  if (dir !== 0) idx.push(ext);
  return idx.map((i) => [t[i], x[i]]);
}

/** Principal axis of 2-D points (largest-eigenvalue eigenvector of the covariance). */
function principalAxis(pts: [number, number][]): [number, number] {
  const n = pts.length;
  const mx = pts.reduce((s, p) => s + p[0], 0) / n, my = pts.reduce((s, p) => s + p[1], 0) / n;
  let a = 0, b = 0, c = 0;
  for (const [x, y] of pts) { a += (x - mx) ** 2; b += (x - mx) * (y - my); c += (y - my) ** 2; }
  const tr = a + c, det = a * c - b * b;
  const l = tr / 2 + Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  let vx = b, vy = l - a;
  if (Math.abs(vx) + Math.abs(vy) < 1e-12) { vx = a >= c ? 1 : 0; vy = a >= c ? 0 : 1; }
  const L = Math.hypot(vx, vy);
  return [vx / L, vy / L];
}

export function pacingSegments(ss: StepSeries, p: BehaviorParams, tid: string): Segment[] {
  if (ss.pos.length < 2 * p.pacing_min_reversals) return [];
  const n = ss.pos.length;
  const mx = ss.pos.reduce((s, q) => s + q[0], 0) / n, my = ss.pos.reduce((s, q) => s + q[1], 0) / n;
  const [vx, vy] = principalAxis(ss.pos);
  const x = ss.pos.map(([a, b]) => (a - mx) * vx + (b - my) * vy);
  const pts = zigzag(x, ss.times, p.pacing_min_amplitude);
  const legs = pts.slice(0, -1).map((q, i) => [q[0], pts[i + 1][0], Math.abs(pts[i + 1][1] - q[1])] as [number, number, number]);
  const groups: [number, number, number][][] = [];
  let cur: [number, number, number][] = [];
  for (const leg of legs) {
    if (leg[1] - leg[0] <= p.pacing_max_leg_s) cur.push(leg);
    else { groups.push(cur); cur = []; }
  }
  groups.push(cur);
  const segs: Segment[] = [];
  for (const run of groups) {
    const rev = run.length - 1;
    if (rev >= p.pacing_min_reversals && run[run.length - 1][1] - run[0][0] >= p.pacing_min_span_s) {
      const amp = median(run.map((l) => l[2]));
      segs.push({ kind: "pacing_candidate", t_start_s: run[0][0], t_end_s: run[run.length - 1][1], reliability: "low",
        detail: `${rev} direction reversals along one axis, median leg ${f(amp, 2)} image diagonals`, track_id: tid,
        measurements: { reversals: rev, median_leg_diagonals: amp, duration_s: run[run.length - 1][1] - run[0][0] }, method: "zigzag turning points on the principal movement axis" });
    }
  }
  return segs;
}

export function zoneSegments(times: number[], boxes: ArrayLike<number>[], zones: ZoneConfig | null, fw: number, fh: number, p: BehaviorParams,
  maxGap: number, tid: string, rel: string): Segment[] {
  if (!zones || !times.length) return [];
  const diag = Math.hypot(fw, fh), sx = fw / diag, sy = fh / diag;
  const segs: Segment[] = [];
  for (const z of zones.zones) {
    const dist = boxes.map((b) => zoneDistance(anchorPoint(b, zones.anchor), z.polygon, sx, sy));
    const inside = dist.map((d) => d <= 0);
    let k = 0;
    while (k < times.length) {
      if (!inside[k]) { k++; continue; }
      const a = k;
      while (k + 1 < times.length && inside[k + 1] && times[k + 1] - times[k] <= maxGap) k++;
      const b = k;
      const name = `'${z.label}' (${z.semantic_tag})`;
      const fromOut = a > 0 && !inside[a - 1] && times[a] - times[a - 1] <= maxGap;
      const toOut = b + 1 < times.length && !inside[b + 1] && times[b + 1] - times[b] <= maxGap;
      if (fromOut) {
        segs.push({ kind: "enter_zone", t_start_s: times[a], t_end_s: times[a], reliability: rel, detail: `entered ${name}`, track_id: tid, measurements: {}, method: `anchor ${zones.anchor} crossed into the owner polygon` });
        let s = a - 1;
        while (s - 1 >= 0 && times[a] - times[s - 1] <= p.approach_lookback_s && times[s] - times[s - 1] <= maxGap && dist[s - 1] >= dist[s] - 0.01) s--;
        const drop = dist[s];
        if (times[a] - times[s] >= p.approach_min_s && drop >= p.approach_min_drop) {
          segs.push({ kind: "approach_zone", t_start_s: times[s], t_end_s: times[a], reliability: rel, detail: `distance to ${name} fell by ${f(drop, 2)} image diagonals before entering`,
            track_id: tid, measurements: { distance_drop_diagonals: drop, duration_s: times[a] - times[s] }, method: "monotone distance decrease" });
        }
      }
      segs.push({ kind: "dwell_zone", t_start_s: times[a], t_end_s: times[b], reliability: rel, detail: `in ${name} for ${f(times[b] - times[a], 1)} s (sampled)`, track_id: tid,
        measurements: { dwell_s: times[b] - times[a] }, method: "consecutive sampled detections inside the polygon" });
      if (toOut) segs.push({ kind: "leave_zone", t_start_s: times[b + 1], t_end_s: times[b + 1], reliability: rel, detail: `left ${name}`, track_id: tid, measurements: {}, method: `anchor ${zones.anchor} left the owner polygon` });
      k++;
    }
  }
  return segs;
}

export function viewSegments(times: number[], boxes: ArrayLike<number>[], mediaStart: number, mediaEnd: number | null, p: BehaviorParams, tid: string, rel: string): Segment[] {
  const segs: Segment[] = [];
  if (!times.length) return segs;
  const edge = (b: ArrayLike<number>) => Math.min(b[0], b[1], 1 - b[2], 1 - b[3]) <= 0.03;
  if (times[0] - mediaStart > p.view_edge_margin_s) segs.push({ kind: "entered_view", t_start_s: times[0], t_end_s: times[0], reliability: rel,
    detail: "first seen after the recording started" + (edge(boxes[0]) ? " at the frame edge" : ""), track_id: tid, measurements: {}, method: "first sampled detection time" });
  if (mediaEnd !== null && mediaEnd - times[times.length - 1] > p.view_edge_margin_s) segs.push({ kind: "left_view", t_start_s: times[times.length - 1], t_end_s: times[times.length - 1], reliability: rel,
    detail: "not seen again before the recording ended" + (edge(boxes[boxes.length - 1]) ? " (last seen at the frame edge)" : ""), track_id: tid, measurements: {}, method: "last sampled detection time" });
  return segs;
}

export function vocalBouts(events: [string, number, number][], p: BehaviorParams): Segment[] {
  const by = new Map<string, [number, number][]>();
  for (const [l, a, b] of events) { if (!by.has(l)) by.set(l, []); by.get(l)!.push([a, b]); }
  const segs: Segment[] = [];
  for (const [label, ivs] of by) {
    ivs.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
    let cur = [ivs[0]];
    for (const iv of [...ivs.slice(1), [Infinity, Infinity] as [number, number]]) {
      if (iv[0] - cur[cur.length - 1][1] <= p.vocal_max_gap_s) { cur.push(iv); continue; }
      if (cur.length >= 2) {
        const a = cur[0][0], b = cur[cur.length - 1][1];
        segs.push({ kind: "vocal_bout", t_start_s: a, t_end_s: b, reliability: "medium", detail: `${cur.length} ${label.replace(/_/g, " ")} events within ${f(b - a, 1)} s`, track_id: null,
          measurements: { events: cur.length, duration_s: b - a }, method: `sound events merged across gaps <= ${p.vocal_max_gap_s} s` });
      }
      cur = [iv];
    }
  }
  return segs;
}

export function startleCandidates(series: Map<string, StepSeries>, onsets: [string, number][], p: BehaviorParams): Segment[] {
  const segs: Segment[] = [];
  for (const [label, tOn] of onsets) {
    for (const [tid, ss] of series) {
      if (!ss.speed.length) continue;
      const base = ss.raw.filter((_, k) => ss.t1[k] <= tOn && ss.t0[k] >= tOn - p.startle_baseline_s);
      if (!base.length) continue;
      const mid = ss.t0.map((v, k) => (v + ss.t1[k]) / 2);
      const after = mid.map((m, k) => [m, k] as [number, number]).filter(([m]) => m > tOn && m <= tOn + p.startle_max_lag_s).map(([, k]) => k);
      if (!after.length) continue;
      const k = after.reduce((best, i) => (ss.raw[i] > ss.raw[best] ? i : best), after[0]);
      const peak = ss.raw[k], b = median(base);
      if (peak >= p.active_speed && peak >= p.startle_ratio * Math.max(b, p.moving_speed / 2)) {
        segs.push({ kind: "startle_candidate", t_start_s: ss.t0[k], t_end_s: ss.t1[k], reliability: "low",
          detail: `speed rose from ${f(b, 2)} to ${f(peak, 2)} diag/s within ${f(p.startle_max_lag_s, 1)} s after ${label.replace(/_/g, " ")}`, track_id: tid,
          measurements: { baseline_speed: b, peak_speed: peak, lag_s: mid[k] - tOn }, method: "speed ratio after a sound onset; temporal coincidence only" });
      }
    }
  }
  return segs;
}
