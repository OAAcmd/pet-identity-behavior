// YAMNet windowing and sound-event extraction.
// Mirrors src/petperception/audio/yamnet.py (score_waveform16k) and
// audio/events.py (extract_events). Scores are raw sigmoid outputs.
import type { AudioAssets } from "./assets";
import { runModel, type LoadedModel } from "./runtime";

export interface WindowScores {
  tStart: Float64Array;
  tEnd: Float64Array;
  scores: Float32Array[]; // [n][521]
  padded: number;
}

export interface SoundEventRaw {
  label: string; // cat_vocalization | dog_vocalization | growl | context:<tag>
  start_s: number;
  end_s: number;
  peak_score: number;
  mean_score: number;
  n_windows: number;
  threshold: number;
  subtypes: Record<string, number>;
  top_audioset_labels: [string, number][];
}

export async function scoreWaveform16k(model: LoadedModel, y: Float32Array, startS: number, hopS: number, a: AudioAssets,
  onWindow?: (k: number, n: number) => void): Promise<WindowScores> {
  const W = a.window_samples;
  const hop = Math.round(hopS * a.sample_rate);
  let nWin: number;
  let padded = 0;
  let buf = y;
  if (y.length < W) {
    buf = new Float32Array(W);
    buf.set(y);
    padded = 1;
    nWin = 1;
  } else {
    nWin = 1 + Math.ceil(Math.max(0, y.length - W) / hop);
    const need = (nWin - 1) * hop + W;
    if (need > y.length) {
      buf = new Float32Array(need);
      buf.set(y);
      padded = 1;
    }
  }
  const scores: Float32Array[] = [];
  const tStart = new Float64Array(nWin), tEnd = new Float64Array(nWin);
  for (let k = 0; k < nWin; k++) {
    const seg = buf.slice(k * hop, k * hop + W);
    const [out] = await runModel(model, [seg]);
    scores.push(out as Float32Array);
    tStart[k] = startS + (k * hop) / a.sample_rate;
    tEnd[k] = tStart[k] + W / a.sample_rate;
    onWindow?.(k + 1, nWin);
  }
  return { tStart, tEnd, scores, padded };
}

function segments(mask: boolean[]): [number, number][] {
  const out: [number, number][] = [];
  let start: number | null = null;
  mask.forEach((m, i) => {
    if (m && start === null) start = i;
    else if (!m && start !== null) {
      out.push([start, i - 1]);
      start = null;
    }
  });
  if (start !== null) out.push([start, mask.length - 1]);
  return out;
}

export function extractEvents(ws: WindowScores, a: AudioAssets): SoundEventRaw[] {
  const n = ws.scores.length;
  if (!n) return [];
  const series: [string, number[], number][] = [];
  for (const [g, idx] of Object.entries(a.groups)) {
    series.push([g, ws.scores.map((s) => Math.max(...idx.map((i) => s[i]))), a.thresholds[g]]);
  }
  series.push(["growl", ws.scores.map((s) => s[a.subtypes.growl]), a.thresholds.growl ?? 0.5]);
  for (const [tag, i] of Object.entries(a.context_tags)) series.push([`context:${tag}`, ws.scores.map((s) => s[i]), a.context_tag_threshold]);
  const events: SoundEventRaw[] = [];
  for (const [label, s, th] of series) {
    const merged: [number, number][] = [];
    for (const [x, y] of segments(s.map((v) => v >= th))) {
      if (merged.length && ws.tStart[x] - ws.tEnd[merged[merged.length - 1][1]] <= a.min_event_gap_s) merged[merged.length - 1][1] = y;
      else merged.push([x, y]);
    }
    for (const [x, y] of merged) {
      const sl = s.slice(x, y + 1);
      const ev: SoundEventRaw = {
        label, start_s: ws.tStart[x], end_s: ws.tEnd[y], peak_score: Math.max(...sl), mean_score: sl.reduce((p, q) => p + q, 0) / sl.length,
        n_windows: y - x + 1, threshold: th, subtypes: {}, top_audioset_labels: [],
      };
      if (label in a.groups || label === "growl") {
        for (const [k, i] of Object.entries(a.subtypes)) {
          let m = 0;
          for (let w = x; w <= y; w++) m = Math.max(m, ws.scores[w][i]);
          if (m >= 0.1) ev.subtypes[k] = m;
        }
        const peak = new Float32Array(a.labels.length);
        for (let w = x; w <= y; w++) for (let i = 0; i < peak.length; i++) peak[i] = Math.max(peak[i], ws.scores[w][i]);
        const top = Array.from(peak.keys()).sort((p, q) => peak[q] - peak[p] || q - p).slice(0, 5);
        ev.top_audioset_labels = top.map((i) => [a.labels[i], peak[i]]);
      }
      events.push(ev);
    }
  }
  events.sort((p, q) => p.start_s - q.start_s || (p.label < q.label ? -1 : p.label > q.label ? 1 : 0));
  return events;
}

/** Levels as in ingest/media.py _audio_levels (dBFS, silence and clipping fractions). */
export function audioLevels(x: Float32Array, sr: number, silenceDbfs: number): { rms_dbfs: number | null; peak_dbfs: number | null; silent_fraction: number | null; clipping_fraction: number | null } {
  if (!x.length) return { rms_dbfs: null, peak_dbfs: null, silent_fraction: null, clipping_fraction: null };
  const eps = 1e-12;
  const n = Math.max(1, Math.trunc(sr * 0.05));
  const m = Math.floor(x.length / n);
  let silent = 0;
  const frames = m === 0 ? 1 : m;
  for (let f = 0; f < frames; f++) {
    const a0 = m === 0 ? 0 : f * n, a1 = m === 0 ? x.length : a0 + n;
    let e = 0;
    for (let j = a0; j < a1; j++) e += x[j] * x[j];
    const rms = Math.sqrt(e / (a1 - a0) + eps);
    if (20 * Math.log10(rms + eps) < silenceDbfs) silent++;
  }
  let s2 = 0, peak = 0, clip = 0;
  for (const v of x) {
    s2 += v * v;
    const av = Math.abs(v);
    if (av > peak) peak = av;
    if (av >= 0.999) clip++;
  }
  return { rms_dbfs: 20 * Math.log10(Math.sqrt(s2 / x.length + eps)), peak_dbfs: 20 * Math.log10(peak + eps),
    silent_fraction: silent / frames, clipping_fraction: clip / x.length };
}
