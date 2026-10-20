// Behaviour / intent inference on the phone ("phone_now" variant).
// Mirrors src/petperception/intent/kinetics.py (pool), intent/heads.py (Head, LateFusion) and
// intent/inference.py (infer) for the channels the phone bundle carries: situation (late fusion of
// audio class scores, animal crop, environment and interaction heads; E28), dog vocal activity
// (YAMNet class scores; E27) and cat meow context (below the evidence gate). Scores are
// temperature-scaled softmax values on the training data, not probabilities of truth.
import { f32FromB64 } from "./assets";

export interface IntentHeadAssets {
  classes: string[]; mean_b64: string; scale_b64: string; coef_b64: string; coef_shape: [number, number];
  intercept: number[]; temperature: number; abstain_below: number; branches?: string[];
}

export interface IntentAssets {
  version: string; taxonomy_version: string; variant: string; vocal_heard: number;
  dog_sounds: string[]; cat_sounds: string[]; object_classes: string[];
  situation: {
    heads: Record<string, IntentHeadAssets>; fusion: IntentHeadAssets; visual_only_abstain_below: number;
    species: Record<string, string>; qualifier: Record<string, string | null>; context_for: Record<string, string[]>;
  };
  dog_activity_hypothesis: Record<string, string | null>;
  dog_audio?: IntentHeadAssets;
  cat_context?: IntentHeadAssets & { evidence_gate_passed: boolean; context_for: Record<string, string[]> };
}

/** One sampled frame, as computed by the engine (see LocalEngine.intentFrame). */
export interface IntentFrame {
  t: number;
  whole: Float32Array; // 1280, backbone on the whole frame resized to 224x224 (area)
  objects: Float32Array; // best score per object class (IntentAssets.object_classes), 0 if not detected
  pet: [number, number, number, number]; // present, detector score, verifier score (-1 if none), 1 if dog
  box: [number, number, number, number];
  crop: Float32Array | null; // 1280, backbone on the square crop of the largest cat/dog
  person: [number, number, number, number]; // people, best person score, best interaction score, best IoU with the animal
  motion: number; // mean |grey difference| to the previous frame on a 64x48 area-resized image, 0..1
}

export interface IntentAudio { scores: Float32Array[]; rmsDb: number[] }

export class LinearHead {
  readonly mean: Float32Array; readonly scale: Float32Array; readonly coef: Float32Array;
  readonly k: number; readonly d: number;
  constructor(readonly a: IntentHeadAssets) {
    this.mean = f32FromB64(a.mean_b64);
    this.scale = f32FromB64(a.scale_b64);
    this.coef = f32FromB64(a.coef_b64);
    [this.k, this.d] = a.coef_shape;
  }
  get classes(): string[] { return this.a.classes; }
  get threshold(): number { return this.a.abstain_below; }
  proba(x: ArrayLike<number>): number[] {
    if (x.length !== this.d) throw new Error(`intent head expects ${this.d} features, got ${x.length}`);
    const z: number[] = [];
    for (let c = 0; c < this.k; c++) {
      let v = this.a.intercept[c];
      const off = c * this.d;
      for (let i = 0; i < this.d; i++) v += ((x[i] - this.mean[i]) / this.scale[i]) * this.coef[off + i];
      z.push(v / this.a.temperature);
    }
    return softmax(z);
  }
}

export function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((p, q) => p + q, 0);
  return e.map((v) => v / s);
}

/** Late fusion over branch probabilities; a null branch is absent (zero block, flag 0). */
export function lateFusion(f: LinearHead, probaByBranch: Record<string, number[] | null>): number[] {
  const branches = f.a.branches ?? [];
  const k = f.classes.length;
  const x: number[] = [];
  const flags: number[] = [];
  for (const b of branches) {
    const p = probaByBranch[b] ?? null;
    if (!p) {
      for (let j = 0; j < k; j++) x.push(0);
      flags.push(0);
      continue;
    }
    const lp = p.map((v) => Math.log(Math.max(v, 1e-6)));
    const m = lp.reduce((s, v) => s + v, 0) / k;
    for (const v of lp) x.push(v - m);
    flags.push(1);
  }
  return f.proba([...x, ...flags]);
}

const mean0 = (rows: ArrayLike<number>[], d: number): Float32Array => {
  const o = new Float32Array(d);
  for (const r of rows) for (let i = 0; i < d; i++) o[i] += r[i];
  for (let i = 0; i < d; i++) o[i] /= Math.max(1, rows.length);
  return o;
};
const max0 = (rows: ArrayLike<number>[], d: number): Float32Array => {
  const o = new Float32Array(d).fill(-Infinity);
  for (const r of rows) for (let i = 0; i < d; i++) o[i] = Math.max(o[i], r[i]);
  return o;
};
const median = (v: number[]): number => {
  const s = [...v].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};
/** numpy.percentile with linear interpolation. */
const percentile = (v: number[], q: number): number => {
  const s = [...v].sort((a, b) => a - b);
  const pos = ((s.length - 1) * q) / 100;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
};

export interface IntentCues {
  missing: Record<string, string>;
  yamnet_max?: Float32Array;
  object_frames?: number[];
  pet_frames?: number;
  frames?: number;
  person_frames?: number;
  interaction_max?: number;
  dog_fraction?: number | null;
}

export type Branches = Record<"audio_scores" | "animal_phone" | "environment" | "interaction", Float32Array | null>;

/** Mirrors intent/kinetics.py pool() for the phone branches. */
export function pool(frames: IntentFrame[] | null, audio: IntentAudio | null, videoState: string, audioState: string,
  silenceDbfs = -60): { branches: Branches; cues: IntentCues } {
  const b: Branches = { audio_scores: null, animal_phone: null, environment: null, interaction: null };
  const cues: IntentCues = { missing: {} };
  let aState = audioState;
  if (aState === "present" && audio && audio.scores.length && Math.max(...audio.rmsDb) < silenceDbfs) aState = "silent";
  if (aState === "present" && audio && audio.scores.length) {
    const n = audio.scores[0].length;
    const mx = max0(audio.scores, n);
    b.audio_scores = Float32Array.from([...mean0(audio.scores, n), ...mx]);
    cues.yamnet_max = mx;
    cues.missing.audio = "YAMNet embedding not computed (phone_now variant)";
  } else {
    cues.missing.audio = `audio ${aState === "present" ? "absent" : aState}`;
  }
  if (videoState !== "present" || !frames || !frames.length) {
    for (const k of ["animal", "animal_phone", "environment", "interaction"]) cues.missing[k] = `video ${videoState === "present" ? "absent" : videoState}`;
    return { branches: b, cues };
  }
  const T = frames.length;
  const C = frames[0].objects.length;
  const objMax = max0(frames.map((f) => f.objects), C);
  b.environment = Float32Array.from([...mean0(frames.map((f) => f.whole), 1280), ...objMax]);
  cues.object_frames = Array.from({ length: C }, (_, j) => frames.filter((f) => f.objects[j] > 0).length);
  const pet = frames.map((f) => f.pet[0] > 0);
  const petIdx = frames.map((_, i) => i).filter((i) => pet[i]);
  const per = frames.map((f) => f.person);
  const mot = frames.map((f) => f.motion).slice(1);
  b.interaction = Float32Array.from([
    per.filter((p) => p[0] > 0).length / T, per.reduce((s, p) => s + p[0], 0) / T, Math.max(...per.map((p) => p[2])),
    petIdx.length ? petIdx.reduce((s, i) => s + per[i][2], 0) / petIdx.length : 0, Math.max(...per.map((p) => p[3])),
    petIdx.length ? petIdx.reduce((s, i) => s + per[i][3], 0) / petIdx.length : 0,
    T > 1 ? mot.reduce((s, v) => s + v, 0) / mot.length : 0, T > 1 ? percentile(mot, 90) : 0]);
  cues.pet_frames = petIdx.length;
  cues.frames = T;
  cues.person_frames = per.filter((p) => p[0] > 0).length;
  cues.interaction_max = Math.max(...per.map((p) => p[2]));
  if (!petIdx.length) {
    cues.missing.animal = cues.missing.animal_phone = "no cat or dog detected in any sampled frame";
    cues.dog_fraction = null;
    return { branches: b, cues };
  }
  const crops = petIdx.map((i) => frames[i].crop as Float32Array);
  const box = petIdx.map((i) => frames[i].box);
  const w = box.map((x) => x[2] - x[0]), h = box.map((x) => x[3] - x[1]);
  const area = w.map((v, i) => v * h[i]);
  const ctr = box.map((x) => [(x[0] + x[2]) / 2, (x[1] + x[3]) / 2]);
  const t = petIdx.map((i) => frames[i].t);
  const sp: number[] = [];
  for (let i = 1; i < ctr.length; i++) sp.push(Math.hypot(ctr[i][0] - ctr[i - 1][0], ctr[i][1] - ctr[i - 1][1]) / Math.max(t[i] - t[i - 1], 1e-3));
  if (!sp.length) sp.push(0);
  const am = area.reduce((s, v) => s + v, 0) / area.length;
  const asd = Math.sqrt(area.reduce((s, v) => s + (v - am) ** 2, 0) / area.length);
  const ver = petIdx.map((i) => frames[i].pet[2]).filter((v) => v >= 0);
  const stats = [am, asd, w.reduce((s, v, i) => s + Math.log(Math.max(v, 1e-4) / Math.max(h[i], 1e-4)), 0) / w.length, median(sp), percentile(sp, 90),
    petIdx.reduce((s, i) => s + frames[i].pet[1], 0) / petIdx.length, ver.length ? ver.reduce((s, v) => s + v, 0) / ver.length : 0.5,
    petIdx.reduce((s, i) => s + frames[i].pet[3], 0) / petIdx.length];
  b.animal_phone = Float32Array.from([...mean0(crops, 1280), ...max0(crops, 1280), ...stats, petIdx.length / T]);
  cues.dog_fraction = stats[7];
  return { branches: b, cues };
}

export interface IntentReading {
  channel: string; model: string; label_space: string; species: string; scores: Record<string, number>; top: string;
  top_score: number; threshold: number; confident: boolean; inputs: string[]; evidence_gate_passed: boolean; note: string;
}
export interface IntentHypothesis {
  name: string; status: string; support: number | null; supporting: string[]; qualifying: string[]; contradicting: string[]; explanation: string;
}
export interface IntentResult {
  version: string; species: string; species_source: string; inputs: Record<string, string>; missing_inputs: Record<string, string>;
  readings: IntentReading[]; situation: { label: string; score: number; status: string; branches_used: string[]; qualifier: string | null } | null;
  qualifiers: string[]; hypotheses: IntentHypothesis[]; conflicts: string[]; status: string; summary: string; caveats: string[];
}

const r4 = (v: number) => Math.round(v * 1e4) / 1e4;
const SOFT_CONFLICTS: Record<string, string[]> = {
  play: ["tense", "warning"], alert: ["relaxed"], tense: ["relaxed", "play"], warning: ["relaxed", "play"], relaxed: ["tense", "warning", "alert"],
  food: [], outside: [], attention: [],
};

export class IntentModels {
  readonly heads: Record<string, LinearHead> = {};
  readonly fusion: LinearHead;
  readonly dogAudio?: LinearHead;
  readonly catContext?: LinearHead;
  constructor(readonly a: IntentAssets) {
    for (const [b, h] of Object.entries(a.situation.heads)) this.heads[b] = new LinearHead(h);
    this.fusion = new LinearHead(a.situation.fusion);
    if (a.dog_audio) this.dogAudio = new LinearHead(a.dog_audio);
    if (a.cat_context) this.catContext = new LinearHead(a.cat_context);
  }

  infer(branches: Branches, cues: IntentCues, yamnetLabels: string[]): IntentResult {
    const a = this.a;
    const missing: Record<string, string> = { ...cues.missing };
    const inputs: Record<string, string> = {};
    for (const [k, v] of Object.entries(branches)) inputs[k] = v ? "present" : "missing";
    const caveats = ["cross-channel intent combination not validated (no synchronized intent-labelled recordings)",
      "scores are temperature-scaled on training data, not probabilities of truth"];
    const ym: Record<string, number> = {};
    if (cues.yamnet_max) yamnetLabels.forEach((n, i) => { ym[n] = cues.yamnet_max![i]; });
    const dogV = Math.max(0, ...a.dog_sounds.map((n) => ym[n] ?? 0));
    const catV = Math.max(0, ...a.cat_sounds.map((n) => ym[n] ?? 0));
    let species: string, src: string;
    if (cues.dog_fraction !== undefined && cues.dog_fraction !== null) {
      species = cues.dog_fraction >= 0.5 ? "dog" : "cat";
      src = `detector (${species} in ${Math.round(Math.max(cues.dog_fraction, 1 - cues.dog_fraction) * 100)}% of ${cues.pet_frames ?? 0} animal frames)`;
    } else if (Math.max(dogV, catV) >= a.vocal_heard) {
      species = dogV >= catV ? "dog" : "cat";
      src = "sound";
    } else {
      species = "unknown";
      src = "no animal seen or heard";
    }
    const readings: IntentReading[] = [];
    const conflicts: string[] = [];
    const qualifiers: string[] = [];
    let situation: IntentResult["situation"] = null;
    const sitBranches = a.situation.fusion.branches ?? [];
    const present = sitBranches.filter((b) => branches[b as keyof Branches]);
    const reading = (channel: string, model: string, space: string, sp: string, classes: string[], p: number[], thr: number, inp: string[],
      gate = true, note = ""): IntentReading => {
      let k = 0;
      p.forEach((v, i) => { if (v > p[k]) k = i; });
      const scores: Record<string, number> = {};
      classes.forEach((c, i) => { scores[c] = r4(p[i]); });
      return { channel, model, label_space: space, species: sp, scores, top: classes[k], top_score: r4(p[k]), threshold: r4(thr),
        confident: p[k] >= thr, inputs: inp, evidence_gate_passed: gate, note };
    };
    if (present.length) {
      const bp: Record<string, number[] | null> = {};
      for (const b of sitBranches) {
        const v = branches[b as keyof Branches];
        bp[b] = v ? this.heads[b].proba(v) : null;
      }
      const p = lateFusion(this.fusion, bp);
      const classes = this.fusion.classes;
      const r = reading("situation", "kinetics_situation_v1", "kinetics_situation", species, classes, p, this.fusion.threshold, present);
      readings.push(r);
      const ab = sitBranches[0];
      const vis = present.filter((b) => b !== ab);
      let sitConflict: string | null = null;
      if (present.includes(ab) && vis.length) {
        const pa = bp[ab]!;
        const vb: Record<string, number[] | null> = {};
        for (const b of sitBranches) vb[b] = vis.includes(b) ? bp[b] : null;
        const pv = lateFusion(this.fusion, vb);
        const aTop = pa.indexOf(Math.max(...pa)), vTop = pv.indexOf(Math.max(...pv));
        if (aTop !== vTop && pa[aTop] >= this.heads[ab].threshold && pv[vTop] >= a.situation.visual_only_abstain_below) {
          sitConflict = `audio suggests '${classes[aTop]}' (${pa[aTop].toFixed(2)}) but video suggests '${classes[vTop]}' (${pv[vTop].toFixed(2)})`;
          conflicts.push("situation: " + sitConflict);
        }
      }
      if ((species === "cat" || species === "dog") && a.situation.species[r.top] !== species && r.confident)
        conflicts.push(`situation '${r.top}' implies a ${a.situation.species[r.top]} but the ${src.split(" ")[0]} indicates a ${species}`);
      const status = sitConflict ? "conflict" : r.confident ? "interpretation" : "uncertain";
      situation = { label: r.top, score: r.top_score, status, branches_used: present, qualifier: status === "interpretation" ? a.situation.qualifier[r.top] ?? null : null };
      if (situation.qualifier) qualifiers.push(situation.qualifier);
    } else {
      missing.situation = "no audio or video branch available";
    }
    const audio = branches.audio_scores;
    if ((species === "dog" || species === "unknown") && audio && dogV >= a.vocal_heard && this.dogAudio) {
      readings.push(reading("dog_vocal_activity", "dog_audio_activity_v1", "barkopedia_activity", "dog", this.dogAudio.classes, this.dogAudio.proba(audio),
        this.dogAudio.threshold, ["audio"]));
      missing.dog_bark_context = "model needs the YAMNet embedding (not in this variant)";
    } else if (species === "dog" || species === "unknown") {
      missing.dog_vocal_activity = !audio ? "audio missing" : this.dogAudio ? "no dog vocalization heard" : "model not in this bundle";
    }
    if ((species === "cat" || species === "unknown") && audio && catV >= a.vocal_heard && this.catContext) {
      readings.push(reading("cat_meow_context", "cat_audio_context_v1", "catmeows_context", "cat", this.catContext.classes, this.catContext.proba(audio),
        this.catContext.threshold, ["audio"], false, "below the evidence gate (E2 leave-one-cat-out macro-F1 0.46): context hint only"));
    } else if (species === "cat" || species === "unknown") {
      missing.cat_meow_context = !audio ? "audio missing" : this.catContext ? "no cat vocalization heard" : "model not in this bundle";
    }
    const hyps = this.hypotheses(readings, qualifiers, conflicts);
    let status: string;
    if (hyps.some((h) => h.status === "conflict") || situation?.status === "conflict") status = "conflict";
    else if (hyps.some((h) => h.status.startsWith("supported"))) status = "interpretation";
    else if (situation?.status === "interpretation") status = "situation_only";
    else status = "insufficient_evidence";
    if (species === "unknown") caveats.push("species unknown: neither seen nor heard");
    return { version: a.version, species, species_source: src, inputs, missing_inputs: missing, readings, situation,
      qualifiers: [...new Set(qualifiers)].sort(), hypotheses: hyps, conflicts, status, summary: summary(status, hyps, situation, conflicts, missing), caveats };
  }

  private hypotheses(readings: IntentReading[], qualifiers: string[], conflicts: string[]): IntentHypothesis[] {
    const a = this.a;
    const out = new Map<string, IntentHypothesis>();
    for (const r of readings.filter((x) => x.channel === "dog_vocal_activity")) {
      const mass = new Map<string, number>();
      for (const [lab, v] of Object.entries(r.scores)) {
        const h = a.dog_activity_hypothesis[lab];
        if (h) mass.set(h, (mass.get(h) ?? 0) + v);
      }
      const topH = a.dog_activity_hypothesis[r.top];
      if (r.top === "taking shower" && r.confident) qualifiers.push("bathing");
      for (const [h, m] of [...mass.entries()].sort((p, q) => q[1] - p[1])) {
        const rel = (r.top === "begging for food" || r.top === "seeking attention") && h === topH ? "annotated intent" : "behaviour consistent with";
        let st: string;
        if (h === topH && r.confident) st = "supported";
        else if (h === topH || m >= 0.3) st = "tentative";
        else continue;
        out.set(h, { name: h, status: st, support: r4(m), qualifying: [], contradicting: [], explanation: "",
          supporting: [h === topH ? `${r.channel}: '${r.top}' ${r.top_score.toFixed(2)} (${rel} '${h}')` : `${r.channel}: ${m.toFixed(2)} of the score mass maps to '${h}'`] });
      }
    }
    for (const r of readings.filter((x) => x.channel === "situation" || x.channel === "cat_meow_context")) {
      if (!r.confident) continue;
      const ctx = r.channel === "situation" ? a.situation.context_for : a.cat_context!.context_for;
      for (const h of ctx[r.top] ?? []) {
        const note = `${r.channel}: context '${r.top}' ${r.top_score.toFixed(2)} makes '${h}' plausible`;
        const e = out.get(h);
        if (e) {
          e.qualifying.push(note);
          if (e.status === "supported" && r.channel === "situation") e.status = "supported_corroborated";
        } else {
          out.set(h, { name: h, status: "context_only", support: null, supporting: [], qualifying: [note], contradicting: [],
            explanation: "context alone is not evidence of what the animal wants" });
        }
      }
    }
    const q = new Set(qualifiers);
    const all = [...out.values()];
    for (const h of all) {
      if ((q.has("bathing") || q.has("handling")) && ["alert", "attention", "play"].includes(h.name) && h.status.startsWith("supported"))
        h.qualifying.push("animal is being handled/bathed: vocal signals may be a reaction to the handling");
      for (const o of all) {
        if ((SOFT_CONFLICTS[h.name] ?? []).includes(o.name) && h.status.startsWith("supported") && o.status.startsWith("supported")) {
          h.contradicting.push(`'${o.name}' is also supported`);
          h.status = "conflict";
          conflicts.push(`hypotheses '${h.name}' and '${o.name}' both supported`);
        }
      }
      if (!h.explanation) h.explanation = ({
        supported: "the channel's top label passes its trained abstention threshold",
        supported_corroborated: "supported, and the synchronized situation reading is consistent",
        tentative: "evidence present but below the channel's abstention threshold",
        conflict: "confident readings disagree; no single interpretation" } as Record<string, string>)[h.status] ?? "";
    }
    const order: Record<string, number> = { conflict: 0, supported_corroborated: 1, supported: 2, tentative: 3, context_only: 4 };
    return all.sort((p, q2) => order[p.status] - order[q2.status] || (q2.support ?? 0) - (p.support ?? 0));
  }
}

function summary(status: string, hyps: IntentHypothesis[], situation: IntentResult["situation"], conflicts: string[], missing: Record<string, string>): string {
  const parts: string[] = [];
  if (status === "conflict") parts.push("Conflicting evidence: " + conflicts.join("; ") + ". No single interpretation.");
  const sup = hyps.filter((h) => h.status.startsWith("supported"));
  if (sup.length) parts.push("Possible: " + sup.map((h) => `${h.name} (${(h.support ?? 0).toFixed(2)})`).join(", ") + ".");
  const tent = hyps.filter((h) => h.status === "tentative");
  if (tent.length) parts.push("Weaker: " + tent.map((h) => h.name).join(", ") + ".");
  if (situation) parts.push(`Situation: ${situation.label} (${situation.score.toFixed(2)}, ${situation.status}).`);
  const mk = Object.keys(missing).sort();
  if (mk.length) parts.push("Missing: " + mk.map((k) => `${k} (${missing[k]})`).join("; ") + ".");
  if (status === "insufficient_evidence" && !sup.length) parts.push("Insufficient evidence for an interpretation.");
  return parts.join(" ");
}
