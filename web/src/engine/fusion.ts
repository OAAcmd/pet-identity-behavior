// Experimental hypothesis fusion with abstention (mirrors fusion/engine.py,
// schemas/ontology.py and explain/templates.py). No weights are fitted; scores
// are uncalibrated and never shown as probabilities.
import type { FusionConfig } from "./assets";

export interface EvidenceItem {
  evidence_id: string;
  kind: string;
  t_start_s: number;
  t_end_s: number;
  summary: string;
  track_id: string | null;
  value: number | null;
  value_kind: string | null;
  reliability: "high" | "medium" | "low";
  provenance: { component: string; model_id: string | null; root: string; method: string };
}

export interface Ev { item: EvidenceItem; root: string; tags: Set<string>; strength: number; attributed_track: string | null }

export interface Hypothesis {
  hypothesis_id: string; name: string; semantic_type: string; status: "experimental_hypothesis" | "insufficient_evidence" | "abstained";
  track_id: string | null; t_start_s: number | null; t_end_s: number | null; supporting_evidence_ids: string[]; conflicting_evidence_ids: string[];
  independent_provenance_roots: string[]; score: number | null; score_kind: string; scorer: { name: string; revision: string; kind: string };
  calibration_status: "uncalibrated"; calibrated_confidence: null; reason_codes: string[]; explanation: string;
}

export interface FusionContext {
  audio_available: boolean; audio_state: string; video_available: boolean; zones_configured: boolean; person_detection_enabled: boolean;
  n_tracks: number; gates: Record<string, number>;
}

const RULES_REVISION = "rules-v1";
const ONTOLOGY_VERSION = "ontology-v1";
const RELIABILITY: Record<string, number> = { high: 1.0, medium: 0.7, low: 0.4 };
const SEMANTIC: Record<string, string> = { food: "need", outside: "need", attention: "need", play: "activity", alert: "state", tense: "state",
  warning: "communicative_signal", relaxed: "state" };

interface Rule { groups: Set<string>[]; min_groups: number; tag_scores: Record<string, number>; conflicts: Set<string>; triggers: Set<string> }
const S = (...x: string[]) => new Set(x);
const REQUEST = S("vocalization");
export const RULES: Record<string, Rule> = {
  food: { groups: [REQUEST, S("zone:food_area")], min_groups: 2, tag_scores: { vocalization: 0.65, "zone:food_area": 0.65 }, conflicts: S(), triggers: S("vocalization", "zone:food_area") },
  outside: { groups: [REQUEST, S("zone:exit_door")], min_groups: 2, tag_scores: { vocalization: 0.65, "zone:exit_door": 0.65 }, conflicts: S(), triggers: S("vocalization", "zone:exit_door") },
  attention: { groups: [REQUEST, S("person_visible")], min_groups: 2, tag_scores: { vocalization: 0.6, person_visible: 0.6 }, conflicts: S(), triggers: S("vocalization") },
  play: { groups: [S("movement:high"), S("zone:play_area"), S("pose:play_bow")], min_groups: 2, tag_scores: { "movement:high": 0.6, "zone:play_area": 0.6, "pose:play_bow": 0.7 },
    conflicts: S("hiss", "growl"), triggers: S("movement:high", "zone:play_area", "pose:play_bow") },
  alert: { groups: [S("bark", "howl", "doorbell", "knock"), S("movement:high")], min_groups: 2, tag_scores: { bark: 0.6, howl: 0.55, doorbell: 0.55, knock: 0.55, "movement:high": 0.55 },
    conflicts: S(), triggers: S("bark", "howl") },
  tense: { groups: [S("hiss", "growl"), S("movement:low")], min_groups: 2, tag_scores: { hiss: 0.65, growl: 0.6, "movement:low": 0.55 }, conflicts: S(), triggers: S("hiss", "growl") },
  warning: { groups: [S("hiss", "growl"), S("animal_visible")], min_groups: 2, tag_scores: { hiss: 0.7, growl: 0.65, animal_visible: 0.55 }, conflicts: S(), triggers: S("hiss", "growl") },
  relaxed: { groups: [S("movement:low"), S("zone:rest_area"), S("purr")], min_groups: 2, tag_scores: { "movement:low": 0.6, "zone:rest_area": 0.6, purr: 0.55 },
    conflicts: S("hiss", "growl", "movement:high", "bark"), triggers: S("movement:low", "zone:rest_area", "purr") },
};

const gap = (a: Ev, b: Ev) => Math.max(0, Math.max(a.item.t_start_s, b.item.t_start_s) - Math.min(a.item.t_end_s, b.item.t_end_s));
const inter = (a: Set<string>, b: Set<string> | string[]) => { const bb = b instanceof Set ? b : new Set(b); return [...a].filter((x) => bb.has(x)); };
const logit = (p: number) => Math.log(p) - Math.log1p(-p);
const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

function* combinations(items: number[], k: number, start = 0, acc: number[] = []): Generator<number[]> {
  if (acc.length === k) { yield acc; return; }
  for (let i = start; i < items.length; i++) yield* combinations(items, k, i + 1, [...acc, items[i]]);
}

function* product(lists: Ev[][], i = 0, acc: Ev[] = []): Generator<Ev[]> {
  if (i === lists.length) { yield acc; return; }
  for (const e of lists[i]) yield* product(lists, i + 1, [...acc, e]);
}

export class FusionEngine {
  readonly scorer: { name: string; revision: string; kind: string };
  constructor(readonly c: FusionConfig) {
    this.scorer = { name: `${c.method}:${RULES_REVISION}`, revision: `${RULES_REVISION}+${ONTOLOGY_VERSION}`, kind: c.method };
  }

  private expert(e: Ev, tag: string, r: Rule): number {
    const base = r.tag_scores[tag] ?? 0.5;
    const q = RELIABILITY[e.item.reliability] * Math.max(0, Math.min(1, e.strength));
    return 0.5 + (base - 0.5) * q;
  }

  private combine(op: Map<string, number>, relevantRoots: Set<string>, ctx: FusionContext): [number, string] {
    const c = this.c;
    if (!op.size) return [0.5, "none"];
    if (c.method === "rule_baseline") return [[...op.values()].reduce((a, b) => a + b, 0) / op.size, "heuristic_support_0_1"];
    const all = new Set([...relevantRoots, ...op.keys()]);
    const w = (r: string) => c.weights[r.split(":")[0]] ?? 1.0;
    if (c.method === "weighted_average") {
      let avail = 0, wa = 0, tot = 0;
      for (const [r, s] of op) { avail += w(r); wa += w(r) * s; }
      for (const r of all) tot += w(r);
      wa /= avail;
      return [0.5 + (wa - 0.5) * (avail / Math.max(tot, 1e-9)), "fusion_experimental_score_0_1"];
    }
    let z = c.prior_logit;
    for (const [r, s0] of op) {
      const s = Math.min(1 - c.smoothing_epsilon, Math.max(c.smoothing_epsilon, s0));
      z += ((ctx.gates[r] ?? 1.0) * w(r) * logit(s)) / c.temperature;
    }
    return [sigmoid(z), "fusion_experimental_score_0_1"];
  }

  evaluate(evidence: Ev[], ctx: FusionContext): Hypothesis[] {
    const tracks = new Set<string>();
    for (const e of evidence) { if (e.item.track_id) tracks.add(e.item.track_id); if (e.attributed_track) tracks.add(e.attributed_track); }
    const subjects: (string | null)[] = [...[...tracks].sort(), null];
    const out: Hypothesis[] = [];
    let counter = 0;
    for (const [name, rule] of Object.entries(RULES)) {
      for (const subject of subjects) {
        const pool = evidence.filter((e) => (subject === null ? e.item.track_id === null && e.attributed_track === null : e.item.track_id === subject || e.attributed_track === subject));
        const relevant = pool.filter((e) => inter(e.tags, Object.keys(rule.tag_scores)).length);
        if (!relevant.some((e) => inter(e.tags, rule.triggers).length)) continue;
        counter++;
        out.push(this.one(`hyp-${counter}`, name, rule, subject, relevant, pool, ctx));
      }
    }
    const seen = new Set(out.filter((h) => h.track_id).map((h) => `${h.name}|${[...h.supporting_evidence_ids].sort().join(",")}`));
    return out.filter((h) => h.track_id || !seen.has(`${h.name}|${[...h.supporting_evidence_ids].sort().join(",")}`));
  }

  private one(hid: string, name: string, rule: Rule, subject: string | null, relevant: Ev[], pool: Ev[], ctx: FusionContext): Hypothesis {
    const c = this.c;
    const reasons = ["fusion_not_validated"];
    const groupItems = rule.groups.map((g) => relevant.filter((e) => inter(e.tags, g).length));
    const satisfied = groupItems.map((it, i) => (it.length ? i : -1)).filter((i) => i >= 0);
    let best: Ev[] = [];
    for (let k = satisfied.length; k > 0 && !best.length; k--) {
      let found: Ev[] | null = null;
      for (const cg of combinations(satisfied, k)) {
        for (const combo of product(cg.map((g) => groupItems[g]))) {
          if (new Set(combo.map((e) => e.root)).size < combo.length) continue;
          if (combo.every((a) => combo.every((b) => gap(a, b) <= c.co_occurrence_window_s))) {
            const sum = (xs: Ev[]) => xs.reduce((s, e) => s + e.strength, 0);
            if (found === null || sum(combo) > sum(found)) found = [...combo];
          }
        }
        if (found) break;
      }
      if (found) best = found;
    }
    const stale: Ev[] = [];
    if (best.length < rule.min_groups) {
      for (const a of relevant) for (const b of relevant) if (a.root !== b.root && c.co_occurrence_window_s < gap(a, b) && gap(a, b) <= c.max_evidence_age_s) stale.push(a);
    }
    const used = best.length ? best : relevant;
    const conflicts = pool.filter((e) => inter(e.tags, rule.conflicts).length && (!best.length || best.some((b) => gap(e, b) <= c.co_occurrence_window_s)));
    const op = new Map<string, number>();
    for (const e of used) for (const t of inter(e.tags, Object.keys(rule.tag_scores))) op.set(e.root, Math.max(op.get(e.root) ?? 0.5, this.expert(e, t, rule)));
    const relevantRoots = new Set(Object.keys(rule.tag_scores).map((t) => ["vocalization", "bark", "howl", "doorbell", "knock", "hiss", "growl", "purr"].includes(t)
      ? "audio:yamnet" : t.startsWith("zone:") ? "context:zones" : t.startsWith("pose:") ? "pose:none" : "vision:detector"));
    const [score, scoreKind] = this.combine(op, relevantRoots, ctx);
    let status: Hypothesis["status"] = "experimental_hypothesis";
    const relRoots = new Set(relevant.map((e) => e.root));
    if (best.length < rule.min_groups) {
      status = "insufficient_evidence";
      if (relRoots.size < 2) reasons.push("single_provenance_root");
      if (relRoots.size === 1 && relRoots.has("context:zones")) reasons.push("context_only_evidence");
      if (stale.length) reasons.push("stale_evidence");
      const missing = new Set([...relevantRoots].filter((r) => !relRoots.has(r)));
      if (missing.has("audio:yamnet") && !ctx.audio_available) reasons.push(ctx.audio_state === "silent" ? "audio_silent" : ctx.audio_state === "present" ? "audio_model_unavailable" : "missing_modality_audio");
      if ((missing.has("vision:detector") || missing.has("context:zones")) && !ctx.video_available) reasons.push("missing_modality_video");
      if (missing.has("context:zones") && ctx.video_available && !ctx.zones_configured) reasons.push("zones_not_configured");
      if (rule.groups.some((g) => [...g].some((t) => t.startsWith("pose:")))) reasons.push("pose_unavailable");
      if (name === "attention" && !ctx.person_detection_enabled) reasons.push("person_detection_disabled");
      if (subject === null && ctx.n_tracks > 1) reasons.push("uncertain_attribution_multiple_animals");
      else if (subject === null && ctx.video_available && ctx.n_tracks === 0) reasons.push("no_visible_animal_for_attribution");
    }
    if (conflicts.length) { status = "abstained"; reasons.push("conflicting_evidence"); }
    if (status === "experimental_hypothesis" && score < c.abstain_below_score) { status = "abstained"; reasons.push("low_support"); }
    if (best.some((e) => e.root === "audio:yamnet") && subject !== null) reasons.push("audio_attribution_unverified");
    if (used.some((e) => e.item.reliability === "low")) reasons.push("low_reliability_evidence");
    if (name === "relaxed") {
      reasons.push("silence_not_used_as_evidence");
      if (used.some((e) => e.tags.has("purr"))) reasons.push("purr_does_not_rule_out_discomfort");
    }
    const t0 = used.length ? Math.min(...used.map((e) => e.item.t_start_s)) : null;
    const t1 = used.length ? Math.max(...used.map((e) => e.item.t_end_s)) : null;
    return { hypothesis_id: hid, name, semantic_type: SEMANTIC[name], status, track_id: subject, t_start_s: t0, t_end_s: t1,
      supporting_evidence_ids: used.map((e) => e.item.evidence_id), conflicting_evidence_ids: conflicts.map((e) => e.item.evidence_id),
      independent_provenance_roots: [...new Set(used.map((e) => e.root))].sort(), score: scoreKind !== "none" ? Math.round(score * 1e4) / 1e4 : null,
      score_kind: scoreKind, scorer: this.scorer, calibration_status: "uncalibrated", calibrated_confidence: null, reason_codes: [...new Set(reasons)], explanation: "" };
  }
}

const REASON_TEXT: Record<string, string> = {
  fusion_not_validated: "the fusion rules have not been validated on paired household data",
  single_provenance_root: "all supporting evidence comes from a single source",
  context_only_evidence: "only scene context (an owner-marked zone) supports it",
  stale_evidence: "the supporting observations were too far apart in time",
  missing_modality_audio: "no usable audio was available",
  audio_silent: "the audio track is silent (silence is not evidence either way)",
  audio_model_unavailable: "the audio-event model is unavailable",
  missing_modality_video: "no usable video was available",
  zones_not_configured: "no scene zones were configured",
  pose_unavailable: "no verified animal pose model is available",
  person_detection_disabled: "person detection is disabled",
  uncertain_attribution_multiple_animals: "several animals were visible, so the sound could not be attributed",
  no_visible_animal_for_attribution: "no animal was visible when the sound occurred (it may be off-screen or from a TV/speaker)",
  conflicting_evidence: "co-occurring evidence points in a different direction",
  low_support: "the experimental score was below the abstention threshold",
  audio_attribution_unverified: "attributing the sound to this animal is unverified (one visible candidate)",
  low_reliability_evidence: "some evidence has low reliability (quality limits)",
  silence_not_used_as_evidence: "silence is never used as evidence of relaxation",
  purr_does_not_rule_out_discomfort: "purring can also occur with discomfort",
};
const STATUS_TEXT: Record<string, string> = { experimental_hypothesis: "Experimental hypothesis", insufficient_evidence: "Insufficient evidence", abstained: "Abstained" };

export function explainHypothesis(h: Hypothesis, ev: Map<string, EvidenceItem>): string {
  const sup = h.supporting_evidence_ids.filter((i) => ev.has(i)).map((i) => `${ev.get(i)!.summary} [${i}]`);
  const con = h.conflicting_evidence_ids.filter((i) => ev.has(i)).map((i) => `${ev.get(i)!.summary} [${i}]`);
  const parts = [`${STATUS_TEXT[h.status]} '${h.name}' (${h.semantic_type})${h.track_id ? ` for ${h.track_id}` : ""}.`];
  if (sup.length) parts.push("Supporting observations: " + sup.join("; ") + ".");
  if (con.length) parts.push("Conflicting observations: " + con.join("; ") + ".");
  const reasons = h.reason_codes.filter((r) => REASON_TEXT[r]).map((r) => REASON_TEXT[r]);
  if (reasons.length) parts.push("Notes: " + reasons.join("; ") + ".");
  if (h.score !== null) parts.push(`Score ${h.score.toFixed(2)} is an uncalibrated ${h.score_kind} value, not a probability.`);
  return parts.join(" ");
}

export interface Suggestion { suggestion_id: string; text: string; based_on_evidence_ids: string[]; kind: string }

export function suggestionsFor(hyps: Hypothesis[], quality: Set<string>, nTracks: number, anyPetSound: boolean): Suggestion[] {
  const out: Suggestion[] = [];
  const add = (text: string, ids: string[], kind = "observation_followup") => out.push({ suggestion_id: `sug-${out.length + 1}`, text, based_on_evidence_ids: ids, kind });
  for (const h of hyps) {
    if (h.status !== "experimental_hypothesis") continue;
    const ids = h.supporting_evidence_ids;
    if (h.name === "food") add("The animal vocalized while in the area you marked for feeding. If it is close to the usual feeding time, you could check whether food and water are available. This is an unvalidated heuristic, not a determination of hunger.", ids);
    else if (h.name === "outside") add("The animal vocalized while in the area you marked as a door. If your routine allows, you could check whether it wants to go out. This is an unvalidated heuristic.", ids);
    else if (h.name === "attention") add("The animal vocalized while a person was visible. This may or may not be a request for attention; the system cannot tell.", ids);
    else if (h.name === "warning" || h.name === "tense") add("Hiss- or growl-like sounds were observed together with a visible animal. Giving animals space and not reaching toward them is a common precaution. If such signals are new or frequent, consider asking a veterinarian or a qualified behavior professional.", ids, "safety_general");
    else if (h.name === "alert") add("Barking or door sounds co-occurred with movement. Checking what happened nearby may help interpret it.", ids);
    else if (h.name === "play") add("Sustained movement in the play area was observed. Movement alone cannot distinguish play from other activity.", ids);
    else if (h.name === "relaxed") add("Low movement in the rest area was observed. This does not confirm comfort or wellbeing.", ids);
  }
  if (nTracks > 1 && anyPetSound) add("Several animals were visible, so sounds were not assigned to a specific animal.", [], "data_quality");
  if (quality.has("low_light")) add("Lighting was low in part of the video; visual observations there are less reliable.", [], "data_quality");
  if (quality.has("bandwidth_limited_audio")) add("The audio has a low sample rate, so high-frequency sound content is missing and sound tagging is less reliable.", [], "data_quality");
  if (quality.has("camera_motion")) add("The camera moved during recording; apparent animal movement is less reliable.", [], "data_quality");
  return out;
}
