// On-device analysis engine: the browser twin of src/petperception/pipeline.py.
// Everything runs locally (LiteRT.js WASM); no media leaves the device.
import type { EngineAssets } from "./assets";
import { audioLevels, extractEvents, scoreWaveform16k, type SoundEventRaw } from "./audio";
import * as BH from "./behavior";
import { Detector } from "./detector";
import { explainHypothesis, FusionEngine, suggestionsFor, type Ev, type EvidenceItem, type Hypothesis } from "./fusion";
import { grayscale, lumaAndLaplacianVar } from "./imageops";
import { decodeAudio16k, probe, sampleFrames, sha256Hex, type Probe } from "./media";
import type { LoadedModel } from "./runtime";
import { detectionFlags, estimateCameraMotion, firstT, lastT, makeTracker, mergeSpecies, movementSummary, score as detScore, species,
  speciesSummary, trackSpecies, type Track } from "./tracking";
import { Interaction } from "./interaction";
import { Backbone, Posture, Verifier } from "./verifier";
import { occupancy, type ZoneConfig } from "./zones";

export const ENGINE_VERSION = "web-0.2.0";

export class AnalysisCancelled extends Error {}

export interface LocalModels { yamnet?: LoadedModel; detector?: LoadedModel; backbone?: LoadedModel }

export interface EngineInfo { bundle_version: string; runtime: string; accelerator: string; models: { model_id: string; sha256: string; file: string }[] }

const STATIC_LIMITATIONS = [
  "Research engine: outputs are observations plus unvalidated heuristics. They are not translations of animal language, readings of thoughts or emotions, or medical assessments.",
  "Track IDs are per-analysis labels for linked detections, not verified pet identities.",
  "Hypothesis scores are uncalibrated; multimodal accuracy has not been evaluated (no paired household validation data).",
  "Sound events use raw YAMNet scores with thresholds selected on ESC-50 clips, which are not household recordings.",
  "Cat/dog detection uses a COCO-trained EfficientDet-Lite model; a second-stage verifier reduces but does not remove false detections (E7, web photos).",
  "Behavior segments (stillness, movement, zone visits, back-and-forth, startle candidates) use fixed thresholds checked only on synthetic and semi-synthetic sequences; they are not validated against labelled household behavior.",
  "For veterinary or behavioral concerns, consult a qualified professional; this system does not diagnose.",
  "Analysed on this device (browser runtime). Results can differ slightly from the server reference because the browser decodes and scales media differently.",
];
const REQUEST_LIKE = new Set(["meow", "caterwaul", "bark", "yip", "bow_wow", "howl", "whimper_dog"]);
const SUBTYPE_MIN = 0.3;
const POSE_FEATURES = ["ear_tip_elevation", "tail_elevation", "play_bow_candidate", "tail_lateral_oscillation"];

export class LocalEngine {
  readonly detector?: Detector;
  readonly verifier?: Verifier;
  readonly posture?: Posture;
  readonly interaction?: Interaction;
  readonly fusion: FusionEngine;
  constructor(readonly a: EngineAssets, readonly m: LocalModels, readonly info: EngineInfo) {
    if (a.detector && m.detector) this.detector = new Detector(m.detector, a.detector);
    if (a.backbone && m.backbone) {
      const bb = new Backbone(m.backbone, a.backbone.norm_mean, a.backbone.norm_std, a.backbone.input_quant ?? null);
      if (a.verifier) this.verifier = new Verifier(bb, a.verifier);
      if (a.posture) this.posture = new Posture(bb, a.posture);
    }
    if (a.interaction && this.detector) this.interaction = new Interaction(a.interaction);
    this.fusion = new FusionEngine(a.fusion);
  }

  components(): { name: string; kind: string; state: string; model_id: string | null; version: string | null; reason: string | null }[] {
    const a = this.a;
    return [
      { name: "audio_event", kind: "audio_event", state: this.m.yamnet && a.audio ? "ok" : "unavailable", model_id: a.audio?.model_id ?? null, version: "yamnet-mediapipe-v1@litertjs", reason: this.m.yamnet ? null : "audio model not in the on-device bundle" },
      { name: "visual_detector", kind: "visual_detector", state: this.detector ? "ok" : "unavailable", model_id: a.detector?.model_id ?? null, version: `${a.detector?.model_id ?? "none"}@litertjs`, reason: this.detector ? null : "detector not in the on-device bundle" },
      { name: "detection_verifier", kind: "visual_verifier", state: this.verifier ? "ok" : "unavailable", model_id: "efficientnet_lite0_mediapipe_v1", version: "verifier-v1:crop", reason: this.verifier ? "track-level median of per-detection rejection scores; thresholds from E7 tuning" : "verifier not in the on-device bundle" },
      { name: `${a.vision.tracker.method}_tracker`, kind: "tracker", state: this.detector ? "ok" : "unavailable", model_id: null, version: a.vision.tracker.method === "kalman" ? "kalman-cv-sort-v1" : "iou-hungarian-v1", reason: null },
      { name: "pose", kind: "pose", state: "unavailable", model_id: null, version: null, reason: "no keypoint model in the on-device bundle (dog keypoints are research-only; no cat keypoint data)" },
      { name: "posture", kind: "pose", state: this.posture ? "ok" : "unavailable", model_id: this.posture ? "efficientnet_lite0_mediapipe_v1" : null, version: a.posture?.version ?? null, reason: this.posture ? "coarse lying/sitting/standing from weak caption labels (E10); uncalibrated" : "posture model not in the on-device bundle" },
      { name: "person_interaction", kind: "behavior", state: this.interaction ? "ok" : "unavailable", model_id: this.interaction ? (a.detector?.model_id ?? null) : null, version: a.interaction?.version ?? null, reason: this.interaction ? "person-pet interaction from box geometry (E13, still images, dogs evaluated); people are detected but not stored" : "interaction model not in the on-device bundle" },
      { name: "behavior_segments", kind: "behavior", state: this.detector && a.behavior.enabled ? "ok" : "unavailable", model_id: null, version: "behavior-rules-v1", reason: "fixed thresholds; validated on synthetic/semi-synthetic sequences only (E8)" },
      { name: "fusion", kind: "fusion", state: "ok", model_id: null, version: this.fusion.scorer.revision, reason: "experimental; not validated on paired data" },
      { name: "runtime", kind: "ingest", state: "ok", model_id: null, version: `${this.info.runtime}/${this.info.accelerator}`, reason: `on-device bundle ${this.info.bundle_version}` },
    ];
  }

  async analyze(file: Blob, opts: { name: string; zones: ZoneConfig | null; modalities: ("audio" | "video")[]; analysisId: string;
    progress?: (stage: string, frac: number) => void; shouldCancel?: () => boolean }): Promise<Record<string, unknown>> {
    const a = this.a;
    // Zone anchor defaults to bottom_center, as in context/zones.py.
    if (opts.zones) opts = { ...opts, zones: { ...opts.zones, anchor: opts.zones.anchor ?? "bottom_center", zones: opts.zones.zones ?? [] } };
    const timings: Record<string, number> = {};
    const tick = () => performance.now();
    const report = (stage: string, frac: number) => {
      if (opts.shouldCancel?.()) throw new AnalysisCancelled(stage);
      opts.progress?.(stage, Math.max(0, Math.min(1, frac)));
    };
    report("probe", 0);
    let t = tick();
    const p: Probe = await probe(file, opts.name);
    const digest = await sha256Hex(file);
    timings.probe_and_hash = tick() - t;
    if (p.duration_s !== null && p.duration_s > a.limits.max_duration_s) throw new Error(`media longer than ${a.limits.max_duration_s} s`);
    const qflags: Record<string, unknown>[] = [];

    // ---------------- audio ----------------
    let audioState = "disabled";
    let rawEvents: SoundEventRaw[] = [];
    let audioSummary: Record<string, unknown> = { state: "disabled", windows_scored: 0 };
    let decodedDuration = 0;
    if (opts.modalities.includes("audio") && a.audio) {
      if (p.kind === "image") audioState = "absent";
      else {
        t = tick();
        const da = await decodeAudio16k(file, p.kind === "audio");
        timings.decode_audio = tick() - t;
        audioState = da.state;
        decodedDuration = da.samples.length / da.sample_rate;
        if (audioState === "present" && decodedDuration < 0.1) audioState = "too_short";
        if (da.state === "present") {
          const lv = audioLevels(da.samples, da.sample_rate, a.audio.silence_dbfs);
          if ((lv.silent_fraction ?? 0) >= 0.999) audioState = "silent";
          if (audioState === "corrupt") qflags.push({ code: "audio_corrupt", severity: "limiting", detail: "audio stream could not be decoded" });
          if (audioState === "silent") qflags.push({ code: "audio_silent", severity: "info", detail: "audio is silent; silence is not evidence of relaxation or of absence of need" });
          if ((lv.clipping_fraction ?? 0) > a.audio.clipping_fraction) qflags.push({ code: "audio_clipping", severity: "warning", detail: `${((lv.clipping_fraction ?? 0) * 100).toFixed(1)}% of samples at full scale` });
          audioSummary = { state: audioState, ...lv, effective_bandwidth_hz: null, windows_scored: 0 };
          report("audio", 0.1);
          if (audioState === "present" && this.m.yamnet) {
            t = tick();
            const clipped = da.samples.map((v) => Math.max(-1, Math.min(1, v)));
            const ws = await scoreWaveform16k(this.m.yamnet, clipped, 0, a.audio.hop_s, a.audio,
              (k, n) => report("audio", 0.1 + 0.2 * (k / n)));
            rawEvents = extractEvents(ws, a.audio);
            timings.audio_inference = tick() - t;
            audioSummary.windows_scored = ws.scores.length;
            if (ws.padded) qflags.push({ code: "audio_end_padded", severity: "info", detail: "final analysis window zero-padded" });
          }
        } else {
          audioSummary = { state: audioState, windows_scored: 0 };
          if (audioState === "corrupt") qflags.push({ code: "audio_corrupt", severity: "limiting", detail: "audio stream could not be decoded" });
        }
      }
    }

    // ---------------- video ----------------
    let videoState = "disabled";
    let tracks: Track[] = [];
    const cameraShift = new Map<number, [number, number]>();
    const cameraMovingTimes: number[] = [], lowLight: number[] = [], blurry: number[] = [];
    let cameraUnknown = 0, nFrames = 0;
    let frameSize: [number, number] = [0, 0];
    const isStill = p.kind === "image";
    if (!opts.modalities.includes("video")) videoState = "disabled";
    else if (p.kind === "audio") videoState = "absent";
    else if (!this.detector) {
      videoState = "present";
      qflags.push({ code: "visual_model_unavailable", severity: "limiting", detail: "detector not in the on-device bundle" });
    } else {
      const v = a.vision;
      const tracker = makeTracker(v.tracker.method, v.tracker.iou_threshold, v.tracker.max_gap_s);
      let prevGray: Uint8Array | null = null;
      let prevBoxes: ArrayLike<number>[] = [];
      const expected = Math.max(1, Math.min(a.limits.max_video_frames, Math.floor((p.duration_s ?? 0) * v.sample_fps) + 1));
      let detMs = 0, verMs = 0, posMs = 0, qMs = 0;
      t = tick();
      for await (const fr of sampleFrames(file, p, v.sample_fps, v.max_side, a.limits.max_video_frames, opts.shouldCancel)) {
        nFrames++;
        report("video", 0.3 + 0.6 * Math.min(1, nFrames / expected));
        frameSize = [fr.rgb.width, fr.rgb.height];
        let tq = tick();
        const gray = grayscale(fr.rgb);
        const { luma, lapVar } = lumaAndLaplacianVar(gray, fr.rgb.width, fr.rgb.height);
        if (luma < v.quality.low_light_luma) lowLight.push(fr.t_s);
        if (lapVar < v.quality.blur_laplacian_var) blurry.push(fr.t_s);
        qMs += tick() - tq;
        let td = tick();
        const classes = this.interaction ? [...this.a.detector!.target_classes, "person"] : this.a.detector!.target_classes;
        const all = await this.detector.detect(fr.rgb, classes, this.a.detector!.score_threshold, this.a.detector!.nms_iou, this.a.detector!.max_detections_per_frame);
        detMs += tick() - td;
        const dets = all.filter((d) => d.label !== "person");
        const adets = mergeSpecies(dets, fr.t_s, fr.index);
        detectionFlags(adets, v.quality.tiny_area_fraction);
        if (this.interaction) {
          const people = all.filter((d) => d.label === "person" && d.score >= this.a.interaction!.min_person_score).map((d) => d.box);
          if (people.length) for (const d of adets) d.interaction = Math.max(...people.map((pb) => this.interaction!.score(pb, d.box, fr.rgb.width, fr.rgb.height, species(d))));
        }
        if (this.verifier) {
          td = tick();
          for (const d of [...adets].sort((x, y) => detScore(y) - detScore(x)).slice(0, this.a.verifier!.max_checks_per_frame)) {
            const r = await this.verifier.verify(fr.rgb, { label: species(d), score: detScore(d), box: d.box });
            d.verify_score = r ? r.score : null;
          }
          verMs += tick() - td;
        }
        if (this.posture) {
          td = tick();
          for (const d of adets.filter((x) => detScore(x) >= this.a.posture!.min_detection_score).sort((x, y) => detScore(y) - detScore(x)).slice(0, this.a.posture!.max_per_frame)) {
            d.posture = await this.posture.predict(fr.rgb, d.box, species(d));
          }
          posMs += tick() - td;
        }
        let shift: [number, number] = [0, 0];
        if (prevGray && !isStill) {
          tq = tick();
          const cm = estimateCameraMotion(prevGray, gray, fr.rgb.width, fr.rgb.height, [...prevBoxes, ...adets.map((d) => d.box)]);
          if (!cm || !cm.reliable) cameraUnknown++;
          else if (Math.hypot(cm.dx, cm.dy) > v.quality.camera_motion_threshold) {
            cameraMovingTimes.push(fr.t_s);
            if (v.camera_motion_compensation) { shift = [cm.dx, cm.dy]; cameraShift.set(fr.t_s, shift); }
          }
          qMs += tick() - tq;
        }
        tracker.update(fr.t_s, adets, shift);
        prevGray = gray;
        prevBoxes = adets.map((d) => d.box);
      }
      timings.video_decode_detect_track = tick() - t;
      timings.video_detector_inference = detMs;
      timings.video_verifier = verMs;
      timings.video_posture = posMs;
      timings.video_quality_and_camera_motion = qMs;
      const minHits = nFrames >= 2 * v.tracker.min_hits ? v.tracker.min_hits : 1;
      tracks = tracker.tracks(minHits);
      if (this.verifier) {
        const kept: Track[] = [], rejected: Track[] = [];
        for (const tr of tracks) {
          const vs = tr.detections.map((d) => d.verify_score).filter((x): x is number => x !== null);
          const cv = this.a.verifier!.params.classes[trackSpecies(tr)];
          if (!cv || !vs.length) { kept.push(tr); continue; }
          const s = [...vs].sort((x, y) => x - y);
          const med = s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
          tr.verification = { checked: vs.length, median: med, tau: cv.tau, accepted: med >= cv.tau };
          (med >= cv.tau ? kept : rejected).push(tr);
        }
        tracks = kept;
        if (rejected.length) qflags.push({ code: "detections_rejected_by_verifier", severity: "info", t_start_s: Math.min(...rejected.map(firstT)),
          t_end_s: Math.max(...rejected.map(lastT)), detail: `${rejected.length} animal-like track(s) did not pass the second-stage check and are not reported as animals` });
      }
      videoState = nFrames > 0 ? "present" : "no_decodable_frames";
      for (const [code, times, det] of [["low_light", lowLight, "mean luma below threshold"], ["blurry_frames", blurry, "low Laplacian variance"]] as const) {
        if (times.length) qflags.push({ code, severity: "warning", t_start_s: Math.min(...times), t_end_s: Math.max(...times), detail: `${det} in ${times.length}/${nFrames} sampled frames` });
      }
      if (cameraMovingTimes.length) qflags.push({ code: "camera_motion", severity: "warning", t_start_s: Math.min(...cameraMovingTimes), t_end_s: Math.max(...cameraMovingTimes),
        detail: "global background motion detected; " + (v.camera_motion_compensation ? "translation-compensated (approximate)" : "not compensated") });
      if (cameraUnknown && nFrames > 1) qflags.push({ code: "camera_motion_unknown", severity: "info", detail: `background motion could not be estimated for ${cameraUnknown} frame pairs` });
      if (isStill) qflags.push({ code: "still_image", severity: "info", detail: "single frame: movement and dwell time cannot be measured" });
    }

    // ---------------- observations & evidence ----------------
    report("fusion", 0.92);
    t = tick();
    const evidence: Ev[] = [];
    const trackModels: Record<string, unknown>[] = [];
    const bp = a.behavior;
    const series = new Map<string, BH.StepSeries>();
    const mediaEnd = p.duration_s;
    const detectorId = a.detector?.model_id ?? null;
    for (const tr of tracks) {
      const tid = tr.track_id;
      const sp = trackSpecies(tr);
      const flags = [...new Set(tr.detections.flatMap((d) => d.flags))].sort();
      const mv = movementSummary(tr, frameSize[0], frameSize[1], a.vision.camera_motion_compensation ? cameraShift : null) as { available: boolean; median_speed: number; p90_speed: number; moving_fraction: number; steps: number; [k: string]: unknown };
      const f0 = firstT(tr), f1 = lastT(tr);
      const spanCamera = cameraMovingTimes.some((x) => f0 <= x && x <= f1);
      const spanDark = lowLight.some((x) => f0 <= x && x <= f1);
      const mvRel = !mv.available || tr.detections.length < 3 || spanDark ? "low" : spanCamera || cameraUnknown ? "medium" : "high";
      const times = tr.detections.map((d) => d.t_s), boxes = tr.detections.map((d) => d.box);
      const zoneOcc = opts.zones ? occupancy(times, boxes, opts.zones, a.vision.tracker.max_gap_s) : [];
      let segs: BH.Segment[] = [];
      if (bp.enabled && !isStill && tr.detections.length >= 2 && frameSize[0]) {
        const ss = BH.stepSeries(times, boxes, frameSize[0], frameSize[1], a.vision.camera_motion_compensation ? cameraShift : null, a.vision.tracker.max_gap_s);
        series.set(tid, ss);
        const brel = spanDark || tr.detections.length < 3 ? "low" : "medium";
        segs = [...BH.motionSegments(ss, bp, tid, brel), ...BH.pacingSegments(ss, bp, tid),
          ...BH.zoneSegments(times, boxes, opts.zones, frameSize[0], frameSize[1], bp, a.vision.tracker.max_gap_s, tid, brel),
          ...BH.viewSegments(times, boxes, 0, mediaEnd, bp, tid, brel),
          ...(this.interaction ? BH.interactionSegments(times, tr.detections.map((d) => d.interaction), this.a.interaction!.threshold, this.a.interaction!.min_frames, a.vision.tracker.max_gap_s, tid) : []),
        ].sort((x, y) => x.t_start_s - y.t_start_s || (x.kind < y.kind ? -1 : 1));
      }
      const meanScore = tr.detections.reduce((s, d) => s + detScore(d), 0) / tr.detections.length;
      let trackRel = meanScore >= 0.6 && !flags.some((x) => x === "tiny_subject" || x === "species_ambiguous") ? "high" : meanScore >= 0.45 ? "medium" : "low";
      if (spanDark) trackRel = "low";
      const evIds: string[] = [];
      const eTrack: EvidenceItem = { evidence_id: `ev-${tid}`, kind: "animal_track", t_start_s: f0, t_end_s: f1, track_id: tid,
        summary: `${sp} detected (${tid}) from ${f0.toFixed(2)}s to ${f1.toFixed(2)}s in ${tr.detections.length} sampled frames`, value: meanScore,
        value_kind: "mean_detector_score_raw", reliability: trackRel as EvidenceItem["reliability"],
        provenance: { component: "visual_detector", model_id: detectorId, root: "vision:detector", method: `efficientdet+${a.vision.tracker.method}_tracker` } };
      evidence.push({ item: eTrack, root: "vision:detector", tags: new Set(["animal_visible", `species:${sp}`]), strength: 1, attributed_track: null });
      evIds.push(eTrack.evidence_id);
      if (mv.available) {
        const tags = new Set<string>();
        if (mv.moving_fraction >= 0.5 && mv.p90_speed >= 0.15) tags.add("movement:high");
        else if (mv.moving_fraction <= 0.1 && f1 - f0 >= 3.0) tags.add("movement:low");
        const desc = tags.has("movement:high") ? "high" : tags.has("movement:low") ? "low" : "moderate";
        const eMv: EvidenceItem = { evidence_id: `ev-move-${tid}`, kind: "movement", t_start_s: f0, t_end_s: f1, track_id: tid,
          summary: `${desc} apparent movement of ${tid} (median ${mv.median_speed.toFixed(3)} image diagonals/s; moving in ${Math.round(mv.moving_fraction * mv.steps)} of ${mv.steps} sampled steps)`,
          value: mv.median_speed, value_kind: "image_diagonals_per_second", reliability: mvRel as EvidenceItem["reliability"],
          provenance: { component: "tracker", model_id: detectorId, root: "vision:detector", method: "centroid displacement, camera-translation compensated when estimated" } };
        evidence.push({ item: eMv, root: "vision:detector", tags, strength: 1, attributed_track: null });
        evIds.push(eMv.evidence_id);
      }
      for (const z of zoneOcc) {
        const ok = z.dwell_s >= a.fusion.min_zone_dwell_s || isStill;
        const eZ: EvidenceItem = { evidence_id: `ev-zone-${tid}-${z.zone_id}`, kind: "zone_occupancy", t_start_s: z.intervals_s[0][0], t_end_s: z.intervals_s[z.intervals_s.length - 1][1],
          track_id: tid, summary: `${tid} in owner-marked zone '${z.label}' (${z.semantic_tag}) for ${z.dwell_s.toFixed(1)}s`, value: z.dwell_s, value_kind: "dwell_seconds_sampled",
          reliability: isStill || spanDark ? "low" : "medium", provenance: { component: "context", model_id: null, root: "context:zones", method: `box ${z.anchor} point in owner polygon` } };
        evidence.push({ item: eZ, root: "context:zones", tags: ok ? new Set([`zone:${z.semantic_tag}`]) : new Set(), strength: 1, attributed_track: null });
        evIds.push(eZ.evidence_id);
      }
      const poseReason = sp === "cat" ? "no cat keypoint model is available" : "no keypoint model in the on-device bundle (dog keypoints are research-only)";
      trackModels.push({
        track_id: tid, identity_note: "per-analysis track label; not a verified pet identity", species: sp, species_scores: speciesSummary(tr),
        score_kind: "detector_sigmoid_raw", first_seen_s: f0, last_seen_s: f1, duration_s: f1 - f0, n_detections: tr.detections.length,
        missed_sampled_frames: tr.missed_frames,
        detections: tr.detections.map((d) => ({ t_s: d.t_s, box_xyxy_norm: d.box, species_scores: d.species_scores, flags: d.flags })),
        movement: { ...mv, reliability: mvRel }, zones: zoneOcc,
        pose_features: POSE_FEATURES.map((name) => ({ name, available: false, value: null, units: null, reason: poseReason })),
        quality_flags: flags, evidence_ids: evIds,
        behavior_segments: segs.map(({ validation, ...g }, k) => ({ segment_id: `beh-${tid}-${k + 1}`, ...g, validation_status: validation ?? "synthetic_and_semi_synthetic_only" })),
        keypoints: [],
        posture: tr.detections.filter((d) => d.posture).map((d) => ({ t_s: d.t_s, label: d.posture!.label, score: Math.round(d.posture!.score * 1000) / 1000, score_kind: "posture_softmax_uncalibrated" })),
        verification: tr.verification ? { verifier_id: "verifier-v1:crop", checked_detections: tr.verification.checked, median_score: tr.verification.median,
          threshold: tr.verification.tau, accepted: tr.verification.accepted, score_kind: "verifier_rejection_score_uncalibrated" } : null,
      });
    }

    const soundModels: Record<string, unknown>[] = [], contextModels: Record<string, unknown>[] = [];
    const videoOk = videoState === "present" && !!this.detector;
    rawEvents.forEach((ev, idx) => {
      const k = idx + 1;
      const eid = `ev-snd-${k}`;
      const isContext = ev.label.startsWith("context:");
      let att: { status: string; candidate_track_ids: string[]; method: string; note: string };
      if (isContext) att = { status: "unknown", candidate_track_ids: [], method: "none_evaluated", note: "environmental sound; not attributed to an animal" };
      else if (!videoOk) att = { status: "video_unavailable", candidate_track_ids: [], method: "none_evaluated", note: "no usable video to relate the sound to a visible animal" };
      else {
        const w = a.fusion.co_occurrence_window_s;
        const cands = tracks.filter((tr) => firstT(tr) - w <= ev.end_s && lastT(tr) + w >= ev.start_s);
        const need = ({ cat_vocalization: "cat", dog_vocalization: "dog" } as Record<string, string>)[ev.label];
        if (!cands.length) att = { status: "no_visible_animal", candidate_track_ids: [], method: "none_evaluated", note: "no animal visible near this time; the source may be off-screen or a TV/speaker" };
        else if (cands.length === 1 && (!need || trackSpecies(cands[0]) === need)) att = { status: "single_visible_candidate", candidate_track_ids: [cands[0].track_id], method: "none_evaluated", note: "exactly one species-consistent animal visible; attribution is unverified" };
        else att = { status: "unknown", candidate_track_ids: cands.map((c) => c.track_id), method: "none_evaluated",
          note: cands.length > 1 ? "several animals visible; no evaluated audio-visual association method, so attribution is unknown" : "the only visible animal's species does not match the sound; source may be off-screen" };
      }
      const sm = { event_id: `snd-${k}`, label: ev.label, t_start_s: ev.start_s, t_end_s: ev.end_s, peak_raw_score: ev.peak_score, mean_raw_score: ev.mean_score,
        threshold: ev.threshold, threshold_source: isContext ? "config context_tag_threshold (not selected on data)" : (a.audio!.threshold_source[ev.label] ?? "config default (not selected on data)"),
        score_kind: "yamnet_sigmoid_raw", calibration_status: "uncalibrated", calibrated_probability: null, subtypes: ev.subtypes, top_audioset_labels: ev.top_audioset_labels,
        attribution: att, evidence_id: eid };
      (isContext ? contextModels : soundModels).push(sm);
      const tags = new Set<string>();
      if (isContext) tags.add(ev.label.split(":")[1]);
      else {
        const strong = new Set(Object.entries(ev.subtypes).filter(([, v]) => v >= SUBTYPE_MIN).map(([k2]) => k2));
        for (const x of ["hiss", "growl", "purr", "bark", "howl"]) if (strong.has(x)) tags.add(x);
        if (ev.label !== "growl" && ([...strong].some((x) => REQUEST_LIKE.has(x)) || !strong.size)) tags.add("vocalization");
        if (ev.label === "growl") tags.add("growl");
      }
      const subTxt = Object.entries(ev.subtypes).sort((x, y) => y[1] - x[1]).slice(0, 3).map(([k2, v2]) => `${k2} ${v2.toFixed(2)}`).join(", ");
      const labelTxt = ev.label.replace("context:", "").replace(/_/g, " ");
      const item: EvidenceItem = { evidence_id: eid, kind: isContext ? "context_sound" : "sound_event", t_start_s: ev.start_s, t_end_s: ev.end_s, track_id: null,
        summary: `${labelTxt} sound ${ev.start_s.toFixed(2)}-${ev.end_s.toFixed(2)}s (peak raw score ${ev.peak_score.toFixed(2)}${subTxt ? "; " + subTxt : ""})`,
        value: ev.peak_score, value_kind: "yamnet_sigmoid_raw", reliability: "high",
        provenance: { component: "audio_event", model_id: a.audio!.model_id, root: "audio:yamnet", method: "YAMNet tags, ontology group max, thresholded" } };
      evidence.push({ item, root: "audio:yamnet", tags, strength: 1, attributed_track: att.status === "single_visible_candidate" ? att.candidate_track_ids[0] : null });
    });

    const behaviorEvents = bp.enabled ? [...BH.vocalBouts(soundModels.map((m) => [m.label as string, m.t_start_s as number, m.t_end_s as number]), bp),
      ...BH.startleCandidates(series, contextModels.map((m) => [(m.label as string).replace("context:", ""), m.t_start_s as number]), bp)]
      .sort((x, y) => x.t_start_s - y.t_start_s || (x.kind < y.kind ? -1 : 1)).map((g, k) => ({ segment_id: `beh-${k + 1}`, ...g, validation_status: "synthetic_and_semi_synthetic_only" })) : [];

    const scene = evidence.filter((e) => e.item.kind === "context_object" || e.item.kind === "context_sound");
    const expanded: Ev[] = [];
    for (const e of evidence) {
      if (scene.includes(e)) for (const tm of trackModels) expanded.push({ ...e, attributed_track: tm.track_id as string });
      expanded.push(e);
    }
    const qcodes = new Set(qflags.map((q) => q.code as string));
    const fctx = { audio_available: audioState === "present" && !!this.m.yamnet, audio_state: audioState, video_available: videoOk,
      zones_configured: !!(opts.zones && opts.zones.zones.length), person_detection_enabled: false, n_tracks: tracks.length,
      gates: { "audio:yamnet": 1.0, "vision:detector": qcodes.has("low_light") || qcodes.has("camera_motion") ? 0.6 : 1.0, "context:zones": 1.0 } };
    const hyps: Hypothesis[] = this.fusion.evaluate(expanded, fctx);
    const evIndex = new Map(evidence.map((e) => [e.item.evidence_id, e.item]));
    for (const h of hyps) h.explanation = explainHypothesis(h, evIndex);
    timings.fusion_and_behavior = tick() - t;

    // ---------------- decision ----------------
    const reasons: string[] = [];
    const petSound = soundModels.length > 0;
    if ((audioState === "absent" || audioState === "disabled") && opts.modalities.includes("audio")) reasons.push("missing_modality_audio");
    if (audioState === "corrupt") reasons.push("audio_corrupt");
    if (audioState === "silent") reasons.push("audio_silent");
    if ((videoState === "absent" || videoState === "disabled") && opts.modalities.includes("video")) reasons.push("missing_modality_video");
    if (videoState === "no_decodable_frames" || videoState === "corrupt") reasons.push("no_decodable_video_frames");
    if ((audioState === "present" && !this.m.yamnet) || (p.kind !== "audio" && opts.modalities.includes("video") && !this.detector)) reasons.push("model_unavailable");
    if (videoOk && !tracks.length) reasons.push("no_pet_detected");
    if (audioState === "present" && this.m.yamnet && !petSound) reasons.push("no_pet_sound_detected");
    if (tracks.length > 1) reasons.push("multiple_animals_visible");
    const experimental = hyps.filter((h) => h.status === "experimental_hypothesis");
    let state: string, interp: string, summary: string;
    if (experimental.length) {
      state = "experimental_hypothesis"; interp = "experimental_hypotheses_present";
      summary = `${experimental.length} experimental hypothesis(es) with co-occurring independent evidence; scores are uncalibrated.`;
    } else if (tracks.length || petSound) {
      state = "supported_observation"; interp = hyps.length ? "insufficient_evidence" : "observations_only";
      summary = `Observations only: ${tracks.length} animal track(s), ${soundModels.length} pet sound event(s). ` + (hyps.length ? "Candidate interpretations lacked sufficient independent evidence." : "No interpretation was attempted.");
    } else {
      state = "abstention"; interp = "unknown"; summary = "No cat or dog was detected in the available modalities; no interpretation is offered.";
      if (!reasons.length) reasons.push("no_pet_detected");
    }
    const avail: string[] = [];
    if (audioState === "present") avail.push("audio");
    if (videoOk && nFrames) avail.push("video");
    report("done", 1);
    return {
      schema_version: "1.2.0", analysis_id: opts.analysisId, created_at: new Date().toISOString(), engine_version: ENGINE_VERSION,
      config_profile: "ondevice", config_hash: this.info.bundle_version,
      media: { analysis_filename: opts.name.replace(/[^A-Za-z0-9._-]/g, "_").slice(0, 120) || "recording", size_bytes: file.size, sha256: digest, container: p.mime,
        duration_s: p.duration_s, start_time_s: 0,
        audio: decodedDuration > 0 ? { codec: "browser-decoded", sample_rate_hz: 16000, channels: 1, first_sample_time_s: 0, decoded_duration_s: decodedDuration } : null,
        video: p.kind !== "audio" ? { codec: "browser-decoded", width: p.width, height: p.height, avg_fps: null, start_time_s: 0, duration_s: p.duration_s, rotation_deg: 0, is_still_image: isStill } : null,
        timeline: "container_pts_seconds" },
      modalities: { requested: opts.modalities, available: avail, audio_state: audioState, video_state: videoState },
      components: this.components(),
      observations: { animal_tracks: trackModels, sound_events: soundModels, context_sounds: contextModels, audio_summary: audioSummary,
        evidence: evidence.map((e) => e.item), sampled_frames: nFrames, sample_fps: nFrames ? a.vision.sample_fps : null, behavior_events: behaviorEvents },
      quality_flags: qflags.map((q) => ({ t_start_s: null, t_end_s: null, detail: null, track_id: null, ...q })),
      hypotheses: hyps,
      decision: { state, overall_interpretation: interp, reason_codes: [...new Set(reasons)], summary },
      suggestions: suggestionsFor(hyps, qcodes, tracks.length, petSound),
      limitations: [...STATIC_LIMITATIONS, ...(isStill ? ["Single still image: movement, dwell time and temporal co-occurrence cannot be measured."] : []),
        "Ear, tail and keypoint-based features are unavailable on-device: dog keypoints are research-only and no cat keypoint model exists.",
        "Dropped-frame and native audio bandwidth checks are not available in the browser runtime.", ...a.notes],
      fusion: { mode: "experimental", method: a.fusion.method, validated: false, multimodal_accuracy: "not_evaluated", weights_fitted: false,
        note: "Experimental fusion; weights/temperature are unfitted configuration values." },
      timings_ms: Object.fromEntries(Object.entries(timings).map(([k, v]) => [k, Math.round(v * 100) / 100])),
      models: this.info.models.map((m) => ({ model_id: m.model_id, name: m.file, revision: `on-device bundle ${this.info.bundle_version}`, sha256: { [m.file]: m.sha256 }, terms: {}, use_restrictions: [] })),
      usage_restrictions: { research_only: a.release_profile === "research", notices: [`Release profile '${a.release_profile}': only assets permitted for this profile are in the on-device bundle.`,
        "Model weights and evaluation data carry third-party terms (see docs/DATA_AND_MODEL_REGISTER.md).", "Media were analysed on this device and were not uploaded."] },
    };
  }
}
