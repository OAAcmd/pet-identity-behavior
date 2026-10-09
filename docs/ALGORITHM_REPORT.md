# Phase 1 Algorithm Report

Scope: an evidence-based behavior interpreter for cats and dogs. The engine
reports **observations** first (detected animals, sounds, movement, zone
occupancy, timestamps, quality limits), then **experimental hypotheses** with
evidence and explicit uncertainty, then **restrained suggestions**. It does not
translate animal language, read minds or emotions, or diagnose.

All numbers below were measured in this workspace (4 vCPU Intel Xeon @ 2.10 GHz,
15.7 GB RAM, no GPU, Python 3.12.3, FFmpeg 6.1.1) and can be regenerated with
the commands shown. Results from different datasets are **not** combined into a
product-level accuracy.

## 1. Architecture

```
media file ──► ingest (ffprobe → limits → decode on one PTS timeline)
                 ├─ audio (native rate) ──► YAMNet tags (16 kHz resample, 0.975 s windows) ──► sound events
                 │                         └─ MFCC/spectral features (CPU baselines, E1/E2)
                 └─ video frames (original pts) ──► EfficientDet-Lite cat/dog ──► species merge
                                                    ├─ quality (luma, blur, tiny, truncation, overlap)
                                                    ├─ camera motion (ORB + RANSAC on background)
                                                    └─ IoU/Hungarian tracker ──► movement, zones, pose(unavailable)
observations + evidence (provenance roots) ──► experimental fusion ──► decision / abstention
                                                                     └─► deterministic explanations + suggestions
```

Package layout: `ingest/`, `audio/`, `vision/`, `context/`, `fusion/`,
`schemas/`, `explain/`, `eval/`, `serve/`, `data/`, plus `pipeline.py` (engine)
and `cli.py`. The CLI and API call the same `AnalysisEngine`.

## 2. Ingestion and synchronization

- `ffprobe` runs before any decoding; size, duration and channel limits are
  enforced (`configs/cpu.yaml: limits`). Errors carry stable codes
  (`corrupt_input`, `file_too_large`, `duration_too_long`, `unsupported_input`).
- **One timeline:** container presentation time in seconds. Video frames are
  *selected* (not re-timed) with an FFmpeg `select` filter under `-copyts`, so
  each frame keeps its original `pts_time`. Audio is decoded at its native rate;
  sample *i* is at `t0 + i/sr`, where `t0` is the timestamp of the first
  *decoded* audio frame (ffprobe), which accounts for AAC priming and Opus
  pre-skip that the stream `start_time` does not.
- Verified with authored flash+click media: alignment error < 0.4 ms in MP4,
  in MKV with a 0.75 s offset (timestamps preserved, not re-based) and in
  WebM/Opus (`tests/test_media.py`). Before the fix, the MKV flash was reported
  0.73 s early and Opus audio 7 ms early.
- States are distinct: audio `present | absent | silent | corrupt | disabled | too_short`;
  video `present | absent | no_decodable_frames | corrupt | disabled`. Cover art
  (`attached_pic`) is not treated as video. Still images are single frames at t = 0.
- Resampling happens only inside model adapters (YAMNet: 16 kHz polyphase).
  Inputs below 16 kHz get a `bandwidth_limited_audio` flag: upsampling 8 kHz audio
  does not recreate content above 4 kHz.
- Supported containers/codecs: anything the installed FFmpeg decodes; tested WAV,
  FLAC, OGG/Vorbis, MP3, MP4/MOV (H.264/AAC), MKV/WebM (VP9/Opus), AVI, JPEG/PNG.
- Windows: audio 0.975 s / 0.48 s hop with a zero-padded final window; frames at
  `vision.sample_fps` (2 fps default) with a frame cap. Packet timestamps are
  scanned for gaps (`dropped_or_missing_frames`).
- Grouping rule for training/evaluation: source groups (animal, source
  recording, Flickr author) are partitioned before any windowing; Phase 1 uses
  no augmentation. Post-event outcomes and labels are never inference inputs;
  owner zones describe the scene and are never used as labels.

## 3. Audio

### 3.1 Frozen audio-event backend (YAMNet, MediaPipe TFLite)

AST could not be used (huggingface.co denied; see register). YAMNet was chosen
because its weights, label map and preprocessing could be verified from
reachable primary sources. Interface facts verified by graph inspection:
in-graph log-mel frontend, in-graph `LOGISTIC` output (independent multi-label
sigmoid scores - softmax is never applied), int8 internals (scores in 1/256
steps). Species groups come from the AudioSet ontology: `cat_vocalization` =
Cat + descendants (Purr, Meow, Hiss, Caterwaul); `dog_vocalization` = Dog +
descendants (Bark, Yip, Howl, Bow-wow, Whimper (dog)). "Growling" is a child of
both Cat and Dog, so it is reported as a species-ambiguous `growl` event.
Group score = max over member classes per window; events = merged windows above
threshold (event boundaries are window-resolution: an onset can be reported up
to ~0.5 s early). Raw scores are reported as `yamnet_sigmoid_raw`,
`calibration_status: uncalibrated`, `calibrated_probability: null`.
Thresholds: cat 0.738, dog 0.801 (F1-optimal on all five ESC-50 folds; their
generalization estimate is the nested E1 result). The growl threshold (0.5) and
context-tag threshold (0.3) are config defaults, labelled "not selected on data".

### 3.2 CPU MFCC/spectral baseline

20 MFCC + deltas (40 mel bands) + spectral centroid, bandwidth, roll-off,
flatness, ZCR, RMS; mean and std over frames; plus active duration (93 features;
`audio/features.py`). Classifier: standardized multinomial logistic regression,
class-balanced, `C` chosen by inner grouped CV on training partitions only.
Models are persisted as numeric JSON with the feature configuration, label map,
training manifest and restrictions (`audio/linear_model.py`; no pickle).

## 4. Vision

- Detector: MediaPipe EfficientDet-Lite0 (default) / Lite2 (optional). Decoding,
  anchors and normalization are parsed from the model's embedded metadata (see
  register); per-class NMS is implemented in the package. Cat and dog boxes that
  overlap with IoU >= 0.7 are merged into one detection with per-species scores
  and a `species_ambiguous` flag when the runner-up is >= 60% of the top score.
- Tracker: Hungarian assignment on IoU (>= 0.3), time-based gap limit (1.5 s),
  minimum 2 hits (1 for very short media). Predicted boxes are shifted by the
  estimated camera translation. Track IDs are per-analysis labels, never identities.
- Camera motion: ORB features + RANSAC similarity on the background (animal boxes
  masked). Background translation > 1% of frame or scale change > 2% raises
  `camera_motion`; the translation is subtracted from centroid displacement
  (approximation; rotation/parallax are not modelled). Unestimable pairs raise
  `camera_motion_unknown` and lower movement reliability.
- Movement: centroid speed in **image diagonals per second** and in
  **box-size units per second**; moving fraction, path length, box-area change.
  Units are image-relative, never physical speed.
- Quality: low light (mean luma < 40), blur (Laplacian variance < 25), tiny
  subjects (< 0.4% of frame), truncation at frame edges, overlapping animals
  (possible occlusion), dropped frames, frame-limit truncation.
- Zones: owner-configured polygons in normalized image coordinates with a fixed
  semantic vocabulary (`food_area`, `water_area`, `exit_door`, `rest_area`,
  `play_area`, `litter_area`, `window`, `other`); membership tested at the box
  bottom-center (ground-contact approximation); dwell intervals per track.
- Pose: provider interface + `UnavailablePoseProvider`. Features are computed only
  from keypoints a provider declares and that pass a visibility threshold; with
  no verified animal pose model, ear angle, tail height, play-bow candidate and
  tail oscillation are reported unavailable with reasons. They are never
  estimated from a bounding box. A human-pose vocabulary yields no animal features (tested).
- Optional VLM adapter: not implemented; config allows only a disabled-by-default
  loopback endpoint. No media is sent to any external API.

## 5. Experimental fusion and abstention

Hypothesis ontology (`schemas/ontology.py`): `food`, `outside`, `attention`
(needs), `play` (activity), `alert`, `tense`, `relaxed` (states), `warning`
(communicative signal). They are scored independently and may coexist;
`relaxed` is in *soft* conflict with `tense`/`warning`/`alert`. `unknown` and
`insufficient_evidence` are decision outcomes.

Rules (`fusion/engine.py`, revision `rules-v1`): each hypothesis has
requirement groups (e.g. `food` = request-like vocalization AND dwell in
`food_area`), trigger tags (animal-originated, discriminative), heuristic
expert opinions (0.5 = neutral), and conflict tags.

- **Provenance roots**: `audio:yamnet`, `vision:detector`, `context:zones`,
  `pose:<provider>`. Items with one root count once; a hypothesis needs >= 2
  distinct roots, so a tag + its embedding + a classifier on that embedding can
  never confirm each other. Zone evidence depends on detector boxes too; its
  primary root is the owner's scene configuration (documented approximation).
- **Co-occurrence**: supporting items must lie within 3 s of each other; pairs
  up to 10 s apart are reported as `stale_evidence`.
- **Attribution**: a sound attaches to a track only if exactly one
  species-consistent animal is visible near that time
  (`single_visible_candidate`, still marked unverified). Several animals ->
  `unknown`; none visible -> `no_visible_animal` (off-screen or TV/speaker);
  species mismatch -> `unknown`. No audio-visual association method has been
  evaluated, so nothing is attached to "the largest box".
- **Guards**: context alone never opens `food`/`outside` beyond
  `insufficient_evidence` (`context_only_evidence`); silence produces no
  evidence (so cannot support `relaxed`); purr adds the note
  `purr_does_not_rule_out_discomfort`; conflicts force `abstained`; low
  reliability (low light, still image, bandwidth limits) shrinks expert opinions
  toward neutral.
- **Scorers** (swappable, `fusion.method`):
  1. `rule_baseline` - mean of per-root expert opinions (`heuristic_support_0_1`);
  2. `weighted_average` - normalized weighted average with a coverage factor that
     shrinks toward neutral when relevant experts are missing;
  3. `product_of_experts` - context-gated PoE for a binary hypothesis computed in
     log space: `sigmoid(prior + sum_r g_r w_r logit(clip(s_r, eps, 1-eps)) / T)`
     with gates lowered for bandwidth-limited audio or low-light/moving-camera video.
  None is a Bayesian posterior; weights (`audio 1, vision 1, context 0.5`),
  temperature (2.0), prior logit (-1) and the abstention threshold (0.5) are
  **unfitted configuration values** - there is no paired development data to fit
  them. Every hypothesis carries `calibration_status: uncalibrated` and
  `calibrated_confidence: null`; the result carries
  `fusion.multimodal_accuracy: not_evaluated`.

Explanations and suggestions are deterministic templates that cite existing
evidence IDs; no medical statements, no first-person pet quotes, no percentages.

## 6. Experiments

Commands: `uv run petperception evaluate --experiment <e1..e6>`; outputs in `results/<exp>/`.

### E1 - Cat/dog vs other sound events (ESC-50, clip level) - COMPLETED

Protocol: 2000 clips (40 cat, 40 dog, 1920 other) with checksums verified;
outer CV = the 5 official source-preserving folds; inner leave-one-fold-out on
the 4 training folds for `C`; YAMNet thresholds F1-optimal on training folds
only; 95% CIs by bootstrapping 1524 Freesound source recordings (1000 resamples).
Decision rule for tags: cat if `s_cat >= t_cat`, dog if `s_dog >= t_dog`, ties by
score/threshold ratio, else other.

| Method | Macro-F1 [95% CI] | Bal. acc. | Cat P / R / F1 | Dog P / R / F1 |
| --- | --- | --- | --- | --- |
| Majority (always "other") | 0.327 | 0.333 | 0 / 0 / 0 | 0 / 0 / 0 |
| Stratified random (200 draws) | 0.333 +/- 0.011 | 0.333 | F1 0.019 | F1 0.020 |
| MFCC + LogReg (CPU baseline) | 0.617 [0.551, 0.677] | 0.734 | 0.33 / 0.57 / 0.42 | 0.35 / 0.68 / 0.46 |
| YAMNet tags, nested thresholds | 0.895 [0.846, 0.935] | 0.906 | 0.78 / 0.80 / 0.79 | 0.88 / 0.93 / 0.90 |
| YAMNet embeddings + LogReg (optional) | 0.956 [0.920, 0.982] | 0.982 | 0.83 / 0.95 / 0.88 | 0.98 / 1.00 / 0.99 |

Confusion matrices (rows true cat/dog/other, columns predicted):
MFCC `[[23,1,16],[0,27,13],[46,49,1825]]`; YAMNet tags `[[32,0,8],[0,37,3],[9,5,1906]]`;
YAMNet embeddings `[[38,0,2],[0,40,0],[8,1,1911]]`.
Per-class F1 CIs: YAMNet tags cat [0.667, 0.878], dog [0.828, 0.962].
Per-fold macro-F1 (tags): 0.942, 0.844, 0.884, 0.932, 0.878. Selected thresholds per
fold: cat 0.50/0.50/0.74/0.74/0.59, dog 0.80/0.80/0.85/0.80/0.74.
Threshold-free ranking of raw tag scores: AP cat 0.891, dog 0.989.

False positives on non-pet sounds stay visible (counts of "other" clips predicted as pet):
- YAMNet tags -> cat: snoring 2, airplane, crying baby, door creaks, insects, pouring water, sheep, vacuum cleaner (1 each); -> dog: pig 2, cow, laughing, rooster.
- MFCC -> cat: door creaks 8, rooster 8, siren 7, crying baby 6, hen 4, ...; -> dog: frog 9, hen 6, siren 6, ...
- YAMNet embeddings -> cat: crying baby 2, door creaks 2, rooster 2, crow, sheep; -> dog: rooster.

Calibration: Brier (pooled) cat raw 0.0068 / nested-Platt 0.0056, dog raw 0.0083 /
nested-Platt 0.0018. ECE/reliability curves are not reported (40 positives per
class). None of this transfers to household audio.

Limitations: clip-level labels only (no event-timing evaluation); ESC-50 is
curated Freesound audio, not household recordings; 4 non-pet sources span two
folds; YAMNet/AudioSet vs Freesound overlap unaudited. The runtime uses tag
thresholds (frame-level, transparent); the embedding classifier is a comparison
trained on 40 positives per class and is not used at runtime.

### E2 - CatMeows recording context - BLOCKED (implemented, not run)

zenodo.org is denied, so no CatMeows metric exists. Implemented protocol:
leave-one-cat-out (plus stricter leave-one-owner-out), inner grouped `C`
selection, standardization inside training partitions, folds with missing
classes reported, cat-level bootstrap, majority/stratified baselines, analysis
at native 8 kHz (fmax 3.9 kHz), random split as a labelled leakage diagnostic,
optional embedding comparison flagged out-of-domain. Mechanics are tested only
on synthetic tones (`tests/test_e2_protocol.py`). Labels keep their original
meaning (`isolation` is not "fear"). Continue with
`uv run petperception fetch-data catmeows && uv run petperception evaluate --experiment e2`.

### E3 - Dog activity/context - PENDING (disabled)

Barkopedia is unreachable and unaudited. Identity-disjoint evaluation:
unknown; cross-dog claims: declined. Real dog *sound-event* detection is covered
by E1. Audit path: `uv run petperception fetch-data barkopedia --audit-only`.

### E4 - Visual cat/dog detection (Open Images V7 validation subset) - COMPLETED

Protocol: per class, only images where the class is human-verified (positive
with boxes, or verified negative); depiction and group-of boxes are ignore
regions; AP@IoU0.5 (all-point interpolation); operating threshold 0.35 fixed in
config before evaluation; 95% CIs by bootstrapping Flickr authors.

| Detector | Class | Eval images (pos/neg) | GT boxes | AP@0.5 [95% CI] | P / R / F1 @0.35 | Image-level P / R / F1 [CI] | Specificity on verified negatives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Lite0 (default), 28 ms/img | cat | 386 (339/47) | 370 | 0.923 [0.897, 0.947] | 0.918 / 0.876 / 0.896 | 0.946 / 0.926 / 0.936 [0.915, 0.954] | 0.617 (29/47) |
| Lite0 | dog | 463 (352/111) | 417 | 0.842 [0.789, 0.886] | 0.783 / 0.820 / 0.801 | 0.865 / 0.892 / 0.878 [0.851, 0.901] | 0.559 (62/111) |
| Lite2 (optional), 69 ms/img | cat | 386 | 370 | 0.938 [0.912, 0.961] | 0.938 / 0.897 / 0.917 | 0.955 / 0.929 / 0.942 [0.922, 0.959] | 0.681 (32/47) |
| Lite2 | dog | 463 | 417 | 0.849 [0.798, 0.895] | 0.815 / 0.837 / 0.826 | 0.883 / 0.901 / 0.892 [0.866, 0.914] | 0.622 (69/111) |

Error analysis (Lite0, operating threshold):
- Recall by size: cat boxes 1-10% of frame 0.27 (n=11), >= 10% 0.89; dog boxes < 1%
  0.25 (n=12), 1-10% 0.53 (n=40), >= 10% 0.87. Occluded boxes: cat 0.55, dog 0.61.
- Verified negatives with detections are mostly other animals: "dog" fired on 49
  negatives (14 contain foxes; also lions and other carnivores), "cat" on 18
  (lions, rabbits, foxes, a raccoon, a skunk, a sculpture). The negatives in Open
  Images are hard (they were verified because a classifier suspected the class), so
  specificity here is a pessimistic stress test, not a household estimate.
- Cross-species diagnostic (other species verified, this one not positive):
  cat fired on 13/350 dog images, dog on 22/337 cat images (unverified negatives).
- The in-sample F1-optimal thresholds (0.38 cat, 0.51 dog) are diagnostics only.
- Contamination: COCO (training) and Open Images both come from Flickr; overlap unaudited.
- Tracking metrics are not reported: no labelled cat/dog video tracking set was
  reachable. Pose metrics are not reported: no pose model or pose evaluation set.

### E5 - Fusion behavior (synthetic pairings) - COMPLETED as behavior checks

14 scenarios pair real public photos (Open Images) and real clips (ESC-50) that
were never recorded together; detection and tagging run for real, pairing is
synthetic. Each scenario is run with all three scorers and checked against
scenario expectations plus global invariants (no calibrated confidence, >= 2 roots
for any experimental hypothesis, evidence IDs resolve, explanations cite existing
evidence, audio only used with a single species-consistent candidate, no
forbidden phrasing, fusion marked unvalidated). **Result: 14/14 pass for each
scorer.** Multimodal accuracy: `not_evaluated`.

| Scenario | Rule baseline | Weighted avg. | PoE |
| --- | --- | --- | --- |
| s01 meow while cat in food zone | food experimental (0.63) | food experimental (0.64) | food abstained (low support 0.36) |
| s02 cat in food zone, silent audio | observations; food insufficient (`context_only_evidence`, `audio_silent`) | same | same |
| s03 cat + dog visible, bark | observations; bark attribution `unknown`; all hypotheses insufficient | same | same |
| s04 no pet, vacuum cleaner | abstention (`no_pet_detected`) | same | same |
| s05 dog visible, meow audio | meow attribution `unknown` (species mismatch) | same | same |
| s06 still cat in rest zone + growl | warning and tense experimental; relaxed **abstained** (conflict) | same | all abstained (low support) |
| s07 audio only | `missing_modality_video`, nothing experimental | same | same |
| s08 video only | `missing_modality_audio`, food insufficient | same | same |
| s09 synthetic camera pan over a still cat | `camera_motion` flag; compensated moving fraction 0.00 (uncompensated 0.53) | same | same |
| s10 cat crop moving in play zone | play experimental (0.59) | play experimental (0.55) | play abstained |
| s11 truncated MP4 | error `corrupt_input` | same | same |
| s12 meow at 8 kHz | `bandwidth_limited_audio` | same | same |
| s13 low light + meow in food zone | `low_light`; food experimental with `low_reliability_evidence` | same | food abstained |
| s14 still cat in rest zone, silent | relaxed experimental from vision + zone only (`silence_not_used_as_evidence`) | same | relaxed abstained |

The PoE candidate with the configured prior/temperature is markedly more
conservative. That is a property of unfitted parameters, not evidence that one
scorer is better.

### E6 - Runtime (CPU, warm engine, median of 3) - COMPLETED

Cold start in a fresh process: imports 1.33 s, model loading 0.60 s (earlier run:
0.97 s / 0.26 s). Peak RSS of the measuring process: 271 MB.

| Input | Wall (s) | Real-time factor | Main stages (ms) |
| --- | --- | --- | --- |
| 8 s video 640x480 + AAC audio (16 frames, 16 audio windows) | 0.95 | 0.119 | probe 60, audio decode 152, YAMNet 24, video total 646 (detector 354, quality+camera 200), fusion 0.3 |
| 64 s video 640x480 + audio (128 frames, 133 windows) | 5.36 | 0.084 | audio decode 383, YAMNet 192, video total 4653 (detector 2752, quality+camera 1636) |
| 5 s audio only, 44.1 kHz | 0.22 | 0.044 | probe 58, decode 130, YAMNet 21 |
| still JPEG | 0.14 | n/a | probe 51, decode+detect 87 (detector 23) |

JSON serialization of a result: < 0.4 ms. Configuration: 2 threads, 2 fps sampling,
EfficientDet-Lite0. No accelerator was available or tested.

## 7. What Phase 1 does not establish

- No household audio/video validation; no intent, need or emotion accuracy.
- No CatMeows or dog-context results (data blocked).
- No animal pose, no pet identity across sessions, no tracking accuracy.
- No calibrated hypothesis confidence; fusion weights are unfitted.
- Historical targets (90% event F1, 60% cat-context accuracy, 70% selective
  intent accuracy, 5-15 point personalization gain) remain hypotheses. Only the
  first relates to a measured task, and only on ESC-50 clips.
