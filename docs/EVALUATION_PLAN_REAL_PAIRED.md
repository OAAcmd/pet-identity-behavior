# Evaluation plan: real paired household recordings

**Status: not started (no data).** Every accuracy number in this repository
comes from public web photos, YouTube video, curated sound clips or synthetic
compositions. None of them shows how the product behaves on a phone camera
in a real home, and none can validate the multimodal interpretations
(hypotheses). This plan specifies the data and protocol that would. It is
written before any data exists so that thresholds and analyses are fixed in
advance.

## 1. What must be validated

| Layer | Question | Current evidence |
| --- | --- | --- |
| Observations | Are cats, dogs, people and furniture found, tracked and measured correctly in household recordings? | web photos (E4, E7, E13, E15), semi-synthetic video (E8) |
| Body cues | Posture, ears, tail (dogs), vocalization type | weak labels (E10), research-only dog data (E9), none for vocal types |
| Temporal behavior | Are activity, zone, interaction, on-furniture and vocal-bout segments correct in time? | semi-synthetic only (E8); interaction and on-furniture per frame on web photos (E13, E15) |
| Interpretation | When the app proposes "possibly food-related" (or abstains), is that right, and is abstention used where it should be? | none (E5 synthetic checks only) |
| Device | Does the phone produce the same result as the reference engine on the same recording? | desktop Chromium only |

## 2. Data to collect

Paired means: each recording has synchronized audio and video **and**
independent labels for what happened.

* **Households:** at least 60 (30 with cats, 30 with dogs), including at
  least 15 multi-pet households and both species together in at least 5.
  Recruit for breed and morphology spread (coat colour and length, ear
  carriage, tail type, brachycephalic breeds, size from toy to giant dogs),
  record the breed as reported by the owner, never inferred.
* **Recordings:** at least 20 per household, 10-60 s, made with the app on
  the owner's phone, covering the contexts the hypotheses address: before
  feeding, waiting at the door, play, greeting, visitor or doorbell, rest,
  conflict or tension when it happens naturally (never provoked), and
  ordinary moments with no particular context. Include pet-free recordings
  of the same rooms (false-positive control) and night or low-light scenes.
* **Owner context labels** at recording time (in-app, before seeing any
  result): what was happening, what the pet did next (e.g. was fed, went
  out, played), and a free-text note.
* **Independent annotation** by two trained annotators per recording, blind
  to the app's output: timestamped animal presence and identity (which pet),
  activity intervals, vocalization events and types, posture per sampled
  frame, person interaction intervals, on-furniture intervals (which piece,
  with "in front of / beside" marked separately because 2-D boxes confuse it
  with "on", E15), and the hypothesis labels with an
  explicit "cannot tell" option. Disagreements are adjudicated; agreement
  (Cohen's kappa per label) is reported because it bounds what any model can
  achieve.
* **Devices:** recordings come from the participants' own phones; the
  device model and browser are recorded so results can be broken down by
  tier.

Consent and privacy: participation is opt-in with written research consent
covering media retention, annotation and publication of aggregate results;
people appearing in recordings must consent or be blurred before annotation;
participants can withdraw and have their media deleted. The consent text
requires legal review before use. The app already stores feedback with an
explicit `consent_research_use` flag; a research upload flow (media only
with that consent) and an annotation tool still need to be built.

## 3. Splits

Split by **household** (never across one household), stratified by species
and multi-pet status: 50 % development (calibration, thresholds, gates),
50 % test, locked until the end. Device models are spread across both. If
new models are trained on household data (TRAINING_PROGRAM.md, G1/G3), their
training households come from a third, separate pool.

## 4. Metrics and acceptance thresholds (fixed in advance)

All with 95 % confidence intervals from household-level bootstrap.

| Component | Metric | Acceptance on test households |
| --- | --- | --- |
| Pet detection | recall of annotated pets per sampled frame; specificity on pet-free recordings | recall ≥ 0.85; pet-free recordings with any reported pet ≤ 5 % |
| Tracking | IDF1 in multi-pet recordings | ≥ 0.75 |
| Activity / zone segments | segment F1 with 1 s tolerance | ≥ 0.70 per segment kind |
| Vocalization | event F1 (cat, dog); type accuracy where the type is annotated | event F1 ≥ 0.80 |
| Posture | agreement with adjudicated posture on reported frames; coverage | agreement ≥ 0.80 at the shipped gate |
| Person interaction | interval F1 | ≥ 0.70 |
| On furniture | interval F1 per object; false "on" rate on "in front of / beside" intervals | F1 ≥ 0.70; false "on" ≤ 0.20 |
| Hypotheses | precision of each non-abstained hypothesis; coverage; risk-coverage curve; calibration error after fitting calibration on development households | precision ≥ 0.70 per hypothesis at the shipped abstention setting, coverage reported as is |
| On-device vs reference | decision agreement and segment F1 between phone and server analysis of the same file | decision agreement ≥ 0.95 |

A component that misses its threshold is reported as not validated and its
outputs stay labelled experimental in the app. Hypotheses that miss their
threshold are disabled or remain abstaining, never shown as findings.

## 5. Sample size rationale

For a hypothesis with true precision 0.70, about 80 non-abstained positive
predictions give a 95 % interval of roughly ±0.10; with household
clustering (assumed design effect 1.5) that needs about 120 predictions per
hypothesis on test households. With 30 households per species on the test
side and 20 recordings each, rarer contexts (tension, warning) will likely
fall short; their results will be reported with their intervals and not
claimed as validated.

## 6. What the product already provides

* On-device and server analysis produce the same result schema
  (`docs/analysis_result.schema.json`), so app output can be scored directly.
* In-app feedback with "What was your pet actually doing?" and a research
  consent flag; on-device feedback is stored locally.
* The device-check harness for per-device runtime evidence.

Missing before collection can start: research upload with consent, an
annotation tool and guidelines, legal review of consent and data handling,
and participants.
