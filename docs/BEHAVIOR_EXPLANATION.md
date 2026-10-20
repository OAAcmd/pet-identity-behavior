# Explaining a pet's behaviour in context

Status: 2026-10-20. Code: `src/petperception/understand/`; command:
`petperception explain`; evaluation: E31
(`src/petperception/eval/e31_behavior_questions.py`, results in
`results/e31/`).

## 1. The task

The user gives the species (cat or dog; trusted, never overridden), a short
scene description and a question, together with a recording, for example:

> In an open living room, my cat keeps turning in circles while biting its
> own tail. What could this behavior mean?

The system must:
1. locate the animal (detection and a single-target track; the detector's
   own species vote is reported, never used to reject the user's species);
2. recognize what the animal does and when (behaviour bouts, temporal
   pattern: staying in place, turning, repetition);
3. combine this with the scene and the question;
4. answer with **candidate explanations grounded in cited sources**, keeping
   requests and intentions apart from play, arousal or stress, compulsive and
   medical explanations, without forcing a human-like intention onto every
   behaviour, and without claiming that a cause is established;
5. say what would tell the candidates apart, which signs need a
   veterinarian, and how uncertain each part is.

## 2. Existing work (targeted review, 2026-10-20)

| Work | What it demonstrates (measured) | What it does not cover | Reuse here |
| --- | --- | --- | --- |
| Animal Kingdom (Ng et al., CVPR 2022) | Action recognition, video grounding and pose over 140 action classes and about 850 species of wild animals | Home pets, owner context, explanations; download behind a request form | Not used (access form; wild animals) |
| MammalNet (Chen et al., CVPR 2023) | 12 common mammal behaviours over 173 categories; zero-shot behaviour recognition was weak (about 18-26 % top-1) | Domestic cats; explanations | Its dog subset was used earlier (E11, research only) |
| EgoPet (Bar et al., 2024) | Egocentric video from pets for locomotion and interaction prediction | Third-person home video of the pet; behaviour meaning | Not used (terms unclear, earlier finding) |
| DECADE / dogTorch (Ehsani et al., CVPR 2018) | Predicting a dog's actions from its own head-mounted camera and IMU | Third-person video; cats; meaning | Not used |
| PetFace (ECCV 2024) | Individual identification from faces | Behaviour | Not used |
| Dussert et al. (Methods Ecol Evol 2025) | Zero-shot vision-language models classify behaviour in camera-trap images of ungulates (reported accuracy CLIP 71.8 %, SigLIP 91.1 %, CogVLM 97.5 %) | Pets, temporal actions, explanations | **Motivates the frozen SigLIP zero-shot recognizer** |
| AnimalMotionCLIP; Animal-CLIP (IJCV); EthoCLIP (CVPR 2026) | CLIP / X-CLIP adapted to animal action recognition (for example 74.6 % mAP on Animal Kingdom with motion cues) | Pets in homes; meaning; on-device budgets | Design reference; X-CLIP weights were retrieved but not evaluated |
| Clinical and ethological literature (Tiira 2012; Burn 2011; Amengual Batle 2019; Waisglass 2006; Gajdos-Kmecova 2023; Cornell, Merck and others; 25 sources in `knowledge.py`) | What a behaviour may mean, which contexts and signs separate the explanations, when to see a veterinarian | Automatic recognition | **The knowledge base: every non-ordinary candidate cites at least one of these** |

No published system was found that takes a home recording of a cat or dog
plus an owner's question and returns source-grounded candidate explanations.
The two halves exist separately: general animal action recognition (mostly
wild animals) and the veterinary-behaviour literature (no automatic
recognition). This work joins them with a deliberately simple, inspectable
middle layer.

## 3. Data (research use; raw media never in Git)

| Source | Clips | Licence / permission | Labels | Groups for splits |
| --- | --- | --- | --- | --- |
| Wikimedia Commons, "Videos of cats" / "Videos of dogs" (depth 3) and searches (`manifests/commons_pets.json`, `data/commons_pets.py`) | 430 records (cat 237, dog 193); first 60 s of 287 downloaded (all with a parsed label plus 70 without); 4 failed (HTTP 429) | per file: mostly CC BY / CC BY-SA / CC0 (recorded per record) | *source*: parsed from title, description and categories (lay, noisy) | uploader |
| Paper supplements (`manifests/paper_videos.json`, `data/paper_videos.py`) | 8: Tiira 2012 S1-S3 (dog tail chasing), Vet Res Commun 2026 M1-M3 (cat, presumptive hyperaesthesia), Frontiers 2023 Dachshund (flank biting), Frontiers 2026 focal seizures | CC BY 4.0 | *expert*: the authors' captions (case level) | paper |
| Internet Archive (`manifests/archive_pets.json`) | 5: two cats chasing their tails, one dog spinning, a cat flicking its tail while resting, a kitten biting an adult cat's tail | **no licence stated**: research analysis only, never redistributed (unresolved permission recorded per record) | *source*: the title | uploader |
| Frame review (`annotations/behavior_clips.json`) | 38 clips (the benchmark clips and one excluded) | - | *assistant*: machine-assisted labels with coarse bouts, from contact sheets reviewed by the AI assistant; stored and scored apart from human labels, never merged | - |

Leakage controls: the Commons copy of Tiira Movie S2 duplicates the paper
file and is removed; cross-validation folds never split an uploader or a
paper; the benchmark scores each clip with the model fold that did not train
on its group. A cartoon found by search was excluded after frame review.

Gaps: very few tail-chasing recordings with usable terms (about ten, five
groups); no cat home recording with an expert label beyond one case report;
no owner-collected questions (the benchmark questions are written from the
sources' captions, see §6); sound labels only from titles.

## 4. Method

```
media ─ decode 6 fps (≤ 60 s, ≤ 480 px) ─ locate: phone detector, cat or dog boxes, single-target track with gap filling
      ├─ motion in the animal's region (optical flow: speed, rotation, share moving) + appearance (repetition)
      ├─ SigLIP image embeddings of the animal crop at 2 fps, scene embedding
      └─ YAMNet scores of the sound (when present)
recognize ─ zero-shot per-frame probabilities over 17 visual behaviours + neutral states (prompts in vocab.py)
          ├─ bouts (frames where a behaviour is the most likely reading, p ≥ 0.35)
          ├─ optional trained head (logistic regression on pooled embeddings; adopted only if E31 says so)
          └─ temporal pattern (stays in place, turning, repetition) - descriptive
question ─ deterministic parsing (aliases, pronoun normalization, crude stemming) → reported behaviours, age, repetition
knowledge ─ entry per (behaviour, species): candidates by kind with sources, context cue words, distinguishing observations, urgent signs
answer ─ what was seen (with times) · what the user reported and whether the recording shows it · candidates by kind, ordered by
         the user's description · request/intention possible or not · observations that separate them · urgent signs · uncertainty · sources
```

Design rules (enforced by `knowledge.validate()` and the tests):
* every candidate outside "ordinary play or maintenance" cites a source with
  its tier (peer-reviewed study or case report, veterinary school, clinical
  review, veterinarian-reviewed advice, animal-welfare organization);
* *kind* separates `request` (the animal seeks something from a person,
  including "stop touching me") from `play_or_normal`, `arousal_or_stress`,
  `compulsive` and `medical`; a behaviour without a request candidate gets an
  explicit "no request is implied" note;
* context cue words change the order of candidates, never their content;
* the answer states that the recording cannot establish why;
* the user's species is used even when the detector votes otherwise (the
  disagreement is reported).

## 5. How to run

```bash
uv sync --extra dev --extra train --extra behavior     # torch (CPU), transformers
# models: SigLIP base patch16-224 in var/models/siglip_base_patch16_224 (see DATA_AND_MODEL_REGISTER.md)
petperception explain clip.mp4 --species cat \
  --scene "open living room" \
  --question "my cat keeps turning in circles while biting its own tail - what could this mean?" \
  --out answer.json --markdown answer.md
```
Data and evaluation:
```bash
python -m petperception.data.commons_pets ...        # manifest + downloads (see module docstring)
python -m petperception.understand.dataset --threads 2   # features for every clip (resumable)
petperception evaluate --experiment e31                  # writes results/e31/, artifacts/understand/
```

## 6. Benchmark (E31)

`benchmarks/behavior_questions.json`: 37 questions on 37 real clips. q08 is
the task's question verbatim (on a clip of a cat turning after its tail in a
living room); the others were written from each source's caption, title or
description without adding facts the source does not state. Categories: tail
chasing or circling (9), tail- and self-directed (3), confusers (4: tail
flicking at rest, a kitten biting another cat's tail, a hyperaesthesia cat
playing with a strap, cats flicking tails at a window), gaps (3: behaviours
or explanations outside the vocabulary or knowledge base), other behaviours
(18). Recognition and explanation support are scored separately; the
protocol and targets were committed before any result (3925a50).

Results: §7 (filled from `results/e31/metrics.json`).
