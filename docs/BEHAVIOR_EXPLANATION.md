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
| Clinical and ethological literature (Tiira 2012; Burn 2011; Amengual Batle 2019; Waisglass 2006; Gajdos-Kmecova 2023; Cornell, Merck and others; 24 sources in `knowledge.py`) | What a behaviour may mean, which contexts and signs separate the explanations, when to see a veterinarian | Automatic recognition | **The knowledge base: every non-ordinary candidate cites at least one of these** |

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
recognize ─ zero-shot per-frame probabilities over the visual behaviours (16; 14 per species) + neutral states (prompts in vocab.py)
          ├─ bouts (frames where a behaviour is the most likely reading, p ≥ 0.35)
          ├─ optional trained head (logistic regression on pooled embeddings; E31: not adopted)
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


## 7. Results (E31, measured; `results/e31/metrics.json`)

All numbers below are measured on real recordings. The protocol, targets and
benchmark were committed before any score (3925a50). One deviation from
"nothing seen before the run": a crash test of the evaluation code on random
features printed the parsing check C1, which does not depend on features
(27 of 34); the parser was not changed afterwards. A second run changed only
the order in which candidates are listed (display); every metric was
identical.

### 7.1 Recognition on the pool (human labels, 5-fold grouped cross-validation)

278 clips with a located animal (of 300; 21 without, 1 duplicate removed),
131 groups (uploaders or papers). Nine behaviours have at least 8 positive
clips from at least 3 groups.

| Recognizer | Macro AP over 9 behaviours [95 % CI, cluster bootstrap] |
| --- | --- |
| R0 zero-shot SigLIP (frozen) | **0.416 [0.351, 0.496]** |
| R1 trained head (logistic regression on pooled embeddings) | 0.248 [0.210, 0.321] |
| R2 mean of R0 and R1 | 0.407 [0.356, 0.485] |
| chance (mean prevalence) | 0.073 |

Gain over R0: R1 -0.167 [-0.243, -0.088], R2 -0.009 [-0.056, +0.038].
Neither qualifies (gain >= 0.03 with the interval above 0): **the zero-shot
recognizer stays; no trained head is deployed.** With about 10-46 weakly
labelled clips per behaviour, from very different footage, a linear head on
1,537 inputs overfits; the frozen encoder's prompts generalize better.

| Behaviour | Positive clips / groups | AP R0 | AP R1 | F1 at the cross-fitted threshold (R0) |
| --- | --- | --- | --- | --- |
| drinking | 15 / 12 | 0.757 | 0.378 | 0.75 (precision 1.00, recall 0.60) |
| kneading | 10 / 8 | 0.687 | 0.388 | 0.62 (0.83 / 0.50) |
| eating | 46 / 27 | 0.613 | 0.488 | 0.48 (0.58 / 0.41) |
| self-grooming | 21 / 15 | 0.392 | 0.323 | 0.34 (0.27 / 0.48) |
| tail chasing | 10 / 5 | 0.365 | 0.034 | 0.35 (0.43 / 0.30) |
| resting | 16 / 14 | 0.289 | 0.111 | 0.31 (0.24 / 0.44) |
| playing with an object | 26 / 11 | 0.285 | 0.122 | 0.38 (0.24 / 0.85) |
| social play | 15 / 13 | 0.190 | 0.108 | 0.34 (0.36 / 0.33) |
| locomotion | 23 / 18 | 0.163 | 0.281 | 0.20 (0.17 / 0.26) |

The labels are uploader titles: positives are often brief within a clip
and absence of a label does not mean absence of the behaviour, so these APs
understate precision on clean labels. Sound: YAMNet's vocal score against
the 'vocalizing' title label, 202 clips with sound, AP 0.441 (base rate
0.104).

### 7.2 Benchmark: recognition (adopted recognizer, cross-fitted)

| Check | Result [Wilson 95 %] |
| --- | --- |
| B1 asked-and-present behaviour recognized, all | 22 / 36 = 0.61 [0.45, 0.75] |
| - **tail chasing** | **3 / 8 = 0.375 [0.14, 0.69]** (Tiira S2, S3; the bathtub cat) |
| - vocalizing (from sound) | 8 / 8 |
| - playing with an object / kneading / eating | 3 / 3, 2 / 2, 1 / 1 |
| - self-grooming, social play | 2 / 3 each |
| - threat display, tail licking, flank biting, staring, tail flicking | 0 / 2, 0 / 1, 0 / 1, 0 / 1, 0 / 2 |
| B2 every present behaviour | 27 / 46 = 0.59 |
| B3 recognized bout overlaps a frame-review bout | 14 / 14 |
| B4 confusers not recognized | 5 / 6 (tail chasing 4 / 4; a dog rubbing itself on the carpet was read as scratching) |

**Tail chasing is not recognized reliably from recordings** (pre-registered
target B1 >= 0.7 and B4 >= 0.8; B1 fails). Zero-shot tail-chasing scores
for the eight clips were 0.19-0.87 and for the four confusers 0.28-0.49; the
threshold chosen on the pool (0.69) rejects all confusers but misses five
positives, among them the cat in the task's own example (small in the frame
on a suitcase: 0.19) and the hyperaesthesia cat circling on a play mat
(0.689, just below). No threshold separates the two groups. Answers about
tail chasing therefore rest on the user's report and say that the recording
did not confirm it.

### 7.3 Benchmark: explanation support (34 questions outside the 3 gap items)

| Check | Pass | Target |
| --- | --- | --- |
| C1 question parsed (asked behaviours found, nothing extra) | 27 / 34 = 0.79 | 0.9: **not met** |
| C2 knowledge entry for every asked behaviour | 34 / 34 | met |
| C3 every non-ordinary candidate cites a source; ids resolve | 34 / 34 | met |
| C4 request / no-request separation as the literature expects | 33 / 34 (29 / 29 among explained behaviours) | met |
| C5 coverage: the paper authors' interpretation is among the candidates | 7 / 7 (lay uploaders' 6 / 6) | met |
| C5 ranking: the first candidate by context has the authors' kind | 6 / 7 | reported |
| C6 distinguishing observations and urgent signs present | 34 / 34 | met |
| C7 no certainty phrase; "cannot establish why" stated | 34 / 34 | met |
| C8 a reported behaviour is called "seen" only when recognized | 34 / 34 | met |
| C9 the user's species kept (3 answers note a detector disagreement) | 34 / 34 | met |

Parsing failures: "lying down" added resting (q13); "biting the adult cat's
tail" not read as social play (q14); "push her paws into the blanket" (q21),
"licks the same favorite places on her body" (q27), "jumps up the wall after
the laser pointer dot" (q33), "bringing his toy" (q34) and "rub her belly"
(q36) matched no alias. In q27 neither the parser nor the recognizer found
grooming, so the answer did not explain it (the C4 failure). Ranking miss:
for the 4-month-old Bull Terrier (q01) the word "puppy" puts play first,
while the authors describe compulsive tail chasing ("several times a day"
is not one of the compulsive cue phrases).

Gap questions (outside the vocabulary or the knowledge base) are answered
wrongly, as expected: a dog dragging itself along the carpet (q17) and a
scratch reflex during petting (q18) are both explained as scratching / itch;
barking on command (q19) gets disturbance, isolation and play candidates but
not "a trained response".

### 7.4 Example (q02, abridged; all 37 answers in `results/e31/answers.md`)

Question: "My 4-year-old German Shepherd chases his tail many times a day,
usually while he waits for his food or when I talk with other people. Why
does he do this?" The answer reports tail chasing in the recording
(0.0-15.5 s and 17.5-23.0 s; also a bark at 12.5 s), lists five candidates
in order of fit to the description - frustration or arousal in waiting
situations (Tiira 2012), canine compulsive disorder (Tiira 2012, Burn 2011;
"German shepherd" matched), self-directed play, a learned attention-getting
behaviour (request; Burn 2011), a physical or neurological cause
(Van Haevermaet 2026, Frontiers 2026) - and asks when it started, whether it
can be interrupted, in which situations it starts, whether the tail is
injured, and whether there are other repetitive behaviours. It also shows
a false recognition: "threat display" at 3.5 and 14.5 s.

## 8. Which questions can be answered today

| Question type | Answerable? | Evidence |
| --- | --- | --- |
| "What could this behaviour mean?" for the 17 behaviours in the knowledge base (cat and dog tail chasing, tail licking or biting, flank biting, staring / air snapping, grooming, scratching, kneading, object play, social play, threat display, vocalizing, tail flicking, being petted, eating, drinking, resting, locomotion) | **yes, from the user's description**: candidates by kind with sources, ordered by the description, with distinguishing questions and urgent signs; never a stated cause | C2-C9 met; the authors' interpretation covered in 7/7 expert cases and ranked first in 6/7 |
| "Is it asking for something?" | **yes, as a separated candidate kind**: a request candidate appears only where the literature supports one (attention-reinforced tail chasing, meowing for food, isolation barking, bringing a toy, "stop touching me"); otherwise "no request is implied" | C4 33/34 |
| "Does the recording show it?" | **partly**: vocalizing, object play, kneading, eating, drinking, grooming are usually recognized; tail chasing in 3 of 8 clips; threat displays, tail licking, flank biting, staring and tail flicking were not recognized | §7.1-7.2 |
| When and how long | **yes for recognized behaviours**: bouts overlap the frame-review bouts in 14/14 cases; movement pattern (stays in place, turning, repetition) is descriptive only | B3 |
| Behaviours or explanations outside the vocabulary / knowledge base | **no**: answered with the nearest entry, which is wrong (q17-q19) | §7.3 |
| Why this particular animal does it | **never claimed**: the answer says the recording cannot establish why | C7 |

## 9. Gaps and next steps

* Tail-chasing recognition needs temporal modelling or more examples: the
  zero-shot frame scores do not separate circling from play or tail
  movement. Candidates: a motion feature for turning in place validated on
  fresh clips, or a video-text model (X-CLIP is retrieved but not evaluated);
  any rule needs a pre-registered test on clips not used here.
* Parsing: 7 of 34 questions missed; aliases are literal. A paraphrase test
  set separate from the benchmark is needed before changing the parser, so
  that the benchmark stays a test.
* Knowledge base: add entries only with a source (rubbing on surfaces, the
  scratch reflex, trained responses); cue words such as "bit" match inside
  other words.
* Data: about ten tail-chasing recordings with usable terms; no cat home
  video with an expert label beyond one case; owner questions written from
  captions, not collected from owners; household recordings with owner
  context would test the whole loop (`docs/household/`).
* Phone: the explanation layer (knowledge base, parser, composer) is plain
  code and could run in the browser engine, but SigLIP (812 MB) does not fit
  the phone budgets; a phone version needs a small encoder evaluated on the
  same benchmark. Nothing here has been run on a phone.
