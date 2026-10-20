# Behaviour and intent data: inventory, label layers, gaps

Source of truth in code: `src/petperception/intent/taxonomy.py` (`DATASETS`, `MISSING`,
`coverage()`); tests in `tests/test_intent_taxonomy.py` keep this document's rules
enforced (a recorded context is never an intent annotation; every link names a
hypothesis of `schemas/ontology.py`).

## 1. Three label layers, kept apart

| Layer | Meaning | Can be checked against the recording? | Examples |
|---|---|---|---|
| `observable_behavior` | what the animal visibly or audibly does | yes | resting, playing, eating, alerting to a sound, barking, being touched by a person, lying, ear/tail keypoints |
| `recorded_context` | the situation the recording was made in (set by the protocol or visible in the scene) | the situation, not the animal's state | brushing session, isolation in an unfamiliar room, waiting for food, stranger at the door, being bathed / walked / trained, location |
| `annotated_intent` | an annotator's judgement of what the animal tries to obtain | no (it is someone's inference) | Barkopedia "begging for food", "seeking attention" (vision-language model proposal, human-verified) |

A recorded context is **not** a mental state. A cat recorded while its owner
prepares food is in a food-related situation; the label does not show that the
cat wanted food at that moment. The product therefore never turns a context into
a hypothesis on its own.

The product's hypotheses are those of `schemas/ontology.py`: food, outside,
attention, play, alert, tense, warning, relaxed. Dataset labels link to them by
one of three relations:

* `annotated` - the label is an intent annotation for the hypothesis;
* `consistent_with` - the behaviour fits the hypothesis without establishing it;
* `context_for` - the situation makes the hypothesis plausible; it can qualify
  or corroborate evidence from the animal, never create a hypothesis.

Context qualifiers change how evidence reads (vocalizing while being bathed reads
as a reaction to handling, not as play): handling, bathing, training, walk,
isolation, disturbance, food_routine, location.

## 2. Inventory of obtained data

| Dataset | Species | Modality | Synchronized A+V | Unit | Grouping / leakage control | Label source | Layer(s) | Terms | On disk |
|---|---|---|---|---|---|---|---|---|---|
| CatMeows | cat | audio | no | meow (8 kHz) | cat (21), owner (12): leave-one-cat-out | owner protocol | context | CC BY-NC 4.0 | 440 clips |
| Yin & McCowan barks (BEANS) | dog | audio | no | bark | dog (10), breed (6): leave-one-dog-out | protocol (CSV terms) | context | none stated; research | features only |
| Barkopedia Dog Act/Env | dog | audio | no (source videos not released) | bark from a video | no ids: near-duplicate pseudo-groups (cos >= 0.95) + max-similarity audit | VLM + human verification | behaviour, intent, context (activity); context (environment) | MIT | 12,480 + 499 clips |
| Kinetics-700 pet classes | cat, dog | audio + video | **yes** | 10 s clip | YouTube id (no id in train and test) + embedding near-duplicate audit | annotators (human action) | context | annotations CC BY 4.0; clips research only | 600 train + 294 val clips |
| MammalNet dogs | dog | video | no | behaviour segment | YouTube id | annotators | behaviour | research | features (E11/E17) |
| Open Images relations / narratives | cat, dog | image | no | box pair / caption | image id | crowd; captions (weak) | behaviour | CC BY 4.0 annotations | subsets |
| FSD50K vocal types | cat, dog | audio | no | freesound clip | uploader | crowd | behaviour (vocal type) | per-clip CC | subset |
| AwA-Pose, dog-pose | cat, dog | image | no | keypoints | image | manual | behaviour geometry (ears, tail) | research | yes |

Only Kinetics provides genuine synchronized audio and video with labels, and its
labels are recorded contexts. Unrelated audio/video pairs are used only as
conflict fixtures (E28 swapped-audio condition), never as multimodal evidence.

## 3. Label mappings

### Barkopedia activity (dog audio; E26, E27)

| Raw label | Layer | Hypothesis link |
|---|---|---|
| begging for food | annotated_intent | food (`annotated`) |
| seeking attention | annotated_intent | attention (`annotated`) |
| playing with human / other animals / toy | observable_behavior | play (`consistent_with`) |
| alerting to sounds | observable_behavior | alert (`consistent_with`) |
| rest | observable_behavior | relaxed (`consistent_with`) |
| taking shower | recorded_context | qualifier `bathing` (no hypothesis) |

The environment column (indoor, near door, near window, near other animals,
bathroom, vehicle interior, outdoor ...) is a recorded context (location).

### CatMeows (cat audio; E2)

| Raw label | Layer | Link |
|---|---|---|
| brushing | context | qualifier `handling` |
| isolation | context | attention (`context_for`), qualifier `isolation` |
| waiting_for_food | context | food (`context_for`), qualifier `food_routine` |

### Yin & McCowan barks (dog audio; E22)

| CSV term | Paper situation (mapping unconfirmed) | Link |
|---|---|---|
| aggression | stranger at the door | alert, warning (`context_for`) |
| contact | isolation from the owner | attention (`context_for`) |
| play | play with the owner | play (`context_for`) |

### Kinetics pet classes (synchronized audio + video; E28)

| Raw label | Link |
|---|---|
| petting cat | qualifier `handling`; attention, relaxed (`context_for`) |
| grooming cat / grooming dog | qualifier `handling` |
| bathing dog | qualifier `bathing` |
| training dog | qualifier `training`; food, attention (`context_for`) |
| walking the dog | qualifier `walk`; outside (`context_for`; the dog is already outside) |

## 4. Hypothesis coverage by trained evidence

| Hypothesis | Dog | Cat |
|---|---|---|
| food | Barkopedia "begging for food" (annotated, audio) | CatMeows waiting_for_food (context only, below gate) |
| attention | Barkopedia "seeking attention" (annotated, audio) | CatMeows isolation (context only, below gate); Kinetics petting (context) |
| play | Barkopedia playing x3 (behaviour, audio); Yin play (context) | none |
| alert | Barkopedia alerting (behaviour, audio); Yin disturbance (context) | none |
| relaxed | Barkopedia rest (behaviour, audio) | Kinetics petting (context) |
| outside | Kinetics walking (context; already outside) | none |
| tense / warning | FSD50K growl (vocal type, below gate, E21) | FSD50K hiss (below gate) |

Cats have no intent annotation in any modality; every cat link above is a
context.

## 5. Missing examples and annotations

Listed exactly in `taxonomy.MISSING`:

1. Synchronized household audio-video with an intent annotation per bout (per
   species >= 300 bouts, >= 40 animals, >= 30 households; onset/offset, animal
   id, household id, owner-reported situation, independent annotator intent
   label with confidence, second annotator on >= 20 %). Without it the
   cross-channel intent layer cannot be fitted or validated.
2. Cat intent annotations in any modality (>= 150 bouts per hypothesis, >= 30
   cats, cat ids).
3. "Outside" requests (door-side bouts with the owner's outcome, >= 100 per
   species).
4. Face / ear / tail annotations linked to situations (e.g. CatFACS/DogFACS-coded
   bouts); keypoint models give geometry but nothing links it to context or intent.
5. Animal and household ids for Barkopedia clips (source video, dog, uploader).
