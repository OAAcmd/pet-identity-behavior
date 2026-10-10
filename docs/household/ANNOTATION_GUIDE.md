# Annotation guide

Annotators label what they can **see and hear** (the `observed` layer),
then, separately, what the situation suggests about the animal's
motivation (the `interpretation` layer). They work blind to the app's
output and never see owner context before finishing the observed layer.
File format: `annotation.schema.json` (unknown fields are rejected). Times
are seconds from the start of the recording; intervals are closed
`[t_start_s, t_end_s]`.

## Observed layer

**Presence** (`presence`): an interval per pet while any part of it is
visible in the frame (gaps under 0.5 s are bridged). Pet ids come from the
recording manifest; if two pets cannot be told apart, use the more likely
id and add a note.

**Activities** (`activities`), one label per pet at a time:

| Label | Definition |
| --- | --- |
| `stationary` | body position unchanged for at least 2 s (head and tail may move) |
| `moving` | locomotion (walking, trotting) for at least 1 s |
| `active_bout` | running, jumping or chasing for at least 1 s |
| `pacing` | at least two back-and-forth passes over the same path within 15 s |
| `entered_view` / `left_view` | the moment the pet enters or leaves the frame (interval of up to 1 s) |
| `eating` / `drinking` | head at food or water, with chewing, licking or lapping |
| `grooming_self` | licking or scratching its own body |
| `playing` | play actions directed at a toy, a person or another animal (pounce, chase, play bow, batting) |
| `sleeping` | lying with eyes closed, no response to ordinary sounds |
| `scratching` | scratching a surface (post, furniture, door) |
| `eliminating` | urinating or defecating, including in a litter box |
| `other` | anything else worth noting |

**Vocalizations** (`vocalizations`): one interval per call or continuous
bout (calls more than 0.5 s apart are separate). Types: cat `meow`, `purr`,
`hiss`, `growl`, `chirp_trill`, `caterwaul`; dog `bark`, `growl`, `howl`,
`whine_whimper`, `yip`; `other_pet_sound`; `unclear` when the type cannot be
told. Give `pet_id` only when the caller is certain.

**Posture** (`posture`): at every whole and half second while the pet is
visible: `lying` (torso on the ground), `sitting` (hindquarters on the
ground, front legs supporting), `standing` (all four legs supporting the
body), `other` (jumping, rolling, held by a person), `not_visible`.

**Person interactions** (`person_interactions`): a person `holding`,
`petting`, `playing_with` or `feeding` the pet, from first to last contact.

**Furniture** (`furniture`): the pet's relation to a bed, couch, chair or
table: `on` (supported by it), `in_front_of` (between the camera and the
furniture, on the floor), `beside`, `under`, `inside`. Label `in_front_of`
and `beside` carefully: they are the cases where the app's 2-D geometry is
expected to fail (E15).

## Interpretation layer

For each pet and each hypothesis the app can propose (`food`, `outside`,
`attention`, `play`, `alert`, `tense`, `warning`, `relaxed`), label `yes`,
`no` or `cannot_tell`, with the evidence in one sentence. Use
`cannot_tell` whenever the footage and the owner context do not settle it;
it is excluded from precision and reported separately. Annotators may now
read the owner context (what happened before and what the pet did next),
which is the main evidence for `food` and `outside`.

## Quality control

* **Training**: annotators label the same 20 training clips; production
  annotation starts when posture kappa is at least 0.70 and activity
  interval agreement (F1 between annotators, 1 s tolerance) at least 0.70.
* **Double annotation**: every test-household recording and at least 25 %
  of development recordings are labelled by two annotators; an adjudicator
  resolves differences and writes the adjudicated file.
* **Agreement** is reported with the results (`household-eval` computes
  posture and interpretation kappa and interval agreement): it bounds what
  any model can achieve.
* **Gold clips**: 5 % of each annotator's queue are previously adjudicated
  clips; a drop below the training thresholds triggers retraining.
* **Timing**: annotators label on the original file at its native frame
  rate; a 1 s tolerance is applied when scoring segments, 0.5 s for sounds.
