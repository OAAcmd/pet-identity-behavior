# G2 annotation specification: cat and dog keypoints, ears and tail

Purpose: keypoint and ear/tail labels that may be used outside research.
Every public set is research-only or has unclear terms (`docs/DATA_SOURCES.md`).
The images come from Open Images V7: CC BY 2.0 images with an author
listed, and CC BY 4.0 boxes. Labels made under this specification belong to
the project and can be released under its own terms. The image credits
travel with the data (`author`, `original_landing_url` in the candidate
file).

## Inputs

* `uv run petperception train-prep g2` writes `var/training/g2/candidates.csv`
  (counts: `results/g2/candidates_summary.json`): 4,813 cat and 6,348 dog
  images (1,193 and 585 fully visible), split by photographer. Each row
  holds:
  * one image with exactly one cat or dog;
  * the animal is not a depiction or a group;
  * its box covers at least 10 % of the image;
  * `truncated` / `occluded` flags. Start a budget with the fully
    visible animals;
  * the split (train / development / test), assigned by Flickr author;
  * an `order` per species. Annotate in that order: any prefix is a
    random sample, so the work can stop at any budget.
* Tool: CVAT (MIT licence) or Label Studio (Apache-2.0), with COCO
  keypoints export. Load the image URL and draw nothing new. The given
  box is the instance.

## Keypoints (25 per animal, both species)

The 24 points of the StanfordExtra / Ultralytics dog-pose schema, the
schema of the research model E9, plus `tail_mid`:

| Group | Points |
| --- | --- |
| Front legs | `front_left_paw`, `front_left_knee`, `front_left_elbow`, `front_right_paw`, `front_right_knee`, `front_right_elbow` |
| Rear legs | `rear_left_paw`, `rear_left_knee`, `rear_left_elbow`, `rear_right_paw`, `rear_right_knee`, `rear_right_elbow` |
| Tail | `tail_start` (base, where the tail leaves the body), `tail_mid` (new: halfway along the visible tail), `tail_end` (tip) |
| Ears | `left_ear_base`, `right_ear_base` (where the front edge of the ear meets the head), `left_ear_tip`, `right_ear_tip` |
| Head | `nose`, `chin`, `left_eye`, `right_eye` (centre of the eye) |
| Body | `withers` (top of the shoulders), `throat` |

"Left" and "right" are the animal's own sides. Visibility follows COCO:
* 0: not labelled (outside the image, or the body part does not exist,
  for example a docked tail beyond `tail_start`);
* 1: hidden but position inferable;
* 2: visible.

The research dog-pose data never labelled eyes, withers or throat. Here
they are required.

## Per-animal categorical labels

Observable only. No emotion or intention words.

| Field | Values |
| --- | --- |
| `ear_position` | forward, neutral, sideways, back, flattened, not_visible |
| `ear_carriage` (morphology) | pricked, semi_pricked, folded, drop, rose, not_visible |
| `tail_position` | raised, horizontal, lowered, tucked, not_visible |
| `tail_shape` (morphology) | straight, curled_over_back, docked, bobbed, not_visible |
| `coat` | hairless, short, medium, long |
| `posture` | lying, sitting, standing, crouching, other |
| `view` | side, front, back, top, other |

Cats use the same fields. `ear_carriage` "folded" covers Scottish Fold
types. `tail_shape` "bobbed" covers bobtail breeds.

## Procedure and quality checks

1. Annotate one species at a time, in `order`, train and development
   first. Record the annotator id and the seconds spent per instance (CVAT
   logs them).
2. Test split: two annotators label every test image independently, and
   a third person adjudicates. Agreement is reported per keypoint group as
   inter-annotator PCK@0.1 (normalised by the longer box side), and per
   categorical field as Cohen's kappa. A group whose agreement is below
   0.8 PCK (or kappa 0.6) is reviewed before any model is scored on it.
3. Skip and record (reason code) any image where:
   * the animal is mostly hidden;
   * the image is not a photo;
   * the animal is not a cat or dog;
   * the image shows something that should not be annotated (a person's
     private information).
4. 5 % of train images get a second annotation, as a drift check.

## Budget

Estimates, not measurements; the first 100 instances will measure them.
* About 60-90 s per instance for 25 points and 7 fields.
* 2,000 cats and 2,000 dogs: about 70-100 annotator hours, plus 20 %
  for the double-annotated test split.
* The first useful checkpoint is 500 per species, with 100 per species
  in test.

## Use

The import into the E9 training format, and G2 training, are specified in
`docs/TRAINING_PROGRAM.md` (job G2). Research-only models, such as the E9
head trained on dog-pose, are **not** used to pre-fill points. That would
carry their data terms into the new labels. Using them anyway would need a
recorded decision by the data owner.
