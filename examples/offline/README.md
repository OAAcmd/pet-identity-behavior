# Offline Examples

Generated with `uv run petperception make-examples` (default config, rule-baseline fusion).

**Synthetic pairings.** Each input pairs a real public photo (Open Images V7,
CC BY 2.0) with a real sound clip (ESC-50) that were never recorded together.
Detection and sound tagging ran for real; the pairing, timing and zones were
constructed for testing. These examples demonstrate timestamped observations and
justified abstentions; they are not evidence of accuracy. Media files are not
committed (regenerate them from the fetched datasets); source attribution is in
`index.json` and in each result's scenario entry.

| Scenario | What it shows | Decision |
| --- | --- | --- |
| `s01_food_zone_meow` | meow (2-7 s) while the only visible cat is in an owner-marked food area: one experimental `food` hypothesis with two independent roots, uncalibrated score, attribution marked unverified | `experimental_hypothesis` |
| `s02_food_zone_silent` | same scene, silent audio: zone context alone gives `food` = `insufficient_evidence` (`context_only_evidence`, `audio_silent`) | `supported_observation` |
| `s03_two_animals_bark` | cat and dog visible, bark: attribution `unknown`, nothing attached to either track | `supported_observation` |
| `s04_no_pet_vacuum` | verified no-pet scene, vacuum sound: abstention with `no_pet_detected`, context sound reported | `abstention` |
| `s06_rest_zone_growl` | still cat in rest area + growl-like sound: `relaxed` abstained (`conflicting_evidence`), `warning`/`tense` experimental | `experimental_hypothesis` |

Files per scenario: `<id>.result.json` (schema 1.0.0), `<id>.report.md`
(readable report), `<id>.zones.json` (zone input, if any).

Sources: cat photo "Fwd: Heh" by Qfamily (CC BY 2.0); cat+dog photo
"Chatelaine the manx cat meets Grady the wheaten terrier" by Helena Jacoba
(CC BY 2.0); no-pet photo "IMG_2448" by Madison Scott-Clary (CC BY 2.0);
ESC-50 clips 4-133047-A-5 (meow), 4-192236-A-0 (bark), 5-182012-A-36 (vacuum),
3-146972-A-5 (growl), all CC0 Freesound sources (see `index.json` for URLs).
