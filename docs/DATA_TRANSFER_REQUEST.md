# Data transfer request

Status: 2026-10-11. Machine-readable source:
[`manifests/mirror_files.json`](../manifests/mirror_files.json). Live status:
`uv run petperception mirror status`.

This document lists the dataset files that this workspace cannot download
itself, in priority order, so that the data owner (or the person running
the project) can place them in an object store this workspace can reach.
Each file keeps its official identity: dataset, exact version, official
filename and official checksum. The mirror only changes where the bytes come
from. It also says what is **not** a download problem: annotation work,
household recordings and real-phone runs cannot be fixed by a transfer, and
are listed separately in section 5.

## 1. How to hand over files

1. Upload the files unchanged (same filename, same bytes) to a bucket or
   prefix you control. Google Cloud Storage works here today:
   `storage.googleapis.com` is reachable from this workspace. S3 and Azure
   Blob URLs are also parsed, but their hosts have not been probed.
2. Create **read-only** download URLs. V4 signed URLs (GET) are best. For
   files over 10 GB, use the maximum lifetime (7 days for GCS V4). At the
   measured 70 MB/s of the Open Images download, a 20 GB file takes about
   5 minutes here (estimate). A transfer can still be interrupted and must
   be resumable later.
3. Give the URLs to the workspace **without pasting them into chat**. Use
   one of these:
   * the environment's settings: one environment variable per file,
     `PETPERCEPTION_MIRROR_<DATASET>_<FILE>` (upper case; every character
     other than a letter or digit becomes `_`). For example:
     `PETPERCEPTION_MIRROR_COP3D_CAT_000_ZIP`. A new session picks these up;
   * or a JSON file `{"<dataset>/<file>": "<url>"}` at
     `var/secrets/mirrors.json`. That path is ignored by git. Point
     `PETPERCEPTION_MIRRORS` at a different path to use another file.
4. `uv run petperception mirror status` then shows, for each file: present,
   partial or missing; whether a URL is set; the host; and the expiry
   (`ok`, `expires_within_1h`, `expired`). It never prints the query
   string.
5. Fetch the files. Each one is resumed, retried, checked against
   Content-Length and the official checksum, and only then moved into
   place, with a provenance record:
   * `uv run petperception mirror fetch --dataset <id>` downloads a
     dataset's files to disk;
   * `uv run petperception mirror verify-remote --dataset <id>` checks a
     file larger than the free disk space by streaming it through the hash
     functions without storing it. The experiments then read only the
     archive members they need with HTTP Range requests
     (`petperception.data.mirror.RemoteZip`, which supports ZIP64 and split
     `.z01` archives).

**Already validated with a real file (2026-10-11):**
`oidv7-class-descriptions-boxable.csv`, 12,064 bytes, from Google Cloud
Storage via a generation-pinned URL held in the uncommitted mirrors file.
The test interrupted the transfer at 5,000 bytes and resumed it with a
Range request. The publisher's MD5 `c5e7cb6b85d0539b2105db2d1974681f`
matched. The provenance record contains no query string.

Nine unit tests cover the rest against a local server
(`tests/test_mirror.py`):
* resume after a dropped connection;
* checksum mismatch;
* the first-retrieval pin;
* expired URLs;
* a 403 reported without its signature;
* expiry parsing for GCS, S3 and Azure;
* environment-over-file precedence;
* signed PUT and resumable uploads;
* a read-only URL refused for upload;
* remote ZIP members, including a split archive and a corrupted member.

**Disk budget:** about 6 GB is free here. The G1 detector data takes about
2.5 GB (images 1.2 GB plus the training cache). Files larger than about
3 GB are therefore verified by streaming and read member by member, not
stored. This applies to CoP3D and to FSD50K audio; their per-file plan is
given in section 2.

## 2. Downloadable labelled datasets, in priority order

The priority follows the current bottleneck: real-video animal detection
first, then audio types and contexts, then research-only keypoint data.

### P1 - CoP3D (Common Pets in 3D), v1

| | |
| --- | --- |
| Source | `https://dl.fbaipublicfiles.com/cop3d/<file>`. The file list and SHA-256 values come from `links/cop3d_sha256.json` in github.com/facebookresearch/cop3d. `dl.fbaipublicfiles.com` is blocked here. |
| What it is | 4,200 smartphone videos of people's own cats and dogs, crowd-sourced. 322 GB in total, as 20 GB zip archives. |
| Annotations | Per sequence: category (cat or dog). Per frame: camera pose and a **model-generated** foreground mask. There are no behaviour labels, no boxes, and no human-verified masks. |
| Terms | CC BY-NC 4.0. Research evaluation only; nothing trained on it ships. |
| Unblocks | A second real-video evaluation of animal detection and tracking on home-style phone video, per species. Measures: frames and sequences with the animal found, species confusion, track fragmentation, and box agreement with mask-derived boxes. It complements Kinetics (YouTube, person-centred). Experiment: **E20**, defined in [TRAINING_PROGRAM.md](TRAINING_PROGRAM.md) G1. |
| Status | Missing, all 5 files. |
| Disk plan | Each 20 GB archive is verified by streaming (`verify-remote`). Only sampled frames and masks are read, by Range. Estimate: 10 frames per sequence, about 30 MB per 100 sequences. Nothing larger is stored. |

| Order | File | Size | SHA-256 (official) |
| --- | --- | --- | --- |
| 1a | `metadata_000.zip` | not published (small) | `ce30c523eefea45d169e9ac060baae328d10f5daa5e993138729c606caf77abd` |
| 1b | `cat_000.zip` | about 20 GB | `e94cea08cd61dfa31f353c766316271be68c397add5586a0f4663f3816508057` |
| 1c | `dog_000.zip` | about 20 GB | `186cc7b6b8c06ff0b51f4b8d4da1d2aee085df5ec80712976580d55b9c8bce55` |
| 1d | `cat_001.zip` | about 20 GB | `7fba200b39c1e9f8a9783e54b9c1e6fb65d31b032a26b9a6f15f3dea78a44187` |
| 1e | `dog_001.zip` | about 20 GB | `6be8e1f92fa3edbb3fe86c5fb925652e2000ee799a23deabc4b09194ebd1a99b` |

Files 1a-1c are enough for a first per-species result. Files 1d and 1e
double the number of sequences, for tighter confidence intervals. The other
14 archives (cat_002-007, dog_002-011) are not requested: under this
sampling plan, more sequences per archive do not change the estimate
enough to justify 20 GB each.

### P2 - FSD50K v1.0 (Zenodo record 4060432)

| | |
| --- | --- |
| Source | `https://zenodo.org/record/4060432/files/<file>`. The MD5 values are those published on the Zenodo record, as copied into soundata's `fsd50k.py`. `zenodo.org` is blocked here. |
| What it is | 51,197 Freesound clips, 108.3 h, with 200 human-verified AudioSet classes. Pet-related eval clips: Dog 178, Bark 122, Growling 51, Cat 132, Meow 78, Purr 60, Hiss 125 (counted from public copies of `eval.csv`). |
| Annotations | Clip-level labels. Per-clip licence and uploader are in `FSD50K.metadata.zip`. |
| Terms | Per clip: CC0, CC BY, CC BY-NC or Sampling+. Attribution is required. A commercial-capable subset (CC0 and CC BY) is selectable from the metadata. |
| Unblocks | **Vocalization type** for both species. The meow, purr and hiss and the bark and growl sub-scores are currently "not evaluated" ([CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md)). Also thresholds for those sub-types, and environmental-context sounds. Experiments: E21 (evaluation, eval split) and G4 (type head training, dev split). |
| Status | Missing, all 11 files. |
| Disk plan | Audio archives are verified by streaming. Only the selected clips (pet classes plus sampled negatives, about 1,500) are read by Range and turned into embeddings; the audio itself is not stored. |

| Order | File | Size | MD5 (official) | Needed for |
| --- | --- | --- | --- | --- |
| 2a | `FSD50K.ground_truth.zip` | small (CSV) | `ca27382c195e37d2269c4c866dd73485` | labels |
| 2b | `FSD50K.metadata.zip` | small | `b9ea0c829a411c1d42adb9da539ed237` | per-clip licences |
| 2c | `FSD50K.doc.zip` | small | `3516162b82dc2945d3e7feba0904e800` | terms and README |
| 2d | `FSD50K.eval_audio.zip` | see note | `6fa47636c3a3ad5c7dfeba99f2637982` | evaluation (E21) |
| 2e | `FSD50K.eval_audio.z01` | see note | `3090670eaeecc013ca1ff84fe4442aeb` | evaluation (E21) |
| 2f-2k | `FSD50K.dev_audio.zip`, `.z01`-`.z05` | see note | `c480d119…5a`, `faa7cf4c…a3`, `8f9b6615…bc`, `1196ef47…59`, `d088ac4e…cc`, `81356521…7f` (full values in the manifest) | training (G4) |

Note on sizes: the per-file sizes are on the Zenodo record, which cannot be
read here. From the durations, 16-bit 44.1 kHz mono PCM gives about 8.9 GB
of eval audio and 25.5 GB of dev audio (estimates). The official MD5 is
authoritative either way.

### P2 - Dog barks, Yin & McCowan (2004), as packaged in BEANS

| | |
| --- | --- |
| Source | `https://storage.googleapis.com/ml-bioacoustics-datasets/dog_barks.zip`, generation 1728757838735376. 901,568,218 bytes; MD5 `e8ef0441a89f8087ce8fcade3dfe6c93` from the publisher's object metadata. |
| What it is | 693 barks from 10 dogs of 6 breeds. Contexts in `annotations.csv`: contact 385, play 209, aggression 99. The paper's situations were disturbance (a stranger at the door), isolation and play. The mapping of the CSV terms to those situations is not confirmed. |
| Terms | No data licence is stated. Research evaluation only. Cite Yin & McCowan 2004. |
| Unblocks | Dog bark context, the dog counterpart of E2 (G4, research profile). Experiment: **E22**. |
| Status | **Present: reachable directly**, so no transfer is needed. It is read by streaming (section 1). |

### P3 - CatMeows v1.0.2 (Zenodo record 4008297): only what E2 still lacks

E2 already ran on all 440 meows: leave-one-cat-out macro-F1 0.46. The audio
came from an unofficial GitHub mirror, pinned per file by SHA-256 at first
retrieval. **The whole dataset is not requested again.** Three things are
still missing, together about 13 MB:

| Order | File | Size | Checksum | Why it is still needed |
| --- | --- | --- | --- | --- |
| 3a | `record.json` (`https://zenodo.org/api/records/4008297`) | small | none (pinned on first retrieval) | It lists the record's file names, sizes and MD5s, and its licence field. That settles which of the two published MD5s belongs to which file, and the terms question in ASSET_TERMS D2 (Zenodo's "research and non-commercial" text versus the University of Milan's CC BY 4.0). |
| 3b | `dataset.zip` | 8.9 MB or 4.1 MB (the record pairs the sizes with MD5 `b5fa911bcd6514e39dfef6876f747df4` and `7663bb5e1de2c808beeb202a52dc99ab`; which belongs to which name is read from 3a) | official MD5 from 3a | It gives byte-level confirmation that the 440 WAVs E2 used are the official release. Each member's SHA-256 is compared with the pins in `manifests/catmeows_mirror_sha256.csv`. If any differ, E2 is re-run. |
| 3c | the record's second file | the other of 8.9 / 4.1 MB | official MD5 from 3a | Its content is unknown here. The mirror does not contain it. If it holds more vocalisations or metadata (for example session or owner details), E2's grouping can be checked against it. |

### P3 - Barkopedia (ArlingtonCL2, IJCAI 2025 challenge)

| | |
| --- | --- |
| Source | `https://huggingface.co/datasets/ArlingtonCL2/BarkopediaDogEmotionClassification_Data`. Requested as a repository snapshot (tar) that includes `README.md`. `huggingface.co` is blocked here. |
| What it is | 1,000 training clips (500 husky, 500 shiba) with arousal (low, medium, high) and valence (negative, neutral, positive) labels. The other challenge sets (breed, sex, age group, individual, vocal detection) cover 156 dogs of 5 breeds. These figures come from search results. |
| Terms | **Unverified**: the dataset card could not be read. The card's licence decides whether it is used at all. |
| Unblocks | Dog vocal context and arousal at scale (G4). Experiment: E22b. |
| Checksum | None published. Pinned by SHA-256 on first retrieval. |
| Status | Missing. |

### P4 - research-only keypoints and faces

| Dataset | File | Source | Terms | Unblocks | Status |
| --- | --- | --- | --- | --- | --- |
| CatFLW (2,079 images, 48 facial landmarks) | `catflw.zip` | kaggle.com/datasets/georgemartvel/catflw (blocked) | CC BY-NC 4.0 | cat facial cues, research profile | missing |
| DogFLW (4,335 images, 46 landmarks) | `dogflw.zip` | kaggle.com/datasets/georgemartvel/dogflw (blocked) | CC BY-NC 4.0 | dog facial cues, research profile | missing |
| AwA2 cat images for AwA-Pose (588 cat annotations: ear base and tip, tail base and end) | `AwA2-cats.zip`, a subset of `AwA2-data.zip` (about 13 GB): `JPEGImages/persian+cat` and `JPEGImages/siamese+cat` only | cvml.ista.ac.at/AwA2 (blocked) | per-image licences from AwA2; the annotations are MIT | first cat ear and tail keypoints, research profile | missing |

None of these has a published checksum. Each is pinned by SHA-256 on first
retrieval.

## 3. Considered and not requested

* **iNatSounds 2024.** It has 14 cat and 8 dog recordings in train and
  14 cat and 10 dog in val, with species labels only and no vocalization
  type. They sit inside 86 GB (train) and 27 GB (val) tarballs that cannot
  be read member by member: tar has no index, so reaching a member means
  streaming everything before it. Forty-six short recordings with no type
  labels do not justify about 113 GB of streaming. Not requested.
* **The other 14 CoP3D archives.** See P1.
* **CatMeows in full.** See P3: only verification files are requested.

## 4. Candidates without usable labels (not a download problem)

These are reachable or requestable, but cannot be scored without new labels:

| Source | What it has | What is missing |
| --- | --- | --- |
| CoP3D (P1) | species per sequence, model-generated masks | Behaviour labels and human-verified boxes. Detection is scored against mask-derived boxes, which is stated as a limitation. |
| Kinetics-700 pet classes (in use: E18, E19) | action class per clip (YouTube) | Per-frame boxes, so missed animals are counted per clip and per frame with the species from the class name. Cat actions beyond the class names. |
| YFCC100M / Open Images videos | none relevant | everything |
| iNatSounds | species per recording | vocalization type and context |

## 5. Gaps that only new collection or annotation can close

These are not network or compute blockers, and no transfer can close them:

| Gap | Needed | Specification |
| --- | --- | --- |
| Household validation: home cameras and phones, own pets, low light, partial views, household negatives (plush toys, pet beds) | consented household recordings | [docs/household/](household/) protocol, consent, annotation guide, evaluator |
| Cat ears, tail and body keypoints for a releasable model; dog keypoints under releasable terms | human annotation of 2,000 cat and 2,000 dog CC BY images (estimate 70-100 h) | [training/G2_ANNOTATION_SPEC.md](training/G2_ANNOTATION_SPEC.md); candidates in `results/g2/` |
| Cat actions (eat, drink, groom, litter, scratch, knead, play, sleep); dog play, greeting, scratching, chewing, waiting at the door | timestamped household video | household annotation guide (G3) |
| Pet-specific objects (litter box, scratching post, pet bed, food and water bowl, crate) | new boxes: no public CC BY set has these classes | G1 extension |
| Pet-pet interactions | household multi-pet recordings | household protocol |
| Owner-reported situations (food, outside, play, tense) for interpretation | household recordings with owner reports | household protocol |
| Real-phone acceptance | runs on real phones | [DEVICE_TEST_PROTOCOL.md](DEVICE_TEST_PROTOCOL.md) |

## 6. Write access for durable checkpoints (separate from the above)

A read-only dataset link does not permit uploads, and the importer refuses
to try (`upload()` rejects GET-signed URLs). Training checkpoints are about
21 MB each (measured), and exported models 3-12 MB. One of these is
needed:

1. **Preferred, no stored credential:** V4 signed URLs with method **PUT**
   for a fixed set of object names under one prefix, for example
   `gs://<bucket>/petperception/g1/slot-00.ckpt` … `slot-09.ckpt`, valid for
   7 days. Give them as environment variables
   `PETPERCEPTION_UPLOAD_SLOT_00` … `_09`. Add matching GET-signed URLs for
   restore (`PETPERCEPTION_RESTORE_SLOT_00` … `_09`).
2. **Or** a service account with `roles/storage.objectCreator` and
   `roles/storage.objectViewer`, with an IAM condition limiting both to
   `resource.name.startsWith("projects/_/buckets/<bucket>/objects/petperception/")`.
   It has no delete permission and no access outside that prefix. Store its
   key as an environment secret (environment settings → Network secrets, or
   an environment variable) named `PETPERCEPTION_GCS_WRITER_KEY`. A new
   session picks it up.

Until one exists, checkpoints are kept durable in two other ways, described
in [TRAINING_PROGRAM.md](TRAINING_PROGRAM.md) G1 "Recoverability":
* copies in the claude.ai artifact store, as base64 text shards;
* their SHA-256 values and the restore procedure, committed to git.

The weights themselves are never committed.
