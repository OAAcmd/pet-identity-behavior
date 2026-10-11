# Public data sources: found, obtained, blocked

This is the readable form of `manifests/data_sources.json`, a survey of
94 public sources made on 2026-10-10. Licence text was read at the primary
source where it was reachable. Every download host was tested from this
workspace through its egress proxy; `petperception network-probe` repeats
those tests.

Each source falls in one of four categories:

| Category | Meaning |
| --- | --- |
| obtainable now | reachable, and terms known and usable at least for research |
| blocked by network only | terms usable, but the download host is blocked by this workspace's egress policy |
| unresolved usage conditions | terms unknown, conflicting or restrictive, whatever the network |
| requires new collection | no suitable public data exists |

The survey treats each area separately, so a dataset relevant to several
areas appears once per area. Kinetics and MammalNet now carry one category
in every area (reconciled 2026-10-10).

## Obtained and used in this project

| Data | Capability it serves | Experiments | Terms (docs/ASSET_TERMS.md) | How obtained |
| --- | --- | --- | --- | --- |
| ESC-50 (40 cat, 40 dog clips) | cat/dog vocalization presence | E1, E1-CC | CC BY-NC 3.0 collection; per-clip CC0 / CC BY / NC | GitHub, pinned |
| CatMeows (440 meows, 3 contexts, 21 cats) | cat vocal context (G4) | E2 | research and non-commercial (Zenodo text) | unofficial GitHub mirror, per-file pins; Zenodo blocked |
| Open Images V7 subsets (cat/dog boxes, verified negatives, confusable animals, person-pet and pet-object relationships, captions) | detection, false-positive rejection, posture, person interaction, pet on furniture | E4, E7, E10, E13, E15 | CC BY 4.0 / CC BY 2.0 | storage.googleapis.com, S3 |
| Open Images V7 train (G1 manifest) | detector training (G1) | G1 data job | CC BY 4.0 / CC BY 2.0 | streamed annotations; images on demand |
| Ultralytics dog-pose (24 keypoints) | dog ears, tail, keypoints | E9 | research only | GitHub release |
| MammalNet dog-breed videos | dog temporal behaviour | E11, E17 | no dataset licence (research) | annotations from S3; video streamed |
| Oxford-IIIT Pet (37 breeds) | breed robustness, cats and dogs | E16 | CC BY-SA 4.0 annotations | fast.ai S3 mirror, pinned |
| Kinetics-700-2020, 6 pet-action classes (294 validation clips with sound; 600 train clips for tuning and holdout) | full engine on real pet video; video operating point | E18, E19, E19b | annotations CC BY 4.0 (secondary); YouTube clips | CVDF S3 mirror, pinned |

## Priorities and what is possible today

**Cat behaviour (video).**
* No public set labels household cat self-maintenance: eating, drinking,
  grooming, litter use, scratching, kneading, sleeping (`requires new
  collection`).
* The closest sources:
  * A2D: cat eat/jump/roll/run/walk/climb on YouTube video. Blocked
    (web.eecs.umich.edu); research only.
  * VidOR: cat tracks and interaction predicates on YFCC100M user video.
    Annotation licence unstated.
  * MammalNet's Felis clips: no licence; used for dogs only so far.
  * Kinetics "petting cat" and "grooming cat": obtained. These are person
    actions, not cat behaviour.
* Evidence for cats therefore comes from the household package
  (`docs/household/`). E18 measures cat presence and person contact on
  real cat video now.

**Licensed dog (and cat) keypoints.**
* Every public dog keypoint set is research-only or unclear:
  * StanfordExtra and Ultralytics dog-pose (Stanford Dogs images);
  * AP-10K (conflicting statements);
  * Animal-Pose (no licence);
  * SuperAnimal (non-commercial, commercial licence through EPFL-TTO);
  * DogFLW and CatFLW faces (CC BY-NC 4.0, blocked on kaggle.com).
* AwA-Pose is the only set with MIT annotations (588 cat images, ears and
  tail). Its AwA2 images are blocked on cvml.ista.ac.at, and their terms
  are per image.
* The route to keypoints usable outside research is new annotation on
  Open Images CC BY 2.0 dog and cat images. This is job G2 in
  `docs/TRAINING_PROGRAM.md`; the image pool is in the G1 manifest.

**Vocal contexts.**
* Cats: CatMeows (obtained; E2).
* Dogs: no obtainable set.
  * Yin & McCowan barks (3 contexts, 10 dogs): reachable on
    storage.googleapis.com inside the BEANS bucket, but no data licence is
    stated (unresolved).
  * Barkopedia (arousal/valence, 2 breeds): huggingface.co is blocked and
    the licence is unverified.
  * DogSpeak: CC BY-NC-SA per mirrors, huggingface.co blocked.
  * Mudi barks in 6 contexts: data on request from the authors.
* Vocalization type (bark/growl/whine/howl, meow/purr/hiss):
  * FSD50K needs zenodo.org (CC licences per clip). It is now on the
    mirror request list ([DATA_TRANSFER_REQUEST.md](DATA_TRANSFER_REQUEST.md), P2).
  * iNatSounds 2024 has species labels only: train 14 cat recordings
    (550 s) and 8 dog (91 s); val 14 cat (249 s) and 10 dog (147 s). The
    audio sits in tarballs of 86 GB (train) and 27 GB (val), which cannot
    be read member by member. It is not requested (decision 2026-10-11).
  * AudioSet Strong labels are obtained, but the audio is on YouTube
    (blocked).

**Pet-to-pet interactions.**
* VidOR is the best candidate (dog chase/lick/bite dog, cat-dog). Its
  annotation licence is unstated, and its hosts (larksuite,
  huggingface.co) are blocked.
* EgoPet is pet point-of-view, CC BY-NC 4.0, and gated on huggingface.co.
* Visual Genome relationships are CC BY 4.0 (secondary) on
  visualgenome.org, which is blocked.
* No pet-pet labels exist in the obtained data. Household recordings are
  required for evaluation.

**Pet-specific objects.**
* Open Images covers bed, couch, chair, table, car, bowl, toy, teddy bear,
  door, window and stairs. These are in the G1 class set.
* LVIS has dog collars (733 masks); dl.fbaipublicfiles.com is blocked.
* No public source with permissive terms labels litter boxes, food and
  water bowls as such, pet beds, crates, scratching posts or cat trees.
  Roboflow Universe projects exist, but the host is blocked and the
  licences are per project. These need new collection (the household
  annotation guide has an object field).

## Network blockers

Each row gives the exact host and the failed operation. The operation is
repeated by `uv run petperception network-probe`, which writes
`results/data_access/network_probe.json`. Allowing a host resumes the work
listed with it.

| Host | Failed operation | Unblocks | Resume |
| --- | --- | --- | --- |
| zenodo.org | GET https://zenodo.org/api/records/4008297 | CatMeows official copy and checksums; FSD50K (vocalization types, per-clip CC); UrbanSound8K; DESED; SuperAnimal data | `uv run petperception fetch-data catmeows` then `evaluate --experiment e2` |
| huggingface.co | GET https://huggingface.co/api/datasets/ArlingtonCL2/Barkopedia_Dog_Act_Env | Barkopedia (dog bark context), DogSpeak, EgoPet, SuperAnimal weights, VidOR labels | `uv run petperception fetch-data barkopedia --audit-only` |
| www.youtube.com and *.googlevideo.com | GET watch page / media | AudioSet and AudioSet Strong audio, VGGSound, AVE, LLP | - |
| dl.fbaipublicfiles.com | GET / | CoP3D (4,200 home phone videos of pets, CC BY-NC), LVIS | - |
| www.kaggle.com | GET /datasets | CatFLW, DogFLW, FSDKaggle | - |
| drive.google.com, drive.usercontent.google.com | GET / | AP-10K, Animal-Pose, AVE/LLP features, DECADE | - |
| www.robots.ox.ac.uk | GET /~vgg/data/pets/ | Oxford-IIIT Pet official copy (mirror used now) | - |
| images.cocodataset.org | GET annotations zip | COCO 2017 (the same bucket answers path-style on S3; not used because the official host is blocked) | - |
| cvml.ista.ac.at | GET /AwA2/ | AwA2 images for AwA-Pose (cat ears and tail, MIT annotations) | - |
| onedrive.live.com | GET / | APT-36K / APTv2 | - |
| fsannotator.upf.edu | GET FSD50K release | FSD50K mirror | - |
| visualgenome.org | GET / | Visual Genome relationships | - |
| ai.google.dev | GET MediaPipe Object Detector page | licence statement of the MediaPipe detector (ASSET_TERMS Q-M2) | - |
| www.deepmind.com | GET Kinetics page | primary licence statement of Kinetics annotations | - |

The data sources also name many smaller hosts that are blocked or do not
resolve (for example web.eecs.umich.edu, data.bris.ac.uk, forms.gle,
universe.roboflow.com). The catalogue below gives them per source.

## Catalogue

Generated from `manifests/data_sources.json`. Long fields are cut. The
manifest has the full text, the citation, the official page and every
reachability test.

| Area | obtainable_now | blocked_by_network_only | unresolved_usage_conditions | requires_new_collection | total |
| --- | --- | --- | --- | --- | --- |
| cats | 3 | 6 | 10 | 2 | 21 |
| dogs | 6 | 7 | 12 | 1 | 26 |
| audio / AV | 3 | 10 | 10 | 3 | 26 |
| interactions, objects | 3 | 5 | 12 | 1 | 21 |
| all | 15 | 28 | 44 | 7 | 94 |

### obtainable_now

| Source | Area | Species | Offers | Acquired here | Usage conditions |
| --- | --- | --- | --- | --- | --- |
| CatMeows | audio / AV | cat | {'context (first letter of the filename)': ['B = brushing', 'I = isolation in unfamiliar environment', 'F = wa | yes (unofficial GitHub mirror, per-file SHA-256 pinned; E2) | Cite the paper. Confirm the Zenodo terms before commercial use. |
| ESC-50 (already obtained) | audio / AV | dog, cat | {'dog': 40, 'cat': 40, 'note': 'dog is in the ESC-10 subset (CC BY); cat is not'} | yes (E1) | non-commercial for the full set; ESC-10 (includes dog) is CC BY |
| iNatSounds 2024 (Felis catus, Canis familiaris) | audio / AV | dog, cat | {'species_only': ['Felis catus (id 5282)', 'Canis familiaris (id 5271)'], 'vocal_type': 'none'} | no | per-file CC terms (attribution; NC where marked) |
| CAT dataset (Cat Head Detection, Zhang, Sun & Tang 2008) - 9 head points | cats | cats only | 9 points per cat head in '<image>.jpg.cat' text files ('9 x1 y1 ... x9 y9', verified on a mirror file): left e | no | non-commercial only; cite the dataset |
| Open Images V7 - Cat class + cat-breed image-level labels | cats | multi-class; 'Cat' (/m/01yrx) is a boxable and segmentation  | bounding boxes, instance segmentation masks, human-verified + machine image-level labels, point labels, relati | yes (subsets: E4, E7, E10, E13, E15) | attribution; per-image licence verification recommended |
| Oxford-IIIT Pet | cats | cats and dogs: 37 breeds (12 cat breeds: Abyssinian, Bengal, | breed + species label, head ROI bounding box (trainval), pixel trimap (pet/background/boundary) | yes (fast.ai mirror, SHA-256 pinned; E16) | CC BY-SA 4.0 attribution + share-alike; image copyright caveat |
| BARC breed data (dog shape model + breed similarity tables) | dogs | dog | data/breed_data/NIHMS866262-supplement-2.xlsx (Parker et al. 2017 supplemental table: breed clades and pairwis | no | non-commercial scientific research |
| MS COCO 2017 - 'dog' category (images with per-image Flickr licences) | dogs | dog (category id 18), cat, others | boxes, instance masks, captions for 'dog'; COCO keypoints exist only for persons - NO animal keypoints | no | filter images by licence id (4, 5, 7, 8 for commercial; 1-3 are NC); attribution needed |
| Open Images V7 - 'Dog' (/m/0bt9lr) boxes, masks, points | dogs | dog (also 'Cat' and many other classes) | image-level labels, bounding boxes (class /m/0bt9lr 'Dog'), instance segmentation masks, V7 point-level labels | yes (subsets: E4, E7, E10, E13, E15) | none beyond CC BY attribution; per-image verification recommended |
| Oxford-IIIT Pet (25 dog breeds + 12 cat breeds) | dogs | dog (25 breeds), cat (12 breeds) | breed and species label, head bounding box (ROI), pixel trimap (pet / background / border) | yes (fast.ai mirror, SHA-256 pinned; E16) | attribution + share-alike for annotations; image rights stay with owners |
| Stanford Dogs (and fast.ai ImageWoof subset) | dogs | dog | breed label + dog bounding box | no | ImageNet terms |
| Ultralytics Dog-Pose (dog-pose.zip; StanfordExtra labels on Stanford Dogs images) - current baseline | dogs | dog | 24 keypoints (x, y, visibility) + box, class 'dog'. Names (dog-pose.yaml): front_left_paw, front_left_knee, fr | yes (E9, research only) | research only (per Ultralytics docs and ImageNet image terms); no registration |
| COCO 2017 (dog/cat + bowl, sports ball, frisbee, teddy bear, bed, couch, chair) | interactions, objects | dog, cat | boxes + instance masks, captions, keypoints (person only) | no | No registration; COCO terms: 'Use of the images must abide by the Flickr Terms of Use' (per secondary sources). |
| Charades (household activities) | interactions, objects | incidental dogs/cats only | temporal action segments, objects, scene type, scripts, free-text descriptions | no | academic/non-profit/government research; 'may also be used for evaluation purposes elsewhere'; no for-profit use; no redistribution |
| Open Images V7 (pet-object boxes, masks, image-level labels, visual relationships) | interactions, objects | dog, cat (plus other animals) | bounding boxes (incl. IsGroupOf/IsOccluded/IsDepiction flags), instance masks (350 classes), image-level label | yes (subsets: E4, E7, E10, E13, E15) | No registration. Attribution required (CC BY 4.0 annotations; CC BY 2.0 images, per-image verification recommended by Google). |

### blocked_by_network_only

| Source | Area | Species | Offers | Blocked host (failure) | Terms once reachable |
| --- | --- | --- | --- | --- | --- |
| AudioSet (weak, clip-level labels, v1) | audio / AV | dog, cat | {'note': 'Counts are clip-level positives computed in the sandbox from balanced_train_segments.csv / eval_segm | www.youtube.com (403); rr1---sn-a5mekn6d.googlevideo.com (403); research.google.com (403) | Labels: attribution. Media: must be obtained from YouTube by the user, and rights are not cleared. |
| AudioSet Strong (temporally-strong labels) | audio / AV | dog, cat | {'note': 'events = labelled intervals; segments = distinct 10-s clips containing the label; sec = summed label | www.youtube.com / *.googlevideo.com (403) | Labels: attribution. Media rights not cleared. |
| DESED (DCASE Task 4 domestic SED: real + synthetic) | audio / AV | dog, cat | {'note': 'Computed in the sandbox from the metadata TSVs in turpaultn/DESED.', 'weak_train': {'Cat': 173, 'Dog | zenodo.org (403); www.youtube.com (403); project.inria.fr (403) | unknown beyond the above |
| DogSpeak | audio / AV | dog | {'metadata': ['breed', 'sex', 'dog_id'], 'vocal_type': 'not labelled (barks)'} | huggingface.co (403) | non-commercial and share-alike, if the mirror licence holds |
| FSD50K | audio / AV | dog, cat | {'note': 'Clip counts computed in the sandbox from FSD50K.ground_truth dev.csv / eval.csv copies found in publ | zenodo.org (403); fsannotator.upf.edu (403); huggingface.co (403); storage.googleapis.com/gresearch/sound_separation/ (403) | Attribution. Clips under CC-BY-NC and Sampling+ must be filtered out for commercial use, which needs the clips_info json |
| FSDKaggle2018 | audio / AV | dog, cat | {'reported_class_names': ['Bark', 'Meow'], 'counts': 'not verified', 'verification': 'Not verified in the sand | zenodo.org (403); www.kaggle.com (403) | unknown |
| FSDKaggle2019 | audio / AV | dog, cat | {'verified_class_names': ['Bark', 'Meow', 'Purr', 'Hiss'], 'counts': 'not verified (metadata on blocked hosts) | zenodo.org (403); www.kaggle.com (403) | unknown |
| NIGENS (general sound events, includes 'dog') | audio / AV | dog | {'dog': 'count unknown'} | zenodo.org (403) | unknown |
| UrbanSound8K | audio / AV | dog | {'dog_bark': '1,000 slices per the published class table (from memory; not re-verified, because metadata/Urban | zenodo.org (403); goo.gl (403); urbansounddataset.weebly.com (403); dagshub.com (403) | non-commercial only |
| VGGSound | audio / AV | dog, cat | {'note': 'Counts are from data/vggsound.csv in hche11/VGGSound (train/test), computed in the sandbox.', 'dog h | www.robots.ox.ac.uk (403); huggingface.co (403); www.youtube.com / *.googlevideo.com (403) | Annotations: CC BY 4.0. Media: user must obtain it; rights not cleared. |
| A2D - Actor-Action Dataset | cats | 7 actors (adult, baby, ball, bird, car, cat, dog) | video-level actor-action labels; pixel-level actor and action segmentation on sampled frames; 8 actions (climb | web.eecs.umich.edu (403) | research only, no commercial training, no redistribution |
| AFHQ (Animal Faces-HQ) - cat domain | cats | cat, dog, wildlife domains (~5,000 images each) | domain label only (cat/dog/wild); no landmarks, no breed | www.dropbox.com (403) | non-commercial, attribution |
| AP-10K | cats | multi-species: 54 species / 23 families; includes category i | 17 keypoints, COCO format: left eye, right eye, nose, neck, root of tail, left shoulder, left elbow, left fron | drive.google.com (403); pan.baidu.com (403) | Research use fine under either reading; commercial use unclear (CC BY 4.0 vs MMPose 'non-commercial' note; image rights  |
| APT-36K (and extension APTv2) | cats | multi-species: 30 species / 15 families; 'cat' is listed amo | 17 keypoints (same scheme as AP-10K: two eyes, nose, neck, root of tail, shoulders, elbows, knees, hips, four  | 1drv.ms (403) | Research use plausible under stated MIT; YouTube-derived frames make commercial use unclear. |
| AwA-Pose (Animals with Attributes Pose) | cats | multi-species: 35 quadruped classes (+background); cat class | 39 keypoints + bbox per image (pickle dicts, data-only opcodes verified with pickletools, not unpickled): nose | cvml.ist.ac.at (403); cvml.ista.ac.at (403); api.openml.org / www.openml.org (403) | Annotations: MIT (attribution). Images: per-image free licences per AwA2 page; must check archive licence files. |
| CatFLW - Cat Facial Landmarks in the Wild | cats | cats only | 48 facial landmarks (scheme of Finka et al. 2019, chosen for relation to facial musculature and CatFACS action | www.kaggle.com (403) | non-commercial, attribution (cite arXiv:2305.04232) |
| AP-10K | dogs | 54 mammal species / 23 families, including 'dog' (Canidae: d | 17 keypoints, COCO style: left eye, right eye, nose, neck, root of tail, left shoulder, left elbow, left front | drive.google.com (403); pan.baidu.com (403); download.openmmlab.com (403); huggingface.co (third-party copies, e.g. tlpss/ap10k-cropped-512x512) (403) | none stated beyond licence; MMPose note says non-commercial |
| APT-36K / APTv2 (animal pose tracking in video) | dogs | 30 species / 15 families (dog inclusion NOT verified - speci | 17 keypoints (two eyes, nose, neck, tail, two shoulders, two elbows, two knees, two hips, four paws), boxes, p | 1drv.ms (403) | none stated |
| DogFLW (Dog Facial Landmarks in the Wild) | dogs | dog | 46 facial landmarks chosen from DogFACS-relevant musculature + face bounding box; per-image JSON with 'labels' | www.kaggle.com (403) | non-commercial; Kaggle account needed |
| PetFace (dog subset) | dogs | 13 animal families incl. dogs and cats | identity labels; fine-grained attributes: sex, breed, colour, pattern (per paper/search) | dahlian00.github.io (403); docs.google.com (403) | Google Form, manually reviewed |
| StanfordExtra | dogs | dog | 20 2D keypoints + silhouette segmentation per dog (keypoint_definitions.csv rows with Stanford=y): left/right  | forms.gle -> docs.google.com (403); vision.stanford.edu (DNS) | Annotation JSON obtained by filling a Google Form (https://forms.gle/sRtbicgxsWvRtRmUA); download link is emailed. Image |
| SuperAnimal-Quadruped (DeepLabCut Model Zoo) model + SuperAnimal-Quadruped-80K data | dogs | quadrupeds incl. dogs (pooled public datasets; AP-10K held o | 39 keypoints (dlclibrary superquadruped.yaml): nose, upper_jaw, lower_jaw, mouth_end_right, mouth_end_left, ri | huggingface.co (403); zenodo.org (403); deeplabcut.github.io (403); files.pythonhosted.org dlclibrary-0.0.12 wheel (DNS) | research / non-commercial only; commercial licence via EPFL-TTO |
| Tsinghua Dogs | dogs | dog | breed label; head and body bounding boxes | cg.cs.tsinghua.edu.cn (403); zenodo.org (403) | academic use |
| CoP3D (Common Pets in 3D) | interactions, objects | cats, dogs | per-frame camera poses, foreground masks, point clouds (no behaviour labels) | dl.fbaipublicfiles.com (403); cop3d.github.io (403) | non-commercial |
| EPIC-KITCHENS-100 (checked - no pets) | interactions, objects | none labelled - noun classes contain no cat/dog/pet (only 'h | action segments (verb/noun), hand-object | data.bris.ac.uk (403); epic-kitchens.github.io (403) | non-commercial |
| LVIS v1.0 | interactions, objects | dog, cat (+ 'pet', 'kitten', 'puppy', several breed categori | instance masks + boxes, federated (neg/not-exhaustive image lists) | dl.fbaipublicfiles.com (403); www.lvisdataset.org (403) | No registration known. |
| V3Det (Vast Vocabulary Visual Detection) | interactions, objects | cat, kitten, puppy and many dog-breed categories (e.g. shiba | boxes with hierarchical category tree | huggingface.co (403); opendatalab.com / v3det.openxlab.org.cn / download.openxlab.org.cn (403); live.staticflickr.com, farm4.staticflickr.com, www.flickr.com (403) | Images must be crawled from Flickr URLs; Flickr terms per image. |
| Visual Genome (relationships incl. pet-pet and pet-object) | interactions, objects | dog, cat (+ all) | object boxes, attributes, free-text relationships, region descriptions, QA | visualgenome.org (403); homes.cs.washington.edu (403); cs.stanford.edu (403) | No registration known. |

### unresolved_usage_conditions

| Source | Area | Species | Offers | Acquired here | Usage conditions |
| --- | --- | --- | --- | --- | --- |
| AVE (Audio-Visual Event localization) | audio / AV | dog, cat | {'Bark': 171, 'Cat': 60, 'note': 'Counted from data/Annotations.txt (category&videoID&quality&start&end)'} | no | unknown |
| AVSBench (audio-visual segmentation; object + semantic) | audio / AV | dog, cat (likely; not confirmed) | {'cat_dog': "'cat meowing' and 'dog barking' appear in the supplement's sample figure. Whether they are S4 cat | no | request by email |
| Barkopedia (IJCAI 2025 challenge datasets, ArlingtonCL2) | audio / AV | dog | {'emotion': 'arousal x valence (husky/shiba)', 'other_tasks': ['breed', 'sex', 'age group', 'individual dog',  | no | unknown; competition-style access through HF |
| CatSound / NAYA_DATA_AUG1X (domestic cat sound categories) | audio / AV | cat | {'classes (folder names)': ['Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind', 'Mating', 'MotherCall', 'P | no | unknown |
| Dog barks (Yin & McCowan 2004) as packaged in BEANS / Earth Species Library | audio / AV | dog | {'note': 'annotations.csv was extracted in the sandbox through HTTP range requests on the zip (columns filenam | no | unknown; cite Yin & McCowan 2004 |
| Kinetics-700-2020 pet action classes (CVDF mirror) | audio / AV | dog, cat | {'val_counts': {'bathing dog': 47, 'grooming cat': 50, 'grooming dog': 50, 'petting animal (not cat)': 48, 'pe | yes (E18: 294 validation clips of 6 pet-action classes, SHA-256-pinned | unknown; research use is customary |
| LLP (Look, Listen and Parse) / AVVP | audio / AV | dog, cat | {'note': 'Counted in the sandbox from the repo CSVs.', 'clip_level': {'Dog': '761 train / 59 val / 110 test (9 | no | unknown |
| MammalNet (trimmed behaviour clips; genus Canis / Felis) | audio / AV | dog (genus canis; includes wild Canis), cat (genus felis; in | {'behaviours': ['drinks_water', 'sleeps', 'eats_food', 'mates', 'nurses', 'pees', 'grooms', 'poops', 'fights', | yes (annotations; dog-breed video streamed frame by frame for E11 and  | unknown |
| Shiba Inu vocal communication datasets (SJTU) | audio / AV | dog | {'types': ['bark', 'whimper', 'growl', 'howl'], 'context': 'location/activity from YouTube video context'} | no | unknown |
| meow_dataset (haydenroche5) | audio / AV | cat | {'meow': 551} | no | unknown |
| Animal Kingdom | cats | 850+ species; domestic cat almost absent: only 4 (animal, ac | multi-label actions (140, incl. eating, drinking, grooming, licking, sleeping, playing, tail swishing, flappin | no | form-gated; terms not published |
| Animal-Pose dataset (Cao et al. 2019) | cats | Part I: 5 categories (dog, cat, cow, horse, sheep) with keyp | 20 keypoints, COCO-like: L_Eye, R_Eye, L_EarBase, R_EarBase, Nose, Throat, TailBase, Withers, L_F_Elbow, R_F_E | no | none stated |
| Animal3D | cats | 40 mammal species; images from PartImageNet and COCO; whethe | 26 keypoints + SMAL pose and shape parameters | no | unknown |
| AnimalWeb | cats | multi-species: 350 species / 21 orders (334 species in arXiv | 9 facial landmarks per face (eyes, nose, mouth region); species hierarchy labels | no | unknown |
| Feline pain / grimace facial image sets (FGS Evangelista 2019; Finka 2019 48-landmark set; Steagall 2023; TiHo Cat Pain) | cats | cats only | FGS: 5 action units (ear position, orbital tightening, muzzle tension, whiskers change, head position), each 0 | no | presumably on request from authors; clinical images of client-owned cats |
| MammalNet | cats | 173 mammal categories (genus level); genus 'felis' has 37 tr | behaviours: drinks_water, sleeps, eats_food, mates, nurses, pees, grooms, poops, fights, hunts, vomits, gives_ | yes (annotations; dog-breed video streamed frame by frame for E11 and  | unstated |
| PetFace | cats | 13 animal families incl. cats and dogs; cat individual/image | identity, breed, sex (94% of individuals), hierarchical colour, pattern; aligned face crops | no | research only; manual approval of Google Form; link sent by email |
| PetFinder.my Adoption Prediction (Kaggle 2019) / TFDS pet_finder | cats | dogs and cats (Type 1=dog, 2=cat) | per-pet tabular fields: Type, Age, Breed1, Breed2, Gender, Color1-3, MaturitySize, FurLength, Vaccinated, Dewo | no | Kaggle competition data typically restricted to competition/non-commercial use; not verified |
| SuperAnimal-Quadruped-80K (DeepLabCut) | cats | multi-species quadrupeds aggregated from AwA-Pose, Animal-Po | 39 keypoints (verified from DeepLabCut superanimal_quadruped.yaml): same names as AwA-Pose (nose, upper/lower  | no | appears non-commercial with commercial licence on request; not verified |
| VidOR (Video Object Relation) | cats | 80 object categories incl. object.animal.mammal.cat and .dog | per-frame bounding-box trajectories + relation triplets with temporal extents; 50 predicates (8 spatial + 42 a | no | annotation terms unstated; per-video CC licence |
| Animal Kingdom | dogs | 850 species across 6 classes; domestic dog is rare | Pose: 23 keypoints (Head_Mid_Top, Eye_Left, Eye_Right, Mouth_Front_Top, Mouth_Back_Left, Mouth_Back_Right, Mou | no | Dataset (videos/frames) via Google Form https://forms.gle/NipvmReDKaD5zUEw6 (README) or Microsoft Form (MMPose docs); form terms not viewabl |
| Animal-Pose (Cao et al. 2019) | dogs | dog, cat, cow, horse, sheep (Part I, keypoints), +7 other sp | 20 keypoints + boxes + visibility: L/R eye, L/R ear base, nose, throat, tail base, withers, L/R front elbow, L | no | unknown |
| Animal3D | dogs | 40 mammal species (dog inclusion not verified) | 26 keypoints + SMAL pose and shape parameters | no | unknown |
| CVSSP synthetic dogs: SyDog, SyDog-Video, MV-SyDog, DigiDogs | dogs | dog (synthetic) | 2D/3D keypoints (33 for SyDog-Video), segmentation, depth; actions implicit in generated motion | no | licence agreement on cvssp.org; contact m.shooter@surrey.ac.uk |
| Columbia Dogs with Parts (CU Dogs) | dogs | dog | 8 face keypoints per image: right eye, left eye, nose, right ear tip, right ear base, top of head, left ear ba | no | unknown |
| DECADE ('Who Let The Dogs Out?') | dogs | dog (one Alaskan Malamute) | per-frame joint movements from 6 IMUs (4 legs, tail, body) as quaternions; synchronised head-mounted GoPro vid | no | unknown |
| Dog emotion image sets (Kaggle 'Dog Emotions Prediction' and similar 4-class sets; Dog Face Emotion) | dogs | dog | single image-level emotion label | no | unknown |
| DogFACS-coded / emotion-labelled dog video sets (Bremhorst Labrador set; CREMD; Mendeley 'Real World Dataset of Canine Emotions') | dogs | dog | Bremhorst: positive anticipation vs frustration, DogFACS action units; CREMD: crowd emotion labels per clip +  | no | Bremhorst data likely on request from authors (not confirmed) |
| Kinetics-700-2020 dog-related classes | dogs | dog (human-dog interaction classes) | one human-action label per ~10 s clip (youtube_id, time_start, time_end) | yes (E18: 294 validation clips of 6 pet-action classes, SHA-256-pinned | none stated by CVDF |
| MammalNet (Canis subset) - already used | dogs | 173 mammal categories; genus 'canis' includes domestic dogs  | 12 behaviours: drinks water, sleeps, eats food, mates with other animals, nurses or breastfeeds its baby, pees | yes (annotations; dog-breed video streamed frame by frame for E11 and  | none stated |
| PASCAL-Part (dog part masks incl. ears and tail) | dogs | dog, cat and the other 18 PASCAL VOC classes | Dog part segmentation masks (verified from CSAILVision/unifiedparsing part2ind.m: dog = cat parts + muzzle): h | no | unknown |
| Roboflow Universe dog keypoint projects (e.g. project-san60/dog-3j6nu, l-ras/animal-pose-7b9yo, animalpose/dog-vvxey) | dogs | dog (and mixed animals) | keypoint detection projects; keypoint schemas not visible in search results (ear/tail presence unverified) | no | export requires Roboflow account / API key |
| ActivityNet v1.3 (pet classes) | interactions, objects | dog, cat | untrimmed video, temporal segments with activity label | no | unknown; full videos historically via request form (Google Drive/Baidu) |
| Animal Kingdom | interactions, objects | 850 species across 6 animal classes, mostly wild; domestic d | multi-label actions per clip, temporal grounding segments, keypoints | no | request via Google Form (forms.gle/NipvmReDKaD5zUEw6) |
| DECADE (dog egocentric video + joint movements) | interactions, objects | dog (single dog) | synchronised egocentric frames + IMU joint angles | no | unknown |
| Ego4D | interactions, objects | incidental dogs/cats; scenario metadata includes pet activit | dense narrations, scenarios, episodic memory, hands-objects, AV, social, forecasting benchmarks | no | signed licence agreement; AWS credentials issued after approval |
| EgoPet | interactions, objects | cats and dogs (validation also has a few eagle/turtle/goose/ | egocentric (pet-mounted / pet-POV) video segments; interaction yes/no + interaction target with start/end time | no | HF gated dataset: 'You need to agree to share your contact information to access this dataset'; access request reportedly needs an academic  |
| HICO-DET (human-pet interaction boxes) | interactions, objects | dog, cat | human box + object box + verb (HOI triplets) | no | unknown |
| Kinetics-700-2020 (pet classes) | interactions, objects | dog, cat (+ 'petting animal (not cat)') | clip-level single action label (YouTube id + start/end seconds) | yes (E18: 294 validation clips of 6 pet-action classes, SHA-256-pinned | none stated by CVDF; YouTube content rights apply |
| MammalNet (domestic-dog subset) | interactions, objects | dogs (genus canis: bulldog 384 videos, terrier 357, malamute | video-level genus + behaviour labels; temporal behaviour segments (start/end seconds); taxonomy | yes (annotations; dog-breed video streamed frame by frame for E11 and  | none stated |
| Objects365 v2 | interactions, objects | dog, cat | boxes | no | unknown (checked: objects365.org blocked; Ultralytics YAML is AGPL-3.0 for the config only) |
| Roboflow Universe pet-object projects (litter box, food bowl, cat-with-bowl) | interactions, objects | cat (dog in some projects) | boxes (project-dependent) | no | Roboflow account/API key for export; per-project licence |
| TAO (Tracking Any Object) | interactions, objects | dog, cat among 833 LVIS categories | track-level boxes at 1 fps with LVIS category names | no | unknown |
| VidOR (Video Object Relation) | interactions, objects | cat, dog (28 animal categories in total, 3 human categories) | per-frame bounding-box trajectories (manual, tracker-generated boxes flagged), subject-predicate-object relati | no | Test videos 'available after participating the grand challenges'; no other terms stated. |

### requires_new_collection

| Source | Area | Species | Offers | Acquired here | Usage conditions |
| --- | --- | --- | --- | --- | --- |
| EPIC-SOUNDS / Ego4D (egocentric AV) | audio / AV | none labelled | {'pets': 'The class list was not downloaded, so pet classes were not verified. The 44 classes are kitchen acti | no | non-commercial |
| Mudi dog barks in 6 contexts (Pongrácz / Molnár et al.) | audio / AV | dog | {'context': ['stranger', 'fight', 'walk', 'alone', 'ball', 'play']} | no | data on request from the authors (ELTE Budapest), if at all |
| YFCC100M videos via Multimedia Commons (and Google AudioScope clip lists) | audio / AV | unlabelled (any) | {'pets': 'No cat/dog labels in the reachable files. Flickr user tags are in yfcc100m_dataset.sql (65.6 GB) in  | no | per-video CC licence; metadata under Webscope terms |
| GAP: household domestic-cat behaviour video (eating, drinking, grooming, sleeping, playing, hunting/stalking, scratching, litter use, kneading, stretching) | cats | cats | needed: temporally segmented behaviour labels on indoor phone/home-camera video | no | n/a |
| GAP: temporal ear/tail movement and facial-expression/emotion labels for cats in video | cats | cats | needed: per-frame ear position/tail position and CatFACS/FGS-style labels on home video | no | n/a |
| NEW COLLECTION NEEDED: properly licensed dog keypoints + ear/tail position + household behaviour video | dogs | dog | Proposed: StanfordExtra/Ultralytics 24-keypoint schema (incl. ear tips/bases, tail start/end) on (a) Open Imag | no | per-image attribution; exclude NC images for commercial profile |
| GAP: in-home pet-camera footage with pet-object and pet-pet interaction labels | interactions, objects | cats, dogs | would need boxes + track IDs + temporal interaction segments | no | n/a |
