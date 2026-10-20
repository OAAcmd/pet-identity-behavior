"""Expert-described pet behaviour videos from open-access papers (supplementary material).

Each entry is a supplementary video whose behaviour and context are described
by the authors (veterinarians or behaviour researchers). Captions are copied
from the article's supplementary-material text or the sentence that cites the
video; ``expert_labels`` translate that description into this project's
behaviour terms, and ``context`` keeps what the authors report about the
situation (for example "chases its tail when it has to wait for food").
Licences are the articles' (all CC BY). Media stay in var/; the manifest
(manifests/paper_videos.json) is committed.
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

UA = "petperception-research/0.2 (academic non-commercial research)"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles"

VIDEOS = [
    {"id": "tiira2012_s1", "species": "dog", "doi": "10.1371/journal.pone.0041684", "licence": "CC BY 4.0 (PLoS ONE)",
     "url": "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0041684.s003",
     "citation": "Tiira K et al. (2012) Environmental effects on compulsive tail chasing in dogs. PLoS ONE 7(7): e41684. Movie S1.",
     "caption": "Tail chasing in Bull Terrier puppy. This female puppy is four months at the time of video shooting, and TC index is 9 (index varies "
                "between 0-12). Dog performs TC multiple times per day, and no clear reason for the behavior can be seen. Dog started the TC behavior "
                "at the age of 9 weeks.",
     "expert_labels": ["tail_chasing"], "context": {"age": "puppy (4 months)", "frequency": "multiple times per day", "trigger": "none apparent",
                                                    "clinical": "compulsive tail chasing (questionnaire TC index 9 of 12)"}},
    {"id": "tiira2012_s2", "species": "dog", "doi": "10.1371/journal.pone.0041684", "licence": "CC BY 4.0 (PLoS ONE)",
     "url": "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0041684.s004",
     "citation": "Tiira K et al. (2012) PLoS ONE 7(7): e41684. Movie S2.",
     "caption": "Tail chasing in German Shepherd male. This is a four year old male at the time of video shooting and has TC index 9. Dog chases its "
                "tail multiple times per day and usually in situations where the dog has to wait for food or when owner talks to other people.",
     "expert_labels": ["tail_chasing"], "context": {"age": "adult (4 years)", "frequency": "multiple times per day",
                                                    "trigger": "waiting for food; owner talking to other people", "clinical": "compulsive tail chasing"}},
    {"id": "tiira2012_s3", "species": "dog", "doi": "10.1371/journal.pone.0041684", "licence": "CC BY 4.0 (PLoS ONE)",
     "url": "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0041684.s005",
     "citation": "Tiira K et al. (2012) PLoS ONE 7(7): e41684. Movie S3.",
     "caption": "Tail chasing in Staffordshire Bull Terrier male. This dog is ten months of age, and all his sisters and brothers and a father (litter "
                "size 7) chase tails, while the mother does not have any stereotypic behavior. Dog has a TC index 7, and the owner reports that the "
                "dog may start TC when nobody pays attention to the dog. This dog also displays a trance-like behavior.",
     "expert_labels": ["tail_chasing"], "context": {"age": "young adult (10 months)", "trigger": "when nobody pays attention to the dog",
                                                    "clinical": "compulsive tail chasing; familial; trance-like behaviour"}},
    {"id": "fhs_cat_2026_m1", "species": "cat", "doi": "10.1007/s11259-026-11255-8", "pmcid": "PMC13156213", "licence": "CC BY 4.0 (Vet Res Commun)",
     "member": "11259_2026_11255_MOESM1_ESM.mp4",
     "citation": "A presumptive diagnosis of feline hyperesthesia syndrome due to stressful human-related condition: case report. Vet Res Commun "
                 "(2026). Supplementary Material 1.",
     "caption": "43-month-old spayed female cat with a one-year history of lumbar tremors, tail chasing and sudden aggressive episodes, temporally "
                "associated with the birth of the owner's child; physical and haematological examination normal; presumptive feline hyperaesthesia "
                "syndrome (article abstract; the supplementary videos document the episodes).",
     "expert_labels": ["tail_chasing", "tail_directed"], "context": {"age": "adult (43 months)", "trigger": "household change (birth of a child)",
                                                                     "clinical": "presumptive feline hyperaesthesia syndrome"}},
    {"id": "fhs_cat_2026_m2", "species": "cat", "doi": "10.1007/s11259-026-11255-8", "pmcid": "PMC13156213", "licence": "CC BY 4.0 (Vet Res Commun)",
     "member": "11259_2026_11255_MOESM2_ESM.mp4", "citation": "Vet Res Commun (2026) case report, Supplementary Material 2.",
     "caption": "Same cat as Supplementary Material 1 (presumptive feline hyperaesthesia syndrome: lumbar tremors, tail chasing, aggressive episodes).",
     "expert_labels": ["tail_chasing", "tail_directed"], "context": {"clinical": "presumptive feline hyperaesthesia syndrome"}},
    {"id": "fhs_cat_2026_m3", "species": "cat", "doi": "10.1007/s11259-026-11255-8", "pmcid": "PMC13156213", "licence": "CC BY 4.0 (Vet Res Commun)",
     "member": "11259_2026_11255_MOESM3_ESM.mp4", "citation": "Vet Res Commun (2026) case report, Supplementary Material 3.",
     "caption": "Same cat as Supplementary Material 1 (presumptive feline hyperaesthesia syndrome: lumbar tremors, tail chasing, aggressive episodes).",
     "expert_labels": ["tail_chasing", "tail_directed"], "context": {"clinical": "presumptive feline hyperaesthesia syndrome"}},
    {"id": "dachshund_flank_2023", "species": "dog", "doi": "10.3389/fvets.2023.1223800", "pmcid": "PMC10414986", "licence": "CC BY 4.0 (Front Vet Sci)",
     "member": "Video_1.MOV",
     "citation": "Case report: Neuropathic pain versus undesirable behavior in a Dachshund after hemilaminectomy surgery. Front Vet Sci (2023). "
                 "Supplementary video 1.",
     "caption": "The video showed the dog barking and biting, sucking and licking the right-side flank, ventrally and slightly caudally to the level "
                "of the surgical incision, 7 days after hemilaminectomy; the behaviour occurred at home, not in the clinic.",
     "expert_labels": ["self_directed_biting", "vocalizing"], "context": {"age": "adult (5.5 years)", "trigger": "after spinal surgery; at home",
                                                                          "clinical": "neuropathic pain vs undesirable behaviour"}},
    {"id": "focal_seizures_dogs_2026", "species": "dog", "doi": "10.3389/fvets.2026.1904889", "pmcid": "PMC13564459", "licence": "CC BY 4.0 (Front Vet Sci)",
     "member": "Video_1.MOV",
     "citation": "Behavioral and autonomic manifestations of suspected focal impaired awareness seizures supported by interictal EEG. Front Vet Sci "
                 "(2026). Supplementary Video 1.",
     "caption": "Episodes of looking behind or to the side as if responding to an unseen stimulus, snapping at the air, hiding, staring into space, "
                "lip licking and fly-catching behaviour with impaired consciousness in three dogs; suspected focal seizures often mistaken for "
                "behavioural disorders.",
     "expert_labels": ["fly_snapping", "staring"], "context": {"clinical": "suspected focal impaired-awareness seizures"}},
]


# Added 2026-10-20 for E32 (fresh test material, never used for development): expert-captioned clips of repetitive movements,
# circling, impaired awareness and other episodes that owners may read as behaviour.
VIDEOS += [
    {"id": "sensory_compulsive_2020", "species": "dog", "doi": "10.1371/journal.pone.0230651", "licence": "CC BY 4.0 (PLoS ONE)", "set": "e32",
     "url": "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0230651.s003", "suffix": ".avi",
     "citation": "Savel S, Sombe P (2020) Are dogs with congenital hearing and/or vision impairments so different from sensory normal dogs? "
                 "PLoS ONE 15(9): e0230651. S1 Video.",
     "caption": "Compulsive behaviours of two sensory impaired dogs filmed before and after behavioural adjustments by owners to the sensory "
                "impairments of their dogs. The initial compulsive behaviours of these two dogs had been foremost considered as neurological signs "
                "prior to medical screening.",
     "expert_labels": ["compulsive_behaviour"], "context": {"clinical": "compulsive behaviour in congenitally deaf and/or blind dogs; first taken "
                                                                       "for neurological signs"}},
    {"id": "hamartoma_circling_2025", "species": "dog", "doi": "10.3389/fvets.2025.1591863", "pmcid": "PMC12150237", "licence": "CC BY 4.0 (Front Vet Sci)",
     "set": "e32", "member": "Video_1.mp4",
     "citation": "Liatis T et al. (2025) Case report: Phenotypic and genetic characterization of a presumptive sporadic hypothalamic hamartoma in a "
                 "standard Schnauzer dog. Front Vet Sci 12:1591863. Supplementary Video 1.",
     "caption": "A 7-month-old male Schnauzer with chronic left head tilt, episodes of imbalance, circling to the left, and continuous repetitive "
                "movements of the head and neck to the left; the episodes could increase with stress or interaction and the dog could not be "
                "distracted out of them; gait analysis showed occasional compulsive tight circling to the left.",
     "expert_labels": ["circling", "head_movements"], "context": {"age": "puppy (7 months)", "clinical": "presumptive hypothalamic hamartoma "
                                                                 "(neurological)", "trigger": "stress or interaction increased episodes"}},
    {"id": "catnip_ethogram_2022", "species": "cat", "doi": "10.1186/s12915-022-01369-1", "pmcid": "PMC9414117", "licence": "CC BY 4.0 (BMC Biol)",
     "set": "e32", "member": "12915_2022_1369_MOESM2_ESM.mp4",
     "url": "https://static-content.springer.com/esm/art%3A10.1186%2Fs12915-022-01369-1/MediaObjects/12915_2022_1369_MOESM2_ESM.mp4",
     "citation": "Bol S et al. (2022) Behavioral differences among domestic cats in the response to cat-attracting plants and their volatile "
                 "compounds. BMC Biol 20:192. Additional file 2.",
     "caption": "Video ethogram showing behavior of domestic cats in response to cat-attracting plants and single compounds; four recordings are "
                "shown for each behavior.",
     "expert_labels": ["catnip_response"], "context": {"trigger": "cat-attracting plants (catnip, silver vine, honeysuckle, valerian)"}},
    {"id": "catnip_cats_2022", "species": "cat", "doi": "10.1186/s12915-022-01369-1", "pmcid": "PMC9414117", "licence": "CC BY 4.0 (BMC Biol)",
     "set": "e32", "member": "12915_2022_1369_MOESM3_ESM.mp4",
     "url": "https://static-content.springer.com/esm/art%3A10.1186%2Fs12915-022-01369-1/MediaObjects/12915_2022_1369_MOESM3_ESM.mp4",
     "citation": "Bol S et al. (2022) BMC Biol 20:192. Additional file 3.",
     "caption": "Video of cats A, H, N, O, V and Z responding to catnip; one representative response is shown for each cat.",
     "expert_labels": ["catnip_response"], "context": {"trigger": "catnip"}},
    {"id": "absence_rr_2018", "species": "dog", "doi": "10.1111/jvim.14892", "pmcid": "PMC5787207", "licence": "CC BY-NC 4.0 (J Vet Intern Med)",
     "set": "e32", "member": "JVIM-32-428-s001.mp4",
     "citation": "Wielaender F et al. (2018) Absence seizures as a feature of juvenile myoclonic epilepsy in Rhodesian Ridgeback dogs. J Vet Intern "
                 "Med 32:428-432. Video S1.",
     "caption": "Absence seizure with generalized 4 Hz spike-and-wave complexes: the dog is nonresponsive to stimuli, slightly swaying and holds the "
                "neck in a flexed position (video-EEG).",
     "expert_labels": ["staring"], "context": {"age": "8 months", "clinical": "absence seizures (staring episodes) in juvenile myoclonic epilepsy"}},
    {"id": "epilepsy_survey_typeD_2025", "species": "dog", "doi": "10.1111/jvim.70077", "pmcid": "PMC12081834", "licence": "CC BY-NC 4.0 (J Vet Intern Med)",
     "set": "e32", "member": "JVIM-39-e70077-s005.mp4",
     "citation": "Matz MS et al. (2025) Development of a novel epilepsy and dyskinesia survey for large-scale characterization of seizure semiology "
                 "in dogs. J Vet Intern Med 39:e70077. Video 4.",
     "caption": "Type D: episodes of impaired awareness (owner-uploaded example of a pragmatic seizure category).",
     "expert_labels": ["staring"], "context": {"clinical": "episode of impaired awareness (possible focal seizure)"}},
    {"id": "epilepsy_survey_typeB_2025", "species": "dog", "doi": "10.1111/jvim.70077", "pmcid": "PMC12081834", "licence": "CC BY-NC 4.0 (J Vet Intern Med)",
     "set": "e32", "member": "JVIM-39-e70077-s009.mp4", "citation": "Matz MS et al. (2025) J Vet Intern Med 39:e70077. Video 2.",
     "caption": "Type B: paroxysmal motor events (owner-uploaded example).", "expert_labels": ["paroxysmal_movements"],
     "context": {"clinical": "paroxysmal motor event (dyskinesia or focal seizure)"}},
    {"id": "head_tremor_2015", "species": "dog", "doi": "10.1155/2015/165463", "pmcid": "PMC4430662", "licence": "CC BY 3.0 (Vet Med Int)",
     "set": "e32", "member": "165463.f1.mp4",
     "citation": "Shell LG et al. (2015) Clinical and breed characteristics of idiopathic head tremor syndrome in 291 dogs. Vet Med Int 2015:165463. "
                 "Video 1.",
     "caption": "A rotational type of head tremor in a 2-year-old miniature Schnauzer (idiopathic head tremor syndrome, an episodic head movement "
                "disorder).",
     "expert_labels": ["head_tremor"], "context": {"age": "2 years", "clinical": "idiopathic head tremor syndrome"}},
    {"id": "narcolepsy_2021", "species": "dog", "doi": "10.1111/jvim.16116", "pmcid": "PMC8162612", "licence": "CC BY-NC-ND 4.0 (J Vet Intern Med)",
     "set": "e32", "member": "JVIM-35-1448-s001.mp4",
     "citation": "Santifort KM et al. (2021) Suspected acquired narcolepsy in 8 dogs. J Vet Intern Med 35:1448-1456. Video S1.",
     "caption": "Footage of included cases: episodes consistent with cataplexy (sudden collapse) in dogs with suspected acquired narcolepsy.",
     "expert_labels": ["collapse"], "context": {"clinical": "suspected acquired narcolepsy with cataplexy"}},
    {"id": "sleep_movements_2016", "species": "dog", "doi": "10.1111/jvim.14575", "pmcid": "PMC5032881", "licence": "CC BY-NC 4.0 (J Vet Intern Med)",
     "set": "e32", "member": "JVIM-30-1681-s003.mov",
     "citation": "Barker EN et al. (2016) Degenerative encephalopathy in Nova Scotia Duck Tolling Retrievers presenting with a rapid eye movement "
                 "sleep behavior disorder. J Vet Intern Med 30:1681-1689. Video S1.",
     "caption": "Home video recording of an affected Nova Scotia Duck Tolling Retriever demonstrating marked involuntary movements during sleep.",
     "expert_labels": ["sleep_movements"], "context": {"clinical": "REM sleep behaviour disorder with degenerative encephalopathy"}},
    {"id": "bruxism_2022", "species": "dog", "doi": "10.1111/jvim.16570", "pmcid": "PMC9708440", "licence": "CC BY 4.0 (J Vet Intern Med)",
     "set": "e32", "member": "JVIM-36-2132-s001.mp4",
     "citation": "Liatis T et al. (2022) Bruxism in awake dogs as a clinical sign of forebrain disease: 4 cases. J Vet Intern Med 36:2132-2140. Video S2.",
     "caption": "Awake bruxism (repetitive clenching or grinding of the teeth) in a dog with a congenital brain anomaly and meningoencephalitis.",
     "expert_labels": ["teeth_grinding"], "context": {"clinical": "awake bruxism as a sign of forebrain disease"}},
    {"id": "eating_seizure_2020", "species": "dog", "doi": "10.1111/jvim.15773", "pmcid": "PMC7255664", "licence": "CC BY 4.0 (J Vet Intern Med)",
     "set": "e32", "member": "JVIM-34-1231-s003.mp4",
     "citation": "Brocal J et al. (2020) Epileptic seizures triggered by eating in dogs. J Vet Intern Med 34:1231-1238. Video S1.",
     "caption": "Focal seizure starting after a few food bites and evolving to become generalized and tonic-clonic.",
     "expert_labels": ["eating", "seizure"], "context": {"clinical": "seizures triggered by eating"}},
]


def _fetch(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=600) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def fetch(data_dir: Path, manifest: Path) -> dict:
    import hashlib

    data_dir.mkdir(parents=True, exist_ok=True)
    zips: dict[str, zipfile.ZipFile] = {}
    out = []
    for v in VIDEOS:
        suffix = v.get("suffix") or Path(v.get("member", "x.wmv")).suffix.lower() or ".wmv"
        p = data_dir / f"{v['id']}{suffix}"
        if not p.exists():
            try:
                if "url" in v:
                    p.write_bytes(_fetch(v["url"]))
                else:
                    if v["pmcid"] not in zips:
                        zips[v["pmcid"]] = zipfile.ZipFile(io.BytesIO(_fetch(EPMC.format(pmcid=v["pmcid"]))))
                    p.write_bytes(zips[v["pmcid"]].read(v["member"]))
            except Exception as exc:  # noqa: BLE001 - one unavailable supplement must not stop the others
                print(f"{v['id']}: not retrieved ({str(exc)[:120]})", flush=True)
                continue
        out.append({**v, "file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest.write_text(json.dumps({"schema": "petperception.paper_videos/1", "videos": out}, indent=1, ensure_ascii=False) + "\n")
    return {"videos": len(out), "bytes": sum(o["bytes"] for o in out)}


if __name__ == "__main__":
    from ..config import REPO_ROOT, load_settings

    print(fetch(load_settings().paths.resolve("data_root") / "paper_videos", REPO_ROOT / "manifests" / "paper_videos.json"))
