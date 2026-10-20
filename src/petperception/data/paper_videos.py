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
        suffix = Path(v.get("member", "x.wmv")).suffix.lower() or ".wmv"
        p = data_dir / f"{v['id']}{suffix}"
        if not p.exists():
            if "url" in v:
                p.write_bytes(_fetch(v["url"]))
            else:
                if v["pmcid"] not in zips:
                    zips[v["pmcid"]] = zipfile.ZipFile(io.BytesIO(_fetch(EPMC.format(pmcid=v["pmcid"]))))
                p.write_bytes(zips[v["pmcid"]].read(v["member"]))
        out.append({**v, "file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest.write_text(json.dumps({"schema": "petperception.paper_videos/1", "videos": out}, indent=1, ensure_ascii=False) + "\n")
    return {"videos": len(out), "bytes": sum(o["bytes"] for o in out)}
