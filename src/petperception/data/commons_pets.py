"""Real pet behaviour recordings from Wikimedia Commons (freely licensed; research use here).

Why Commons: it holds hundreds of real cat and dog videos whose behaviour is
named by people - the uploader's title and description, and categories that
editors assign ("Videos of grooming cats", "Videos of kneading cats",
"Videos of playing cats", ...). It also mirrors supplementary videos of
open-access papers (e.g. Tiira et al. 2012, PLoS ONE, compulsive tail chasing
in dogs, CC BY). Every file carries its own licence (mostly CC BY / CC BY-SA
/ CC0), author and SHA-1, recorded in the manifest.

Retrieval: the Commons API (no key). Videos are downloaded as the smallest
transcode of at least 360 p (the original when no transcode exists), checked
against the size the API reports, and hashed (SHA-256) on first retrieval.
Raw media stay in var/ (never committed); the manifest
(manifests/commons_pets.json) is committed and lets anyone re-fetch.

Label provenance kept apart:
* ``source_labels``: behaviour terms parsed from the uploader's title and
  description and from editor-assigned categories (human-written, not expert);
* ``paper_labels``: the caption of a paper's supplementary video (expert);
* machine-assisted proposals live in the annotation file with
  ``label_source: assistant_frame_review`` and are never merged into the above.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "petperception-research/0.2 (academic non-commercial research; github.com/OAAcmd/pet-identity-behavior)"
ROOT_CATEGORIES = ("Videos of cats", "Videos of dogs")
EXTRA_SEARCHES = ("pone.0041684", "dog spinning", "dog chasing its tail", "cat hissing", "cat growl", "dog scratching", "cat scratching",
                  "dog licking", "cat licking", "play bow", "dog growling", "dog barking", "cat meowing", "dog playing ball",
                  "cat chasing", "kitten playing", "puppy playing", "dog rolling", "cat rolling", "dog sleeping", "cat sleeping",
                  "dog eating", "dog drinking", "cat stretching", "dog digging", "dog howling", "dog shaking off water")
SKIP_CATEGORY_WORDS = ("anatomy", "animations", "veterinary", "camera traps", "giving birth", "mating", "cat fancy", "dog shows", "by country",
                       "lab cats", "recorded by cats")
MAX_BYTES = 80_000_000
MAX_SECONDS = 180.0

# behaviour vocabulary: label -> regexes over title / description / categories (lower case)
BEHAVIOR_TERMS = {
    "tail_chasing": [r"chas\w* (its |his |her |their )?(own )?tail", r"tail[- ]chas", r"spinning", r"spins? (around|in circles)"],
    "tail_directed": [r"licking (its |his |her )?tail", r"bit\w* (its |his |her )?(own )?tail", r"tail bit"],
    "self_grooming": [r"groom\w* (itself|himself|herself|themselves)", r"self[- ]?(grooming|cleaning)", r"washing (itself|himself|herself)",
                      r"cleaning (itself|himself|herself)", r"grooming cats", r"licking cats", r"licks? (its|his|her) (paw|fur|leg)"],
    "allogrooming": [r"grooms? (a|another|the other|each other)", r"cleaning each other", r"mutual groom"],
    "scratching_self": [r"scratch\w* (its|his|her) (back|ear|neck|head)", r"scratch reflex", r"scratching (his|her|its) back"],
    "kneading": [r"knead"],
    "playing_object": [r"play\w* with (a |the )?(toy|ball|mouse toy|string|laser|bubbles|stick|cotton|feather|box|bag|paper|beans|swab)",
                       r"laser pointer", r"fetch", r"playing (in|with) (the )?(snow|water)", r"tug"],
    "social_play": [r"play[- ]?fight", r"playing together", r"dogs playing", r"cats playing", r"kittens playing", r"puppies play", r"wrestl",
                    r"play bow"],
    "fighting": [r"fighting cats", r"\bfight\b(?!.*play)", r"attack"],
    "hiss_growl": [r"hiss", r"growl", r"spitting"],
    "vocalizing": [r"meow", r"miaow", r"bark", r"howl", r"yowl", r"purr", r"whin", r"cat sounds"],
    "eating": [r"\beat(s|ing)?\b", r"eating", r"\bfeed(ing)?\b", r"chewing"],
    "drinking": [r"drink", r"lapping"],
    "resting": [r"sleep", r"\bnap", r"rest(s|ing)\b", r"relax", r"lying", r"yawn"],
    "jumping": [r"jump"],
    "hunting_chasing": [r"chas\w* (a |the )?(mouse|pigeon|bird|shadow|fly|squirrel|cat|dog)", r"hunt", r"stalk"],
    "being_handled": [r"pett", r"stroking", r"brush(ing|ed) (the |a )?(cat|dog)", r"bath(ing|ed)? (the |a )?(cat|dog)"],
    "locomotion": [r"\bwalk", r"\brun(s|ning)?\b", r"trot", r"swimming"],
}


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001 - network retry with backoff
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Commons API failed: {params}")


def category_tree(root: str, depth: int = 3) -> tuple[set[str], dict[str, set[str]]]:
    """Files under a category tree -> {file title: set of categories it was found in}."""
    seen_cats, files = set(), {}
    frontier = [(root, 0)]
    while frontier:
        cat, d = frontier.pop()
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        cont: dict = {}
        while True:
            q = _get({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}", "cmlimit": 500, **cont})
            for m in q["query"]["categorymembers"]:
                if m["ns"] == 6:
                    files.setdefault(m["title"], set()).add(cat)
                elif m["ns"] == 14 and d < depth:
                    sub = m["title"].removeprefix("Category:")
                    if not any(w in sub.lower() for w in SKIP_CATEGORY_WORDS):
                        frontier.append((sub, d + 1))
            if "continue" not in q:
                break
            cont = q["continue"]
    return seen_cats, files


def search(term: str, limit: int = 50) -> list[str]:
    q = _get({"action": "query", "list": "search", "srsearch": f"{term} filetype:video", "srnamespace": 6, "srlimit": limit})
    return [x["title"] for x in q["query"]["search"]]


def file_info(titles: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(titles), 40):
        q = _get({"action": "query", "titles": "|".join(titles[i:i + 40]), "prop": "videoinfo|categories", "cllimit": 500,
                  "viprop": "url|size|mime|sha1|user|extmetadata|metadata|derivatives", "clshow": "!hidden"})
        for p in q["query"].get("pages", {}).values():
            vi = (p.get("videoinfo") or [None])[0]
            if not vi:
                continue
            meta = {m["name"]: m["value"] for m in (vi.get("metadata") or []) if isinstance(m, dict) and "name" in m}
            em = vi.get("extmetadata") or {}
            out[p["title"]] = {
                "title": p["title"], "url": vi["url"], "bytes": vi.get("size"), "mime": vi.get("mime"), "sha1": vi.get("sha1"), "uploader": vi.get("user"),
                "duration_s": float(meta.get("playtime_seconds") or 0.0) or None, "width": vi.get("width"), "height": vi.get("height"),
                "license": _plain(em.get("LicenseShortName", {}).get("value")), "artist": _plain(em.get("Artist", {}).get("value")),
                "description": _plain(em.get("ImageDescription", {}).get("value"))[:1500],
                "categories": [c["title"].removeprefix("Category:") for c in p.get("categories", [])],
                "derivatives": [{"src": d["src"], "height": d.get("height"), "type": d.get("type")} for d in (vi.get("derivatives") or [])],
            }
    return out


def _plain(html: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


def source_labels(rec: dict) -> list[str]:
    text = " ".join([rec["title"].removeprefix("File:"), rec.get("description") or "", " ".join(rec.get("categories") or [])]).lower()
    text = text.replace("_", " ")
    return sorted({lab for lab, pats in BEHAVIOR_TERMS.items() if any(re.search(p, text) for p in pats)})


def species_of(rec: dict) -> str | None:
    text = " ".join([rec["title"], rec.get("description") or "", " ".join(rec.get("categories") or [])]).lower()
    cat = bool(re.search(r"\bcats?\b|kitten|felis|feline|\bchat\b|katze|gato|mačk|кот|кош", text))
    dog = bool(re.search(r"\bdogs?\b|pupp|canis lupus familiaris|canine|\bchien|hund|perro|pies|собак|terrier|retriever|shepherd|husky|collie|poodle|pug\b|labrador|malamute|beagle", text))
    if cat and not dog:
        return "cat"
    if dog and not cat:
        return "dog"
    return "both" if cat and dog else None


def pick_media(rec: dict) -> tuple[str, str]:
    """Smallest WebM transcode with height >= 360, else the original."""
    ders = [d for d in rec["derivatives"] if d.get("height") and d["height"] >= 360 and "webm" in (d.get("type") or "")]
    if ders:
        d = min(ders, key=lambda x: x["height"])
        return d["src"], f"transcode_{d['height']}p"
    return rec["url"], "original"


def build_manifest(out: Path, depth: int = 3) -> dict:
    found: dict[str, set[str]] = {}
    for root in ROOT_CATEGORIES:
        _, files = category_tree(root, depth)
        for t, cats in files.items():
            found.setdefault(t, set()).update(cats)
    for term in EXTRA_SEARCHES:
        for t in search(term):
            found.setdefault(t, set()).add(f"search:{term}")
    info = file_info(sorted(found))
    recs = []
    for t, rec in sorted(info.items()):
        rec["found_via"] = sorted(found[t])
        rec["species_from_text"] = species_of(rec)
        rec["source_labels"] = source_labels(rec)
        if rec["species_from_text"] not in ("cat", "dog"):
            continue
        if (rec["bytes"] or 0) > MAX_BYTES * 4 or (rec["duration_s"] or 0) > MAX_SECONDS * 4:
            continue
        rec["media_url"], rec["media_kind"] = pick_media(rec)
        recs.append(rec)
    m = {"schema": "petperception.commons_pets/1", "retrieved_at": time.strftime("%Y-%m-%d"), "api": API, "root_categories": list(ROOT_CATEGORIES),
         "searches": list(EXTRA_SEARCHES), "records": recs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n")
    return {"files_found": len(found), "with_info": len(info), "kept": len(recs)}


def key(rec: dict) -> str:
    return hashlib.sha1(rec["title"].encode()).hexdigest()[:12]


CLIP_SECONDS = 60


def download(manifest: Path, data_dir: Path, keys: list[str] | None = None) -> dict:
    """First CLIP_SECONDS of each selected record (stream copy with FFmpeg, no re-encoding; Range reads)."""
    import subprocess

    m = json.loads(manifest.read_text())
    data_dir.mkdir(parents=True, exist_ok=True)
    got, failed = 0, []
    for rec in m["records"]:
        k = key(rec)
        if keys is not None and k not in keys:
            continue
        ext = ".webm" if "webm" in rec["media_url"] else (Path(urllib.parse.urlparse(rec["media_url"]).path).suffix or ".webm")
        p = data_dir / f"{k}{ext}"
        if p.exists() and p.stat().st_size > 0:
            continue
        tmp = data_dir / f"{k}.part{ext}"
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-user_agent", UA, "-i", rec["media_url"], "-t", str(CLIP_SECONDS),
                            "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy", str(tmp)], capture_output=True, timeout=600)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(p)
            got += 1
        else:
            tmp.unlink(missing_ok=True)
            failed.append({"key": k, "error": r.stderr.decode(errors="replace")[-200:]})
        time.sleep(0.3)
    return {"downloaded": got, "failed": failed}


def media_path(data_dir: Path, rec: dict) -> Path | None:
    k = key(rec)
    for p in data_dir.glob(f"{k}.*"):
        if ".part" not in p.name:
            return p
    return None
