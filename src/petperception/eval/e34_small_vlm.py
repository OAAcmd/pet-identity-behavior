"""E34: published small vision-language encoders as phone behaviour encoders, zero-shot on the E31 pool.

E33's distilled student kept 58 % of SigLIP's zero-shot recognition above chance (bar 80 %). E34 asks whether an
off-the-shelf small CLIP model, trained on far more image-text pairs than E33's 20k crops, does better.

Pre-registered (committed before any candidate score was computed):
* Candidates (MIT licence, Hugging Face CLIP format, pinned revisions):
  TinyCLIP ViT-8M/16 + Text-3M (YFCC15M) and TinyCLIP ViT-39M/16 + Text-19M (YFCC15M).
* Zero-shot exactly as in E31/E33: the frozen E31 prompts embedded by the candidate's own text tower, per-frame
  softmax over the species' behaviours (+ neutral) with the candidate's logit scale (no bias), clip score = mean of
  the top quarter of frames, macro AP over E31's 9 eligible behaviours on the E31 pool (278 clips, the teacher's own
  crops), paired difference to SigLIP with a cluster bootstrap over groups (1,000, seed 0).
* Criterion (same as E33): a candidate qualifies if its macro AP >= chance + 0.8 (AP_SigLIP - chance) = 0.347.
  Size matters for the phone (download budget 50 MB with 44.8 MB already used by the default bundle and runtime,
  docs/DEVICE_TIERS_AND_BUDGETS.md): image-tower parameters are reported; a qualifying candidate larger than about
  10 MB at int8 would only fit as an optional download or with the int8 bundle.
* Reported regardless: per-behaviour AP, clip top-behaviour agreement with SigLIP, desktop CPU time per crop (PyTorch).
Nothing here is a phone or browser result; export is a later step for a qualifying candidate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT, Settings
from ..utils import utc_now, write_json

CANDIDATES = {
    "tinyclip_8m16_yfcc": ("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", "a2a8c6eaa2549ad66eb7c31b85022bf58273a26c"),
    "tinyclip_39m16_yfcc": ("wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M", "07a4b0bc751cb64fecd2b661c048c1dd98d69444"),
}
FILES = ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json")


def fetch(models_root: Path, key: str) -> Path:
    import urllib.request

    repo, rev = CANDIDATES[key]
    d = models_root / key
    d.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        p = d / f
        if not p.exists():
            urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/{rev}/{f}", str(p.with_suffix(p.suffix + ".tmp")))
            p.with_suffix(p.suffix + ".tmp").replace(p)
    return d


class ClipEncoder:
    """A Hugging Face CLIP model with the ``ZeroShot`` encoder interface (images, texts, scale, bias)."""

    def __init__(self, model_dir: Path, threads: int = 2):
        import torch
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = CLIPModel.from_pretrained(str(model_dir)).eval()
        self.proc = CLIPImageProcessor.from_pretrained(str(model_dir))
        self.tok = CLIPTokenizerFast(tokenizer_file=str(model_dir / "tokenizer.json"))
        self.scale = float(self.model.logit_scale.exp())
        self.bias = 0.0
        self.image_params = sum(p.numel() for p in self.model.vision_model.parameters()) + sum(p.numel() for p in self.model.visual_projection.parameters())

    def images(self, imgs: list[np.ndarray], batch: int = 32) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(imgs), batch):
                x = self.proc(images=imgs[i:i + batch], return_tensors="pt")
                e = self.model.get_image_features(**x)
                out.append(self.torch.nn.functional.normalize(e, dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.config.projection_dim), np.float32)

    def texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            y = self.tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
            e = self.model.get_text_features(input_ids=y["input_ids"], attention_mask=y["attention_mask"])
            return self.torch.nn.functional.normalize(e, dim=-1).numpy()


def run(settings: Settings, out_dir: Path) -> dict:
    from . import e31_behavior_questions as E31
    from . import e33_phone_student as E33

    e31 = json.loads((REPO_ROOT / "results" / "e31" / "metrics.json").read_text())
    eligible = e31["part_a"]["eligible"]
    rows = E33.pool(settings)
    groups = np.array([c.group for c, _, _ in rows])
    Y = {b: np.array([b in lab for _, _, lab in rows], int) for b in eligible}
    bank = E33._TextBank(E33.work_dir(settings) / "text_e31.npz")
    St, _ = E33.macro_scores(rows, [E33._Z(emb=z["emb"].astype(np.float32)) for _, z, _ in rows], bank, eligible)
    models_root = settings.paths.resolve("models_root")
    encs = {k: ClipEncoder(fetch(models_root, k), settings.runtime.num_threads) for k in CANDIDATES}
    embs = {k: [] for k in encs}
    secs = {k: 0.0 for k in encs}
    n_crops = 0
    for i, (c, z, _) in enumerate(rows):
        crops = E33.recut(c.path, z)
        n_crops += len(crops)
        for k, enc in encs.items():
            t0 = time.perf_counter()
            embs[k].append(E33._Z(emb=enc.images(crops)))
            secs[k] += time.perf_counter() - t0
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)} clips", flush=True)

    def macro(S, idx=None):
        idx = np.arange(len(rows)) if idx is None else idx
        v = [E31.ap(Y[b][idx], S[idx, j]) for j, b in enumerate(eligible)]
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    chance = float(np.mean([Y[b].mean() for b in eligible]))
    need = chance + 0.8 * (macro(St) - chance)
    res = {"experiment": "E34", "created_at": utc_now().isoformat(timespec="seconds"), "clips": len(rows), "crops": n_crops, "eligible": eligible,
           "chance_macro_ap": chance, "siglip_macro_ap": macro(St), "required": need, "candidates": {}}
    for k, enc in encs.items():
        S, _ = E33.macro_scores(rows, embs[k], enc, eligible)
        res["candidates"][k] = {"repo": CANDIDATES[k][0], "revision": CANDIDATES[k][1], "image_tower_params": int(enc.image_params),
                                "macro_ap": macro(S), "macro_ap_ci95": E31.cluster_boot(groups, lambda idx, S=S: macro(S, idx)),
                                "difference_to_siglip": macro(S) - macro(St),
                                "difference_ci95": E31.cluster_boot(groups, lambda idx, S=S: macro(S, idx) - macro(St, idx)),
                                "per_behavior_ap": {b: E31.ap(Y[b], S[:, j]) for j, b in enumerate(eligible)},
                                "clip_top_behavior_agreement": float(np.mean(S.argmax(1) == St.argmax(1))),
                                "cpu_ms_per_crop_pytorch": round(secs[k] / max(n_crops, 1) * 1000, 2), "qualifies": bool(macro(S) >= need)}
    write_json(out_dir / "metrics.json", res)
    return res


def main() -> None:
    import argparse

    from ..config import load_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    s = load_settings()
    if a.threads:
        s.runtime.num_threads = a.threads
    r = run(s, s.paths.resolve("results_root") / "e34")
    print(json.dumps({k: (v if k != "candidates" else {n: {x: c[x] for x in ("macro_ap", "macro_ap_ci95", "difference_ci95", "qualifies",
                                                                            "image_tower_params", "cpu_ms_per_crop_pytorch")} for n, c in v.items()})
                      for k, v in r.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
