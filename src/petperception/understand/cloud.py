"""Cloud-assisted behaviour understanding: local preprocessing, then a stronger hosted model (E35).

The phone (or the server, for an uploaded file) prepares an evidence packet; the backend sends it to the hosted model
with the API key that only the backend holds, and returns a structured answer that keeps apart what was observed in
the recording, what the user reported, and which explanations are inferred (each a hypothesis with cited sources).

Provider: Anthropic Messages API. Its documented inputs are text and images (JPEG, PNG, GIF, WebP); it takes no
video or audio, so the recording is preprocessed here: video into time-stamped frames or pet-focused crops in
temporal order, sound into a time-stamped event timeline from the local YAMNet model. Documented limits used:
up to 100 images per request on 200k-context models, image cost ceil(w/28) x ceil(h/28) visual tokens
(docs: platform.claude.com/docs/en/build-with-claude/vision, read 2026-10-31).

Input strategies (compared in ``eval/e35_cloud_strategies.py``):
* ``full_frames`` - high-information baseline: whole frames at 2 fps (up to 60, long edge 768 px) + sound timeline;
* ``local_preprocessed`` - phone-style: the local detector's track gives pet-focused crops every 0.5 s (up to 40,
  336 px) and three context views (512 px), plus the local motion summary, detector context and sound timeline;
* ``features_only`` - no pixels: the local recognizer's readings (behaviour bouts and scores, turning / staying in
  place, recording context, sound events) as text.
All three get the same question, species, scene and the same knowledge-base entries (the only citable sources).
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import CloudConfig
from . import knowledge as K
from . import vocab as V

STRATEGIES = ("full_frames", "local_preprocessed", "features_only")
API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
PROMPT_VERSION = "cloud-understanding-v1"
SOUND_MIN = 0.2


def image_tokens(w: int, h: int) -> int:
    return math.ceil(w / 28) * math.ceil(h / 28)


def _jpeg(rgb: np.ndarray, long_edge: int, quality: int = 80) -> tuple[bytes, int, int]:
    h, w = rgb.shape[:2]
    k = min(1.0, long_edge / max(h, w))
    if k < 1.0:
        rgb = cv2.resize(rgb, (max(2, int(w * k)), max(2, int(h * k))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes(), rgb.shape[1], rgb.shape[0]


def _pick(n: int, k: int) -> list[int]:
    return list(range(n)) if n <= k else [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


@dataclass
class Packet:
    strategy: str
    species: str
    scene: str
    question: str
    images: list[dict] = field(default_factory=list)  # {"label", "t", "jpeg", "w", "h"}
    evidence_text: str = ""
    preprocess_s: float = 0.0

    def stats(self, system_text: str = "") -> dict:
        img_bytes = sum(len(i["jpeg"]) for i in self.images)
        text = system_text + self.evidence_text + self.question + self.scene
        return {"strategy": self.strategy, "images": len(self.images), "image_bytes": img_bytes,
                "upload_bytes_base64": int(math.ceil(img_bytes / 3) * 4) + len(text.encode()),
                "image_tokens_est": sum(image_tokens(i["w"], i["h"]) for i in self.images), "text_tokens_est": len(text) // 4,
                "preprocess_s": round(self.preprocess_s, 2)}


def sound_timeline(z, labels: list[str] | None) -> list[str]:
    if labels is None or "a_scores" not in z.files:
        return ["(no usable sound in the recording)"]
    s, t = z["a_scores"].astype(np.float32), z["a_t"]
    out = []
    for i in range(len(t)):
        top = [(labels[j], float(s[i, j])) for j in np.argsort(-s[i])[:4] if s[i, j] >= SOUND_MIN and labels[j] not in ("Silence",)]
        if top:
            out.append(f"{t[i]:.1f}-{t[i] + 0.96:.1f} s: " + ", ".join(f"{n} {v:.2f}" for n, v in top))
    return out or ["(sound present; no sound class above 0.2)"]


def motion_summary(pattern: dict) -> str:
    if not pattern.get("available"):
        return "animal not located reliably; no motion summary"
    keys = ("animal_found_fraction", "stays_in_place", "turned_rad", "repetition_strength", "repetition_period_s", "detector_dog_vote")
    return "; ".join(f"{k}={pattern[k]:.2f}" if isinstance(pattern.get(k), float) else f"{k}={pattern.get(k)}" for k in keys if k in pattern)


def knowledge_text(species: str) -> str:
    lines = []
    for (b, sp), e in sorted(K.ENTRIES.items()):
        if sp != species:
            continue
        lines.append(f"## {b}: {e.summary}")
        lines += [f"- [{c.id}] ({c.kind}) {c.text} (sources: {', '.join(c.sources) or 'none'})" for c in e.candidates]
        if e.distinguish:
            lines.append("  distinguishing observations: " + " | ".join(e.distinguish))
        if e.urgent:
            lines.append("  see a veterinarian promptly if: " + " | ".join(e.urgent))
    return "\n".join(lines)


def system_prompt(species: str) -> str:
    vocab = ", ".join(sorted(b for b, x in V.BEHAVIORS.items() if species in x.species))
    return (
        "You analyse a short recording of a pet to answer its owner's question. The species and the scene description are given by the "
        "owner and are correct. The recording is given as time-stamped images in temporal order and as text evidence (a sound-event "
        "timeline from a sound classifier, and possibly readings from on-device detectors). You never hear the audio yourself.\n"
        "Rules: (1) keep three things apart: what the recording shows (observed, with times), what the owner reports, and inferred "
        "explanations; (2) an action you see does not establish its cause - every explanation is a hypothesis; (3) cite only the "
        "knowledge-base candidate ids given below; (4) say what would tell the candidates apart and when to see a veterinarian; "
        "(5) if a behaviour is not visible, say so instead of assuming the owner's description.\n"
        f"Behaviour labels to use for observed behaviour (or 'other: <short text>'): {vocab}.\n"
        "Answer with one JSON object only, no prose outside it, with keys: observed (list of {behavior, t_start_s, t_end_s, evidence, "
        "confidence: low|medium|high}), not_observed (list of behaviours the owner mentions that the recording does not show), "
        "scene_evidence (list of strings), sound_evidence (list), owner_report_used (list), candidate_explanations (list of {kind: "
        "request|play_or_normal|arousal_or_stress|compulsive|medical, explanation, supported_by_observations (list), "
        "knowledge_ids (list)}), uncertain (list), would_distinguish (list), see_vet_if (list), answer_summary (string).\n\n"
        f"Knowledge base for {species}s (the only citable sources):\n" + knowledge_text(species))


def build_packet(media: Path, species: str, scene: str, question: str, strategy: str, evidence: dict | None = None) -> Packet:
    """``evidence``: local features and answer from ``local_evidence`` (needed for local_preprocessed and features_only)."""
    from .features import _region, decode

    t0 = time.perf_counter()
    p = Packet(strategy, species, scene, question)
    lines = [f"Species (given by the owner): {species}", f"Scene (given by the owner): {scene or '-'}", f"Question: {question or '-'}"]
    if strategy == "full_frames":
        frames, t = decode(Path(media), fps=2.0, max_seconds=60.0, max_side=768)
        for i in _pick(len(frames), 60):
            jb, w, h = _jpeg(frames[i], 768)
            p.images.append({"label": f"frame t={t[i]:.1f} s", "t": float(t[i]), "jpeg": jb, "w": w, "h": h})
        if evidence is not None:
            lines += ["Sound-event timeline (YAMNet, top classes >= 0.2 per 0.96 s):"] + sound_timeline(evidence["z"], evidence["yam"])
    else:
        if evidence is None:
            raise ValueError(f"{strategy} needs local evidence")
        z, ans = evidence["z"], evidence["answer"]
        if strategy == "local_preprocessed":
            frames, t = decode(Path(media), fps=6.0, max_seconds=60.0, max_side=640)
            box = z["box"] if "box" in z.files else np.zeros((0, 4))
            idx = [i for i in range(0, min(len(frames), len(box)), 3) if not np.isnan(box[i, 0])]
            for i in _pick(len(idx), 40) if idx else []:
                k = idx[i]
                x0, y0, x1, y1 = _region(box[k], frames[k].shape[:2], 1.4)
                crop = cv2.resize(frames[k][y0:y1, x0:x1], (336, 336), interpolation=cv2.INTER_AREA)
                jb, w, h = _jpeg(crop, 336)
                p.images.append({"label": f"pet crop t={t[k]:.1f} s", "t": float(t[k]), "jpeg": jb, "w": w, "h": h})
            for q in (1 / 6, 1 / 2, 5 / 6):
                k = min(len(frames) - 1, int(len(frames) * q)) if frames else None
                if k is not None:
                    jb, w, h = _jpeg(frames[k], 512)
                    p.images.append({"label": f"context view t={t[k]:.1f} s", "t": float(t[k]), "jpeg": jb, "w": w, "h": h})
            p.images.sort(key=lambda x: x["t"])
            lines.append("On-device motion summary of the tracked animal: " + motion_summary(ans["temporal_pattern"]))
            lines.append("On-device detector context: " + ("; ".join(f"{c['what']} ({c['from']}, {c['score']})" for c in ans.get("recording_context", [])) or "none"))
        else:
            lines.append("On-device behaviour readings (zero-shot recognizer and turning-in-place model; score >= threshold means reported as seen):")
            lines += [f"- {b['behavior']}: score {b['score']:.2f} (threshold {b['threshold']:.2f}), bouts {b['bouts_s'][:6]}" for b in ans["behaviors_seen"]] or ["- nothing above threshold"]
            lines.append("On-device motion summary: " + motion_summary(ans["temporal_pattern"]))
            lines.append("On-device detector context: " + ("; ".join(f"{c['what']} ({c['from']}, {c['score']})" for c in ans.get("recording_context", [])) or "none"))
        lines += ["Sound-event timeline (YAMNet, top classes >= 0.2 per 0.96 s):"] + sound_timeline(z, evidence["yam"])
    p.evidence_text = "\n".join(lines)
    p.preprocess_s = time.perf_counter() - t0
    return p


def local_evidence(media: Path, species: str, scene: str, question: str, explainer) -> dict:
    """Run the local (version 2) feature extraction and answer once; reused by the packets and as the local comparison."""
    from .run import _Z

    t0 = time.perf_counter()
    z = _Z(explainer.extractor.extract(Path(media)))
    ans = explainer.answer(z, species, scene, question).to_dict()
    yam = explainer.extractor.yamnet.labels if explainer.extractor.yamnet is not None and "a_scores" in z.files else None
    return {"z": z, "answer": ans, "yam": yam, "seconds": time.perf_counter() - t0}


def request_body(p: Packet, model: str, max_tokens: int = 2000) -> dict:
    content: list[dict] = [{"type": "text", "text": p.evidence_text}]
    for im in p.images:
        content.append({"type": "text", "text": im["label"]})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(im["jpeg"]).decode()}})
    content.append({"type": "text", "text": "Answer the owner's question following the rules; JSON only."})
    return {"model": model, "max_tokens": max_tokens, "system": system_prompt(p.species), "messages": [{"role": "user", "content": content}]}


def parse_answer(text: str) -> dict | None:
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def validate_answer(a: dict, species: str) -> list[str]:
    """Problems that make an answer unusable or ungrounded (empty list = usable)."""
    probs = []
    for k in ("observed", "candidate_explanations", "uncertain", "would_distinguish", "answer_summary"):
        if k not in a:
            probs.append(f"missing {k}")
    ids = {c.id for (_, sp), e in K.ENTRIES.items() if sp == species for c in e.candidates}
    for c in a.get("candidate_explanations", []):
        if c.get("kind") not in K.KINDS:
            probs.append(f"unknown kind {c.get('kind')}")
        bad = [i for i in c.get("knowledge_ids", []) if i not in ids]
        if bad:
            probs.append(f"unknown knowledge ids {bad}")
    return probs


def available(cfg: CloudConfig) -> bool:
    return bool(os.environ.get(cfg.api_key_env))


def call(p: Packet, cfg: CloudConfig) -> dict:
    """One request; returns {"status", "answer", "problems", "usage", "latency_s", "cost_usd"}. The key is read from the
    backend's environment and never logged or returned."""
    key = os.environ.get(cfg.api_key_env)
    if not key:
        return {"status": "unavailable", "reason": f"no API key in {cfg.api_key_env} on the backend"}
    body = json.dumps(request_body(p, cfg.model, cfg.max_tokens)).encode()
    req = urllib.request.Request(API_URL, data=body, method="POST",
                                 headers={"x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return {"status": "error", "reason": f"HTTP {exc.code}", "latency_s": round(time.perf_counter() - t0, 2)}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"status": "error", "reason": type(exc).__name__, "latency_s": round(time.perf_counter() - t0, 2)}
    lat = time.perf_counter() - t0
    text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
    usage = out.get("usage", {})
    cost = usage.get("input_tokens", 0) * cfg.price_in_per_mtok / 1e6 + usage.get("output_tokens", 0) * cfg.price_out_per_mtok / 1e6
    ans = parse_answer(text)
    return {"status": "ok" if ans is not None else "unparseable", "answer": ans, "raw_text": None if ans is not None else text[:2000],
            "problems": validate_answer(ans, p.species) if ans is not None else ["no JSON object"], "usage": usage,
            "latency_s": round(lat, 2), "cost_usd": round(cost, 5), "model": out.get("model", cfg.model), "prompt_version": PROMPT_VERSION}


def understand(media: Path, species: str, scene: str, question: str, explainer, strategy: str = "local_preprocessed",
               cfg: CloudConfig | None = None, dry_run: bool = False) -> dict:
    """End-to-end: local evidence, packet, cloud call (or the local answer when the cloud is unavailable)."""
    cfg = cfg or CloudConfig()
    ev = local_evidence(media, species, scene, question, explainer)
    p = build_packet(media, species, scene, question, strategy, ev)
    stats = p.stats(system_prompt(species))
    stats["estimated_cost_usd"] = round((stats["image_tokens_est"] + stats["text_tokens_est"]) * cfg.price_in_per_mtok / 1e6
                                        + cfg.max_tokens * 0.5 * cfg.price_out_per_mtok / 1e6, 5)
    out = {"strategy": strategy, "packet": stats, "local_answer": ev["answer"], "local_seconds": round(ev["seconds"], 2),
           "species": species, "scene": scene, "question": question, "provenance": {"observed": "recording", "owner_report": "question and scene",
                                                                                   "explanations": "hypotheses (cloud model or local knowledge base)"}}
    if dry_run:
        out["cloud"] = {"status": "dry_run"}
    else:
        out["cloud"] = call(p, cfg)
    out["answer_source"] = "cloud" if out["cloud"].get("status") == "ok" else "local"
    return out


def render_markdown(out: dict) -> str:
    """Readable answer: what was observed, what the owner said, which explanations are hypotheses, what is uncertain."""
    from .explain import render_markdown as local_md

    lines = [f"# Behaviour question ({out['species']})", "", f"**Scene (owner):** {out['scene'] or '-'}  ", f"**Question (owner):** {out['question'] or '-'}", "",
             f"**Answer from:** {out['answer_source']} (strategy {out['strategy']}; cloud status {out['cloud'].get('status')})", ""]
    st = out.get("packet")
    if st:
        lines.append(f"Packet: {st['images']} images, {st['upload_bytes_base64'] / 1e6:.2f} MB upload, about {st['image_tokens_est'] + st['text_tokens_est']} input "
                     f"tokens (estimate {st.get('estimated_cost_usd')} USD)")
    a = out["cloud"].get("answer") if out["cloud"].get("status") == "ok" else None
    if a:
        lines += ["", "## Observed in the recording (cloud model; each item needs checking)"]
        lines += [f"- {o.get('behavior')} {o.get('t_start_s')}-{o.get('t_end_s')} s ({o.get('confidence')}): {o.get('evidence')}" for o in a.get("observed", [])] or ["- nothing"]
        if a.get("not_observed"):
            lines += ["", "**Mentioned by the owner but not visible:** " + ", ".join(map(str, a["not_observed"]))]
        lines += ["", "**Scene evidence:** " + "; ".join(map(str, a.get("scene_evidence", []))), "**Sound evidence:** " + "; ".join(map(str, a.get("sound_evidence", []))),
                  "", "## Possible explanations (hypotheses, not established causes)"]
        lines += [f"- [{c.get('kind')}] {c.get('explanation')} (supported by: {'; '.join(map(str, c.get('supported_by_observations', [])))}; sources: "
                  f"{', '.join(c.get('knowledge_ids', []))})" for c in a.get("candidate_explanations", [])]
        lines += ["", "## Uncertain"] + [f"- {u}" for u in a.get("uncertain", [])]
        lines += ["", "## Would tell the explanations apart"] + [f"- {u}" for u in a.get("would_distinguish", [])]
        lines += ["", "## See a veterinarian promptly if"] + [f"- {u}" for u in a.get("see_vet_if", [])]
        if out["cloud"].get("problems"):
            lines += ["", "**Validation problems:** " + "; ".join(out["cloud"]["problems"])]
        lines += ["", f"Cloud cost {out['cloud'].get('cost_usd')} USD, latency {out['cloud'].get('latency_s')} s, model {out['cloud'].get('model')}."]
    lines += ["", "## On-device / local answer", "", local_md(out["local_answer"])]
    return "\n".join(lines)
