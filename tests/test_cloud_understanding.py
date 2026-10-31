"""Cloud-assisted behaviour understanding (E35): evidence packets, the provider call (stubbed), the app path with
behaviour questions, and research samples with provisional / owner-reported / human-confirmed annotations."""

from __future__ import annotations

import io
import json
import shutil
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_app_backend import PW, make, register, start

from petperception.config import CloudConfig
from petperception.understand import cloud as CL

pytestmark = pytest.mark.synthetic


class FakeZ(dict):
    @property
    def files(self):
        return list(self.keys())


def _video(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg missing")
    p = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=320x240:r=12:d=4", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def _evidence():
    T = 24  # 4 s at 6 fps
    box = np.tile([0.3, 0.3, 0.7, 0.7], (T, 1)).astype(np.float32)
    z = FakeZ(box=box, a_scores=np.array([[0.9, 0.05], [0.1, 0.6]], np.float32), a_t=np.array([0.0, 0.96]))
    ans = {"behaviors_seen": [{"behavior": "tail_chasing", "score": 0.8, "threshold": 0.64, "bouts_s": [[1.0, 3.0]]}],
           "temporal_pattern": {"available": True, "stays_in_place": True, "turned_rad": 7.1}, "recording_context": []}
    return {"z": z, "answer": ans, "yam": ["Meow", "Purr"]}


def test_packets_keep_time_order_and_report_size(tmp_path):
    v, ev = _video(tmp_path), _evidence()
    stats = {}
    for strat in CL.STRATEGIES:
        p = CL.build_packet(v, "cat", "open living room", "Why does my cat circle while biting its tail?", strat, ev)
        ts = [im["t"] for im in p.images]
        assert ts == sorted(ts)
        assert "Species (given by the owner): cat" in p.evidence_text and "Meow 0.90" in p.evidence_text
        stats[strat] = p.stats(CL.system_prompt("cat"))
    assert stats["features_only"]["images"] == 0 and "tail_chasing: score 0.80" in CL.build_packet(v, "cat", "", "q", "features_only", ev).evidence_text
    assert stats["full_frames"]["images"] == 8  # 4 s at 2 fps
    assert stats["local_preprocessed"]["images"] == 8 + 3  # crops every 0.5 s + 3 context views
    assert stats["full_frames"]["upload_bytes_base64"] > stats["features_only"]["upload_bytes_base64"]
    assert CL.image_tokens(336, 336) == 144


def test_request_body_labels_images_and_cites_only_the_knowledge_base(tmp_path):
    p = CL.build_packet(_video(tmp_path), "cat", "living room", "q", "local_preprocessed", _evidence())
    b = CL.request_body(p, "m")
    kinds = [c["type"] for c in b["messages"][0]["content"]]
    assert kinds.count("image") == len(p.images) and kinds[1] == "text" and kinds[2] == "image"  # a time label before each image
    assert "cat_tc_" in b["system"] and "never hear the audio" in b["system"]


def test_validate_and_parse_answers():
    good = {"observed": [], "candidate_explanations": [{"kind": "medical", "knowledge_ids": ["cat_tc_neuro_circling"]}], "uncertain": [],
            "would_distinguish": [], "answer_summary": "x"}
    assert CL.validate_answer(good, "cat") == []
    bad = dict(good, candidate_explanations=[{"kind": "fate", "knowledge_ids": ["made_up"]}])
    assert len(CL.validate_answer(bad, "cat")) == 2
    assert CL.parse_answer('Here: {"a": 1} done') == {"a": 1} and CL.parse_answer("no json") is None


def test_call_keeps_the_key_on_the_backend(monkeypatch, tmp_path):
    p = CL.build_packet(_video(tmp_path), "cat", "", "q", "features_only", _evidence())
    cfg = CloudConfig(api_key_env="TEST_CLOUD_KEY")
    monkeypatch.delenv("TEST_CLOUD_KEY", raising=False)
    assert CL.call(p, cfg)["status"] == "unavailable"
    monkeypatch.setenv("TEST_CLOUD_KEY", "sk-secret-value")
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        seen["headers"] = dict(req.header_items())
        body = {"model": "m", "usage": {"input_tokens": 1000, "output_tokens": 200},
                "content": [{"type": "text", "text": json.dumps({"observed": [], "candidate_explanations": [], "uncertain": [], "would_distinguish": [],
                                                                  "answer_summary": "s"})}]}
        return Resp(json.dumps(body).encode())

    monkeypatch.setattr(CL.urllib.request, "urlopen", fake_urlopen)
    out = CL.call(p, cfg)
    assert out["status"] == "ok" and out["problems"] == [] and out["cost_usd"] == pytest.approx(1000 * 3e-6 + 200 * 15e-6)
    assert seen["headers"]["X-api-key"] == "sk-secret-value" and "sk-secret-value" not in json.dumps(out)


class FakeExplainer:
    def __init__(self):
        self.extractor = type("X", (), {"extract": staticmethod(lambda path: dict(_evidence()["z"])), "yamnet": None})()

    def answer(self, z, species, scene, question):
        a = _evidence()["answer"] | {"species": species}
        return type("A", (), {"to_dict": lambda self: a})()

    def explain(self, media, species, scene, question):
        return self.answer(None, species, scene, question)


def test_app_question_research_sample_and_annotation_layers(app_settings, tmp_path):
    app, runner, _ = make(app_settings)
    runner.explainer_factory = lambda version: FakeExplainer()
    q = {"species": "cat", "scene": "open living room", "question": "Why does my cat keep circling and biting its tail?"}
    with TestClient(app) as c:
        csrf = register(c)
        h = {"X-CSRF-Token": csrf}
        r = c.post("/api/v1/analyses", json={"upload_id": "x", "question": q | {"understanding": "cloud"}}, headers=h)
        assert r.status_code in (404, 422)
        from test_app_backend import upload, wav_bytes

        uid = upload(c, csrf, wav_bytes())
        r = c.post("/api/v1/analyses", json={"upload_id": uid, "question": q | {"understanding": "cloud"}}, headers=h)
        assert r.status_code == 422 and r.json()["error"]["code"] == "cloud_disabled"
        r = c.post("/api/v1/analyses", json={"upload_id": uid, "question": q | {"retain_for_research": True}}, headers=h)
        assert r.status_code == 422 and r.json()["error"]["code"] == "research_consent_missing"
        aid = start(c, csrf, question=q | {"retain_for_research": True, "research_consent_version": "rs-1"})
        assert runner.run_pending() == 1
        a = c.get(f"/api/v1/analyses/{aid}").json()
        bq = a["behavior_question"]
        assert bq["status"] == "answered" and bq["answer"]["answer_source"] == "local" and bq["answer"]["local_answer"]["species"] == "cat"
        samples = c.get("/api/v1/research-samples").json()
        assert len(samples) == 1 and samples[0]["annotations"][0]["source"] == "model" and samples[0]["annotations"][0]["status"] == "provisional"
        sid = samples[0]["id"]
        assert list((tmp_path / "research").rglob("sample-*"))  # kept although the analysis media was deleted (retention 0)
        fb = c.post(f"/api/v1/research-samples/{sid}/feedback", json={"observed_behavior": "spun 3 times, then groomed", "subsequent_events": "slept"}, headers=h)
        assert fb.status_code == 201 and fb.json()["annotations"][-1]["status"] == "owner_reported"
        rv = c.post(f"/api/v1/research-samples/{sid}/review", json={"behaviors": [{"behavior": "tail_chasing", "t0": 1, "t1": 3}]}, headers=h)
        assert rv.status_code == 201 and rv.json()["annotations"][-1]["status"] == "human_confirmed"  # the first account is the administrator
        assert [a["status"] for a in rv.json()["annotations"]] == ["provisional", "owner_reported", "human_confirmed"]
        assert c.delete(f"/api/v1/research-samples/{sid}", headers=h).status_code == 204
        assert not list((tmp_path / "research").rglob("sample-*")) and c.get("/api/v1/research-samples").json() == []


def test_cloud_mode_without_key_falls_back_to_the_local_answer(app_settings, tmp_path, monkeypatch):
    app_settings.cloud.enabled = True
    app_settings.cloud.api_key_env = "NO_SUCH_TEST_KEY"
    monkeypatch.delenv("NO_SUCH_TEST_KEY", raising=False)
    app, runner, _ = make(app_settings)
    runner.explainer_factory = lambda version: FakeExplainer()
    with TestClient(app) as c:
        csrf = register(c)
        aid = start(c, csrf, question={"species": "cat", "scene": "", "question": "why the circling?", "understanding": "cloud", "strategy": "features_only"})
        runner.run_pending()
        ans = c.get(f"/api/v1/analyses/{aid}").json()["behavior_question"]["answer"]
        assert ans["cloud"]["status"] == "unavailable" and ans["answer_source"] == "local" and ans["packet"]["images"] == 0
        assert PW not in json.dumps(ans)


def test_correction_on_a_donated_recording_becomes_an_owner_report(app_settings):
    app, runner, _ = make(app_settings)
    runner.explainer_factory = lambda version: FakeExplainer()
    with TestClient(app) as c:
        csrf = register(c)
        h = {"X-CSRF-Token": csrf}
        q = {"species": "cat", "scene": "", "question": "why does it spin?", "retain_for_research": True, "research_consent_version": "rs-1"}
        aid = start(c, csrf, question=q)
        runner.run_pending()
        r = c.post(f"/api/v1/analyses/{aid}/feedback", json={"target_type": "observation", "target_id": "tail_chasing", "verdict": "disagree",
                                                              "actual_behavior": "playing_object", "note": "it was chasing a toy"}, headers=h)
        assert r.status_code == 201
        ann = c.get("/api/v1/research-samples").json()[0]["annotations"]
        assert [a["status"] for a in ann] == ["provisional", "owner_reported"]
        assert ann[1]["payload"]["actual_behavior"] == "playing_object" and ann[1]["payload"]["kind"] == "correction"
        caps = c.get("/api/v1/capabilities").json()["behavior_question"]
        assert caps["cloud_available"] is False and any(b["name"] == "tail_chasing" for b in caps["behaviors"])
