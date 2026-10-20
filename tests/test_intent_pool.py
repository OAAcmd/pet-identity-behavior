from __future__ import annotations

import numpy as np

from petperception.intent import kinetics as K
from petperception.intent.av_features import empty_video


class Z:
    def __init__(self, d):
        self._d = d
        self.files = list(d)

    def __getitem__(self, k):
        return self._d[k]


def _clip(T=4, W=3, pets=(1, 1, 0, 1), audio="present", emb=True):
    d = empty_video(T, 77, 2.0)
    d["video_state"] = np.array("present")
    rng = np.random.default_rng(0)
    d["v_whole"][:] = rng.normal(size=(T, 1280))
    for i, p in enumerate(pets):
        if p:
            d["v_pet"][i] = [1, 0.8, 0.6, 1.0]
            d["v_box"][i] = [0.1 + 0.05 * i, 0.2, 0.5 + 0.05 * i, 0.7]
            d["v_crop"][i] = rng.normal(size=1280)
    d["v_person"][1] = [1, 0.9, 0.7, 0.2]
    d["v_motion"][:] = [0.0, 0.1, 0.2, 0.3][:T]
    d["audio_state"] = np.array(audio)
    if audio == "present":
        d["a_scores"] = rng.random((W, 521)).astype(np.float16)
        d["a_rms_db"] = np.full(W, -20.0, np.float32)
        if emb:
            d["a_emb"] = rng.normal(size=(W, 1024)).astype(np.float16)
    return d


def test_branch_dimensions_and_cues():
    br, cues = K.pool(Z(_clip()))
    assert br["audio"].shape == (2048,) and br["audio_scores"].shape == (1042,)
    assert br["animal"].shape == (2644,) and br["animal_phone"].shape == (2569,)
    assert br["environment"].shape == (1357,) and br["interaction"].shape == (8,)
    assert cues["pet_frames"] == 3 and cues["frames"] == 4 and cues["person_frames"] == 1
    assert cues["dog_fraction"] == 1.0 and cues["missing"] == {}
    assert br["interaction"][0] == 0.25 and np.isclose(br["interaction"][6], 0.2)  # person share, mean motion after frame 0


def test_missing_inputs_are_explicit():
    br, cues = K.pool(Z(_clip(pets=(0, 0, 0, 0))))
    assert br["animal"] is None and br["animal_phone"] is None and br["environment"] is not None
    assert cues["missing"]["animal"].startswith("no cat or dog") and cues["dog_fraction"] is None
    br, cues = K.pool(Z(_clip(audio="silent")))
    assert br["audio"] is None and br["audio_scores"] is None and cues["missing"]["audio"] == "audio silent"
    br, cues = K.pool(Z(_clip(emb=False)))
    assert br["audio"] is None and br["audio_scores"] is not None and "embedding" in cues["missing"]["audio"]
    d = _clip()
    d["video_state"] = np.array("absent")
    br, cues = K.pool(Z(d))
    assert all(br[k] is None for k in ("animal", "animal_phone", "environment", "interaction"))
    assert cues["missing"]["environment"] == "video absent"


def test_split_listing_refuses_sealed_set_and_shared_ids(tmp_path):
    root = tmp_path / "kinetics_pets"
    for d, name in (("train_sample", "abcdefghijk_000001_000011.mp4"), ("clips", "zzzzzzzzzzz_000001_000011.mp4")):
        (root / d / "petting cat").mkdir(parents=True)
        (root / d / "petting cat" / name).write_bytes(b"")
    clips = K.list_clips(tmp_path)
    assert {c.split for c in clips} == {"train", "test"} and all(c.youtube_id in ("abcdefghijk", "zzzzzzzzzzz") for c in clips)
    (root / "clips" / "petting cat" / "abcdefghijk_000020_000030.mp4").write_bytes(b"")
    try:
        K.list_clips(tmp_path)
        raise AssertionError("shared YouTube id not detected")
    except RuntimeError as exc:
        assert "shared" in str(exc)
