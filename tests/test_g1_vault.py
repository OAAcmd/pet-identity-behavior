"""training/g1/vault.py: checkpoint shards round-trip, and every corruption is refused."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _vault():
    spec = importlib.util.spec_from_file_location("g1_vault", REPO / "training" / "g1" / "vault.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_run(tmp: Path) -> Path:
    run = tmp / "runs" / "r1"
    (run / "ckpt").mkdir(parents=True)
    prefix = run / "ckpt" / "ckpt-7"
    Path(str(prefix) + ".index").write_bytes(b"index-bytes")
    Path(str(prefix) + ".data-00000-of-00001").write_bytes(bytes(range(256)) * 300)
    state = {"checkpoint": str(prefix), "epoch": 2, "step_in_epoch": 15, "optimizer_iterations": 515, "total_epochs": 10,
             "config": {"stages": "10:all"}, "config_sha256": "c" * 64, "data_sha256": {"train": "t" * 64}, "code_sha256": "s" * 64,
             "weights_sha256": "w" * 64}
    Path(str(prefix) + ".state.json").write_text(json.dumps(state))
    (run / "latest.json").write_text(json.dumps(state))
    return run


def test_pack_unpack_round_trip(tmp_path):
    V = _vault()
    run = _fake_run(tmp_path)
    m = V.pack(run, None, tmp_path / "vault", shard_mb=0.02, include_exports=False, name=None)
    assert len(m["shards"]) > 1 and all(s["bytes"] % 4 == 0 for s in m["shards"][:-1])
    out = V.unpack(tmp_path / "vault" / f"{m['name']}.manifest.json", tmp_path / "vault", tmp_path / "restored")
    assert out["epoch"] == 2 and out["step_in_epoch"] == 15 and out["members_verified"] == 3
    restored = tmp_path / "restored"
    assert (restored / "ckpt" / "ckpt-7.data-00000-of-00001").read_bytes() == bytes(range(256)) * 300
    latest = json.loads((restored / "latest.json").read_text())
    assert latest["checkpoint"] == str(restored / "ckpt" / "ckpt-7") and latest["restored_from_vault"] == m["name"]


def test_corrupt_shard_and_occupied_target_are_refused(tmp_path):
    V = _vault()
    run = _fake_run(tmp_path)
    m = V.pack(run, None, tmp_path / "vault", shard_mb=0.02, include_exports=False, name="x")
    man = tmp_path / "vault" / "x.manifest.json"
    with pytest.raises(SystemExit):
        V.unpack(man, tmp_path / "vault", run)  # a directory that already holds a run
    shard = tmp_path / "vault" / m["shards"][0]["file"]
    b = bytearray(shard.read_bytes())
    b[10] = ord("A") if b[10] != ord("A") else ord("B")
    shard.write_bytes(bytes(b))
    with pytest.raises(SystemExit):
        V.unpack(man, tmp_path / "vault", tmp_path / "restored")
