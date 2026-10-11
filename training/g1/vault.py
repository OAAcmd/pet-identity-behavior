"""G1: pack a training checkpoint into verifiable text shards for durable storage, and restore it.

The workspace's disk does not outlive the environment, and git must not hold
weights. A checkpoint is therefore copied to durable storage outside git:
an object-store prefix when write access exists
(docs/DATA_TRANSFER_REQUEST.md section 6), otherwise the claude.ai artifact
store, which accepts text files of up to about 16 MB. This script makes the
copy independent of where it goes:

* `pack` collects one checkpoint of a run (TF checkpoint index and data
  files, its `state.json` sidecar with position, RNG states, configuration
  and data hashes) and optionally the exported models and `final.json`,
  writes them into an uncompressed tar, base64-encodes it and splits it into
  shards of at most --shard-mb MB. A manifest records the SHA-256 of every
  member, of the tar and of every shard, plus the training position.
* `unpack` checks every shard and the tar against the manifest, extracts
  with tar's "data" filter (no links outside the target, no special files)
  into a new run directory and points that run's `latest.json` at the
  restored checkpoint, so `train_resumable.py --run-dir <new>` continues
  from it and `--validate-only` confirms the restored weights match the
  recorded weights SHA-256.

The manifest (small, no weights) can be committed; the shards cannot.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pack(run: Path, checkpoint: str | None, out: Path, shard_mb: float, include_exports: bool, name: str | None) -> dict:
    state_p = Path(checkpoint + ".state.json") if checkpoint else run / "latest.json"
    state = json.loads(state_p.read_text())
    prefix = Path(state["checkpoint"])
    files = sorted(p for p in prefix.parent.glob(prefix.name + ".*") if p.is_file() and not p.name.endswith(".state.json"))
    if not any(p.name.endswith(".index") for p in files):
        raise SystemExit(f"no checkpoint index next to {prefix}")
    members = [(f"ckpt/{p.name}", p.read_bytes()) for p in files]
    members.append(("state.json", json.dumps(state).encode()))
    if include_exports:
        members += [(f"export/{p.name}", p.read_bytes()) for p in sorted((run / "export").glob("*.tflite"))]
        if (run / "final.json").exists():
            members.append(("final.json", (run / "final.json").read_bytes()))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for arc, data in members:
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
            tf.addfile(ti, io.BytesIO(data))
    tar = buf.getvalue()
    text = base64.b64encode(tar)
    step = int(shard_mb * 2**20) // 4 * 4  # whole base64 quanta per shard
    name = name or f"{run.name}-{prefix.name}"
    out.mkdir(parents=True, exist_ok=True)
    shards = []
    for i in range(0, len(text), step):
        part = text[i:i + step]
        f = out / f"{name}.b64.{len(shards):03d}.txt"
        f.write_bytes(part)
        shards.append({"file": f.name, "bytes": len(part), "sha256": sha256_bytes(part)})
    manifest = {
        "schema": "petperception.g1_vault/1", "name": name, "run": run.name, "checkpoint": prefix.name,
        "epoch": state["epoch"], "step_in_epoch": state["step_in_epoch"], "optimizer_iterations": state["optimizer_iterations"],
        "total_epochs": state["total_epochs"], "config": state["config"], "config_sha256": state["config_sha256"],
        "data_sha256": state["data_sha256"], "code_sha256": state["code_sha256"], "weights_sha256": state["weights_sha256"],
        "tar_bytes": len(tar), "tar_sha256": sha256_bytes(tar),
        "members": [{"name": arc, "bytes": len(data), "sha256": sha256_bytes(data)} for arc, data in members],
        "shards": shards,
    }
    (out / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def unpack(manifest_p: Path, shards_dir: Path, run: Path) -> dict:
    m = json.loads(manifest_p.read_text())
    parts = []
    for s in m["shards"]:
        b = (shards_dir / s["file"]).read_bytes().strip()
        if sha256_bytes(b) != s["sha256"]:
            raise SystemExit(f"shard {s['file']} does not match the manifest")
        parts.append(b)
    tar = base64.b64decode(b"".join(parts), validate=True)
    if sha256_bytes(tar) != m["tar_sha256"]:
        raise SystemExit("reassembled tar does not match the manifest")
    if (run / "latest.json").exists():
        raise SystemExit(f"{run} already holds a run; restore into a new directory")
    run.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar), mode="r") as tf:
        tf.extractall(run, filter="data")
    want = {x["name"]: x["sha256"] for x in m["members"]}
    for name, digest in want.items():
        if sha256_bytes((run / name).read_bytes()) != digest:
            raise SystemExit(f"member {name} does not match the manifest")
    state = json.loads((run / "state.json").read_text())
    state["checkpoint"] = str(run / "ckpt" / m["checkpoint"])
    state["restored_from_vault"] = m["name"]
    (run / "ckpt" / (m["checkpoint"] + ".state.json")).write_text(json.dumps(state))
    (run / "latest.json").write_text(json.dumps(state))
    return {"restored": m["name"], "run_dir": str(run), "checkpoint": state["checkpoint"], "epoch": state["epoch"],
            "step_in_epoch": state["step_in_epoch"], "weights_sha256": state["weights_sha256"], "members_verified": len(want)}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", help="checkpoint prefix (default: the run's latest)")
    p.add_argument("--out", required=True)
    p.add_argument("--shard-mb", type=float, default=14.0)
    p.add_argument("--include-exports", action="store_true")
    p.add_argument("--name")
    u = sub.add_parser("unpack")
    u.add_argument("--manifest", required=True)
    u.add_argument("--shards-dir", required=True)
    u.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "pack":
        m = pack(Path(a.run_dir), a.checkpoint, Path(a.out), a.shard_mb, a.include_exports, a.name)
        print(json.dumps({k: m[k] for k in ("name", "epoch", "step_in_epoch", "tar_bytes", "tar_sha256", "weights_sha256")} | {"shards": len(m["shards"])}))
    else:
        print(json.dumps(unpack(Path(a.manifest), Path(a.shards_dir), Path(a.run_dir))))


if __name__ == "__main__":
    main()
