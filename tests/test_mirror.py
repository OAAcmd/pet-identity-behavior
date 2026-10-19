"""Mirrored dataset ingestion: resume, retries, verification, expiry, redaction, uploads (local HTTP server)."""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from petperception.data import mirror as MR

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


class Handler(http.server.BaseHTTPRequestHandler):
    calls: list[dict] = []
    drop_first = False
    uploads: dict[str, bytes] = {}

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):  # noqa: N802
        Handler.calls.append({"path": self.path, "range": self.headers.get("Range")})
        if self.path.startswith("/forbidden"):
            self.send_error(403)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
        body = PAYLOAD[start:]
        if Handler.drop_first and len(Handler.calls) == 1:
            # announce the full length but send only part of it, then close
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD[: len(PAYLOAD) // 3])
            self.wfile.flush()
            self.close_connection = True
            return
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(n)
        if self.path.startswith("/readonly"):
            self.send_error(403)
            return
        cr = self.headers.get("Content-Range")
        key = self.path.split("?")[0]
        Handler.uploads[key] = Handler.uploads.get(key, b"") + data if cr else data
        if cr and not cr.endswith(f"/{int(cr.split('/')[1])}") or (cr and int(cr.split("-")[1].split("/")[0]) + 1 < int(cr.split("/")[1])):
            self.send_response(308)
            self.send_header("Range", f"bytes=0-{len(Handler.uploads[key]) - 1}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture()
def server():
    Handler.calls, Handler.drop_first, Handler.uploads = [], False, {}
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _cf(size=len(PAYLOAD), checksums=None) -> MR.CatalogFile:
    return MR.CatalogFile(dataset="demo", version="1.0", file="data.bin", official_url="https://official.example/data.bin", target="demo/data.bin",
                          size=size, checksums=checksums if checksums is not None else {"md5": hashlib.md5(PAYLOAD).hexdigest()})


def test_resume_after_dropped_connection_and_verify(server, tmp_path: Path) -> None:
    Handler.drop_first = True
    secret = "X-Goog-Signature=deadbeef&X-Goog-Date=20991231T000000Z&X-Goog-Expires=3600"
    logs: list[str] = []
    out = MR.fetch_file(_cf(), f"{server}/obj?{secret}", tmp_path, tmp_path / "pins.csv", log=logs.append, sleep=lambda s: None)
    assert out["state"] == "fetched"
    assert (tmp_path / "demo" / "data.bin").read_bytes() == PAYLOAD
    assert Handler.calls[1]["range"] == f"bytes={len(PAYLOAD) // 3}-"  # second request resumed, did not restart
    prov = json.loads((tmp_path / "demo" / "data.bin.provenance.json").read_text())
    assert prov["official_url"] == "https://official.example/data.bin" and prov["verified_against"] == ["md5"]
    blob = json.dumps(prov) + "".join(logs) + json.dumps(out)
    assert "deadbeef" not in blob and "Signature" not in blob  # signature never written


def test_checksum_mismatch_is_rejected_and_nothing_placed(server, tmp_path: Path) -> None:
    with pytest.raises(MR.MirrorError, match="md5 mismatch"):
        MR.fetch_file(_cf(checksums={"md5": "0" * 32}), f"{server}/obj", tmp_path, tmp_path / "pins.csv", sleep=lambda s: None)
    assert not (tmp_path / "demo" / "data.bin").exists() and not (tmp_path / "demo" / "data.bin.part").exists()


def test_first_retrieval_pin_then_enforced(server, tmp_path: Path) -> None:
    cf = _cf(checksums={})
    MR.fetch_file(cf, f"{server}/obj", tmp_path, tmp_path / "pins.csv", sleep=lambda s: None)
    assert (tmp_path / "pins.csv").read_text().strip().endswith(hashlib.sha256(PAYLOAD).hexdigest())
    (tmp_path / "demo" / "data.bin").unlink()
    (tmp_path / "pins.csv").write_text("key,sha256\ndemo/data.bin," + "1" * 64 + "\n")
    with pytest.raises(MR.MirrorError, match="pinned"):
        MR.fetch_file(cf, f"{server}/obj", tmp_path, tmp_path / "pins.csv", sleep=lambda s: None)


def test_expired_url_refused_and_forbidden_reported_without_query(server, tmp_path: Path) -> None:
    with pytest.raises(MR.MirrorError, match="expired"):
        MR.fetch_file(_cf(), f"{server}/obj?X-Amz-Date=20200101T000000Z&X-Amz-Expires=60&X-Amz-Signature=abc", tmp_path, tmp_path / "p.csv")
    with pytest.raises(MR.MirrorError) as e:
        MR.fetch_file(_cf(), f"{server}/forbidden?sig=topsecret&se=2099-01-01T00:00:00Z", tmp_path, tmp_path / "p.csv", sleep=lambda s: None)
    assert "403" in str(e.value) and "topsecret" not in str(e.value)


def test_expiry_parsing_for_each_signing_scheme() -> None:
    now = datetime(2026, 10, 11, tzinfo=UTC)
    assert MR.expiry_report("https://s/o?X-Goog-Date=20261011T000000Z&X-Goog-Expires=1800&X-Goog-Signature=x", now)["state"] == "expires_within_1h"
    assert MR.expiry_report("https://s/o?X-Amz-Date=20261011T000000Z&X-Amz-Expires=604800&X-Amz-Signature=x", now)["state"] == "valid"
    assert MR.expiry_report("https://s/o?Expires=1700000000&Signature=x", now)["state"] == "expired"
    assert MR.expiry_report("https://acct.blob.core.windows.net/c/o?se=2026-10-12T00:00:00Z&sig=x", now)["state"] == "valid"
    assert MR.expiry_report("https://storage.googleapis.com/b/o", now)["state"] == "no_expiry"
    assert MR.redact("https://h/p/o.zip?X-Goog-Signature=s#f") == "https://h/p/o.zip"


def test_urls_come_from_env_or_untracked_file(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "mirrors.json"
    f.write_text(json.dumps({"demo/data.bin": "https://from-file/x"}))
    monkeypatch.setenv("PETPERCEPTION_MIRRORS", str(f))
    cf = _cf()
    assert MR.url_for(cf, MR.mirror_urls()) == "https://from-file/x"
    monkeypatch.setenv(MR.env_key(cf.key), "https://from-env/x")
    assert MR.url_for(cf, MR.mirror_urls()) == "https://from-env/x"


def test_upload_signed_put_and_resumable_session(server, tmp_path: Path) -> None:
    p = tmp_path / "ckpt.bin"
    p.write_bytes(PAYLOAD)
    assert MR.upload(p, f"{server}/bucket/ckpt.bin?X-Goog-Signature=s")["bytes"] == len(PAYLOAD)
    assert Handler.uploads["/bucket/ckpt.bin"] == PAYLOAD
    MR.upload(p, f"{server}/session/ckpt.bin?upload_id=abc", chunk=300_000)
    assert Handler.uploads["/session/ckpt.bin"] == PAYLOAD
    with pytest.raises(MR.MirrorError, match="does not grant upload"):
        MR.upload(p, f"{server}/readonly/x?X-Goog-Signature=s", sleep=lambda s: None)


class FileHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with HTTP Range support (single range), like an object store."""

    def log_message(self, *a):
        pass

    def send_head(self):  # noqa: D401
        path = self.translate_path(self.path.split("?")[0])
        try:
            data = open(path, "rb").read()
        except OSError:
            self.send_error(404)
            return None
        rng = self.headers.get("Range")
        if rng:
            a_, b_ = rng.split("=")[1].split("-")
            start, end = int(a_), min(int(b_) if b_ else len(data) - 1, len(data) - 1)
            body = data[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        import io

        return io.BytesIO(body)


@pytest.fixture()
def file_server(tmp_path: Path):
    import functools

    root = tmp_path / "srv"
    root.mkdir()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(FileHandler, directory=str(root)))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _members() -> dict[str, bytes]:
    import random as _r

    rng = _r.Random(0)
    return {f"seq{i}/frame_{j:03d}.bin": bytes(rng.getrandbits(8) for _ in range(20_000 + 997 * j)) for i in range(3) for j in range(4)} | {
        "meta/info.txt": b"hello " * 5000}


def test_remote_zip_reads_members_without_full_download(file_server) -> None:
    import zipfile

    root, base = file_server
    mem = _members()
    with zipfile.ZipFile(root / "a.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for k, v in mem.items():
            z.writestr(k, v)
    rz = MR.RemoteZip([f"{base}/a.zip?X-Goog-Signature=s"], sleep=lambda s: None)
    assert set(rz.names()) == set(mem)
    assert rz.read("seq1/frame_002.bin") == mem["seq1/frame_002.bin"] and rz.read("meta/info.txt") == mem["meta/info.txt"]
    full = MR.stream_digests(f"{base}/a.zip")
    assert full["sha256"] == hashlib.sha256((root / "a.zip").read_bytes()).hexdigest()


def test_remote_zip_split_archive_and_crc_check(file_server, tmp_path: Path) -> None:
    import subprocess
    import zipfile

    root, base = file_server
    mem = _members()
    src = tmp_path / "src"
    for k, v in mem.items():
        (src / k).parent.mkdir(parents=True, exist_ok=True)
        (src / k).write_bytes(v)
    subprocess.run(["zip", "-q", "-r", "-0", "-s", "64k", str(root / "s.zip"), "."], cwd=src, check=True)  # stored, split into 64 KiB parts
    parts = sorted(p.name for p in root.glob("s.z[0-9]*")) + ["s.zip"]
    assert len(parts) > 2
    rz = MR.RemoteZip([f"{base}/{p}" for p in parts], sleep=lambda s: None)
    for k in ("./seq2/frame_003.bin", "seq2/frame_003.bin"):
        if k in rz.entries:
            assert rz.read(k) == mem["seq2/frame_003.bin"]
            break
    else:
        pytest.fail(f"member not found: {rz.names()[:5]}")
    # a corrupted member is rejected by its CRC
    with zipfile.ZipFile(root / "c.zip", "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("x.bin", b"A" * 5000)
    data = bytearray((root / "c.zip").read_bytes())
    data[100] ^= 0xFF
    (root / "c.zip").write_bytes(bytes(data))
    with pytest.raises(MR.MirrorError, match="CRC"):
        MR.RemoteZip([f"{base}/c.zip"], sleep=lambda s: None).read("x.bin")


def test_local_split_zip_reads_members_across_parts(tmp_path: Path) -> None:
    import subprocess

    mem = _members()
    src = tmp_path / "src"
    for k, v in mem.items():
        (src / k).parent.mkdir(parents=True, exist_ok=True)
        (src / k).write_bytes(v)
    out = tmp_path / "out"
    out.mkdir()
    subprocess.run(["zip", "-q", "-r", "-6", "-s", "64k", str(out / "s.zip"), "."], cwd=src, check=True)  # deflated, split into 64 KiB parts
    parts = [str(p) for p in sorted(out.glob("s.z[0-9]*"))] + [str(out / "s.zip")]
    assert len(parts) > 2
    lz = MR.LocalSplitZip(parts)
    found = 0
    for k, v in mem.items():
        name = k if k in lz.entries else f"./{k}"
        assert lz.read(name) == v
        found += 1
    assert found == len(mem)
