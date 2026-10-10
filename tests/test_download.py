"""The download helper never keeps a truncated or mismatching file (pins are taken on first retrieval)."""

import http.server
import threading
from pathlib import Path

import pytest

from petperception.utils import DownloadError, download

BODY = b"x" * 5000


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY) if self.path == "/ok" else len(BODY) + 100))  # /short declares more than it sends
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server(monkeypatch):
    for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_complete_download_is_kept(server, tmp_path: Path):
    rec = download(f"{server}/ok", tmp_path / "f.bin")
    assert rec["bytes"] == len(BODY) and (tmp_path / "f.bin").read_bytes() == BODY


def test_truncated_download_is_rejected_and_not_kept(server, tmp_path: Path):
    with pytest.raises(DownloadError, match="truncated|IncompleteRead|incomplete"):
        download(f"{server}/short", tmp_path / "f.bin", timeout=5)
    assert not (tmp_path / "f.bin").exists() and not list(tmp_path.glob(".part-*"))
