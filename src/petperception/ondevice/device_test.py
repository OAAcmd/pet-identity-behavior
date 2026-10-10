"""Static HTTPS test build for real-phone acceptance (the "phone test").

Assembles the phone test page (web/devicetest/page.html + the harness script
built by `npm run build:devicetest`), the LiteRT.js WebAssembly runtime and an
exported on-device bundle into one directory that any static HTTPS host can
serve (it was built for a claude.ai Artifact page, which needs no server of
ours). That host serves only web media types, so binary model files are
published as Base64 text parts (``.b64.txt``, at most ``MAX_PART`` decoded bytes
each); the page decodes and reassembles them and checks the SHA-256 of every
decoded part and of the whole file before use. Nothing in this directory is committed: it contains model
weights (see manifests/resources.json for their terms).
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
from pathlib import Path

from ..utils import sha256_file, utc_now

SCHEMA = "petperception.device_test_site/1"
HARNESS_VERSION = "phone-test-1"
MAX_PART = 9 * 2**20  # decoded bytes per part; Base64 text of it stays below the 16 MB per-file text limit of the Artifact host
BINARY = (".tflite",)  # served as Base64 text parts
RUNTIME_FILES = ("litert_wasm_internal.js", "litert_wasm_internal.wasm", "litert_wasm_compat_internal.js", "litert_wasm_compat_internal.wasm")
CONTENT_TYPES = {".json": "application/json", ".js": "text/javascript", ".wasm": "application/wasm", ".mp4": "video/mp4", ".webm": "video/webm",
                 ".html": "text/html"}


def content_type(path: str) -> str:
    return CONTENT_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def credits(bundle_dir: Path, manifest: dict) -> list[dict]:
    """Attribution shown on the page: models (manifests/resources.json), runtime, reference clip images."""
    from ..registry import resource

    out = []
    for rid in dict.fromkeys(f["resource_id"] for f in manifest["files"] if f["kind"] == "model" and f["resource_id"]):
        r = resource(rid)
        weights = r.get("terms", {}).get("weights", "")
        short = re.split(r":|\. ", weights, maxsplit=1)[0]
        out.append({"item": r["name"], "credit": r.get("attribution", ""), "terms": f"weights licence: {short}"})
    out += [{"item": "LiteRT.js runtime (@litertjs/core 2.5.3)", "credit": "Google", "terms": "Apache-2.0"},
            {"item": "Preact 10.29.8", "credit": "Jason Miller and contributors", "terms": "MIT"}]
    att = bundle_dir / "device_check" / "ATTRIBUTION.json"
    if att.exists():
        a = json.loads(att.read_text())
        for role in ("animal", "background"):
            if a.get(role):
                out.append({"item": f"Reference clip {role} photo (Open Images {a[role]['image_id']})", "credit": a[role].get("author") or "",
                            "terms": f"{a[role].get('license')} - {a[role].get('source')}"})
    return out


def export_site(bundle_dir: Path, web_dir: Path, out: Path) -> dict:
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    app_js = web_dir / "dist-devicetest" / "app.js"
    if not app_js.exists():
        raise FileNotFoundError(f"{app_js} missing: run `cd web && npm run build:devicetest` first")
    if out.exists():
        shutil.rmtree(out)
    (out / "bundle").mkdir(parents=True)
    parts: dict[str, list[dict]] = {}
    published: dict[str, str] = {}
    for f in manifest["files"]:
        src = bundle_dir / f["path"]
        if sha256_file(src) != f["sha256"]:
            raise RuntimeError(f"{f['path']}: does not match the bundle manifest")
        if Path(f["path"]).suffix in BINARY or f["bytes"] > MAX_PART:
            data = src.read_bytes()
            plist = []
            for k in range(0, len(data), MAX_PART):
                rel = f"bundle/{f['path']}.part{k // MAX_PART}.b64.txt"
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                chunk = data[k:k + MAX_PART]
                dst.write_bytes(base64.b64encode(chunk))
                plist.append({"path": rel, "sha256": hashlib.sha256(chunk).hexdigest(), "bytes": len(chunk), "encoding": "base64"})
                published[rel] = "text/plain"
            parts[f["path"]] = plist
        else:
            rel = f"bundle/{f['path']}"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out / rel)
            published[rel] = content_type(rel)
    site = {"schema": SCHEMA, "harness_version": HARNESS_VERSION, "created_at": utc_now().isoformat(timespec="seconds"), "bundle": manifest, "parts": parts,
            "credits": credits(bundle_dir, manifest)}
    (out / "bundle" / "site_manifest.json").write_text(json.dumps(site, indent=1))
    published["bundle/site_manifest.json"] = "application/json"
    (out / "litert").mkdir()
    for name in RUNTIME_FILES:
        shutil.copyfile(web_dir / "node_modules" / "@litertjs" / "core" / "wasm" / name, out / "litert" / name)
        published[f"litert/{name}"] = content_type(name)
    shutil.copyfile(app_js, out / "app.js")
    published["app.js"] = "text/javascript"
    shutil.copyfile(web_dir / "devicetest" / "page.html", out / "index.html")
    sizes = {p: (out / p).stat().st_size for p in published}
    # Map for publishing: published path -> {from, contentType} (the page itself is index.html).
    files = {p: {"from": str((out / p).resolve()), "contentType": t} for p, t in published.items()}
    (out.parent / f"{out.name}_files.json").write_text(json.dumps(files, indent=1))
    return {"out": str(out), "bundle_version": manifest["bundle_version"], "release_profile": manifest["release_profile"], "files": len(published) + 1,
            "total_bytes": sum(sizes.values()) + (out / "index.html").stat().st_size, "largest_file_bytes": max(sizes.values()), "split_files": sorted(parts)}
