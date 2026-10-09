"""Command-line interface: ``petperception <command> [options]`` (see README)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path

from . import __version__
from .config import REPO_ROOT, load_settings

EXPERIMENTS = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10"]
DATASETS = ["esc50", "openimages-catdog", "openimages-tuning", "dog-pose", "posture-weak", "catmeows", "barkopedia"]
PROBE_HOSTS = ["https://pypi.org/simple/", "https://storage.googleapis.com/", "https://raw.githubusercontent.com/",
               "https://zenodo.org/", "https://huggingface.co/", "https://docs.ultralytics.com/"]


def _settings(args: argparse.Namespace):
    return load_settings(args.config)


def cmd_doctor(args: argparse.Namespace) -> int:
    from .registry import check_model, load_resources

    s = _settings(args)
    rep: dict = {"engine_version": __version__, "python": sys.version.split()[0], "executable": sys.executable,
                 "platform": platform.platform(), "config": args.config or "configs/cpu.yaml", "config_hash": s.config_hash()}
    try:
        rep["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        rep["mem_total_gb"] = round(int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1]) / 1048576, 1)
    except (OSError, StopIteration):
        pass
    rep["cpu_count"] = os.cpu_count()
    du = shutil.disk_usage(REPO_ROOT)
    rep["disk_free_gb"] = round(du.free / 1e9, 1)
    rep["gpu"] = "nvidia device present" if list(Path("/dev").glob("nvidia*")) else "none detected"
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        rep[tool] = subprocess.run([tool, "-version"], capture_output=True, text=True).stdout.splitlines()[0] if path else "MISSING"
    pkgs = ["numpy", "scipy", "scikit-learn", "librosa", "soundfile", "opencv-python-headless", "ai-edge-litert", "pydantic", "fastapi", "uvicorn"]
    rep["packages"] = {p: _ver(p) for p in pkgs}
    rep["models"] = {}
    for rid, r in load_resources().items():
        if r["kind"] == "model":
            m = check_model(s, rid)
            rep["models"][rid] = {"status": m.status, "reason": m.reason}
    data_root = s.paths.resolve("data_root")
    rep["datasets"] = {
        "esc50": (data_root / "esc50" / "meta" / "esc50.csv").exists(),
        "openimages-catdog": (data_root / "openimages" / "images").exists(),
        "catmeows": (data_root / "catmeows" / "dataset").exists(),
        "barkopedia": (data_root / "barkopedia" / "audit_review.json").exists(),
    }
    if args.network:
        rep["network"] = {}
        for u in PROBE_HOSTS:
            try:
                with urllib.request.urlopen(urllib.request.Request(u, method="HEAD"), timeout=10) as resp:
                    rep["network"][u] = resp.status
            except urllib.error.HTTPError as exc:  # the host answered: reachable
                rep["network"][u] = f"reachable (HTTP {exc.code})"
            except Exception as exc:  # noqa: BLE001 - report any failure verbatim
                rep["network"][u] = f"unreachable: {type(exc).__name__}: {str(exc)[:80]}"
    print(json.dumps(rep, indent=2))
    ready = all(v["status"] == "ready" for k, v in rep["models"].items() if k in (s.audio.yamnet.model_id, s.vision.detector))
    return 0 if ready and rep["ffmpeg"] != "MISSING" else 1


def _ver(p: str) -> str:
    try:
        return metadata.version(p)
    except metadata.PackageNotFoundError:
        return "not installed"


def cmd_fetch_models(args: argparse.Namespace) -> int:
    from .registry import default_model_ids, fetch_model, load_resources

    s = _settings(args)
    ids = args.only or default_model_ids(s)
    if args.include_optional:
        ids += [rid for rid, r in load_resources().items() if r["kind"] == "model" and r["status"] == "available_optional"]
    rc = 0
    for mid in dict.fromkeys(ids):
        out = fetch_model(s, mid)
        print(json.dumps(out, indent=2))
        rc |= 0 if out["status"] == "ready" else 1
    return rc


def cmd_fetch_data(args: argparse.Namespace) -> int:
    s = _settings(args)
    root = s.paths.resolve("data_root")
    man = s.paths.resolve("manifests_root")
    if args.dataset == "esc50":
        from .data import esc50

        d = esc50.fetch(root)
        v = esc50.verify_against_manifest(d, man / "esc50_files.csv")
        print(json.dumps({"dataset": "esc50", "path": str(d), "checked": v["checked"], "mismatched_or_missing": v["mismatched_or_missing"][:10]}, indent=2))
        return 0 if not v["mismatched_or_missing"] else 1
    if args.dataset == "openimages-catdog":
        from .data import openimages as OI
        from .registry import resource

        expected = {f["name"]: f["sha256"] for f in resource("openimages_v7_val_catdog")["files"]}
        ann = OI.fetch_annotations(root, expected)
        if not (man / "openimages_catdog_images.csv").exists() or args.rebuild_subset:
            print(json.dumps(OI.build_subset(root, man), indent=2))
        out = OI.fetch_images(root, man)
        print(json.dumps({"annotations": {k: v.get("sha256") for k, v in ann.items()}, "images": {k: v for k, v in out.items() if k != "errors"},
                          "errors": list(out["errors"].items())[:10]}, indent=2))
        return 0 if not out["errors"] else 1
    if args.dataset == "openimages-tuning":
        from .data import openimages_tuning as OT

        OT.fetch_annotations(root)
        if not (man / "oi_test_tuning_images.csv").exists():
            print(json.dumps(OT.build(root, man), indent=2))
        out = OT.fetch_images(root, man)
        print(json.dumps({k: v for k, v in out.items() if k != "errors"} | {"errors": list(out.get("errors", {}).items())[:10]}, indent=2))
        return 0 if not out.get("errors") else 1
    if args.dataset == "posture-weak":
        from .data import posture_weak as PW

        PW.fetch_captions(root)
        if not (man / "posture_weak_images.csv").exists():
            print(json.dumps(PW.build(root, man), indent=2))
        out = PW.fetch_images(root, man)
        print(json.dumps({k: v for k, v in out.items() if k != "errors"} | {"errors": list(out.get("errors", {}).items())[:10]}, indent=2))
        return 0 if not out.get("errors") else 1
    if args.dataset == "dog-pose":
        from .data import dog_pose

        print(json.dumps(dog_pose.fetch(root), indent=2))
        return 0
    if args.dataset == "catmeows":
        from .data import catmeows

        out = catmeows.fetch(root)
        print(json.dumps(out, indent=2))
        return 0 if out.get("status") == "ok" else 2
    if args.dataset == "barkopedia":
        from .data import barkopedia

        if not args.audit_only:
            print(json.dumps({"status": "disabled", "reason": "media ingestion is disabled pending provenance/grouping review; use --audit-only"}))
            return 2
        out = barkopedia.audit(root)
        print(json.dumps(out, indent=2))
        return 0 if out.get("status") != "blocked" else 2
    return 2


def _run_experiment(name: str, args: argparse.Namespace) -> dict:
    s = _settings(args)
    out_dir = s.paths.resolve("results_root") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if name == "e1":
        from .eval import e1_audio_events as E

        return E.run(s, out_dir)
    if name == "e2":
        from .eval import e2_cat_context as E

        return E.run(s, out_dir)
    if name == "e3":
        from .eval import e3_dog_context as E

        return E.run(s, out_dir)
    if name == "e4":
        from .eval import e4_visual as E

        dets = [s.vision.detector] + ([] if args.no_optional else ["efficientdet_lite2_mediapipe_v1"])
        return E.run(s, out_dir, dets)
    if name == "e5":
        from .eval import e5_fusion as E

        return E.run(s, out_dir)
    if name == "e6":
        from .eval import e6_runtime as E

        return E.run(s, out_dir)
    if name == "e7":
        from .eval import e7_fp_rejection as E

        return E.run(s, out_dir, params_out=REPO_ROOT / "artifacts" / "verifier")
    if name == "e8":
        from .eval import e8_behavior_sequences as E

        return E.run(s, out_dir)
    if name == "e10":
        from .eval import e10_posture as E

        return E.run(s, out_dir, artifact_dir=REPO_ROOT / "artifacts" / "posture")
    if name == "e9":
        from .eval import e9_dog_keypoints as E

        return E.run(s, out_dir, artifact_dir=REPO_ROOT / "artifacts" / "keypoints")
    raise ValueError(name)


def cmd_evaluate(args: argparse.Namespace) -> int:
    names = EXPERIMENTS if args.experiment == "all" else [args.experiment]
    for n in names:
        r = _run_experiment(n, args)
        print(json.dumps({"experiment": n, "status": r.get("status", "completed"), "results_dir": f"results/{n}"}))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    # Training produces the deployment artifacts of an experiment's protocol.
    if args.experiment not in ("e1", "e2"):
        print(json.dumps({"error": "trainable experiments: e1 (thresholds + MFCC baseline), e2 (CatMeows context)"}))
        return 2
    r = _run_experiment(args.experiment, args)
    print(json.dumps({"experiment": args.experiment, "status": r.get("status", "completed"), "artifacts_dir": "artifacts/"}))
    return 0 if r.get("status") != "blocked" else 2


def cmd_analyze(args: argparse.Namespace) -> int:
    from .context.zones import ZoneConfig
    from .explain.report import render_markdown
    from .ingest.media import MediaError
    from .pipeline import AnalysisEngine
    from .schemas.analysis import AnalysisError

    overrides = {"fusion": {"method": args.fusion_method}} if args.fusion_method else None
    s = load_settings(args.config, overrides)
    zones = ZoneConfig.model_validate(json.loads(Path(args.zones).read_text())) if args.zones else None
    mods = tuple(m.strip() for m in args.modalities.split(","))
    eng = AnalysisEngine(s).load()
    try:
        res = eng.analyze(args.media, zones=zones, modalities=mods)
    except MediaError as exc:
        err = AnalysisError(error_code=exc.code, message=str(exc))
        print(err.model_dump_json(indent=2))
        return 3
    js = res.model_dump_json(indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(js + "\n", encoding="utf-8")
    else:
        print(js)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(render_markdown(res), encoding="utf-8")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import ipaddress

    import uvicorn

    from .app.main import create_app

    s = _settings(args)
    host = args.host or s.serve.host
    port = args.port or s.serve.port
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback and not s.app.auth.cookie_secure:
        print(f"refusing to bind {host}: non-loopback serving requires app.auth.cookie_secure=true and an HTTPS "
              "reverse proxy in front (see docs/DEPLOYMENT.md).", file=sys.stderr)
        return 2
    if args.workers is not None:
        s.app.jobs.workers = args.workers
    uvicorn.run(create_app(s), host=host, port=port, workers=1, log_config=None, proxy_headers=s.serve.trusted_proxy_hops > 0,
                forwarded_allow_ips="*" if s.serve.trusted_proxy_hops > 0 else None)
    return 0


def _store(args: argparse.Namespace):
    from .app.db import Database
    from .app.store import Store

    s = _settings(args)
    db = Database(s.app.storage.resolve(s.app.storage.database_path))
    db.migrate()
    return s, db, Store(s, db)


def cmd_worker(args: argparse.Namespace) -> int:
    """Run job workers in a separate process (same database as the API)."""
    import signal
    import threading

    from .app.logs import configure
    from .app.main import _engine_factory
    from .app.worker import JobRunner

    s, db, store = _store(args)
    configure(s.app.log_format, s.app.log_level)
    runner = JobRunner(s, db, store, _engine_factory(s))
    if args.once:
        print(json.dumps({"processed": runner.run_pending()}))
        return 0
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    th = runner.start()
    stop.wait()
    runner.stop()
    th.join(timeout=30)
    return 0


def cmd_db(args: argparse.Namespace) -> int:
    s, db, _ = _store(args)
    if args.action == "backup":
        import sqlite3

        if not args.out:
            print(json.dumps({"error": "--out is required for backup"}))
            return 2
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        src = sqlite3.connect(str(db.path))
        dst = sqlite3.connect(str(out))
        with dst:
            src.backup(dst)  # online, consistent copy (WAL-safe)
        src.close()
        dst.close()
        print(json.dumps({"database": str(db.path), "backup": str(out), "bytes": out.stat().st_size, "schema_version": db.schema_version()}))
        return 0
    print(json.dumps({"database": str(db.path), "schema_version": db.schema_version()}))
    return 0


def cmd_release_check(args: argparse.Namespace) -> int:
    from .registry import check_model, release_report, resource

    s = _settings(args)
    rep = release_report(s)
    for d in rep["decisions"]:
        if resource(d["resource_id"]).get("kind") == "model":
            r = check_model(s, d["resource_id"])
            d["files"] = r.status
    print(json.dumps(rep, indent=2))
    return 0 if rep["all_allowed"] else 3


def cmd_user(args: argparse.Namespace) -> int:
    import getpass

    s, db, store = _store(args)
    if args.action == "list":
        print(json.dumps(store.list_users(), indent=2))
        return 0
    if not args.username:
        print("--username is required", file=sys.stderr)
        return 2
    if args.action in ("create", "reset-password"):
        pw = os.environ.get("PETPERCEPTION_NEW_PASSWORD") or getpass.getpass("New password: ")
        if args.action == "create":
            u = store.create_user(args.username, pw, is_admin=args.admin)
            print(json.dumps({"created": u["username"], "is_admin": bool(u["is_admin"])}))
        else:
            r = db.one("SELECT id FROM users WHERE username = ?", (args.username,))
            if r is None:
                print("unknown user", file=sys.stderr)
                return 1
            store.set_password(r["id"], pw)
            print(json.dumps({"password_reset": args.username, "sessions_revoked": True}))
        return 0
    if args.action in ("disable", "enable"):
        store.set_disabled(args.username, args.action == "disable")
        print(json.dumps({args.action + "d": args.username}))
        return 0
    return 2


def cmd_cleanup(args: argparse.Namespace) -> int:
    from .app.worker import JobRunner

    s, db, store = _store(args)
    recovered = JobRunner(s, db, store, lambda: None, worker_id="cleanup-cli").recover_expired()
    print(json.dumps({"recovered_jobs": recovered, **store.cleanup()}))
    return 0


def cmd_export_openapi(args: argparse.Namespace) -> int:
    import tempfile

    from .app.main import create_app
    from .schemas.analysis import AnalysisResult

    s = _settings(args)
    with tempfile.TemporaryDirectory() as td:
        s.app.storage.database_path = str(Path(td) / "openapi.sqlite3")
        s.app.storage.media_root = str(Path(td) / "media")
        app = create_app(s, engine_factory=lambda: None, start_workers=False)  # schema generation needs no models
        doc = app.openapi()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    Path(args.schema_out).write_text(json.dumps(AnalysisResult.model_json_schema(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"openapi": str(out), "analysis_result_schema": args.schema_out}))
    return 0


def cmd_make_examples(args: argparse.Namespace) -> int:
    from .eval import scenarios
    from .explain.report import render_markdown
    from .pipeline import AnalysisEngine

    s = _settings(args)
    media_dir = s.paths.resolve("data_root").parent / "examples"
    scs = {x.scenario_id: x for x in scenarios.build(s, media_dir)}
    eng = AnalysisEngine(s).load()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for sid in args.scenarios.split(","):
        sc = scs[sid]
        res = eng.analyze(sc.media, zones=sc.zones, modalities=sc.modalities, analysis_id=f"example-{sid}")
        (out / f"{sid}.result.json").write_text(res.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (out / f"{sid}.report.md").write_text(render_markdown(res), encoding="utf-8")
        if sc.zones:
            (out / f"{sid}.zones.json").write_text(sc.zones.model_dump_json(by_alias=True, indent=2) + "\n", encoding="utf-8")
        index.append({"scenario_id": sid, "description": sc.description, "synthetic_pairing": True, "sources": sc.sources,
                      "decision": res.decision.state, "reason_codes": res.decision.reason_codes})
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="petperception", description="Pet perception research engine (Phase 1)")
    p.add_argument("--config", default=None, help="YAML config (default: configs/cpu.yaml or $PETPERCEPTION_CONFIG)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", help="inspect environment, models and datasets")
    d.add_argument("--network", action="store_true", help="also probe reachability of data/model hosts")
    d.set_defaults(func=cmd_doctor)
    fm = sub.add_parser("fetch-models", help="explicitly download pinned model files")
    fm.add_argument("--only", nargs="*", help="model IDs from manifests/resources.json")
    fm.add_argument("--include-optional", action="store_true")
    fm.set_defaults(func=cmd_fetch_models)
    fd = sub.add_parser("fetch-data", help="explicitly download a dataset")
    fd.add_argument("dataset", choices=DATASETS)
    fd.add_argument("--audit-only", action="store_true", help="barkopedia: fetch metadata only")
    fd.add_argument("--rebuild-subset", action="store_true", help="openimages: reselect the subset (changes the benchmark)")
    fd.set_defaults(func=cmd_fetch_data)
    tr = sub.add_parser("train", help="fit an experiment's deployment artifacts")
    tr.add_argument("--experiment", required=True, choices=["e1", "e2"])
    tr.add_argument("--no-optional", action="store_true")
    tr.set_defaults(func=cmd_train)
    ev = sub.add_parser("evaluate", help="run an experiment protocol and write results/<exp>/")
    ev.add_argument("--experiment", required=True, choices=EXPERIMENTS + ["all"])
    ev.add_argument("--no-optional", action="store_true", help="E4: skip the optional EfficientDet-Lite2 comparison")
    ev.set_defaults(func=cmd_evaluate)
    an = sub.add_parser("analyze", help="analyze one local media file")
    an.add_argument("media")
    an.add_argument("--zones", help="zone configuration JSON (petperception.zones/1)")
    an.add_argument("--modalities", default="audio,video")
    an.add_argument("--fusion-method", choices=["rule_baseline", "weighted_average", "product_of_experts"])
    an.add_argument("--out", help="write JSON result here (default: stdout)")
    an.add_argument("--report", help="write a Markdown report here")
    an.set_defaults(func=cmd_analyze)
    sv = sub.add_parser("serve", help="run the web app + API (+ in-process workers); loopback by default")
    sv.add_argument("--host")
    sv.add_argument("--port", type=int)
    sv.add_argument("--workers", type=int, help="in-process job workers (0 = use a separate `worker` process)")
    sv.set_defaults(func=cmd_serve)
    wk = sub.add_parser("worker", help="run job workers as a separate process")
    wk.add_argument("--once", action="store_true", help="process due jobs and exit")
    wk.set_defaults(func=cmd_worker)
    dbp = sub.add_parser("db", help="apply database migrations and print the schema version")
    dbp.add_argument("action", choices=["migrate", "backup"])
    dbp.add_argument("--out", help="backup destination file (action backup)")
    dbp.set_defaults(func=cmd_db)
    us = sub.add_parser("user", help="manage accounts (password via prompt or $PETPERCEPTION_NEW_PASSWORD)")
    us.add_argument("action", choices=["create", "list", "disable", "enable", "reset-password"])
    us.add_argument("--username")
    us.add_argument("--admin", action="store_true")
    us.set_defaults(func=cmd_user)
    rc = sub.add_parser("release-check", help="check every configured asset against the release profile (exit 3 if any is not allowed)")
    rc.set_defaults(func=cmd_release_check)
    cl = sub.add_parser("cleanup", help="recover expired jobs, expire uploads/media/sessions, remove orphan files")
    cl.set_defaults(func=cmd_cleanup)
    oa = sub.add_parser("export-openapi", help="write the OpenAPI document and the result JSON schema")
    oa.add_argument("--out", default=str(REPO_ROOT / "docs" / "openapi.json"))
    oa.add_argument("--schema-out", default=str(REPO_ROOT / "docs" / "analysis_result.schema.json"))
    oa.set_defaults(func=cmd_export_openapi)
    mx = sub.add_parser("make-examples", help="regenerate synthetic-pairing example media and analyses")
    mx.add_argument("--out-dir", default=str(REPO_ROOT / "examples" / "offline"))
    mx.add_argument("--scenarios", default="s01_food_zone_meow,s02_food_zone_silent,s03_two_animals_bark,s04_no_pet_vacuum,s06_rest_zone_growl")
    mx.set_defaults(func=cmd_make_examples)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
