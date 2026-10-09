"""Barkopedia dog activity/environment candidate (Hugging Face dataset card).

Status: DISABLED / PENDING REVIEW. The card advertises MIT, but the
underlying media provenance, the availability and format of each split,
and whether animal or source IDs exist were not verified (huggingface.co is
denied in this workspace). Until an audit file records these facts, the
importer only performs a metadata audit and never ingests media; E3 reports
``pending`` and makes no cross-dog claim. Activity labels, if ever used,
remain activity/context labels, not mental states.
"""

from __future__ import annotations

from pathlib import Path

from ..utils import DownloadError, download, read_json, utc_now, write_json

API = "https://huggingface.co/api/datasets/ArlingtonCL2/Barkopedia_Dog_Act_Env"
AUDIT_FIELDS = ["media_provenance_reviewed", "license_confirmed", "splits_inspected", "animal_or_source_ids_available", "test_labels_available"]


def dataset_dir(data_root: Path) -> Path:
    return data_root / "barkopedia"


def audit(data_root: Path) -> dict:
    """Fetch only repository metadata (file list, revision, card data) for review."""
    d = dataset_dir(data_root)
    d.mkdir(parents=True, exist_ok=True)
    try:
        download(API, d / "hf_dataset_info.json", timeout=60)
    except DownloadError as exc:
        st = {"status": "blocked", "error": str(exc), "checked_at": utc_now().isoformat(timespec="seconds")}
        write_json(d / "audit_status.json", st)
        return st
    info = read_json(d / "hf_dataset_info.json")
    st = {
        "status": "metadata_fetched_pending_review",
        "revision": info.get("sha"),
        "card_license": (info.get("cardData") or {}).get("license"),
        "files": [s.get("rfilename") for s in info.get("siblings", [])],
        "review_checklist": {k: None for k in AUDIT_FIELDS},
        "note": "Fill review_checklist in audit_review.json after manual inspection; media ingestion stays disabled until all are true.",
        "checked_at": utc_now().isoformat(timespec="seconds"),
    }
    write_json(d / "audit_status.json", st)
    return st


def ingestion_allowed(data_root: Path) -> tuple[bool, str]:
    review = dataset_dir(data_root) / "audit_review.json"
    if not review.exists():
        return False, "no audit_review.json (provenance, splits and grouping unreviewed)"
    r = read_json(review)
    missing = [k for k in AUDIT_FIELDS if r.get(k) is not True]
    return (not missing), ("ok" if not missing else f"review incomplete: {missing}")
