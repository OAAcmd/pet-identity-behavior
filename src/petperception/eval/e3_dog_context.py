"""E3: dog activity/context - executes only with audited data and labels.

Barkopedia is not reachable and unaudited in this workspace (see
data/barkopedia.py), so E3 reports ``pending``. Identity-disjoint (cross-dog)
evaluation cannot be assessed until animal or source IDs are verified; no
dog-intent or dog-context model is claimed. Real dog *sound-event* detection is
covered by E1 instead.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..data import barkopedia as BK
from ..utils import utc_now, write_json


def run(settings: Settings, out_dir: Path) -> dict:
    allowed, why = BK.ingestion_allowed(settings.paths.resolve("data_root"))
    status = {
        "experiment": "E3",
        "status": "pending" if not allowed else "ready_to_implement_protocol",
        "reason": "Barkopedia metadata/media not reachable (huggingface.co denied) and not audited: " + why,
        "identity_disjoint_evaluation_possible": "unknown (animal/source IDs unverified)",
        "cross_dog_claims": "declined",
        "dog_sound_event_detection": "see E1 (ESC-50 dog vs other)",
        "next_command": "uv run petperception fetch-data barkopedia --audit-only  # then complete var/data/barkopedia/audit_review.json",
        "created_at": utc_now().isoformat(timespec="seconds"),
    }
    write_json(out_dir / "status.json", status)
    return status
