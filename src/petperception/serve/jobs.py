"""Single-worker in-process job store with private temporary storage.

* Each job gets a private (0700) directory under ``paths.jobs_root``.
* The uploaded media file is deleted as soon as its analysis finishes; only the
  JSON result is kept until it is deleted or its TTL expires.
* ``delete`` is deterministic: a queued/finished job's directory is removed
  before the call returns; a running job is marked and removed by the worker
  immediately after the analysis returns (the API reports this as
  ``deletion_scheduled``), and it is never readable again either way.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..context.zones import ZoneConfig
from ..ingest.media import MediaError
from ..schemas.analysis import AnalysisError, AnalysisResult
from ..utils import private_tempdir, utc_now

log = logging.getLogger("petperception.jobs")


@dataclass
class Job:
    analysis_id: str
    workdir: Path
    media_path: Path
    display_name: str
    zones: ZoneConfig | None
    modalities: tuple[str, ...]
    status: str = "queued"  # queued | running | succeeded | failed
    created_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: utc_now().isoformat(timespec="seconds"))
    created_mono: float = field(default_factory=time.monotonic)
    result: AnalysisResult | None = None
    error: AnalysisError | None = None
    delete_requested: bool = False


class JobStore:
    def __init__(self, root: Path, analyze: Callable[..., AnalysisResult], ttl_s: float, max_jobs: int):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.analyze = analyze
        self.ttl_s = ttl_s
        self.max_jobs = max_jobs
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.q: queue.Queue[str | None] = queue.Queue()
        self.worker = threading.Thread(target=self._run, name="analysis-worker", daemon=True)
        self.worker.start()

    # -- lifecycle -----------------------------------------------------------
    def new_workdir(self) -> tuple[str, Path]:
        aid = str(uuid.uuid4())
        return aid, private_tempdir(self.root, prefix=f"{aid}-")

    def submit(self, aid: str, workdir: Path, media: Path, display_name: str, zones: ZoneConfig | None, modalities: tuple[str, ...]) -> Job:
        self.expire()
        with self.lock:
            if len(self.jobs) >= self.max_jobs:
                shutil.rmtree(workdir, ignore_errors=True)
                raise OverflowError("job store is full; delete finished analyses first")
            job = Job(aid, workdir, media, display_name, zones, modalities)
            self.jobs[aid] = job
        self.q.put(aid)
        return job

    def get(self, aid: str) -> Job | None:
        self.expire()
        with self.lock:
            j = self.jobs.get(aid)
            return None if j is None or j.delete_requested else j

    def delete(self, aid: str) -> str:
        with self.lock:
            j = self.jobs.get(aid)
            if j is None or j.delete_requested:
                return "not_found"
            if j.status == "running":
                j.delete_requested = True
                return "deletion_scheduled"
            self._purge(j)
            return "deleted"

    def _purge(self, j: Job) -> None:
        shutil.rmtree(j.workdir, ignore_errors=True)
        self.jobs.pop(j.analysis_id, None)

    def expire(self) -> None:
        now = time.monotonic()
        with self.lock:
            for j in list(self.jobs.values()):
                if j.status in ("succeeded", "failed") and now - j.created_mono > self.ttl_s:
                    self._purge(j)

    def shutdown(self) -> None:
        self.q.put(None)
        with self.lock:
            for j in list(self.jobs.values()):
                if j.status != "running":
                    self._purge(j)

    # -- worker ----------------------------------------------------------------
    def _run(self) -> None:
        while True:
            aid = self.q.get()
            if aid is None:
                return
            with self.lock:
                j = self.jobs.get(aid)
                if j is None or j.delete_requested:
                    continue
                j.status = "running"
                j.updated_at = utc_now().isoformat(timespec="seconds")
            result, error = None, None
            try:
                result = self.analyze(j.media_path, zones=j.zones, modalities=j.modalities, analysis_id=aid, display_name=j.display_name)
            except MediaError as exc:
                error = AnalysisError(analysis_id=aid, error_code=exc.code, message=str(exc))
            except Exception as exc:  # noqa: BLE001 - reported to the client as a failed job
                log.exception("analysis %s failed", aid)
                error = AnalysisError(analysis_id=aid, error_code="internal_error", message=type(exc).__name__)
            finally:
                try:
                    j.media_path.unlink(missing_ok=True)  # never retain uploaded media after analysis
                except OSError:
                    pass
            with self.lock:
                j.result, j.error = result, error
                j.status = "succeeded" if result is not None else "failed"
                j.updated_at = utc_now().isoformat(timespec="seconds")
                if j.delete_requested:
                    self._purge(j)
