"""Persistent background job processing.

State machine (table ``analyses``)::

    queued --claim--> running --ok--> succeeded
       ^                 |--input error--> failed (not retried)
       |                 |--transient error, attempts < max--> queued (backoff)
       |                 |--transient error, attempts = max--> failed
       |                 |--cancel requested--> cancelled
       |                 '--lease expired (worker died)--> queued / failed(worker_lost)

* Claims use ``BEGIN IMMEDIATE`` so several worker threads or processes can
  share one database safely; a claim sets a lease that the worker extends
  while it makes progress. ``recover_expired`` requeues jobs whose lease ran
  out (crash, kill -9, power loss) and counts that as an attempt.
* Cancellation and the processing timeout are cooperative: the engine polls
  ``should_cancel`` between stages and for every sampled frame.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ..config import Settings
from ..context.zones import ZoneConfig
from ..ingest.media import MediaError
from ..pipeline import AnalysisCancelled
from ..schemas.analysis import AnalysisResult
from .db import Database, now_iso
from .errors import MEDIA_ERROR_TEXT
from .logs import user_ref
from .store import RETRYABLE, Store

log = logging.getLogger("petperception.worker")


class _Timeout(Exception):
    pass


def summarize(res: AnalysisResult) -> dict:
    o = res.observations
    return {
        "tracks": [{"track_id": t.track_id, "species": t.species, "first_seen_s": t.first_seen_s, "last_seen_s": t.last_seen_s} for t in o.animal_tracks],
        "sound_events": [{"label": e.label, "t_start_s": e.t_start_s, "t_end_s": e.t_end_s} for e in o.sound_events],
        "hypotheses": [{"name": h.name, "status": h.status} for h in res.hypotheses],
        "decision": {"state": res.decision.state, "interpretation": res.decision.overall_interpretation, "summary": res.decision.summary},
        "quality_flags": sorted({q.code for q in res.quality_flags}),
        "modalities": {"audio": res.modalities.audio_state, "video": res.modalities.video_state},
    }


class JobRunner:
    def __init__(self, settings: Settings, db: Database, store: Store, engine_factory: Callable[[], object], worker_id: str | None = None,
                 explainer_factory: Callable[[int], object] | None = None):
        self.s = settings
        self.db = db
        self.store = store
        self.engine_factory = engine_factory
        self.explainer_factory = explainer_factory
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}"
        self._stop = threading.Event()
        self._engine = None
        self._explainers: dict[int, object] = {}

    def explainer(self, version: int):
        """Behaviour-explanation system: version 1 (default, local answers), version 2 (evidence for cloud packets)."""
        if version not in self._explainers:
            if self.explainer_factory is not None:
                self._explainers[version] = self.explainer_factory(version)
            else:
                from ..understand.run import Explainer

                self._explainers[version] = Explainer.load(self.s.runtime.num_threads, version=version)
        return self._explainers[version]

    def answer_question(self, analysis_id: str, media_path: str) -> None:
        """Answer the owner's behaviour question (local or cloud-assisted) and donate the recording if the owner opted in."""
        q = self.store.question(analysis_id)
        if q is None:
            return
        from ..understand import cloud as CL

        try:
            if q["understanding"] == "cloud":
                out = CL.understand(Path(media_path), q["species"], q["scene"], q["question"], self.explainer(2), q["strategy"], self.s.cloud)
            else:
                ans = self.explainer(1).explain(Path(media_path), q["species"], q["scene"], q["question"]).to_dict()
                out = {"strategy": "local", "answer_source": "local", "local_answer": ans, "cloud": {"status": "not_requested"},
                       "species": q["species"], "scene": q["scene"], "question": q["question"]}
            self.store.store_answer(analysis_id, "answered", out)
        except Exception:  # noqa: BLE001 - the analysis result stays valid when the question step fails
            log.exception("question_error", extra={"analysis_id": analysis_id})
            self.store.store_answer(analysis_id, "failed", {"error": "the behaviour question could not be answered"})
            out = None
        if q["retain_for_research"]:
            ann = None
            if out is not None:
                c = out.get("cloud") or {}
                ann = {"origin": "cloud model" if out["answer_source"] == "cloud" else "local recognizer", "status_note": "provisional; not a label",
                       "observed": (c.get("answer") or {}).get("observed") if out["answer_source"] == "cloud"
                       else [{k: b[k] for k in ("behavior", "score", "bouts_s")} for b in out["local_answer"]["behaviors_seen"]],
                       "model": c.get("model"), "prompt_version": c.get("prompt_version")}
            self.store.donate_sample(analysis_id, ann)

    # ---------------------------------------------------------------- queue ops
    def recover_expired(self) -> int:
        now = time.time()
        n = 0
        with self.db.tx() as c:
            rows = c.execute("SELECT id, attempts, max_attempts FROM analyses WHERE status = 'running' AND lease_expires_at < ?", (now,)).fetchall()
            for r in rows:
                n += 1
                if r["attempts"] < r["max_attempts"]:
                    c.execute("UPDATE analyses SET status = 'queued', stage = 'queued', lease_owner = NULL, lease_expires_at = NULL, "
                              "next_attempt_at = ?, error_code = 'worker_lost', error_message = ?, error_action = NULL, updated_at = ? WHERE id = ?",
                              (now + self.s.app.jobs.retry_backoff_s, "previous attempt was interrupted; retrying", now_iso(), r["id"]))
                else:
                    msg, action = MEDIA_ERROR_TEXT["worker_lost"]
                    c.execute("UPDATE analyses SET status = 'failed', stage = 'failed', lease_owner = NULL, lease_expires_at = NULL, "
                              "error_code = 'worker_lost', error_message = ?, error_action = ?, finished_at = ?, updated_at = ? WHERE id = ?",
                              (msg, action, now_iso(), now_iso(), r["id"]))
        for r in rows:
            log.warning("job_lease_expired", extra={"analysis_id": r["id"], "attempts": r["attempts"]})
            self.store.apply_retention_after_terminal(r["id"])
        return n

    def claim(self) -> dict | None:
        now = time.time()
        with self.db.tx() as c:
            r = c.execute("SELECT id FROM analyses WHERE status = 'queued' AND next_attempt_at <= ? ORDER BY created_at LIMIT 1", (now,)).fetchone()
            if r is None:
                return None
            c.execute("UPDATE analyses SET status = 'running', stage = 'starting', attempts = attempts + 1, lease_owner = ?, lease_expires_at = ?, "
                      "started_at = coalesce(started_at, ?), updated_at = ? WHERE id = ?",
                      (self.worker_id, now + self.s.app.jobs.lease_s, now_iso(), now_iso(), r["id"]))
            job = dict(c.execute("SELECT * FROM analyses WHERE id = ?", (r["id"],)).fetchone())
        return job

    def heartbeat(self) -> None:
        now = time.time()
        self.db.execute("INSERT INTO worker_heartbeats (worker_id, started_at, last_seen) VALUES (?, ?, ?) "
                        "ON CONFLICT(worker_id) DO UPDATE SET last_seen = excluded.last_seen", (self.worker_id, now, now))

    def _finish(self, job_id: str, status: str, **cols) -> bool:
        sets = ", ".join(f"{k} = ?" for k in cols)
        params = list(cols.values())
        n = self.db.execute(f"UPDATE analyses SET status = ?, lease_owner = NULL, lease_expires_at = NULL, updated_at = ?{', ' + sets if sets else ''} "
                            "WHERE id = ? AND lease_owner = ?", (status, now_iso(), *params, job_id, self.worker_id))
        return n == 1

    # -------------------------------------------------------------- execution
    def engine(self):
        if self._engine is None:
            self._engine = self.engine_factory()
        return self._engine

    def run_one(self, job: dict) -> str:
        jid = job["id"]
        t0 = time.time()
        deadline = t0 + self.s.app.jobs.job_timeout_s
        last = {"t": 0.0, "cancel": False}

        def should_cancel() -> bool:
            now = time.time()
            if now > deadline:
                raise _Timeout()
            if now - last["t"] >= 1.0:
                r = self.db.one("SELECT cancel_requested FROM analyses WHERE id = ?", (jid,))
                last["cancel"] = r is None or bool(r["cancel_requested"])  # deleted rows also stop work
                last["t"] = now
            return last["cancel"]

        prog = {"t": 0.0}

        def progress(stage: str, frac: float) -> None:
            now = time.time()
            if now - prog["t"] >= 0.5 or frac >= 1.0:
                self.db.execute("UPDATE analyses SET stage = ?, progress = ?, lease_expires_at = ?, updated_at = ? WHERE id = ? AND lease_owner = ?",
                                (stage, round(frac, 3), now + self.s.app.jobs.lease_s, now_iso(), jid, self.worker_id))
                prog["t"] = now

        zones = ZoneConfig.model_validate(json.loads(job["zones_json"])) if job["zones_json"] else None
        outcome = "failed"
        try:
            if not job["media_path"] or not Path(job["media_path"]).exists():
                raise MediaError("not_found", "media file missing")
            res = self.engine().analyze(job["media_path"], zones=zones, modalities=tuple(job["modalities"].split(",")), analysis_id=jid,
                                        display_name=job["original_filename"], progress=progress, should_cancel=should_cancel)
            progress("question", 0.99)
            self.answer_question(jid, job["media_path"])  # before 'succeeded', so a finished analysis already carries its answer
            with self.db.tx() as c:
                owner = c.execute("SELECT lease_owner FROM analyses WHERE id = ?", (jid,)).fetchone()
                if owner is None or owner["lease_owner"] != self.worker_id:
                    outcome = "discarded"
                else:
                    c.execute("INSERT OR REPLACE INTO analysis_results (analysis_id, schema_version, result_json, created_at) VALUES (?, ?, ?, ?)",
                              (jid, res.schema_version, res.model_dump_json(), now_iso()))
                    c.execute("UPDATE analyses SET status = 'succeeded', stage = 'done', progress = 1, lease_owner = NULL, lease_expires_at = NULL, "
                              "error_code = NULL, error_message = NULL, error_action = NULL, decision_state = ?, summary_json = ?, finished_at = ?, updated_at = ? WHERE id = ?",
                              (res.decision.state, json.dumps(summarize(res)), now_iso(), now_iso(), jid))
                    outcome = "succeeded"
        except AnalysisCancelled:
            outcome = "cancelled" if self._finish(jid, "cancelled", stage="cancelled", finished_at=now_iso()) else "discarded"
        except _Timeout:
            msg, action = MEDIA_ERROR_TEXT["analysis_timeout"]
            self._finish(jid, "failed", stage="failed", error_code="analysis_timeout", error_message=msg, error_action=action, finished_at=now_iso())
        except MediaError as exc:
            msg, action = MEDIA_ERROR_TEXT.get(exc.code, (str(exc)[:200], None))
            self._finish(jid, "failed", stage="failed", error_code=exc.code, error_message=msg, error_action=action, finished_at=now_iso())
        except Exception:  # noqa: BLE001 - transient/unknown errors are retried with a bound
            log.exception("job_error", extra={"analysis_id": jid, "attempt": job["attempts"]})
            if job["attempts"] < job["max_attempts"]:
                backoff = self.s.app.jobs.retry_backoff_s * (2 ** (job["attempts"] - 1))
                self._finish(jid, "queued", stage="queued", next_attempt_at=time.time() + backoff, error_code="internal_error",
                             error_message="processing failed; will retry automatically", error_action=None)
                outcome = "requeued"
            else:
                msg, action = MEDIA_ERROR_TEXT["internal_error"]
                self._finish(jid, "failed", stage="failed", error_code="internal_error", error_message=msg, error_action=action, finished_at=now_iso())
        if outcome != "requeued":
            self.store.apply_retention_after_terminal(jid)
        log.info("job_finished", extra={"analysis_id": jid, "outcome": outcome, "attempt": job["attempts"],
                                        "duration_ms": round((time.time() - t0) * 1000), "user": user_ref(job["user_id"])})
        return outcome

    def run_pending(self, max_jobs: int | None = None) -> int:
        """Process queued jobs until none are due (used by tests and `worker --once`)."""
        n = 0
        self.recover_expired()
        while max_jobs is None or n < max_jobs:
            job = self.claim()
            if job is None:
                break
            self.run_one(job)
            n += 1
        return n

    # ------------------------------------------------------------------ loop
    def loop(self) -> None:
        last_cleanup = 0.0
        last_beat = 0.0
        while not self._stop.is_set():
            try:
                now = time.time()
                if now - last_beat >= self.s.app.jobs.heartbeat_s:
                    self.heartbeat()
                    last_beat = now
                if now - last_cleanup >= self.s.app.storage.cleanup_interval_s:
                    self.recover_expired()
                    stats = self.store.cleanup()
                    if any(stats.values()):
                        log.info("cleanup", extra=stats)
                    last_cleanup = now
                job = self.claim()
                if job is None:
                    self._stop.wait(self.s.app.jobs.poll_interval_s)
                    continue
                self.heartbeat()
                self.run_one(job)
            except Exception:  # noqa: BLE001 - the loop must survive unexpected DB/engine errors
                log.exception("worker_loop_error")
                self._stop.wait(2.0)

    def start(self) -> threading.Thread:
        th = threading.Thread(target=self.loop, name=f"worker-{self.worker_id}", daemon=True)
        th.start()
        return th

    def stop(self) -> None:
        self._stop.set()


def retryable(code: str | None) -> bool:
    return code in RETRYABLE
