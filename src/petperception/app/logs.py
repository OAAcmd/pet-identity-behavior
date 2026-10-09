"""Structured JSON logging with request IDs.

Logs carry IDs, codes, durations and pseudonymous user references only.
They never contain media content, file names, zone geometry, passwords,
session tokens, or feedback text.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import sys
import time

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            out["request_id"] = rid
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            out["exc"] = self.formatException(record.exc_info)[-2000:]
        return json.dumps(out, default=str)


def configure(fmt: str = "json", level: str = "INFO") -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_petperception", False):
            root.removeHandler(h)
    h = logging.StreamHandler(sys.stderr)
    h._petperception = True  # type: ignore[attr-defined]
    h.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(h)
    root.setLevel(level)
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def user_ref(user_id: str | None) -> str | None:
    """Stable pseudonymous reference for logs (not reversible to the account name)."""
    return hashlib.sha256(user_id.encode()).hexdigest()[:12] if user_id else None
