"""SQLite persistence with ordered SQL migrations.

* WAL journal, foreign keys on, busy timeout; one connection per thread.
* Migrations are ``migrations/NNNN_name.sql`` applied in order inside a
  transaction and recorded in ``schema_migrations`` with their SHA-256; an
  applied migration whose file later changes is reported as an error.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

MIGRATIONS_PACKAGE = "petperception.app.migrations"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class MigrationError(RuntimeError):
    pass


def available_migrations() -> list[tuple[str, str, str]]:
    """(version, name, sql) sorted by version."""
    out = []
    for entry in resources.files(MIGRATIONS_PACKAGE).iterdir():
        if entry.name.endswith(".sql"):
            version = entry.name.split("_", 1)[0]
            out.append((version, entry.name, entry.read_text(encoding="utf-8")))
    return sorted(out)


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        self._local = threading.local()

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=True)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    @contextmanager
    def tx(self, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        """Transaction; ``BEGIN IMMEDIATE`` takes the write lock up front."""
        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        return self.connect().execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        return self.connect().execute(sql, params).fetchone()

    def execute(self, sql: str, params: tuple | dict = ()) -> int:
        cur = self.connect().execute(sql, params)
        return cur.rowcount

    # -- migrations ------------------------------------------------------------
    def migrate(self) -> list[str]:
        conn = self.connect()
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, name TEXT NOT NULL, "
                     "sha256 TEXT NOT NULL, applied_at TEXT NOT NULL)")
        applied = {r["version"]: r for r in conn.execute("SELECT * FROM schema_migrations").fetchall()}
        done = []
        for version, name, sql in available_migrations():
            digest = hashlib.sha256(sql.encode()).hexdigest()
            if version in applied:
                if applied[version]["sha256"] != digest:
                    raise MigrationError(f"migration {name} changed after being applied")
                continue
            conn.execute("BEGIN IMMEDIATE")
            try:
                for stmt in _split_sql(sql):
                    conn.execute(stmt)
                conn.execute("INSERT INTO schema_migrations VALUES (?, ?, ?, ?)", (version, name, digest, now_iso()))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            done.append(name)
        return done

    def schema_version(self) -> str | None:
        try:
            r = self.one("SELECT max(version) AS v FROM schema_migrations")
            return r["v"] if r else None
        except sqlite3.OperationalError:
            return None


def _split_sql(sql: str) -> list[str]:
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def row(r: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(r) if r is not None else None
