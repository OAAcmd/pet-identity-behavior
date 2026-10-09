-- 0001: initial product schema (users, sessions, pets, places, uploads,
-- analyses/jobs, results, feedback, audit, worker heartbeats).
-- All user-owned rows carry user_id and cascade on account deletion.

CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0,
    disabled INTEGER NOT NULL DEFAULT 0,
    retain_media_days INTEGER NOT NULL DEFAULT 0,
    feedback_research_consent INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    csrf_token TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    last_seen_at REAL NOT NULL
);
CREATE INDEX sessions_user ON sessions(user_id);

CREATE TABLE login_failures (
    key TEXT NOT NULL,
    at REAL NOT NULL
);
CREATE INDEX login_failures_key ON login_failures(key, at);

CREATE TABLE pets (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    species TEXT NOT NULL CHECK (species IN ('cat', 'dog')),
    birth_year INTEGER,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX pets_user ON pets(user_id);

CREATE TABLE places (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    zones_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX places_user ON places(user_id);

CREATE TABLE uploads (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    ext TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    received_bytes INTEGER NOT NULL DEFAULT 0,
    sha256_expected TEXT,
    status TEXT NOT NULL CHECK (status IN ('receiving', 'complete', 'consumed', 'aborted')),
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX uploads_user ON uploads(user_id, status);

CREATE TABLE analyses (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    place_id TEXT REFERENCES places(id) ON DELETE SET NULL,
    subject_mode TEXT NOT NULL CHECK (subject_mode IN ('unknown', 'single_known', 'multiple_known')),
    modalities TEXT NOT NULL,
    zones_json TEXT,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    stage TEXT,
    progress REAL NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_expires_at REAL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    error_code TEXT,
    error_message TEXT,
    error_action TEXT,
    media_path TEXT,
    media_ext TEXT,
    media_size INTEGER,
    media_sha256 TEXT,
    media_retained_until REAL,
    original_filename TEXT NOT NULL,
    duration_s REAL,
    decision_state TEXT,
    summary_json TEXT,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX analyses_user ON analyses(user_id, created_at);
CREATE INDEX analyses_queue ON analyses(status, next_attempt_at);

CREATE TABLE analysis_pets (
    analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    pet_id TEXT NOT NULL REFERENCES pets(id) ON DELETE CASCADE,
    PRIMARY KEY (analysis_id, pet_id)
);

CREATE TABLE analysis_results (
    analysis_id TEXT PRIMARY KEY REFERENCES analyses(id) ON DELETE CASCADE,
    schema_version TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE feedback (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    target_type TEXT NOT NULL CHECK (target_type IN ('overall', 'hypothesis', 'observation', 'sound_event', 'track')),
    target_id TEXT,
    verdict TEXT NOT NULL CHECK (verdict IN ('agree', 'disagree', 'unsure')),
    actual_behavior TEXT,
    note TEXT NOT NULL DEFAULT '',
    consent_research_use INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX feedback_analysis ON feedback(analysis_id);

CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    action TEXT NOT NULL,
    object_type TEXT,
    object_id TEXT,
    at TEXT NOT NULL
);

CREATE TABLE worker_heartbeats (
    worker_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    last_seen REAL NOT NULL
);
