-- 0004: behaviour questions on an analysis (local or cloud-assisted understanding, E35) and research samples that the
-- owner explicitly donated, with annotations kept apart by origin: model output is provisional, owner feedback is
-- owner-reported, and only a reviewer's label is human-confirmed. Files live under storage.research_root, never in Git.

CREATE TABLE analysis_questions (
    analysis_id TEXT PRIMARY KEY REFERENCES analyses(id) ON DELETE CASCADE,
    species TEXT NOT NULL CHECK (species IN ('cat', 'dog')),
    scene TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL,
    understanding TEXT NOT NULL CHECK (understanding IN ('local', 'cloud')),
    strategy TEXT NOT NULL,
    retain_for_research INTEGER NOT NULL DEFAULT 0,
    research_consent_version TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'answered', 'failed')),
    answer_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE research_samples (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    analysis_id TEXT REFERENCES analyses(id) ON DELETE SET NULL,
    media_path TEXT NOT NULL,
    media_sha256 TEXT NOT NULL,
    species TEXT NOT NULL,
    scene TEXT NOT NULL,
    question TEXT NOT NULL,
    consent_version TEXT NOT NULL,
    consent_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX research_samples_user ON research_samples(user_id);

CREATE TABLE sample_annotations (
    id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL REFERENCES research_samples(id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK (source IN ('model', 'owner', 'reviewer')),
    status TEXT NOT NULL CHECK (status IN ('provisional', 'owner_reported', 'human_confirmed')),
    author_id TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK ((source = 'model' AND status = 'provisional') OR (source = 'owner' AND status = 'owner_reported')
           OR (source = 'reviewer' AND status = 'human_confirmed'))
);
CREATE INDEX sample_annotations_sample ON sample_annotations(sample_id);
