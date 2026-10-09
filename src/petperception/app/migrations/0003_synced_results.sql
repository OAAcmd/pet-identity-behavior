-- On-device analysis results the user chose to back up (opt-in). Result JSON and
-- feedback only: recordings never leave the device.
CREATE TABLE synced_results (
    id TEXT NOT NULL,                 -- the on-device analysis id ("dev...")
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,         -- when the analysis was made on the device
    synced_at TEXT NOT NULL,
    filename TEXT NOT NULL,
    bundle_version TEXT,
    decision_state TEXT,
    result_json TEXT NOT NULL,
    feedback_json TEXT NOT NULL DEFAULT '[]',
    bytes INTEGER NOT NULL,
    PRIMARY KEY (user_id, id)
);
CREATE INDEX synced_results_user ON synced_results(user_id, created_at);
