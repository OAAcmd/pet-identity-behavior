-- Device-check reports (sent only when the user taps "Send report"; no media).
CREATE TABLE device_reports (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    bundle_version TEXT,
    passed INTEGER,
    user_agent TEXT,
    report_json TEXT NOT NULL
);
CREATE INDEX device_reports_user ON device_reports(user_id, created_at);
