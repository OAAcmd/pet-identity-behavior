import { useEffect, useState } from "preact/hooks";
import { ConfirmDialog, ErrorBanner, Field, InfoBanner } from "../components/ui";
import { api } from "../lib/api";
import type { Me } from "../lib/types";
import { useApp } from "../state";

export function SettingsPage() {
  const { me, setMe, caps, toast } = useApp();
  const [days, setDays] = useState(String(me?.retain_media_days ?? 0));
  const [consent, setConsent] = useState(me?.feedback_research_consent ?? false);
  const [pw, setPw] = useState({ current: "", next: "" });
  const [delPw, setDelPw] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [newUser, setNewUser] = useState({ username: "", password: "" });
  const [users, setUsers] = useState<{ id: string; username: string; is_admin: number; disabled: number }[]>([]);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    if (me?.is_admin) api.get<typeof users>("/admin/users").then(setUsers).catch(() => undefined);
  }, [me]);

  const run = async (fn: () => Promise<unknown>, msg: string) => {
    setError(null);
    try {
      await fn();
      toast(msg);
    } catch (e) {
      setError(e);
    }
  };

  const maxDays = caps?.retention.max_retain_media_days ?? 30;
  return (
    <section>
      <h1 tabIndex={-1}>Settings</h1>
      <ErrorBanner error={error} />

      <form class="card" onSubmit={(e) => { e.preventDefault(); void run(async () => setMe(await api.patch<Me>("/settings", { retain_media_days: Number(days), feedback_research_consent: consent })), "Settings saved."); }}>
        <h2>Privacy</h2>
        <Field label="Keep recordings on the server" id="retain" hint="0 = delete each recording as soon as it has been analyzed (default). Keeping recordings lets you replay them next to results.">
          <select id="retain" value={days} onChange={(e) => setDays((e.target as HTMLSelectElement).value)}>
            <option value="0">Don't keep (delete after analysis)</option>
            {[1, 7, 14, 30].filter((d) => d <= maxDays).map((d) => <option key={d} value={String(d)}>{d} day(s)</option>)}
          </select>
        </Field>
        <label class="check"><input type="checkbox" checked={consent} onChange={() => setConsent(!consent)} /> By default, allow my feedback to be used for evaluating the system</label>
        <button type="submit" class="btn btn-primary">Save</button>
      </form>

      <div class="card">
        <h2>Your data</h2>
        <p>Download everything stored for your account (pets, places, results and feedback) as JSON.</p>
        <a class="btn btn-secondary" href="/api/v1/export" download="petperception-export.json">Download my data</a>
      </div>

      <form class="card" onSubmit={(e) => { e.preventDefault(); void run(() => api.post("/auth/password", { current_password: pw.current, new_password: pw.next }), "Password changed; other sessions signed out."); }}>
        <h2>Password</h2>
        <Field label="Current password" id="pw-cur"><input id="pw-cur" type="password" autoComplete="current-password" required value={pw.current} onInput={(e) => setPw({ ...pw, current: (e.target as HTMLInputElement).value })} /></Field>
        <Field label="New password" id="pw-new" hint="At least 10 characters."><input id="pw-new" type="password" autoComplete="new-password" required minLength={10} value={pw.next} onInput={(e) => setPw({ ...pw, next: (e.target as HTMLInputElement).value })} /></Field>
        <button type="submit" class="btn btn-secondary">Change password</button>
      </form>

      {me?.is_admin && (
        <form class="card" onSubmit={(e) => { e.preventDefault(); void run(async () => { await api.post("/admin/users", newUser); setUsers(await api.get("/admin/users")); setNewUser({ username: "", password: "" }); }, "Account created."); }}>
          <h2>Accounts (administrator)</h2>
          <ul class="list">{users.map((u) => <li key={u.id} class="list-item">{u.username}{u.is_admin ? " (admin)" : ""}{u.disabled ? " - disabled" : ""}</li>)}</ul>
          <Field label="New username" id="nu-name"><input id="nu-name" required minLength={3} maxLength={32} value={newUser.username} onInput={(e) => setNewUser({ ...newUser, username: (e.target as HTMLInputElement).value })} /></Field>
          <Field label="Initial password" id="nu-pw"><input id="nu-pw" type="password" autoComplete="new-password" required minLength={10} value={newUser.password} onInput={(e) => setNewUser({ ...newUser, password: (e.target as HTMLInputElement).value })} /></Field>
          <button type="submit" class="btn btn-secondary">Create account</button>
        </form>
      )}

      <div class="card">
        <h2>Session</h2>
        <button type="button" class="btn btn-secondary" onClick={() => run(async () => { await api.post("/auth/logout"); setMe(null); }, "Signed out.")}>Sign out</button>
      </div>

      <div class="card danger-zone">
        <h2>Delete account</h2>
        <InfoBanner tone="warn">Deletes your account, pets, places, all analyses, feedback and stored recordings. This cannot be undone.</InfoBanner>
        <Field label="Password" id="del-pw"><input id="del-pw" type="password" autoComplete="current-password" value={delPw} onInput={(e) => setDelPw((e.target as HTMLInputElement).value)} /></Field>
        <button type="button" class="btn btn-danger" disabled={!delPw} onClick={() => setConfirmDelete(true)}>Delete my account</button>
      </div>
      <ConfirmDialog open={confirmDelete} title="Delete your account permanently?" danger confirmLabel="Delete everything" body={<p>All your data on this server will be removed.</p>}
        onCancel={() => setConfirmDelete(false)}
        onConfirm={() => { setConfirmDelete(false); void run(async () => { await api.post("/auth/delete-account", { password: delPw, confirm: "DELETE" }); setMe(null); }, "Account deleted."); }} />
      <p class="muted"><a href="#/about">About this app and its limits</a> - <a href="#/places">Places</a></p>
    </section>
  );
}
