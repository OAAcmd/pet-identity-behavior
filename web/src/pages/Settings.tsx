import { useEffect, useState } from "preact/hooks";
import { ConfirmDialog, ErrorBanner, Field, InfoBanner, ProgressBar } from "../components/ui";
import { api } from "../lib/api";
import { fmtBytes, fmtDate } from "../lib/format";
import { localAnalyses } from "../lib/localstore";
import { serverAnalysisEnabled, setServerAnalysis, useOnDevice, WARMUP } from "../lib/ondevice";
import { listSynced, restoreMissing, setSyncEnabled, syncEnabled, syncPending } from "../lib/sync";
import type { Me } from "../lib/types";
import { offlineCache, useApp } from "../state";

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
      <OnDeviceCard />
      <BackupCard />

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

      <ResearchSamples />
      <AndroidServer />
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
        <button type="button" class="btn btn-secondary" onClick={() => run(async () => { await api.post("/auth/logout"); offlineCache.clear(); setMe(null); }, "Signed out.")}>Sign out</button>
      </div>

      <div class="card danger-zone">
        <h2>Delete account</h2>
        <InfoBanner tone="warn">Deletes your account, pets, places, all analyses, feedback and stored recordings. This cannot be undone.</InfoBanner>
        <Field label="Password" id="del-pw"><input id="del-pw" type="password" autoComplete="current-password" value={delPw} onInput={(e) => setDelPw((e.target as HTMLInputElement).value)} /></Field>
        <button type="button" class="btn btn-danger" disabled={!delPw} onClick={() => setConfirmDelete(true)}>Delete my account</button>
      </div>
      <ConfirmDialog open={confirmDelete} title="Delete your account permanently?" danger confirmLabel="Delete everything" body={<p>All your data on this server will be removed.</p>}
        onCancel={() => setConfirmDelete(false)}
        onConfirm={() => { setConfirmDelete(false); void run(async () => { await api.post("/auth/delete-account", { password: delPw, confirm: "DELETE" });
          if (me) for (const a of await localAnalyses.all(me.id)) await localAnalyses.remove(a.id);
          offlineCache.clear(); setMe(null); }, "Account deleted."); }} />
      <p class="muted"><a href="#/about">About this app and its limits</a> - <a href="#/places">Places</a></p>
    </section>
  );
}

function BackupCard() {
  const { me, toast } = useApp();
  const [on, setOn] = useState(syncEnabled());
  const [count, setCount] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const refresh = () => listSynced().then((r) => setCount(r.length), () => setCount(null));
  useEffect(() => { if (navigator.onLine !== false) refresh(); }, []);
  const act = async (fn: () => Promise<string>) => {
    setBusy(true); setError(null);
    try { toast(await fn()); await refresh(); } catch (e) { setError(e); } finally { setBusy(false); }
  };
  return (
    <div class="card">
      <h2>Back up results</h2>
      <p>On-device results are stored only on this device unless you turn this on. Backups contain the analysis results and your feedback;
        <strong> recordings are never uploaded</strong>.</p>
      <label class="check">
        <input type="checkbox" checked={on} onChange={() => {
          const next = !on; setSyncEnabled(next); setOn(next);
          if (next && me) void act(async () => `Backed up ${await syncPending(me.id)} result(s).`);
        }} />
        Back up on-device results to my account (off by default)
      </label>
      {count !== null && <p class="muted">{count} result(s) backed up in your account.</p>}
      <ErrorBanner error={error} />
      <div class="row wrap">
        {on && me && <button type="button" class="btn btn-secondary" disabled={busy || navigator.onLine === false}
          onClick={() => void act(async () => `Backed up ${await syncPending(me.id)} result(s).`)}>Back up now</button>}
        {me && <button type="button" class="btn btn-secondary" disabled={busy || navigator.onLine === false}
          onClick={() => void act(async () => `Restored ${await restoreMissing(me.id)} result(s) to this device (without recordings).`)}>Restore backed-up results here</button>}
      </div>
    </div>
  );
}

function OnDeviceCard() {
  const { status, installOrUpdate, remove, checkForUpdate } = useOnDevice();
  const [server, setServer] = useState(serverAnalysisEnabled());
  const [confirm, setConfirm] = useState(false);
  const inst = status.installed;
  const update = inst && status.remote && status.remote.bundle_version !== inst.bundle_version;
  const p = status.progress;
  return (
    <div class="card">
      <h2>On-device analysis</h2>
      <p>Recordings are analysed on this device with downloaded models. They are not uploaded for analysis.</p>
      {status.state === "unsupported" && <InfoBanner tone="warn">{status.error}</InfoBanner>}
      {inst ? (
        <p>Installed: model bundle <strong>{inst.bundle_version}</strong> ({fmtBytes(inst.total_bytes)}, built {fmtDate(inst.created_at)}).
          {status.runtime === "threads" ? " Multi-threaded runtime." : status.runtime === "single-thread" ? " Single-threaded runtime (slower on this browser)." : ""}</p>
      ) : status.state !== "unsupported" && <p class="muted">Not installed. The download is about {status.remote ? fmtBytes(status.remote.total_bytes) : "40 MB"} and is needed once (and for updates).</p>}
      {update && <InfoBanner tone="info">An update ({status.remote!.bundle_version}, {fmtBytes(status.remote!.total_bytes)}) is available. The current version stays in use until the update is fully downloaded and verified.</InfoBanner>}
      {status.state === "installing" && <ProgressBar value={p && p.total_bytes ? p.done_bytes / p.total_bytes : 0} label={p?.file === WARMUP ? "Starting the models on this device" : `Downloading and verifying ${p?.file ?? ""}`} />}
      {status.error && status.state !== "unsupported" && <p class="field-error" role="alert">{status.error}</p>}
      <div class="row wrap">
        {status.state !== "unsupported" && (!inst || update) && (
          <button type="button" class="btn btn-primary" disabled={status.state === "installing" || navigator.onLine === false} onClick={() => void installOrUpdate()}>
            {inst ? "Update models" : "Install models"}
          </button>
        )}
        {inst && <button type="button" class="btn btn-secondary" onClick={() => void checkForUpdate()}>Check for updates</button>}
        {inst && <a class="btn btn-secondary" href="#/device-check">Run device check</a>}
        {inst && <button type="button" class="btn btn-secondary" onClick={() => setConfirm(true)}>Remove models</button>}
      </div>
      <fieldset class="field">
        <legend>Server analysis</legend>
        <label class="check">
          <input type="checkbox" checked={server} onChange={() => { setServerAnalysis(!server); setServer(!server); }} />
          Upload recordings to my server for analysis instead (off by default)
        </label>
        <p class="hint">When on, each recording you analyse is uploaded to your server. When off, nothing is uploaded for analysis.</p>
      </fieldset>
      <ConfirmDialog open={confirm} title="Remove on-device models?" body="Analysis on this device will need the models to be downloaded again. Your stored results are kept."
        confirmLabel="Remove" onCancel={() => setConfirm(false)} onConfirm={() => { setConfirm(false); void remove(); }} />
    </div>
  );
}


/** Recordings the owner chose to keep as research samples: list and delete (docs/RESEARCH_SAMPLES.md). */
/** Shown only inside the Android test app: go back to its start screen to change the server address. */
function AndroidServer() {
  if (!navigator.userAgent.includes("PetObserverAndroid")) return null;
  return (
    <div class="card">
      <h2>Server</h2>
      <p class="muted">This app is connected to {location.origin}.</p>
      <a class="btn btn-secondary" href="http://localhost/#change">Change server</a>
    </div>
  );
}

function ResearchSamples() {
  const [items, setItems] = useState<{ id: string; question: string; created_at: string; annotations: { status: string }[] }[] | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const load = () => api.get<typeof items>("/research-samples").then(setItems).catch(setErr);
  useEffect(() => { void load(); }, []);
  return (
    <div class="card">
      <h2>Research samples</h2>
      <p class="hint">Recordings you chose to keep to help improve the app. Your notes are kept as your reports; nothing is used as a confirmed label until a person has reviewed it.</p>
      <ErrorBanner error={err} onRetry={load} />
      {items && items.length === 0 && <p class="muted">None.</p>}
      <ul class="list">
        {(items ?? []).map((x) => (
          <li class="list-item row space">
            <span>"{x.question}" <span class="muted">{x.created_at.slice(0, 10)} - {x.annotations.filter((a) => a.status === "owner_reported").length} note(s) from you</span></span>
            <button type="button" class="btn btn-danger small" onClick={async () => { await api.del(`/research-samples/${x.id}`); void load(); }}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
