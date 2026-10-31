import { useEffect, useState } from "preact/hooks";
import { AnalysisListItem } from "../components/AnalysisListItem";
import { Empty, ErrorBanner, InfoBanner, ProgressBar } from "../components/ui";
import { api } from "../lib/api";
import { fmtBytes } from "../lib/format";
import { pendingStore } from "../lib/idb";
import type { Analysis } from "../lib/types";
import { useApp } from "../state";

export function HomePage() {
  const { me, pets, pending, progress, startPending, refreshPending } = useApp();
  const [recent, setRecent] = useState<Analysis[] | null>(null);
  const [error, setError] = useState<unknown>(null);

  const load = () => api.get<Analysis[]>("/analyses?limit=5").then(setRecent).catch(setError);
  useEffect(() => {
    load();
  }, []);

  return (
    <section>
      <h1 tabIndex={-1}>Hello, {me?.username}</h1>
      <a class="btn btn-primary btn-hero" href="#/ask">Ask about my pet's behaviour</a>
      <p class="hint">Describe what you noticed, add a short video, and get what the recording shows and what it could mean.</p>
      <div class="actions-grid">
        <a class="action" href="#/capture"><span class="action-icon" aria-hidden="true">●</span>Observe a clip (no question)</a>
        <a class="action" href="#/history"><span class="action-icon" aria-hidden="true">☰</span>Past analyses</a>
        <a class="action" href="#/pets"><span class="action-icon" aria-hidden="true">♥</span>{pets.length ? `${pets.length} pet profile(s)` : "Add your pets"}</a>
        <a class="action" href="#/places"><span class="action-icon" aria-hidden="true">▣</span>Mark places (bowl, door, bed)</a>
      </div>

      {pending.length > 0 && (
        <div class="card">
          <h2>Unfinished on this device</h2>
          <ul class="list">
            {pending.map((p) => {
              const pr = progress[p.localId];
              return (
                <li key={p.localId} class="list-item">
                  <div>
                    <strong>{p.filename}</strong> <span class="muted">({fmtBytes(p.blob.size)})</span>
                    <div class="muted">State: {p.state}{p.lastError ? ` - ${p.lastError}` : ""}</div>
                    {pr && pr.phase !== "failed" && <ProgressBar value={pr.total ? pr.sent / pr.total : 0} label={`Upload ${p.filename}`} />}
                    {pr?.message && <div class="muted">{pr.message}</div>}
                  </div>
                  <div class="row">
                    <button type="button" class="btn btn-secondary" onClick={() => void startPending(p)}>Resume</button>
                    <button type="button" class="btn btn-link" onClick={async () => { await pendingStore.remove(p.localId); await refreshPending(); }}>Discard</button>
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <div class="card">
        <h2>Recent analyses</h2>
        <ErrorBanner error={error} onRetry={load} />
        {recent === null && !error && <p class="muted">Loading...</p>}
        {recent && recent.length === 0 && (
          <Empty title="No analyses yet">
            <p>Record a short clip (10-60 seconds works well) of your pet, ideally with the whole animal in view.</p>
          </Empty>
        )}
        {recent && recent.length > 0 && (
          <ul class="list">
            {recent.map((a) => <AnalysisListItem key={a.id} a={a} pets={pets} />)}
          </ul>
        )}
      </div>

      <InfoBanner>
        Results describe what the system observed. Interpretations such as "possibly food-related" are experimental guesses that have not been
        validated on real household recordings. <a href="#/about">How it works and its limits</a>
      </InfoBanner>
    </section>
  );
}
