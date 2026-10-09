import { useEffect, useState } from "preact/hooks";
import { AnalysisListItem } from "../components/AnalysisListItem";
import { Empty, ErrorBanner } from "../components/ui";
import { api, ApiError } from "../lib/api";
import { fmtDate } from "../lib/format";
import { localAnalyses, type LocalAnalysis } from "../lib/localstore";
import type { Analysis } from "../lib/types";
import { useApp } from "../state";

const PAGE = 20;

export function HistoryPage() {
  const { pets, me } = useApp();
  const [local, setLocal] = useState<LocalAnalysis[]>([]);
  useEffect(() => {
    if (me && localAnalyses.available) localAnalyses.all(me.id).then(setLocal).catch(() => undefined);
  }, [me]);
  const [items, setItems] = useState<Analysis[]>([]);
  const [petFilter, setPetFilter] = useState(() => {
    const v = sessionStorage.getItem("historyPet") ?? "";
    sessionStorage.removeItem("historyPet");
    return v;
  });
  const [more, setMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  const load = async (reset: boolean) => {
    setLoading(true);
    setError(null);
    try {
      const before = !reset && items.length ? `&before=${encodeURIComponent(items[items.length - 1].created_at)}` : "";
      const pet = petFilter ? `&pet_id=${petFilter}` : "";
      const page = await api.get<Analysis[]>(`/analyses?limit=${PAGE}${pet}${before}`);
      setItems(reset ? page : [...items, ...page]);
      setMore(page.length === PAGE);
    } catch (e) {
      if (!(e instanceof ApiError && e.isNetwork)) setError(e); // offline: on-device results are still listed
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load(true);
  }, [petFilter]);

  return (
    <section>
      <h1 tabIndex={-1}>History</h1>
      <div class="field">
        <label for="petfilter">Show analyses for</label>
        <select id="petfilter" value={petFilter} onChange={(e) => setPetFilter((e.target as HTMLSelectElement).value)}>
          <option value="">All recordings</option>
          {pets.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
      </div>
      <ErrorBanner error={error} onRetry={() => load(true)} />
      {!loading && items.length === 0 && local.length === 0 && !error && <Empty title="Nothing here yet"><a href="#/capture">Record or upload a clip</a></Empty>}
      {local.length > 0 && <h2>Analysed on this device</h2>}
      <ul class="list" aria-label="Analysed on this device">
        {local.filter((a) => !petFilter || a.pet_ids.includes(petFilter)).map((a) => (
          <li key={a.id} class="list-item">
            <a href={`#/local/${a.id}`}>
              <strong>{a.filename}</strong>
              <span class="muted"> - {fmtDate(new Date(a.createdAt).toISOString())}{a.synced_at ? " - backed up" : ""} - {a.result.decision.summary}</span>
            </a>
          </li>
        ))}
      </ul>
      {items.length > 0 && <h2>Analysed on the server</h2>}
      <ul class="list">
        {items.map((a) => <AnalysisListItem key={a.id} a={a} pets={pets} />)}
      </ul>
      {loading && <p class="muted" role="status">Loading...</p>}
      {more && !loading && <button type="button" class="btn btn-secondary btn-block" onClick={() => load(false)}>Load more</button>}
    </section>
  );
}
