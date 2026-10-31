import type { Analysis, BehaviorAnswer } from "../lib/types";

type Obs = { behavior?: string; t_start_s?: number; t_end_s?: number; evidence?: string; confidence?: string };
type Cand = { kind?: string; explanation?: string; supported_by_observations?: string[]; knowledge_ids?: string[] };

const list = (x: unknown): string[] => (Array.isArray(x) ? x.map(String) : []);

/** The owner's behaviour question and its answer: observed (recording), reported (owner) and inferred (hypotheses) kept apart. */
export function BehaviorAnswerView({ q }: { q: NonNullable<Analysis["behavior_question"]> }) {
  const a: BehaviorAnswer | null = q.answer;
  return (
    <section class="card" aria-labelledby="bq-h">
      <h2 id="bq-h">Your question</h2>
      <p><strong>{q.question}</strong></p>
      <p class="muted">You said: {q.species}{q.scene ? `, ${q.scene}` : ""}. These are used as given.</p>
      {q.status === "pending" && <p role="status">Answering...</p>}
      {q.status === "failed" && <p role="alert">The question could not be answered; the analysis result below is still valid.</p>}
      {a && <AnswerBody a={a} />}
    </section>
  );
}

function AnswerBody({ a }: { a: BehaviorAnswer }) {
  const c = a.cloud?.status === "ok" ? (a.cloud.answer as Record<string, unknown>) : null;
  if (c) {
    const obs = (c.observed as Obs[]) ?? [];
    const cands = (c.candidate_explanations as Cand[]) ?? [];
    return (
      <div>
        <p class="hint">Answered by the cloud model from {a.packet?.images ?? 0} frames or close-ups and a sound summary. Check what it says against the recording.</p>
        <h3>Seen in the recording</h3>
        <ul>{obs.length ? obs.map((o) => <li>{o.behavior} {o.t_start_s}-{o.t_end_s} s ({o.confidence}): {o.evidence}</li>) : <li>Nothing specific.</li>}</ul>
        {list(c.not_observed).length > 0 && <p>You mentioned, but the recording does not show: {list(c.not_observed).join(", ")}</p>}
        <h3>Possible explanations (hypotheses, not causes)</h3>
        <ul>{cands.map((x) => <li>[{x.kind}] {x.explanation} <span class="muted">(sources: {(x.knowledge_ids ?? []).join(", ")})</span></li>)}</ul>
        <h3>Still uncertain</h3>
        <ul>{list(c.uncertain).map((u) => <li>{u}</li>)}</ul>
        <h3>What would tell them apart</h3>
        <ul>{list(c.would_distinguish).map((u) => <li>{u}</li>)}</ul>
        {list(c.see_vet_if).length > 0 && (<><h3>See a veterinarian promptly if</h3><ul>{list(c.see_vet_if).map((u) => <li>{u}</li>)}</ul></>)}
      </div>
    );
  }
  const l = a.local_answer;
  if (!l) return <p role="alert">{a.error ?? "No answer."}</p>;
  return (
    <div>
      {a.cloud && a.cloud.status !== "not_requested" && <p class="hint">Cloud model unavailable ({a.cloud.reason ?? a.cloud.status}); this is the on-server answer.</p>}
      <h3>Seen in the recording</h3>
      <ul>{l.behaviors_seen.length ? l.behaviors_seen.map((b) => <li>{b.behavior} ({b.bouts_s.slice(0, 4).map((x) => `${x[0]}-${x[1]} s`).join(", ") || "clip level"})</li>) : <li>Nothing above threshold.</li>}</ul>
      <h3>Possible explanations (hypotheses, not causes)</h3>
      {l.explained_behaviors.map((e) => (
        <div><p><strong>{e.behavior}</strong> <span class="muted">({e.basis})</span></p>
          <ul>{Object.entries(e.candidates_by_kind).flatMap(([k, cs]) => cs.map((x) => <li>[{k}] {x.explanation} <span class="muted">({x.sources.join(", ")})</span></li>))}</ul></div>
      ))}
      <h3>What would tell them apart</h3>
      <ul>{l.observations_needed.map((u) => <li>{u}</li>)}</ul>
      {l.urgent_signs.length > 0 && (<><h3>See a veterinarian promptly if</h3><ul>{l.urgent_signs.map((u) => <li>{u}</li>)}</ul></>)}
      <h3>Uncertainty</h3>
      <ul>{l.uncertainty.map((u) => <li>{u}</li>)}</ul>
    </div>
  );
}
