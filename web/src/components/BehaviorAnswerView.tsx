import { useState } from "preact/hooks";
import { api } from "../lib/api";
import { navigate } from "../lib/router";
import type { Analysis, BehaviorAnswer } from "../lib/types";
import { useApp } from "../state";

/** The owner's question and its answer in plain language. What the recording shows, what the owner said and possible
 * explanations (hypotheses) stay visibly apart; scores, thresholds and model names are not shown here. */

const KIND_TEXT: Record<string, string> = {
  request: "Asking you for something", play_or_normal: "Normal play or everyday behaviour", arousal_or_stress: "Excitement, frustration or stress",
  compulsive: "A repetitive habit that can become compulsive", medical: "A possible health cause - worth checking with a vet",
};
const QUALITY_TEXT: Record<string, string> = {
  low_light: "The recording is quite dark; more light would help.", blurry_frames: "Parts of the recording are blurry; hold the phone steadier.",
  camera_motion: "The camera moved a lot; a steadier recording would help.", audio_silent: "There is no sound in the recording.",
  still_image: "This is a photo; movement over time could not be seen.", frame_limit_reached: "Only the first part of a long recording was looked at.",
};
const HIDE = ["results/e31", "vision-language", "zero-shot", "scores are relative"];

type Obs = { behavior?: string; t_start_s?: number; t_end_s?: number; evidence?: string };
type Cand = { kind?: string; explanation?: string };

export function BehaviorAnswerView({ a, q, seek }: { a: Analysis; q: NonNullable<Analysis["behavior_question"]>; seek?: (t: number) => void }) {
  const { caps } = useApp();
  const labels: Record<string, string> = Object.fromEntries((caps?.behavior_question?.behaviors ?? []).map((b) => [b.name, b.label]));
  const name = (b?: string) => (b ? labels[b] ?? b.replace(/_/g, " ") : "");
  const ans: BehaviorAnswer | null = q.answer;
  const running = a.status === "queued" || a.status === "running";
  return (
    <section class="card answer" aria-labelledby="bq-h">
      <h2 id="bq-h">Your question</h2>
      <p class="quote">"{q.question}"</p>
      <p class="muted">{q.species === "cat" ? "Cat" : "Dog"}{q.scene ? ` - ${q.scene}` : ""}</p>
      {running && <Steps stage={a.stage} />}
      {a.status === "failed" && <p role="alert">The recording could not be analysed. Use Retry below, or try a shorter or clearer clip.</p>}
      {q.status === "failed" && <p role="alert">Your question could not be answered this time. The observations below are still valid; you can add another recording.</p>}
      {ans && ans.local_answer && <Answer ans={ans} name={name} seek={seek} />}
      {a.status === "succeeded" && a.result && <Quality codes={a.result.quality_flags.map((f) => f.code)} />}
      {a.status === "succeeded" && <Correct a={a} ans={ans} name={name} />}
      {a.status !== "queued" && a.status !== "running" && (
        <div class="row wrap">
          <button type="button" class="btn btn-primary" onClick={() => navigate("/ask")}>Add another recording</button>
          <a class="btn btn-secondary" href="#/history">History</a>
        </div>
      )}
    </section>
  );
}

function Steps({ stage }: { stage: string | null }) {
  const s = stage ?? "queued";
  const answering = s === "question";
  return (
    <ol class="progress-steps" aria-live="polite">
      <li class="done">Recording received</li>
      <li class={answering ? "done" : "now"}>Looking at the recording{!answering && s !== "queued" ? "..." : s === "queued" ? " (waiting for a free slot)" : ""}</li>
      <li class={answering ? "now" : ""}>Answering your question{answering ? "..." : ""}</li>
      <li class="hint">This usually takes one to three minutes. You can leave this page; the answer will be in History.</li>
    </ol>
  );
}

function Answer({ ans, name, seek }: { ans: BehaviorAnswer; name: (b?: string) => string; seek?: (t: number) => void }) {
  const c = ans.cloud?.status === "ok" ? (ans.cloud.answer as Record<string, unknown>) : null;
  const l = ans.local_answer;
  const seen: Obs[] = c ? ((c.observed as Obs[]) ?? []) : l.behaviors_seen.flatMap((b) => (b.bouts_s.length ? b.bouts_s.slice(0, 4).map((x) => ({ behavior: b.behavior, t_start_s: x[0], t_end_s: x[1] })) : [{ behavior: b.behavior }]));
  const notSeen: string[] = c ? ((c.not_observed as string[]) ?? []) : (l as unknown as { user_reported: { behavior: string; status: string }[] }).user_reported
    .filter((u) => !u.status.startsWith("also seen")).map((u) => name(u.behavior));
  const groups: { kind: string; items: string[] }[] = [];
  const add = (kind: string, text: string) => {
    const g = groups.find((x) => x.kind === kind) ?? (groups.push({ kind, items: [] }), groups[groups.length - 1]);
    if (!g.items.includes(text)) g.items.push(text);
  };
  if (c) ((c.candidate_explanations as Cand[]) ?? []).forEach((x) => add(x.kind ?? "", x.explanation ?? ""));
  else l.explained_behaviors.slice(0, 2).forEach((e) => Object.entries(e.candidates_by_kind).forEach(([k, cs]) => cs.forEach((x) => add(k, x.explanation))));
  const uncertain = (c ? ((c.uncertain as string[]) ?? []) : l.uncertainty).filter((u) => !HIDE.some((h) => u.includes(h)));
  const watch = c ? ((c.would_distinguish as string[]) ?? []) : l.observations_needed;
  const vet = c ? ((c.see_vet_if as string[]) ?? []) : l.urgent_signs;
  const firstSeen = [...new Set(seen.map((o) => name(o.behavior)))];
  const summary = `${firstSeen.length ? `In this recording we saw: ${firstSeen.slice(0, 3).join("; ")}.` : "We could not recognise a specific behaviour in this recording."}`
    + `${notSeen.length ? ` We could not see ${notSeen.join(" or ")} in this clip.` : ""}`
    + `${groups.length ? ` Possible explanations to check: ${groups.slice(0, 3).map((g) => (KIND_TEXT[g.kind] ?? g.kind).toLowerCase()).join("; ")}.` : ""}`
    + " A recording shows what happens, not why; the notes below help tell the explanations apart.";
  return (
    <div>
      <p class="answer-summary">{summary}</p>
      <h3>What we saw</h3>
      <ul class="obs">
        {seen.length ? seen.map((o) => (
          <li>
            {name(o.behavior)}
            {o.t_start_s !== undefined && (
              <button type="button" class="btn btn-link time" onClick={() => seek?.(Number(o.t_start_s))} aria-label={`Play from ${Math.round(Number(o.t_start_s))} seconds`}>
                {fmtT(o.t_start_s)}-{fmtT(o.t_end_s)}
              </button>
            )}
          </li>
        )) : <li>Nothing specific.</li>}
      </ul>
      <p class="hint">You said: "{ans.question}" - that is your description; the list above is what the recording shows.</p>
      <h3>What it could mean</h3>
      {groups.map((g) => (
        <details class="cand"><summary>{KIND_TEXT[g.kind] ?? g.kind}</summary><ul>{g.items.map((t) => <li>{t}</li>)}</ul></details>
      ))}
      {uncertain.length > 0 && (<><h3>Not sure about</h3><ul>{uncertain.slice(0, 4).map((u) => <li>{u}</li>)}<li>Automatic recognition makes mistakes, especially with brief or unusual behaviour.</li></ul></>)}
      {watch.length > 0 && (<><h3>Watch for next time</h3><ul>{watch.slice(0, 6).map((u) => <li>{u}</li>)}</ul></>)}
      {vet.length > 0 && (<><h3>See a vet promptly if</h3><ul class="vet">{vet.map((u) => <li>{u}</li>)}</ul></>)}
      {!c && ans.cloud && !["not_requested", "ok"].includes(ans.cloud.status) && <p class="hint">The extended AI service was not available; this answer was made on your server.</p>}
      {(l as unknown as { sources?: { id: string; citation: string; url?: string }[] }).sources?.length ? (
        <details class="sources"><summary>Sources</summary><ul>{(l as unknown as { sources: { citation: string; url?: string }[] }).sources.map((s) => <li>{s.url ? <a href={s.url} target="_blank" rel="noreferrer">{s.citation}</a> : s.citation}</li>)}</ul></details>
      ) : null}
    </div>
  );
}

function fmtT(t?: number): string {
  if (t === undefined || t === null || Number.isNaN(Number(t))) return "";
  const s = Math.round(Number(t));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function Quality({ codes }: { codes: string[] }) {
  const notes = [...new Set(codes)].map((c) => QUALITY_TEXT[c]).filter(Boolean);
  return notes.length ? <div class="banner banner-warn"><p class="banner-title">About this recording</p><ul>{notes.map((n) => <li>{n}</li>)}</ul></div> : null;
}

function Correct({ a, ans, name }: { a: Analysis; ans: BehaviorAnswer | null; name: (b?: string) => string }) {
  const { caps, toast } = useApp();
  const [open, setOpen] = useState(false);
  const [actual, setActual] = useState("");
  const [own, setOwn] = useState("");
  const [note, setNote] = useState("");
  const [done, setDone] = useState<Record<string, string>>({});
  const [err, setErr] = useState<string | null>(null);
  const seen = [...new Set((ans?.local_answer?.behaviors_seen ?? []).map((b) => b.behavior))];
  const choices = (caps?.behavior_question?.behaviors ?? []).filter((b) => b.species.includes(a.behavior_question?.species ?? "cat"));
  const post = async (body: Record<string, unknown>, key: string) => {
    try {
      setErr(null);
      await api.post(`/analyses/${a.id}/feedback`, body);
      setDone((d) => ({ ...d, [key]: "saved" }));
      toast("Thank you - saved.");
    } catch {
      setErr("Could not save; check the connection and try again.");
    }
  };
  return (
    <div class="correct">
      <h3>Was this right?</h3>
      {seen.map((b) => (
        <div class="row wrap">
          <span>{name(b)}</span>
          {done[b] ? <span class="muted">Saved</span> : (<>
            <button type="button" class="btn btn-secondary small" onClick={() => post({ target_type: "observation", target_id: b, verdict: "agree" }, b)}>Right</button>
            <button type="button" class="btn btn-secondary small" onClick={() => post({ target_type: "observation", target_id: b, verdict: "disagree" }, b)}>Not right</button>
          </>)}
        </div>
      ))}
      {!open ? <button type="button" class="btn btn-link" onClick={() => setOpen(true)}>Tell us what your pet actually did</button> : done.overall ? <p class="muted">Saved. Thank you.</p> : (
        <div>
          <label for="actual">What did your pet actually do?</label>
          <select id="actual" value={actual} onChange={(e) => setActual((e.target as HTMLSelectElement).value)}>
            <option value="">Choose...</option>
            {choices.map((b) => <option value={b.name}>{b.label}</option>)}
            <option value="other">Something else</option>
          </select>
          {actual === "other" && <input aria-label="In your own words" maxLength={120} value={own} onInput={(e) => setOwn((e.target as HTMLInputElement).value)} placeholder="In your own words" />}
          <label for="ctx">What happened just before or after? (optional)</label>
          <textarea id="ctx" maxLength={1000} rows={2} value={note} onInput={(e) => setNote((e.target as HTMLTextAreaElement).value)} />
          <button type="button" class="btn btn-primary" disabled={!actual && !note.trim()}
            onClick={() => post({ target_type: "overall", verdict: "disagree", actual_behavior: actual === "other" ? own.trim() || "other" : actual || null, note: note.trim() }, "overall")}>Save</button>
        </div>
      )}
      {err && <p class="field-error" role="alert">{err}</p>}
      <p class="hint">Your corrections are saved as your report. They help improve the app only if you chose to keep the recording as a research sample, and they are reviewed before use.</p>
    </div>
  );
}
