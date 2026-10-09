import { useEffect, useRef, useState } from "preact/hooks";
import { api } from "../lib/api";
import {
  ATTRIBUTION_TEXT, DECISION_HELP, fmtSpan, fmtTime, HYPOTHESIS_LABEL, QUALITY_TEXT, REASON_TEXT, soundLabel, STATUS_LABEL, supportWord,
} from "../lib/format";
import type { AnalysisResult, Feedback, Hypothesis } from "../lib/types";
import { useApp } from "../state";
import { Chip, ErrorBanner, InfoBanner } from "./ui";

const BEHAVIOR_TEXT: Record<string, string> = {
  stationary: "Still", moving: "Moving", active_bout: "Fast movement", pacing_candidate: "Back-and-forth movement (possible)",
  approach_zone: "Approached a marked place", enter_zone: "Entered a marked place", dwell_zone: "Stayed in a marked place",
  leave_zone: "Left a marked place", entered_view: "Came into view", left_view: "Left the view",
  vocal_bout: "Repeated vocalizations", startle_candidate: "Sudden movement right after a sound (possible)",
  person_interaction_candidate: "Held by or interacting with a person (possible)",
};

const POSE_TEXT: Record<string, string> = {
  ear_tip_elevation: "Ear tips relative to ear bases", tail_elevation: "Tail end relative to tail start",
  play_bow_candidate: "Play-bow-like position (possible)", tail_lateral_oscillation: "Tail side-to-side swings",
};

const ACTUAL_BEHAVIORS = ["eating or waiting for food", "wanting to go out", "seeking attention", "playing", "resting or sleeping",
  "alert / reacting to something", "fearful or hiding", "warning / aggressive", "grooming", "other", "not sure"];

/** Where feedback is stored: the server account, or only this device (local analyses). */
export interface FeedbackSink {
  where: "server" | "device";
  list(): Promise<Feedback[]>;
  add(body: { target_type: Feedback["target_type"]; target_id?: string | null; verdict: Feedback["verdict"]; actual_behavior?: string | null; note?: string; consent_research_use: boolean }): Promise<void>;
}

export function serverFeedback(analysisId: string): FeedbackSink {
  return {
    where: "server",
    list: () => api.get<Feedback[]>(`/analyses/${analysisId}/feedback`),
    add: async (body) => { await api.post(`/analyses/${analysisId}/feedback`, body); },
  };
}

export function ResultView({ feedback, result, mediaUrl, mediaKind }: { feedback: FeedbackSink; result: AnalysisResult; mediaUrl: string | null;
  mediaKind: "video" | "audio" | "image" }) {
  const mediaRef = useRef<HTMLVideoElement & HTMLAudioElement>(null);
  const [playhead, setPlayhead] = useState(0);
  const ev = Object.fromEntries(result.observations.evidence.map((e) => [e.evidence_id, e]));
  const duration = Math.max(result.media.duration_s ?? 0, ...result.observations.animal_tracks.map((t) => t.last_seen_s),
    ...result.observations.sound_events.map((s) => s.t_end_s), 0.1);

  const seek = (t: number) => {
    const m = mediaRef.current;
    if (m) {
      m.currentTime = Math.max(0, t);
      m.focus();
    }
  };

  useEffect(() => {
    const m = mediaRef.current;
    if (!m) return;
    const on = () => setPlayhead(m.currentTime);
    m.addEventListener("timeupdate", on);
    return () => m.removeEventListener("timeupdate", on);
  }, [mediaUrl]);

  const exp = result.hypotheses.filter((h) => h.status === "experimental_hypothesis");
  const other = result.hypotheses.filter((h) => h.status !== "experimental_hypothesis");
  const tracks = result.observations.animal_tracks;
  const sounds = result.observations.sound_events;
  const behaviors = [...tracks.flatMap((t) => t.behavior_segments ?? []), ...(result.observations.behavior_events ?? [])]
    .sort((a, b) => a.t_start_s - b.t_start_s);

  return (
    <div class="result">
      <div class={`card decision decision-${result.decision.state}`}>
        <h2>{STATUS_LABEL[result.decision.state] ?? result.decision.state}</h2>
        <p>{result.decision.summary}</p>
        <p class="muted">{DECISION_HELP[result.decision.state]}</p>
        {result.decision.reason_codes.length > 0 && (
          <ul class="reasons">
            {result.decision.reason_codes.map((r) => <li key={r}>{REASON_TEXT[r] ?? r}</li>)}
          </ul>
        )}
      </div>

      {mediaUrl && mediaKind !== "image" && (
        <div class="card">
          {mediaKind === "video" ? (
            <video ref={mediaRef} class="preview" src={mediaUrl} controls playsInline preload="metadata" aria-label="Recording" />
          ) : (
            <audio ref={mediaRef} src={mediaUrl} controls preload="metadata" aria-label="Recording" />
          )}
        </div>
      )}
      {mediaUrl && mediaKind === "image" && <div class="card"><img class="preview" src={mediaUrl} alt="Analyzed photo" /></div>}

      <div class="card">
        <h2>Timeline</h2>
        <p class="hint">Bars show when things were observed. {mediaUrl ? "Select a bar to jump there in the recording." : ""}</p>
        <div class="timeline" role="list" aria-label="Observation timeline">
          {tracks.map((t) => (
            <TimelineRow key={t.track_id} label={`${t.species} (${t.track_id})`} start={t.first_seen_s} end={t.last_seen_s} total={duration}
              tone="animal" onSelect={seek} />
          ))}
          {sounds.map((s) => (
            <TimelineRow key={s.event_id} label={soundLabel(s.label)} start={s.t_start_s} end={s.t_end_s} total={duration} tone="sound" onSelect={seek} />
          ))}
          {behaviors.map((b) => (
            <TimelineRow key={b.segment_id} label={BEHAVIOR_TEXT[b.kind] ?? b.kind} start={b.t_start_s} end={b.t_end_s} total={duration}
              tone="behavior" onSelect={seek} />
          ))}
          {result.quality_flags.filter((q) => q.t_start_s !== null && q.t_end_s !== null).map((q, i) => (
            <TimelineRow key={`q${i}`} label={QUALITY_TEXT[q.code] ?? q.code} start={q.t_start_s!} end={q.t_end_s!} total={duration} tone="quality" onSelect={seek} />
          ))}
          {mediaUrl && <div class="playhead" style={{ left: `calc(38% + ${(62 * Math.min(playhead, duration)) / duration}%)` }} aria-hidden="true" />}
        </div>
        <div class="axis" aria-hidden="true"><span>0:00</span><span>{fmtTime(duration)}</span></div>
      </div>

      <div class="card">
        <h2>What was observed</h2>
        {tracks.length === 0 && sounds.length === 0 && <p>No cat or dog was seen or heard.</p>}
        {tracks.map((t) => (
          <div key={t.track_id} class="obs">
            <h3>{t.species === "cat" ? "Cat" : "Dog"} seen <span class="muted">({t.track_id})</span></h3>
            <p>
              <button type="button" class="btn btn-link" onClick={() => seek(t.first_seen_s)}>{fmtSpan(t.first_seen_s, t.last_seen_s)}</button>
              {" "}- {t.n_detections} detections. <span class="muted">{t.identity_note}.</span>
            </p>
            {t.movement.available ? (
              <p>Apparent movement: median {t.movement.median_speed?.toFixed(3)} image diagonals per second (relative to the picture, not
                physical speed); moving in {Math.round((t.movement.moving_fraction ?? 0) * (t.movement.steps ?? 0))} of {t.movement.steps ?? 0} sampled
                steps. Reliability: {t.movement.reliability}.</p>
            ) : (
              <p class="muted">Movement not measured: {t.movement.reason}</p>
            )}
            {t.zones.map((z) => <p key={z.zone_id}>In "{z.label}" ({z.semantic_tag.replace(/_/g, " ")}) for {z.dwell_s.toFixed(1)} s.</p>)}
            {(t.behavior_segments ?? []).length > 0 && (
              <ul class="seg-list" aria-label="Behavior segments">
                {t.behavior_segments!.map((b) => (
                  <li key={b.segment_id}>
                    <button type="button" class="btn btn-link" onClick={() => seek(b.t_start_s)}>{fmtSpan(b.t_start_s, b.t_end_s)}</button>{" "}
                    {BEHAVIOR_TEXT[b.kind] ?? b.kind}: {b.detail}{b.reliability === "low" ? " (low reliability)" : ""}
                  </li>
                ))}
              </ul>
            )}
            {(t.posture ?? []).length > 0 && (
              <p>Posture estimate (experimental, from single frames; not a probability): {summarizePosture(t.posture!)}</p>
            )}
            {t.verification && (
              <p class="muted">Second check: {t.verification.accepted ? "passed" : "not passed"} in {t.verification.checked_detections} checked frames.</p>
            )}
            {t.quality_flags.length > 0 && <p class="muted">Limits: {t.quality_flags.map((f) => f.replace(/_/g, " ")).join(", ")}</p>}
            <details>
              <summary>Body language details</summary>
              <ul>{t.pose_features.map((p) => (
                <li key={p.name}>{POSE_TEXT[p.name] ?? p.name.replace(/_/g, " ")}: {p.available ? poseValue(p) : `not available - ${p.reason}`}</li>
              ))}</ul>
              {(t.keypoints ?? []).length > 0 && <p class="muted">Body points estimated in {t.keypoints!.length} frames (dogs only; research model tested on web photos, not home video).</p>}
            </details>
          </div>
        ))}
        {sounds.map((s) => (
          <div key={s.event_id} class="obs">
            <h3>{soundLabel(s.label)}</h3>
            <p>
              <button type="button" class="btn btn-link" onClick={() => seek(s.t_start_s)}>{fmtSpan(s.t_start_s, s.t_end_s)}</button>
              {Object.keys(s.subtypes).length > 0 && <> - sounded like: {Object.entries(s.subtypes).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k]) => k.replace(/_/g, " ")).join(", ")}</>}
            </p>
            <p class="muted">{ATTRIBUTION_TEXT[s.attribution.status] ?? s.attribution.note}</p>
          </div>
        ))}
        {result.observations.context_sounds.length > 0 && (
          <p class="muted">Other sounds: {result.observations.context_sounds.map((c) => `${soundLabel(c.label)} (${fmtTime(c.t_start_s)})`).join(", ")}</p>
        )}
      </div>

      <div class="card exp-card">
        <h2>Possible interpretations <Chip tone="exp">Experimental</Chip></h2>
        <p class="hint">
          These are rule-based guesses from the observations above. They have not been checked against real households, the scores are not
          probabilities, and the app may be wrong. Use your own knowledge of your pet.
        </p>
        {exp.length === 0 && <p>No interpretation had enough independent evidence.</p>}
        {exp.map((h) => <HypothesisCard key={h.hypothesis_id} h={h} ev={ev} seek={seek} sink={feedback} />)}
        {other.length > 0 && (
          <details>
            <summary>Considered but not supported ({other.length})</summary>
            {other.map((h) => <HypothesisCard key={h.hypothesis_id} h={h} ev={ev} seek={seek} sink={feedback} compact />)}
          </details>
        )}
      </div>

      {result.suggestions.length > 0 && (
        <div class="card">
          <h2>Suggestions</h2>
          <ul>{result.suggestions.map((s) => <li key={s.suggestion_id}>{s.text}</li>)}</ul>
        </div>
      )}

      <details class="card">
        <summary><h2 class="inline">Quality and limitations</h2></summary>
        {result.quality_flags.length > 0 && (
          <ul>{result.quality_flags.map((q, i) => <li key={i}>{QUALITY_TEXT[q.code] ?? q.code}{q.detail ? ` - ${q.detail}` : ""}</li>)}</ul>
        )}
        <ul>{result.limitations.map((l, i) => <li key={i}>{l}</li>)}</ul>
        <p class="muted">Components: {result.components.map((c) => `${c.name}: ${c.state}`).join(", ")}</p>
      </details>

      <FeedbackForm sink={feedback} />
    </div>
  );
}

function poseValue(p: { name: string; value?: number | boolean | null; units?: string | null }): string {
  if (typeof p.value === "boolean") return p.value ? "seen in at least one frame" : "not seen";
  if (typeof p.value !== "number") return "measured";
  if (p.name === "ear_tip_elevation") return p.value > 0 ? "tips above bases (depends strongly on ear type)" : "tips at or below bases (depends strongly on ear type)";
  if (p.name === "tail_elevation") return p.value > 0.05 ? "tail end higher than its start" : p.value < -0.05 ? "tail end lower than its start" : "tail roughly level";
  if (p.name === "tail_lateral_oscillation") return `${p.value} side changes`;
  return `${p.value.toFixed(2)} ${p.units ?? ""}`;
}

function summarizePosture(p: { label: string }[]): string {
  const counts: Record<string, number> = {};
  p.forEach((x) => (counts[x.label] = (counts[x.label] ?? 0) + 1));
  return Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, n]) => `${k.replace(/_/g, " ")} in ${n} frame(s)`).join(", ");
}

function TimelineRow({ label, start, end, total, tone, onSelect }: { label: string; start: number; end: number; total: number; tone: string;
  onSelect: (t: number) => void }) {
  const left = (100 * Math.max(0, start)) / total;
  const width = Math.max(1.5, (100 * Math.max(0.05, end - start)) / total);
  return (
    <div class="tl-row" role="listitem">
      <span class="tl-label">{label}</span>
      <div class="tl-track">
        <button type="button" class={`tl-bar tl-${tone}`} style={{ left: `${left}%`, width: `${Math.min(width, 100 - left)}%` }}
          onClick={() => onSelect(start)} aria-label={`${label}, ${fmtSpan(start, end)}`} title={`${label} ${fmtSpan(start, end)}`} />
      </div>
    </div>
  );
}

function HypothesisCard({ h, ev, seek, sink, compact }: { h: Hypothesis; ev: Record<string, { summary: string; t_start_s: number }>;
  seek: (t: number) => void; sink: FeedbackSink; compact?: boolean }) {
  const tone = h.status === "experimental_hypothesis" ? "exp" : h.status === "abstained" ? "warn" : "neutral";
  return (
    <div class={`hyp ${compact ? "compact" : ""}`}>
      <h3>
        {HYPOTHESIS_LABEL[h.name] ?? h.name} <Chip tone={tone}>{STATUS_LABEL[h.status] ?? h.status}</Chip>
      </h3>
      <p class="muted">
        {h.semantic_type.replace(/_/g, " ")}{h.track_id ? ` - ${h.track_id}` : ""}{h.t_start_s !== null ? ` - ${fmtSpan(h.t_start_s, h.t_end_s)}` : ""}
        {" "}- support: {supportWord(h.score)} (uncalibrated)
      </p>
      <p>Based on:</p>
      <ul>
        {h.supporting_evidence_ids.map((id) => ev[id] && (
          <li key={id}><button type="button" class="btn btn-link" onClick={() => seek(ev[id].t_start_s)}>{fmtTime(ev[id].t_start_s)}</button> {ev[id].summary}</li>
        ))}
      </ul>
      {h.conflicting_evidence_ids.length > 0 && (
        <>
          <p>Against it:</p>
          <ul>{h.conflicting_evidence_ids.map((id) => ev[id] && <li key={id}>{ev[id].summary}</li>)}</ul>
        </>
      )}
      <ul class="reasons">{h.reason_codes.filter((r) => r !== "fusion_not_validated").map((r) => <li key={r}>{REASON_TEXT[r] ?? r.replace(/_/g, " ")}</li>)}</ul>
      {!compact && <QuickFeedback sink={sink} targetType="hypothesis" targetId={h.hypothesis_id} />}
    </div>
  );
}

function QuickFeedback({ sink, targetType, targetId }: { sink: FeedbackSink; targetType: Feedback["target_type"]; targetId: string }) {
  const { me, toast } = useApp();
  const [sent, setSent] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const send = async (verdict: Feedback["verdict"]) => {
    try {
      await sink.add({ target_type: targetType, target_id: targetId, verdict, consent_research_use: me?.feedback_research_consent ?? false });
      setSent(verdict);
      toast(sink.where === "device" ? "Thanks - feedback saved on this device." : "Thanks - feedback saved.");
    } catch (e) {
      setError(e);
    }
  };
  if (sent) return <p class="muted">Your feedback: {sent}.</p>;
  return (
    <div class="row wrap" role="group" aria-label="Does this match what happened?">
      <span class="muted">Does this match what happened?</span>
      <button type="button" class="btn btn-small" onClick={() => send("agree")}>Yes</button>
      <button type="button" class="btn btn-small" onClick={() => send("disagree")}>No</button>
      <button type="button" class="btn btn-small" onClick={() => send("unsure")}>Not sure</button>
      <ErrorBanner error={error} />
    </div>
  );
}

function FeedbackForm({ sink }: { sink: FeedbackSink }) {
  const { me, toast } = useApp();
  const [items, setItems] = useState<Feedback[]>([]);
  const [verdict, setVerdict] = useState<Feedback["verdict"]>("unsure");
  const [actual, setActual] = useState("");
  const [note, setNote] = useState("");
  const [consent, setConsent] = useState(me?.feedback_research_consent ?? false);
  const [error, setError] = useState<unknown>(null);
  const load = () => sink.list().then(setItems).catch(setError);
  useEffect(() => {
    load();
  }, [sink]);
  const submit = async (e: Event) => {
    e.preventDefault();
    try {
      await sink.add({ target_type: "overall", verdict, actual_behavior: actual || null, note, consent_research_use: consent });
      setNote("");
      toast(sink.where === "device" ? "Feedback saved on this device." : "Feedback saved.");
      load();
    } catch (err) {
      setError(err);
    }
  };
  return (
    <form class="card" onSubmit={submit}>
      <h2>Your feedback</h2>
      <p class="hint">Telling us what actually happened is the only way these interpretations can ever be checked.
        {sink.where === "device" ? " This feedback stays on this device unless you turn on result sync." : ""}</p>
      <fieldset class="field">
        <legend>Overall, was the result useful?</legend>
        {(["agree", "disagree", "unsure"] as const).map((v) => (
          <label key={v} class="radio"><input type="radio" name="verdict" checked={verdict === v} onChange={() => setVerdict(v)} /> {v === "agree" ? "Yes" : v === "disagree" ? "No" : "Not sure"}</label>
        ))}
      </fieldset>
      <div class="field">
        <label for="actual">What was your pet actually doing?</label>
        <select id="actual" value={actual} onChange={(e) => setActual((e.target as HTMLSelectElement).value)}>
          <option value="">(choose)</option>
          {ACTUAL_BEHAVIORS.map((b) => <option key={b} value={b}>{b}</option>)}
        </select>
      </div>
      <div class="field">
        <label for="fbnote">Note</label>
        <textarea id="fbnote" maxLength={1000} rows={2} value={note} onInput={(e) => setNote((e.target as HTMLTextAreaElement).value)} />
      </div>
      <label class="check"><input type="checkbox" checked={consent} onChange={() => setConsent(!consent)} /> Allow this feedback to be used to evaluate and improve the system</label>
      <ErrorBanner error={error} />
      <button type="submit" class="btn btn-primary">Save feedback</button>
      {items.length > 0 && (
        <details>
          <summary>Saved feedback ({items.length})</summary>
          <ul>
            {items.map((f) => (
              <li key={f.id}>
                {f.target_type}: {f.verdict}{f.actual_behavior ? ` - ${f.actual_behavior}` : ""}{f.note ? ` - ${f.note}` : ""}{" "}
                <button type="button" class="btn btn-link" onClick={async () => { await api.del(`/feedback/${f.id}`); load(); }}>Delete</button>
              </li>
            ))}
          </ul>
        </details>
      )}
      <InfoBanner>Feedback never changes past results; it is stored with this analysis and deleted with it.</InfoBanner>
    </form>
  );
}
