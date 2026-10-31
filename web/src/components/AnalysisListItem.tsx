import { fmtDate, STATUS_LABEL } from "../lib/format";
import type { Analysis, Pet } from "../lib/types";
import { Chip } from "./ui";

const STATUS_TONE: Record<string, "neutral" | "ok" | "warn" | "bad" | "exp"> = {
  queued: "neutral", running: "neutral", succeeded: "ok", failed: "bad", cancelled: "warn",
};

export function subjectText(a: Analysis, pets: Pet[]): string {
  if (a.subject_mode === "unknown") return "Animal not specified";
  const names = a.pet_ids.map((id) => pets.find((p) => p.id === id)?.name ?? "deleted pet");
  return a.subject_mode === "single_known" ? `You said: ${names[0]}` : `You said: ${names.join(", ")}`;
}

export function AnalysisListItem({ a, pets }: { a: Analysis; pets: Pet[] }) {
  const s = a.summary;
  return (
    <li class="list-item">
      <a href={`#/analyses/${a.id}`} class="list-link">
        <div class="row space">
          <strong class="truncate">{a.behavior_question ? `"${a.behavior_question.question}"` : a.original_filename}</strong>
          <Chip tone={STATUS_TONE[a.status]}>{a.status === "running" ? `${Math.round(a.progress * 100)}% processed` : a.status}</Chip>
        </div>
        <div class="muted">{fmtDate(a.created_at)} - {subjectText(a, pets)}</div>
        {a.behavior_question && <div class="muted">Question - {a.behavior_question.status === "answered" ? "answered" : a.behavior_question.status === "failed" ? "could not be answered" : "in progress"}</div>}
        {s && !a.behavior_question && (
          <div class="muted">
            {s.tracks.length} animal track(s), {s.sound_events.length} pet sound(s)
            {a.decision_state && <> - {STATUS_LABEL[a.decision_state] ?? a.decision_state}</>}
          </div>
        )}
        {a.error && a.status === "failed" && <div class="error-text">{a.error.message}</div>}
      </a>
    </li>
  );
}
