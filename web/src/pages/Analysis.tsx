import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { subjectText } from "../components/AnalysisListItem";
import { BehaviorAnswerView } from "../components/BehaviorAnswerView";
import { ResultView, serverFeedback } from "../components/ResultView";
import { ConfirmDialog, ErrorBanner, InfoBanner, ProgressBar, Spinner } from "../components/ui";
import { ApiError, api } from "../lib/api";
import { fmtDate, isTerminal } from "../lib/format";
import { localMedia } from "../lib/localMedia";
import { navigate } from "../lib/router";
import type { Analysis } from "../lib/types";
import { useApp } from "../state";

const STAGE_TEXT: Record<string, string> = {
  queued: "Waiting in the queue", starting: "Starting", probe: "Checking the file", audio: "Listening for sounds", video: "Looking for animals",
  fusion: "Combining observations", done: "Done",
};

export function AnalysisPage({ id }: { id: string }) {
  const { pets, toast } = useApp();
  const [a, setA] = useState<Analysis | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [confirm, setConfirm] = useState<null | "delete" | "media">(null);
  const failures = useRef(0);
  const lastStatus = useRef(0);

  const load = async () => {
    try {
      const next = await api.get<Analysis>(`/analyses/${id}`);
      setA(next);
      setError(null);
      failures.current = 0;
      return next;
    } catch (e) {
      failures.current += 1;
      lastStatus.current = e instanceof ApiError ? e.status : 0;
      setError(e);
      return null;
    }
  };

  useEffect(() => {
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      const next = await load();
      if (stop) return;
      if (next && isTerminal(next.status)) return;
      if (!next && (lastStatus.current === 404 || lastStatus.current === 401)) return;
      timer = setTimeout(tick, Math.min(10000, 1500 * 2 ** Math.min(failures.current, 3)));
    };
    tick();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [id]);

  const localBlob = localMedia.get(id);
  const localUrl = useMemo(() => (localBlob ? URL.createObjectURL(localBlob) : null), [localBlob]);
  useEffect(() => () => { if (localUrl) URL.revokeObjectURL(localUrl); }, [localUrl]);

  const act = async (fn: () => Promise<unknown>, msg: string) => {
    try {
      await fn();
      toast(msg);
      await load();
    } catch (e) {
      setError(e);
    }
  };

  if (!a && error) {
    return (
      <section>
        <h1 tabIndex={-1}>Analysis</h1>
        <ErrorBanner error={error} onRetry={load} />
        <a href="#/history">Back to history</a>
      </section>
    );
  }
  if (!a) return <Spinner label="Loading analysis..." />;

  const mediaUrl = localUrl ?? (a.media_available ? `/api/v1/analyses/${a.id}/media` : null);
  const mediaKind = a.modalities.includes("video") ? (a.original_filename.match(/\.(jpe?g|png)$/i) ? "image" : "video") : "audio";

  return (
    <section>
      <h1 tabIndex={-1} class="truncate">{a.original_filename}</h1>
      <p class="muted">{fmtDate(a.created_at)} - {subjectText(a, pets)}{a.note ? ` - "${a.note}"` : ""}</p>
      <ErrorBanner error={error} onRetry={load} />

      {(a.status === "queued" || a.status === "running") && (
        <div class="card" aria-live="polite">
          <ProgressBar value={a.progress} label={a.cancel_requested ? "Cancelling..." : STAGE_TEXT[a.stage ?? "queued"] ?? a.stage ?? "Processing"} />
          {a.error?.code === "worker_lost" && <p class="muted">A previous attempt was interrupted; retrying automatically.</p>}
          {a.error?.code === "internal_error" && <p class="muted">A temporary problem occurred; retrying automatically (attempt {a.attempts} of {a.max_attempts}).</p>}
          <button type="button" class="btn btn-secondary" disabled={a.cancel_requested}
            onClick={() => act(() => api.post(`/analyses/${a.id}/cancel`), "Cancellation requested.")}>Cancel</button>
        </div>
      )}

      {a.status === "failed" && a.error && (
        <div class="banner banner-error" role="alert">
          <p class="banner-title">{a.error.message}</p>
          {a.error.action && <p>{a.error.action}</p>}
          <p class="banner-code">Code: {a.error.code}</p>
          {["internal_error", "worker_lost", "analysis_timeout"].includes(a.error.code) && (
            <button type="button" class="btn btn-secondary" onClick={() => act(() => api.post(`/analyses/${a.id}/retry`), "Queued again.")}>Retry</button>
          )}
        </div>
      )}

      {a.status === "cancelled" && (
        <InfoBanner tone="warn">
          This analysis was cancelled.{" "}
          {a.media_available && <button type="button" class="btn btn-link" onClick={() => act(() => api.post(`/analyses/${a.id}/retry`), "Queued again.")}>Run it again</button>}
        </InfoBanner>
      )}

      {a.behavior_question && <BehaviorAnswerView q={a.behavior_question} />}
      {a.status === "succeeded" && a.result && <ResultView feedback={serverFeedback(a.id)} result={a.result} mediaUrl={mediaUrl} mediaKind={mediaKind} />}

      <div class="card">
        <h2>Manage</h2>
        <p class="muted">
          {a.media_available ? "The recording is stored on the server until it expires or you delete it." : "The recording is not stored on the server."}
        </p>
        <div class="row wrap">
          {a.media_available && <button type="button" class="btn btn-secondary" onClick={() => setConfirm("media")}>Delete recording only</button>}
          <button type="button" class="btn btn-danger" onClick={() => setConfirm("delete")}>Delete analysis</button>
        </div>
      </div>

      <ConfirmDialog open={confirm === "delete"} title="Delete this analysis?" danger confirmLabel="Delete"
        body={<p>The result, your feedback and any stored recording will be permanently removed.</p>}
        onCancel={() => setConfirm(null)}
        onConfirm={async () => {
          setConfirm(null);
          try {
            await api.del(`/analyses/${a.id}`);
            localMedia.delete(a.id);
            toast("Analysis deleted.");
            navigate("/history");
          } catch (e) {
            setError(e);
          }
        }} />
      <ConfirmDialog open={confirm === "media"} title="Delete the stored recording?" danger confirmLabel="Delete recording"
        body={<p>The observations stay; the recording can no longer be replayed from the server.</p>}
        onCancel={() => setConfirm(null)}
        onConfirm={() => { setConfirm(null); void act(() => api.del(`/analyses/${a.id}/media`), "Recording deleted."); }} />
    </section>
  );
}
