import { useEffect, useMemo, useState } from "preact/hooks";
import { ResultView, type FeedbackSink } from "../components/ResultView";
import { ConfirmDialog, ErrorBanner, InfoBanner, Spinner } from "../components/ui";
import { fmtDate } from "../lib/format";
import { localAnalyses, localFeedback, type LocalAnalysis } from "../lib/localstore";
import { navigate } from "../lib/router";
import type { Feedback } from "../lib/types";
import { useApp } from "../state";

export function LocalAnalysisPage({ id }: { id: string }) {
  const { pets, places, toast } = useApp();
  const [a, setA] = useState<LocalAnalysis | null | undefined>(undefined);
  const [error, setError] = useState<unknown>(null);
  const [confirm, setConfirm] = useState(false);

  useEffect(() => {
    localAnalyses.get(id).then((x) => setA(x ?? null)).catch(setError);
  }, [id]);

  const mediaUrl = useMemo(() => (a?.media ? URL.createObjectURL(a.media) : null), [a?.media]);
  useEffect(() => () => { if (mediaUrl) URL.revokeObjectURL(mediaUrl); }, [mediaUrl]);

  const sink = useMemo<FeedbackSink>(() => ({
    where: "device",
    list: () => localFeedback.list(id),
    add: async (body) => {
      const f: Feedback = { id: `fb-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`, analysis_id: id, target_type: body.target_type,
        target_id: body.target_id ?? null, verdict: body.verdict, actual_behavior: body.actual_behavior ?? null, note: body.note ?? "",
        consent_research_use: body.consent_research_use, created_at: new Date().toISOString() };
      await localFeedback.add(f);
    },
  }), [id]);

  if (error) return <section><h1 tabIndex={-1}>Analysis</h1><ErrorBanner error={error} /></section>;
  if (a === undefined) return <Spinner label="Loading..." />;
  if (a === null) {
    return (
      <section>
        <h1 tabIndex={-1}>Analysis not found</h1>
        <p>This on-device analysis is not stored on this device (it may have been deleted, or it was made on another device).</p>
        <a class="btn btn-primary" href="#/history">Back to history</a>
      </section>
    );
  }
  const petNames = a.pet_ids.map((pid) => pets.find((p) => p.id === pid)?.name ?? "a pet").join(", ");
  const place = places.find((p) => p.id === a.place_id);
  return (
    <section>
      <h1 tabIndex={-1}>{a.filename}</h1>
      <p class="muted">{fmtDate(new Date(a.createdAt).toISOString())}{place ? ` - ${place.name}` : ""}{a.subject_mode !== "unknown" ? ` - You said: ${petNames}` : ""}</p>
      <InfoBanner tone="info">Analysed on this device. The recording and the result stay on this device{a.synced_at ? "; the result (not the recording) was synced to your account" : ""}.</InfoBanner>
      <ResultView feedback={sink} result={a.result} mediaUrl={mediaUrl} mediaKind={a.kind} />
      <div class="card">
        <h2>On this device</h2>
        <p class="muted">Model bundle {a.bundle_version}. {a.media ? "The recording is kept for playback." : "The recording has been removed; the result remains."}</p>
        <div class="row wrap">
          {a.media && (
            <button type="button" class="btn btn-secondary" onClick={async () => { await localAnalyses.removeMedia(a.id); setA({ ...a, media: null }); toast("Recording removed from this device."); }}>
              Remove recording, keep result
            </button>
          )}
          <button type="button" class="btn btn-danger" onClick={() => setConfirm(true)}>Delete analysis</button>
        </div>
      </div>
      <ConfirmDialog open={confirm} title="Delete this analysis?" body="The result, the recording and your feedback on this device are deleted. This cannot be undone."
        confirmLabel="Delete" onCancel={() => setConfirm(false)}
        onConfirm={async () => { setConfirm(false); await localAnalyses.remove(a.id); toast("Deleted."); navigate("/history"); }} />
    </section>
  );
}
