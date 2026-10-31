import { useEffect, useMemo, useState } from "preact/hooks";
import { ErrorBanner, Field, InfoBanner, ProgressBar } from "../components/ui";
import { ApiError } from "../lib/api";
import { fmtBytes } from "../lib/format";
import { pendingStore, type BehaviorQuestionIn, type PendingUpload } from "../lib/idb";
import { localMedia } from "../lib/localMedia";
import { navigate } from "../lib/router";
import { newLocalId } from "../lib/uploader";
import { useApp } from "../state";

/** Guided journey: pet & concern -> recording -> send. The answer and its progress are shown on the analysis page. */

const DRAFT_KEY = "ask-draft-v1";
const PLACES = ["living room", "bedroom", "kitchen", "garden", "at night", "when alone", "with visitors"];

interface Draft { petId: string; species: "cat" | "dog" | ""; concern: string; scene: string; donate: boolean }
const EMPTY: Draft = { petId: "", species: "", concern: "", scene: "", donate: false };

function loadDraft(): Draft {
  try {
    return { ...EMPTY, ...JSON.parse(localStorage.getItem(DRAFT_KEY) ?? "{}") };
  } catch {
    return EMPTY;
  }
}

interface Picked { blob: Blob; name: string; duration: number | null; problem: string | null; warn: string | null }

function probeDuration(blob: Blob): Promise<number | null> {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    const url = URL.createObjectURL(blob);
    const done = (d: number | null) => { URL.revokeObjectURL(url); resolve(d); };
    v.preload = "metadata";
    v.onloadedmetadata = () => done(Number.isFinite(v.duration) ? v.duration : null);
    v.onerror = () => done(null);
    setTimeout(() => done(null), 8000);
    v.src = url;
  });
}

export function AskPage() {
  const { me, pets, caps, startPending, refreshPending, progress } = useApp();
  const [step, setStep] = useState<1 | 2>(1);
  const [d, setD] = useState<Draft>(loadDraft);
  const [clip, setClip] = useState<Picked | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const maxBytes = caps?.upload.max_bytes ?? 100 * 1024 * 1024;
  const maxS = caps?.upload.max_duration_s ?? 600;

  useEffect(() => { localStorage.setItem(DRAFT_KEY, JSON.stringify(d)); }, [d]);
  const set = (p: Partial<Draft>) => setD((x) => ({ ...x, ...p }));
  const pet = pets.find((p) => p.id === d.petId);
  const species = (pet?.species ?? d.species) as "cat" | "dog" | "";
  const step1ok = species !== "" && d.concern.trim().length >= 3;
  const previewUrl = useMemo(() => (clip ? URL.createObjectURL(clip.blob) : null), [clip]);
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  const pick = async (f: File | undefined) => {
    if (!f) return;
    const isMedia = f.type.startsWith("video/") || f.type.startsWith("audio/") || /\.(mp4|mov|m4v|webm|mkv|3gp)$/i.test(f.name);
    if (!isMedia) return setClip({ blob: f, name: f.name, duration: null, problem: "This file is not a video. Choose a video of your pet.", warn: null });
    if (f.size > maxBytes) return setClip({ blob: f, name: f.name, duration: null, problem: `The video is larger than ${fmtBytes(maxBytes)}. Record a shorter clip.`, warn: null });
    const dur = await probeDuration(f);
    let problem: string | null = null;
    let warn: string | null = null;
    if (dur !== null && dur < 3) problem = "The clip is shorter than 3 seconds. Record at least 10 seconds of the behaviour.";
    else if (dur !== null && dur > maxS) problem = `The clip is longer than ${Math.round(maxS / 60)} minutes. Trim it or record a shorter one.`;
    else if (dur !== null && dur > 60) warn = "Only the first minute is looked at closely. If the behaviour happens later, trim the clip.";
    else if (dur === null) warn = "The length could not be read here; the server will check the file.";
    setClip({ blob: f, name: f.name || `recording-${Date.now()}.mp4`, duration: dur, problem, warn });
  };

  const send = async () => {
    if (!me || !clip || clip.problem || !species) return;
    setError(null);
    const question: BehaviorQuestionIn = {
      species, scene: d.scene.trim(), question: d.concern.trim(), understanding: caps?.behavior_question?.cloud_available ? "cloud" : "local",
      strategy: "local_preprocessed", retain_for_research: d.donate,
      research_consent_version: d.donate ? (caps?.behavior_question?.research_consent_version ?? "rs-1") : null,
    };
    const item: PendingUpload = {
      localId: newLocalId(), userId: me.id, blob: clip.blob, filename: clip.name, sha256: null, uploadId: null,
      analysis: { subject_mode: pet ? "single_known" : "unknown", pet_ids: pet ? [pet.id] : [], place_id: null, modalities: ["audio", "video"],
        note: d.concern.trim().slice(0, 500), question },
      createdAt: Date.now(), state: "pending", lastError: null, analysisId: null,
    };
    await pendingStore.put(item).catch(() => undefined); // survives the app being closed; resumes from Home
    await refreshPending();
    setBusy(item.localId);
    const out = await startPending(item);
    setBusy(null);
    if ("id" in out) {
      localMedia.set(out.id, clip.blob);
      localStorage.removeItem(DRAFT_KEY);
      navigate(`/analyses/${out.id}`);
    } else if (out.error instanceof ApiError && out.error.isNetwork) {
      setError(new ApiError(0, "upload_paused", "The upload paused because the connection dropped.", "It is saved on this device and continues when you are back online (or tap Resume on Home)."));
    } else {
      setError(out.error);
    }
  };

  const pr = busy ? progress[busy] : null;
  return (
    <section class="journey">
      <p class="steps" aria-label={`Step ${step} of 2`}><span class={step === 1 ? "on" : ""}>1 Your pet</span> <span class={step === 2 ? "on" : ""}>2 Recording</span> <span>3 Answer</span></p>
      {step === 1 && (
        <div>
          <h1 tabIndex={-1}>What would you like to understand?</h1>
          <fieldset class="field">
            <legend>Your pet</legend>
            <div class="chips">
              {pets.map((p) => (
                <button type="button" class={`chip-btn ${d.petId === p.id ? "on" : ""}`} aria-pressed={d.petId === p.id} onClick={() => set({ petId: p.id, species: p.species as "cat" | "dog" })}>
                  {p.name} ({p.species})
                </button>
              ))}
              {(["cat", "dog"] as const).map((sp) => (
                <button type="button" class={`chip-btn ${!d.petId && d.species === sp ? "on" : ""}`} aria-pressed={!d.petId && d.species === sp} onClick={() => set({ petId: "", species: sp })}>
                  {sp === "cat" ? "A cat" : "A dog"}
                </button>
              ))}
            </div>
          </fieldset>
          <Field label="What did you notice?" id="concern" hint="In your own words, for example: my cat keeps turning in circles and bites its tail.">
            <textarea id="concern" maxLength={1000} rows={3} value={d.concern} onInput={(e) => set({ concern: (e.target as HTMLTextAreaElement).value })} />
          </Field>
          <Field label="Where or when does it happen? (optional)" id="scene">
            <input id="scene" maxLength={300} value={d.scene} onInput={(e) => set({ scene: (e.target as HTMLInputElement).value })} placeholder="e.g. open living room, in the evening" />
          </Field>
          <div class="chips">{PLACES.map((p) => <button type="button" class="chip-btn small" onClick={() => set({ scene: d.scene ? `${d.scene}, ${p}` : p })}>+ {p}</button>)}</div>
          <div class="row sticky-actions">
            <button type="button" class="btn btn-primary btn-wide" disabled={!step1ok} onClick={() => setStep(2)}>Next: add a recording</button>
          </div>
          {!step1ok && <p class="hint">Choose your pet and describe what you noticed to continue.</p>}
        </div>
      )}
      {step === 2 && (
        <div>
          <h1 tabIndex={-1}>Add a recording</h1>
          <div class="card guidance">
            <ul>
              <li>Record 10 to 60 seconds while it is happening.</li>
              <li>Keep the whole animal in view and hold the phone steady.</li>
              <li>Good light helps; keep the sound on.</li>
              <li>If it is rare, record it next time - or choose an earlier video.</li>
            </ul>
          </div>
          <div class="row wrap">
            <label class="btn btn-primary btn-wide">
              Record video
              <input type="file" accept="video/*" capture="environment" class="visually-hidden" onChange={(e) => void pick((e.target as HTMLInputElement).files?.[0])} />
            </label>
            <label class="btn btn-secondary btn-wide">
              Choose a video
              <input type="file" accept="video/*" class="visually-hidden" onChange={(e) => void pick((e.target as HTMLInputElement).files?.[0])} />
            </label>
          </div>
          <p class="hint">If the camera does not open, allow camera and microphone access for this app in your phone settings, or choose a video you already have.</p>
          {clip && (
            <div class="card">
              {previewUrl && !clip.problem && <video class="preview" src={previewUrl} controls playsInline muted />}
              <p><strong>{clip.name}</strong> - {fmtBytes(clip.blob.size)}{clip.duration ? `, ${Math.round(clip.duration)} s` : ""}</p>
              {clip.problem && <p class="field-error" role="alert">{clip.problem}</p>}
              {clip.warn && <InfoBanner tone="warn">{clip.warn}</InfoBanner>}
            </div>
          )}
          <details class="options">
            <summary>Options</summary>
            <label class="check"><input type="checkbox" checked={d.donate} onChange={(e) => set({ donate: (e.target as HTMLInputElement).checked })} />
              {" "}Also keep this recording as a research sample to help improve the app. You can delete it at any time in Settings.</label>
            {caps?.behavior_question?.cloud_available
              ? <p class="hint">Frames, pet close-ups and a sound summary of this clip are sent from your server to an AI service for the answer.</p>
              : <p class="hint">Answers are made on your server; the extended AI service is not available here.</p>}
          </details>
          <p class="hint">The video is sent to your server to answer your question.</p>
          <ErrorBanner error={error} />
          {pr && <ProgressBar value={pr.total ? pr.sent / pr.total : 0} label={pr.phase === "hashing" ? "Preparing the video" : pr.phase === "starting" ? "Starting" : "Uploading"} />}
          <div class="row sticky-actions">
            <button type="button" class="btn btn-secondary" onClick={() => setStep(1)}>Back</button>
            <button type="button" class="btn btn-primary btn-wide" disabled={!clip || !!clip.problem || busy !== null} onClick={send}>
              {busy ? "Uploading..." : "Send and get an answer"}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
