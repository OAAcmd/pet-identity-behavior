import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { ErrorBanner, Field, InfoBanner, ProgressBar } from "../components/ui";
import { ApiError } from "../lib/api";
import { localMedia } from "../lib/localMedia";
import { pendingStore, type PendingUpload } from "../lib/idb";
import { captureSupport, ClipRecorder, describeMediaError, grabPhoto, openStream, stopStream, type CaptureError, type CaptureMode } from "../lib/recorder";
import { navigate } from "../lib/router";
import type { SubjectMode } from "../lib/types";
import { fmtBytes } from "../lib/format";
import { newLocalId } from "../lib/uploader";
import { useApp } from "../state";

type Source = "video" | "audio" | "photo" | "file";

interface Clip {
  blob: Blob;
  filename: string;
  kind: "video" | "audio" | "image";
  note?: string;
}

export function CapturePage() {
  const { me, pets, places, caps, startPending, progress, refreshPending } = useApp();
  const [source, setSource] = useState<Source>("video");
  const [facing, setFacing] = useState<"environment" | "user">("environment");
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [capErr, setCapErr] = useState<CaptureError | null>(captureSupport());
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [limitS, setLimitS] = useState(60);
  const [clip, setClip] = useState<Clip | null>(null);
  const [subject, setSubject] = useState<SubjectMode>("unknown");
  const [petIds, setPetIds] = useState<string[]>([]);
  const [placeId, setPlaceId] = useState<string>("");
  const [note, setNote] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const recRef = useRef<ClipRecorder | null>(null);

  const maxS = Math.min(limitS, caps?.upload.max_duration_s ?? 600);
  const live = source === "video" || source === "audio" || source === "photo";

  // Open/close the live stream when the source changes.
  useEffect(() => {
    let cancelled = false;
    setClip(null);
    if (!live || captureSupport()) return;
    const mode: CaptureMode = source === "audio" ? "audio" : "video";
    openStream(mode, facing).then((s) => {
      if (cancelled) return stopStream(s);
      setStream(s);
      setCapErr(null);
    }).catch((e) => setCapErr(describeMediaError(e)));
    return () => {
      cancelled = true;
      setStream((s) => {
        stopStream(s);
        return null;
      });
    };
  }, [source, facing]);

  useEffect(() => {
    if (videoRef.current && stream && source !== "audio") {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(() => undefined);
    }
  }, [stream, source]);

  useEffect(() => {
    if (!recording) return;
    const t = setInterval(() => setElapsed(recRef.current?.elapsedMs ?? 0), 250);
    return () => clearInterval(t);
  }, [recording]);

  const startRec = async () => {
    if (!stream) return;
    setError(null);
    const r = new ClipRecorder(stream, source === "audio" ? "audio" : "video", maxS * 1000);
    recRef.current = r;
    setRecording(true);
    try {
      const rec = await r.start();
      const why = rec.stoppedReason === "hidden" ? "Recording stopped because the app went to the background."
        : rec.stoppedReason === "limit" ? `Recording stopped at the ${maxS}s limit.` : rec.stoppedReason === "track_ended" ? "The camera or microphone stopped." : undefined;
      setClip({ blob: rec.blob, filename: rec.filename, kind: source === "audio" ? "audio" : "video", note: why });
    } catch (e) {
      setCapErr(describeMediaError(e));
    } finally {
      setRecording(false);
    }
  };

  const takePhoto = async () => {
    if (!videoRef.current) return;
    const blob = await grabPhoto(videoRef.current);
    setClip({ blob, filename: `photo-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}.jpg`, kind: "image" });
  };

  const onFile = (e: Event) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    const ext = f.name.includes(".") ? `.${f.name.split(".").pop()!.toLowerCase()}` : "";
    const allowed = caps?.upload.allowed_extensions ?? [];
    if (allowed.length && !allowed.includes(ext)) {
      setError({ message: `Files of type '${ext || "unknown"}' are not supported.`, action: `Use one of: ${allowed.join(", ")}`, code: "unsupported_file_type" });
      return;
    }
    if (caps && f.size > caps.upload.max_bytes) {
      setError({ message: `The file is larger than ${fmtBytes(caps.upload.max_bytes)}.`, action: "Trim it or record a shorter clip.", code: "upload_too_large" });
      return;
    }
    setError(null);
    setClip({ blob: f, filename: f.name, kind: f.type.startsWith("image/") ? "image" : f.type.startsWith("audio/") ? "audio" : "video" });
  };

  const togglePet = (id: string) => setPetIds((ids) => (ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]));
  const subjectValid = subject === "unknown" ? true : subject === "single_known" ? petIds.length === 1 : petIds.length >= 2;

  const analyze = async () => {
    if (!clip || !me) return;
    const item: PendingUpload = {
      localId: newLocalId(), userId: me.id, blob: clip.blob, filename: clip.filename, sha256: null, uploadId: null,
      analysis: {
        subject_mode: subject, pet_ids: subject === "unknown" ? [] : petIds, place_id: placeId || null,
        modalities: clip.kind === "audio" ? ["audio"] : clip.kind === "image" ? ["video"] : ["audio", "video"], note,
      },
      createdAt: Date.now(), state: "pending", lastError: null, analysisId: null,
    };
    await pendingStore.put(item).catch(() => undefined);
    await refreshPending();
    setBusyId(item.localId);
    stopStream(stream);
    const out = await startPending(item);
    setBusyId(null);
    if ("id" in out) {
      localMedia.set(out.id, clip.blob);
      navigate(`/analyses/${out.id}`);
    } else if (out.error instanceof ApiError && !out.error.isNetwork && out.error.status !== 429) {
      setError(out.error); // server explains what to do (e.g. choose a different file)
    } else {
      const saved = await pendingStore.get(item.localId).catch(() => undefined);
      setError({ message: saved?.lastError ?? "The upload did not finish.", action: "It is saved on this device and resumes automatically when you are back online (or tap Resume on Home).",
        code: saved?.state ?? "upload_paused" });
    }
  };

  const pr = busyId ? progress[busyId] : undefined;
  const clipUrl = useMemo(() => (clip ? URL.createObjectURL(clip.blob) : null), [clip]);
  useEffect(() => () => { if (clipUrl) URL.revokeObjectURL(clipUrl); }, [clipUrl]);

  return (
    <section>
      <h1 tabIndex={-1}>Record or upload</h1>
      <div class="segmented" role="tablist" aria-label="Source">
        {(["video", "audio", "photo", "file"] as Source[]).map((s) => (
          <button key={s} type="button" role="tab" aria-selected={source === s} class={source === s ? "seg active" : "seg"} disabled={recording}
            onClick={() => setSource(s)}>
            {s === "video" ? "Video" : s === "audio" ? "Sound only" : s === "photo" ? "Photo" : "Upload file"}
          </button>
        ))}
      </div>

      {live && capErr && (
        <div class="banner banner-warn" role="alert">
          <p class="banner-title">{capErr.message}</p>
          <p>{capErr.action}</p>
          <button type="button" class="btn btn-secondary" onClick={() => setSource("file")}>Upload a file instead</button>
        </div>
      )}

      {live && !capErr && !clip && (
        <div class="card capture">
          {source !== "audio" ? (
            <video ref={videoRef} class="preview" muted playsInline autoPlay aria-label="Camera preview" />
          ) : (
            <div class="audio-only" aria-hidden="true">🎙</div>
          )}
          {source !== "photo" && (
            <div class="row">
              <label for="limit">Max length</label>
              <select id="limit" value={String(limitS)} disabled={recording} onChange={(e) => setLimitS(Number((e.target as HTMLSelectElement).value))}>
                {[15, 30, 60, 120, 300].map((v) => <option key={v} value={v}>{v} s</option>)}
              </select>
            </div>
          )}
          <div class="row center">
            {source === "photo" ? (
              <button type="button" class="btn btn-primary btn-round" onClick={takePhoto} disabled={!stream}>Take photo</button>
            ) : !recording ? (
              <button type="button" class="btn btn-record" onClick={startRec} disabled={!stream} aria-label="Start recording">● Record</button>
            ) : (
              <button type="button" class="btn btn-stop" onClick={() => recRef.current?.stop("user")} aria-label="Stop recording">■ Stop ({(elapsed / 1000).toFixed(0)} / {maxS} s)</button>
            )}
            {source !== "audio" && !recording && (
              <button type="button" class="btn btn-secondary" onClick={() => setFacing((f) => (f === "environment" ? "user" : "environment"))}>Switch camera</button>
            )}
          </div>
          <p class="hint">Tips: keep the whole animal in view, avoid TV or music in the background, and record 10-60 seconds.</p>
        </div>
      )}

      {source === "file" && !clip && (
        <div class="card">
          <Field label="Choose a video, sound recording or photo" id="file" hint={caps ? `Up to ${fmtBytes(caps.upload.max_bytes)}; ${caps.upload.allowed_extensions.join(" ")}` : undefined}>
            <input id="file" type="file" accept="video/*,audio/*,image/jpeg,image/png" onChange={onFile} />
          </Field>
        </div>
      )}

      <ErrorBanner error={error} />

      {clip && (
        <div class="card">
          <h2>Review</h2>
          {clip.note && <InfoBanner tone="warn">{clip.note}</InfoBanner>}
          {clip.kind === "video" && clipUrl && <video class="preview" src={clipUrl} controls playsInline aria-label="Recorded clip" />}
          {clip.kind === "audio" && clipUrl && <audio src={clipUrl} controls aria-label="Recorded sound" />}
          {clip.kind === "image" && clipUrl && <img class="preview" src={clipUrl} alt="Captured photo" />}
          <p class="muted">{clip.filename} - {fmtBytes(clip.blob.size)}</p>

          <fieldset class="field">
            <legend>Which animal is this?</legend>
            <label class="radio"><input type="radio" name="subject" checked={subject === "unknown"} onChange={() => setSubject("unknown")} /> Not sure / not listed</label>
            <label class="radio"><input type="radio" name="subject" checked={subject === "single_known"} onChange={() => setSubject("single_known")} disabled={!pets.length} /> One of my pets</label>
            <label class="radio"><input type="radio" name="subject" checked={subject === "multiple_known"} onChange={() => setSubject("multiple_known")} disabled={pets.length < 2} /> Several of my pets</label>
            {subject !== "unknown" && (
              <div class="checks">
                {pets.map((p) => (
                  <label key={p.id} class="check">
                    <input type={subject === "single_known" ? "radio" : "checkbox"} name="pet" checked={petIds.includes(p.id)}
                      onChange={() => (subject === "single_known" ? setPetIds([p.id]) : togglePet(p.id))} />
                    {p.name} ({p.species})
                  </label>
                ))}
              </div>
            )}
            <p class="hint">This is your note, not a recognition result: the app does not identify individual pets.</p>
            {!subjectValid && <p class="field-error" role="alert">{subject === "single_known" ? "Select one pet." : "Select at least two pets."}</p>}
          </fieldset>

          <Field label="Place (optional)" id="place" hint="Places mark where the bowl, door or bed are in this camera view.">
            <select id="place" value={placeId} onChange={(e) => setPlaceId((e.target as HTMLSelectElement).value)}>
              <option value="">No place</option>
              {places.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
          </Field>
          <Field label="Note (optional)" id="note">
            <input id="note" maxLength={500} value={note} onInput={(e) => setNote((e.target as HTMLInputElement).value)} placeholder="e.g. before dinner" />
          </Field>

          {pr && <ProgressBar value={pr.total ? pr.sent / pr.total : 0} label={pr.phase === "hashing" ? "Preparing" : pr.phase === "starting" ? "Starting analysis" : "Uploading"} />}
          {pr?.message && <p class="muted" role="status">{pr.message}</p>}
          <div class="row">
            <button type="button" class="btn btn-primary" onClick={analyze} disabled={!subjectValid || busyId !== null}>
              {busyId ? "Uploading..." : "Analyze"}
            </button>
            <button type="button" class="btn btn-secondary" disabled={busyId !== null} onClick={() => { setClip(null); if (source !== "file") setSource(source); }}>
              Discard
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
