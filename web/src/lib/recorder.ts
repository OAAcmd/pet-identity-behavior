// Camera / microphone capture with MediaRecorder.
//
// Browser realities handled here:
// * getUserMedia needs a secure context (HTTPS or localhost);
// * Chrome/Firefox/Android record WebM (VP8/VP9 + Opus); iOS/macOS Safari
//   records MP4 (H.264 + AAC) - the container is negotiated, never assumed;
// * permission prompts can be denied or dismissed; devices can be missing or busy;
// * iOS may stop capture when the page is hidden, so recording stops on
//   `visibilitychange` and keeps what was captured.

export type CaptureMode = "video" | "audio";

// Preference order: WebM where offered (Chrome, Firefox, Android); MP4 (H.264/AAC) for Safari/iOS.
const VIDEO_TYPES = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm", "video/mp4;codecs=avc1.42E01E,mp4a.40.2", "video/mp4"];
const AUDIO_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4;codecs=mp4a.40.2", "audio/mp4"];

export function pickMimeType(mode: CaptureMode, isSupported: (t: string) => boolean): string | null {
  for (const t of mode === "video" ? VIDEO_TYPES : AUDIO_TYPES) if (isSupported(t)) return t;
  return null;
}

export function extensionFor(mime: string, mode: CaptureMode): string {
  const base = mime.split(";")[0].trim();
  if (base === "video/mp4") return ".mp4";
  if (base === "audio/mp4") return ".m4a";
  if (base === "audio/ogg") return ".ogg";
  if (base.endsWith("/webm")) return ".webm";
  return mode === "video" ? ".mp4" : ".m4a";
}

export interface CaptureError {
  code: "insecure_context" | "unsupported" | "permission_denied" | "no_device" | "device_busy" | "constraints" | "unknown";
  message: string;
  action: string;
}

export function describeMediaError(e: unknown): CaptureError {
  const name = (e as { name?: string })?.name ?? "";
  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return { code: "permission_denied", message: "Camera or microphone access was blocked.",
        action: "Allow access in the browser's site settings (address bar lock icon), then try again. You can also upload an existing file." };
    case "NotFoundError":
    case "DevicesNotFoundError":
      return { code: "no_device", message: "No camera or microphone was found.", action: "Connect a device or upload an existing recording instead." };
    case "NotReadableError":
    case "TrackStartError":
      return { code: "device_busy", message: "The camera or microphone is in use by another app.", action: "Close other apps using it, then try again." };
    case "OverconstrainedError":
      return { code: "constraints", message: "The device does not support the requested settings.", action: "Try the other camera or audio-only recording." };
    case "SecurityError":
      return { code: "insecure_context", message: "Recording requires a secure (HTTPS) connection.", action: "Open the app over HTTPS." };
    default:
      return { code: "unknown", message: "Recording could not start.", action: "Try again, or upload an existing file." };
  }
}

export function captureSupport(): CaptureError | null {
  if (typeof window !== "undefined" && window.isSecureContext === false) {
    return { code: "insecure_context", message: "Recording requires a secure (HTTPS) connection.", action: "Open the app over HTTPS, or upload a file." };
  }
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    return { code: "unsupported", message: "This browser cannot record audio or video here.", action: "Upload an existing recording instead." };
  }
  return null;
}

export async function openStream(mode: CaptureMode, facing: "environment" | "user"): Promise<MediaStream> {
  const audio = { echoCancellation: false, noiseSuppression: false, autoGainControl: false };
  const constraints: MediaStreamConstraints =
    mode === "video" ? { audio, video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } } } : { audio };
  try {
    return await navigator.mediaDevices.getUserMedia(constraints);
  } catch (e) {
    if ((e as { name?: string })?.name === "OverconstrainedError" && mode === "video") {
      return navigator.mediaDevices.getUserMedia({ audio, video: true });
    }
    throw e;
  }
}

export function stopStream(stream: MediaStream | null): void {
  stream?.getTracks().forEach((t) => t.stop());
}

export interface Recording {
  blob: Blob;
  mimeType: string;
  filename: string;
  durationMs: number;
  stoppedReason: "user" | "limit" | "hidden" | "track_ended";
}

export class ClipRecorder {
  private rec: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private started = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private reason: Recording["stoppedReason"] = "user";
  private onHidden = () => {
    if (document.visibilityState === "hidden") this.stop("hidden");
  };

  constructor(private stream: MediaStream, private mode: CaptureMode, private maxMs: number) {}

  start(): Promise<Recording> {
    const mime = pickMimeType(this.mode, (t) => MediaRecorder.isTypeSupported(t));
    this.rec = mime ? new MediaRecorder(this.stream, { mimeType: mime, videoBitsPerSecond: 2_500_000, audioBitsPerSecond: 128_000 }) : new MediaRecorder(this.stream);
    const actualMime = this.rec.mimeType || mime || (this.mode === "video" ? "video/webm" : "audio/webm");
    this.chunks = [];
    return new Promise<Recording>((resolve, reject) => {
      const r = this.rec!;
      r.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) this.chunks.push(ev.data);
      };
      r.onerror = (ev) => reject((ev as unknown as { error?: Error }).error ?? new Error("recorder error"));
      r.onstop = () => {
        document.removeEventListener("visibilitychange", this.onHidden);
        if (this.timer) clearTimeout(this.timer);
        const blob = new Blob(this.chunks, { type: actualMime.split(";")[0] });
        const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
        resolve({ blob, mimeType: actualMime, filename: `recording-${stamp}${extensionFor(actualMime, this.mode)}`, durationMs: Date.now() - this.started,
          stoppedReason: this.reason });
      };
      this.stream.getTracks().forEach((t) => t.addEventListener("ended", () => this.stop("track_ended")));
      document.addEventListener("visibilitychange", this.onHidden);
      this.started = Date.now();
      r.start(1000); // timeslice: data is flushed every second, so an interrupted capture keeps most content
      this.timer = setTimeout(() => this.stop("limit"), this.maxMs);
    });
  }

  stop(reason: Recording["stoppedReason"] = "user"): void {
    if (this.rec && this.rec.state !== "inactive") {
      this.reason = reason;
      this.rec.stop();
    }
  }

  get elapsedMs(): number {
    return this.started ? Date.now() - this.started : 0;
  }
}

export async function grabPhoto(video: HTMLVideoElement): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d")!.drawImage(video, 0, 0);
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("photo failed"))), "image/jpeg", 0.9));
}
