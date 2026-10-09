// Browser media decoding for on-device analysis.
// Differences from the Python reference (FFmpeg), measured by engine parity:
// * audio is decoded and resampled to 16 kHz by the browser (decodeAudioData
//   on a 16 kHz OfflineAudioContext); the native sample rate is not exposed,
//   so the "bandwidth limited" check is not available on-device;
// * video frames are taken by seeking an HTMLVideoElement to a 1/sample_fps
//   grid and drawing to a canvas (browser scaler instead of FFmpeg "area");
//   timestamps are the element's currentTime after the seek.
import { fromRGBA, type RGBImage } from "./imageops";

export interface Probe {
  kind: "video" | "audio" | "image";
  mime: string;
  duration_s: number | null;
  width: number;
  height: number;
}

const IMAGE_TYPES = /^image\/(jpeg|png|webp)$/;

function once(el: HTMLMediaElement, ok: string, timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => { cleanup(); reject(new Error(`media ${ok} timeout`)); }, timeoutMs);
    const onOk = () => { cleanup(); resolve(); };
    const onErr = () => { cleanup(); reject(new Error(el.error?.message || "media could not be decoded")); };
    const cleanup = () => { clearTimeout(t); el.removeEventListener(ok, onOk); el.removeEventListener("error", onErr); };
    el.addEventListener(ok, onOk);
    el.addEventListener("error", onErr);
  });
}

export async function probe(file: Blob, name = ""): Promise<Probe> {
  const mime = file.type || (name.match(/\.(jpe?g|png|webp)$/i) ? "image/jpeg" : "");
  if (IMAGE_TYPES.test(mime)) {
    const bmp = await createImageBitmap(file);
    const p: Probe = { kind: "image", mime, duration_s: null, width: bmp.width, height: bmp.height };
    bmp.close();
    return p;
  }
  const url = URL.createObjectURL(file);
  const v = document.createElement("video");
  v.muted = true;
  v.playsInline = true;
  v.preload = "metadata";
  v.src = url;
  try {
    await once(v, "loadedmetadata", 20000);
    const duration = Number.isFinite(v.duration) ? v.duration : null;
    return { kind: v.videoWidth > 0 ? "video" : "audio", mime: mime || "application/octet-stream", duration_s: duration, width: v.videoWidth, height: v.videoHeight };
  } finally {
    v.removeAttribute("src");
    v.load();
    URL.revokeObjectURL(url);
  }
}

export interface DecodedAudio { state: "present" | "absent" | "corrupt" | "silent"; samples: Float32Array; sample_rate: number; channels: number; warning?: string }

export async function decodeAudio16k(file: Blob, expectAudio: boolean): Promise<DecodedAudio> {
  const buf = await file.arrayBuffer();
  try {
    const ctx = new OfflineAudioContext(1, 16000, 16000);
    const ab = await ctx.decodeAudioData(buf);
    const n = ab.length;
    const mono = new Float32Array(n);
    for (let c = 0; c < ab.numberOfChannels; c++) {
      const d = ab.getChannelData(c);
      for (let i = 0; i < n; i++) mono[i] += d[i] / ab.numberOfChannels;
    }
    for (let i = 0; i < n; i++) if (!Number.isFinite(mono[i])) mono[i] = 0;
    return { state: n ? "present" : "corrupt", samples: mono, sample_rate: ab.sampleRate, channels: ab.numberOfChannels };
  } catch (e) {
    return { state: expectAudio ? "corrupt" : "absent", samples: new Float32Array(0), sample_rate: 16000, channels: 0, warning: String(e) };
  }
}

/** Output size as in ingest/media.py _output_size (max side, even dimensions). */
export function outputSize(w: number, h: number, maxSide: number): [number, number] {
  const scale = Math.max(w, h) > 0 ? Math.min(1, maxSide / Math.max(w, h)) : 1;
  return [Math.max(2, Math.round((w * scale) / 2) * 2), Math.max(2, Math.round((h * scale) / 2) * 2)];
}

export interface Frame { index: number; t_s: number; rgb: RGBImage }

export async function* sampleFrames(file: Blob, p: Probe, sampleFps: number, maxSide: number, maxFrames: number,
  shouldStop?: () => boolean): AsyncGenerator<Frame> {
  const [ow, oh] = outputSize(p.width, p.height, maxSide);
  const canvas = document.createElement("canvas");
  canvas.width = ow;
  canvas.height = oh;
  const g = canvas.getContext("2d", { willReadFrequently: true })!;
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = "high";
  if (p.kind === "image") {
    const bmp = await createImageBitmap(file);
    g.drawImage(bmp, 0, 0, ow, oh);
    bmp.close();
    yield { index: 0, t_s: 0, rgb: fromRGBA(g.getImageData(0, 0, ow, oh).data, ow, oh) };
    return;
  }
  const url = URL.createObjectURL(file);
  const v = document.createElement("video");
  v.muted = true;
  v.playsInline = true;
  v.preload = "auto";
  v.src = url;
  try {
    await once(v, "loadeddata", 20000);
    const dur = Number.isFinite(v.duration) ? v.duration : (p.duration_s ?? 0);
    const interval = 1 / sampleFps;
    let index = 0;
    for (let t = 0; t <= dur + 1e-6 && index < maxFrames; t += interval) {
      if (shouldStop?.()) return;
      const target = Math.min(t, Math.max(0, dur - 0.001));
      if (Math.abs(v.currentTime - target) > 1e-4 || index > 0) {
        v.currentTime = target;
        await once(v, "seeked", 15000);
      }
      g.drawImage(v, 0, 0, ow, oh);
      yield { index, t_s: v.currentTime, rgb: fromRGBA(g.getImageData(0, 0, ow, oh).data, ow, oh) };
      index++;
      if (target >= dur - 0.001) break;
    }
  } finally {
    v.removeAttribute("src");
    v.load();
    URL.revokeObjectURL(url);
  }
}

export async function sha256Hex(file: Blob): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(d)).map((b) => b.toString(16).padStart(2, "0")).join("");
}
