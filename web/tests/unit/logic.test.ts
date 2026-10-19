// Unit tests (synthetic inputs / mocked transport) for client-side logic.
import { describe, expect, it } from "vitest";
import { ApiError } from "../../src/lib/api";
import { fmtTime, shownSubtypes, supportWord } from "../../src/lib/format";
import { describeMediaError, extensionFor, pickMimeType } from "../../src/lib/recorder";
import { parseHash } from "../../src/lib/router";
import type { Upload } from "../../src/lib/types";
import { sendResumable, type UploadDeps } from "../../src/lib/uploader";
import { rectsToZones } from "../../src/pages/Places";

function fakeServer(size: number, opts: { failOnce?: number[]; offlineFor?: number; maxChunk?: number } = {}) {
  let received = 0;
  const calls: string[] = [];
  let offline = opts.offlineFor ?? 0;
  const failOnce = new Set(opts.failOnce ?? []);
  const up = (): Upload => ({ id: "u1", filename: "f.webm", size_bytes: size, received_bytes: received, status: received === size ? "complete" : "receiving",
    max_chunk_bytes: opts.maxChunk ?? 1 << 30, expires_at: 0 });
  const deps: UploadDeps = {
    createUpload: async () => { calls.push("create"); return up(); },
    getUpload: async () => { calls.push(`get@${received}`); return up(); },
    putChunk: async (_id, offset, chunk) => {
      calls.push(`put@${offset}`);
      if (offset !== received) throw new ApiError(409, "upload_offset_mismatch", "mismatch", null, { expected_offset: received });
      if (failOnce.has(offset)) {
        failOnce.delete(offset);
        received = offset + chunk.size; // the bytes arrived but the response was lost
        throw new ApiError(0, "network_error", "lost");
      }
      received = offset + chunk.size;
      return up();
    },
    online: () => offline-- <= 0,
    wait: async () => undefined,
  };
  return { deps, calls, get received() { return received; } };
}

describe("resumable upload", () => {
  const blob = new Blob([new Uint8Array(5 * 1024 * 1024 + 123)]);

  it("uploads in chunks and completes", async () => {
    const s = fakeServer(blob.size);
    const up = await sendResumable({ blob, filename: "f.webm", sha256: null, uploadId: null }, () => undefined, () => undefined, s.deps);
    expect(up.status).toBe("complete");
    expect(s.calls.filter((c) => c.startsWith("put")).length).toBe(3);
  });

  it("resumes from the server offset after a lost response", async () => {
    const s = fakeServer(blob.size, { failOnce: [2 * 1024 * 1024] });
    const up = await sendResumable({ blob, filename: "f.webm", sha256: null, uploadId: null }, () => undefined, () => undefined, s.deps);
    expect(up.status).toBe("complete");
    expect(s.calls).toContain(`get@${4 * 1024 * 1024}`); // asked the server where to continue
    expect(s.calls.filter((c) => c === `put@${2 * 1024 * 1024}`).length).toBe(1); // chunk not re-sent
  });

  it("waits while offline and continues an existing upload id", async () => {
    const s = fakeServer(blob.size, { offlineFor: 3 });
    const phases: string[] = [];
    const up = await sendResumable({ blob, filename: "f.webm", sha256: null, uploadId: "u1" }, () => undefined, (p) => phases.push(p.phase), s.deps);
    expect(up.status).toBe("complete");
    expect(phases).toContain("waiting_for_network");
    expect(s.calls[0]).toBe("get@0"); // resumed, did not create a new upload
  });

  it("respects a smaller server chunk limit", async () => {
    const s = fakeServer(blob.size, { maxChunk: 1024 * 1024 });
    await sendResumable({ blob, filename: "f.webm", sha256: null, uploadId: null }, () => undefined, () => undefined, s.deps);
    expect(s.calls.filter((c) => c.startsWith("put")).length).toBe(6);
  });

  it("gives up on non-retryable errors", async () => {
    const s = fakeServer(blob.size);
    s.deps.putChunk = async () => { throw new ApiError(413, "chunk_too_large", "too big"); };
    await expect(sendResumable({ blob, filename: "f.webm", sha256: null, uploadId: null }, () => undefined, () => undefined, s.deps)).rejects.toThrow("too big");
  });
});

describe("recording formats", () => {
  it("prefers WebM, falls back to MP4 for Safari", () => {
    expect(pickMimeType("video", (t) => t.startsWith("video/webm"))).toBe("video/webm;codecs=vp9,opus");
    expect(pickMimeType("video", (t) => t.startsWith("video/mp4"))).toBe("video/mp4;codecs=avc1.42E01E,mp4a.40.2");
    expect(pickMimeType("audio", (t) => t === "audio/mp4")).toBe("audio/mp4");
    expect(pickMimeType("audio", () => false)).toBeNull();
    expect(extensionFor("video/mp4;codecs=avc1", "video")).toBe(".mp4");
    expect(extensionFor("audio/mp4", "audio")).toBe(".m4a");
    expect(extensionFor("audio/webm;codecs=opus", "audio")).toBe(".webm");
  });

  it("maps getUserMedia errors to actionable messages", () => {
    expect(describeMediaError({ name: "NotAllowedError" }).code).toBe("permission_denied");
    expect(describeMediaError({ name: "NotFoundError" }).code).toBe("no_device");
    expect(describeMediaError({ name: "NotReadableError" }).action).toMatch(/Close other apps/);
    expect(describeMediaError(new Error("x")).code).toBe("unknown");
  });
});

describe("display rules", () => {
  it("never renders scores as percentages", () => {
    for (const v of [0.3, 0.55, 0.65, 0.9, null]) expect(supportWord(v)).not.toMatch(/%|percent|probab/i);
    expect(fmtTime(75.25)).toBe("1:15.3");
  });

  it("routes hash paths", () => {
    expect(parseHash("#/analyses/abc123")).toEqual({ name: "analysis", params: { id: "abc123" } });
    expect(parseHash("").name).toBe("home");
    expect(parseHash("#/nope").name).toBe("not_found");
  });

  it("converts drawn rectangles to valid normalized zones", () => {
    const z = rectsToZones([{ tag: "food_area", label: "Bowl", x1: 0.8, y1: 0.9, x2: 0.5, y2: 1.2 }]);
    expect(z.zones[0].polygon).toEqual([[0.5, 0.9], [0.8, 0.9], [0.8, 1], [0.5, 1]]);
    expect(z.schema).toBe("petperception.zones/1");
  });
});

describe("shownSubtypes", () => {
  it("lists only validated subtypes, strongest first", () => {
    const s = { subtypes: { purr: 0.9, meow: 0.4, bark: 0.6, hiss: 0.8 }, subtype_validation: { purr: "failed", meow: "passed", bark: "passed", hiss: "not_evaluable" } };
    expect(shownSubtypes(s)).toEqual(["bark", "meow"]);
  });
  it("shows nothing for results without validation statuses", () => {
    expect(shownSubtypes({ subtypes: { meow: 0.9 } })).toEqual([]);
  });
});
