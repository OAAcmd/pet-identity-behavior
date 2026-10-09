// Recordings made in this session stay available for local playback next to
// their results even when the server deletes media after processing.
const blobs = new Map<string, Blob>();

export const localMedia = {
  set: (analysisId: string, blob: Blob) => blobs.set(analysisId, blob),
  get: (analysisId: string) => blobs.get(analysisId) ?? null,
  delete: (analysisId: string) => blobs.delete(analysisId),
};
