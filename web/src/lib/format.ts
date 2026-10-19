// Display helpers. Scores are never rendered as percentages or "confidence".

export function fmtTime(s: number | null | undefined): string {
  if (s === null || s === undefined || !isFinite(s)) return "-";
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return `${m}:${sec.toFixed(1).padStart(4, "0")}`;
}

export function fmtSpan(a: number | null | undefined, b: number | null | undefined): string {
  return `${fmtTime(a)}-${fmtTime(b)}`;
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

/** Uncalibrated support score: shown as a coarse word, never as a percentage. */
export function supportWord(score: number | null): string {
  if (score === null) return "no score";
  if (score < 0.5) return "below abstention threshold";
  if (score < 0.6) return "weak";
  if (score < 0.7) return "moderate";
  return "stronger (still unvalidated)";
}

export const HYPOTHESIS_LABEL: Record<string, string> = {
  food: "Possibly food-related",
  outside: "Possibly wants to go outside",
  attention: "Possibly seeking attention",
  play: "Possibly playing",
  alert: "Possibly alert",
  tense: "Possibly tense",
  warning: "Possible warning signal",
  relaxed: "Possibly relaxed",
};

export const STATUS_LABEL: Record<string, string> = {
  experimental_hypothesis: "Experimental hypothesis",
  insufficient_evidence: "Not enough evidence",
  abstained: "Abstained",
  supported_observation: "Observations only",
  abstention: "No interpretation",
};

export const DECISION_HELP: Record<string, string> = {
  experimental_hypothesis: "Some observations line up with an experimental interpretation. It has not been validated on real household data.",
  supported_observation: "We report what was seen and heard, without interpreting it.",
  abstention: "No cat or dog evidence was found, so nothing is interpreted.",
};

export const REASON_TEXT: Record<string, string> = {
  fusion_not_validated: "The interpretation rules have not been validated on real household recordings.",
  single_provenance_root: "All supporting evidence comes from one source.",
  context_only_evidence: "Only the place (zone) supports it, which is not enough.",
  stale_evidence: "The supporting observations were too far apart in time.",
  missing_modality_audio: "There was no usable audio.",
  missing_modality_video: "There was no usable video.",
  audio_silent: "The audio was silent (silence is not evidence either way).",
  audio_model_unavailable: "The sound model is unavailable on the server.",
  model_unavailable: "A required model is unavailable on the server.",
  zones_not_configured: "No places/zones were marked for this recording.",
  pose_unavailable: "Body-posture details (ears, tail) are not available.",
  person_detection_disabled: "People are not detected (by design).",
  uncertain_attribution_multiple_animals: "Several animals were visible, so the sound could not be assigned to one.",
  no_visible_animal_for_attribution: "No animal was visible when the sound happened; it may come from off-screen or a TV.",
  conflicting_evidence: "Other observations at the same time point in a different direction.",
  low_support: "The experimental score was too low.",
  audio_attribution_unverified: "Assigning the sound to this animal is a guess (only one animal was visible).",
  low_reliability_evidence: "Some evidence has quality problems (for example low light).",
  silence_not_used_as_evidence: "Silence is never treated as a sign of relaxation.",
  purr_does_not_rule_out_discomfort: "Purring can also happen when an animal is uncomfortable.",
  no_pet_detected: "No cat or dog was detected in the video.",
  no_pet_sound_detected: "No cat or dog sounds were detected.",
  multiple_animals_visible: "More than one animal was visible.",
  audio_corrupt: "The audio track could not be read.",
  no_decodable_video_frames: "No video frames could be read.",
  temporal_context_only: "Only timing patterns support it.",
};

export const QUALITY_TEXT: Record<string, string> = {
  low_light: "Low light",
  blurry_frames: "Blurry frames",
  camera_motion: "Camera moved",
  camera_motion_unknown: "Camera motion could not be checked",
  bandwidth_limited_audio: "Low-quality audio (low sample rate)",
  audio_silent: "Silent audio",
  audio_clipping: "Audio too loud (clipping)",
  still_image: "Single photo (no movement information)",
  dropped_or_missing_frames: "Dropped video frames",
  frame_limit_reached: "Only part of the video was analysed",
  audio_end_padded: "End of audio padded",
  video_decode_warnings: "Video decoding warnings",
  audio_decode_warnings: "Audio decoding warnings",
  visual_model_unavailable: "Visual model unavailable",
  detections_rejected_by_verifier: "Something animal-like did not pass the second check (not reported as a pet)",
};

export function soundLabel(label: string): string {
  if (label === "cat_vocalization") return "Cat sound";
  if (label === "dog_vocalization") return "Dog sound";
  if (label === "growl") return "Growl-like sound (cat or dog)";
  return label.replace("context:", "").replace(/_/g, " ");
}

export const ATTRIBUTION_TEXT: Record<string, string> = {
  single_visible_candidate: "One matching animal was visible (assignment not verified).",
  unknown: "Not assigned to an animal.",
  no_visible_animal: "No animal visible at that time - may be off-screen or from a TV/speaker.",
  video_unavailable: "No video to relate it to an animal.",
};

export function isTerminal(status: string): boolean {
  return status === "succeeded" || status === "failed" || status === "cancelled";
}

/** Subtypes of a sound event that passed validation (E21 gate), strongest first, at most three.
 * Results without validation statuses (schema < 1.4.0) show none. */
export function shownSubtypes(s: { subtypes: Record<string, number>; subtype_validation?: Record<string, string> }): string[] {
  const v = s.subtype_validation ?? {};
  return Object.entries(s.subtypes).filter(([k]) => v[k] === "passed").sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k]) => k.replace(/_/g, " "));
}

/** Heading of a sound event: a growl event whose type has not passed validation says so. */
export function soundHeading(s: { label: string; subtype_validation?: Record<string, string> }): string {
  if (s.label === "growl" && s.subtype_validation?.growl !== "passed") return "Possible growl (sound type not validated)";
  return soundLabel(s.label);
}
