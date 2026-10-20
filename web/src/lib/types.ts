// Types mirroring the backend API (docs/API_CONTRACT.md) and result schema 1.x.

export interface Me {
  id: string;
  username: string;
  is_admin: boolean;
  csrf_token: string;
  retain_media_days: number;
  feedback_research_consent: boolean;
}

export interface Pet {
  id: string;
  name: string;
  species: "cat" | "dog";
  birth_year: number | null;
  notes: string;
  created_at: string;
  updated_at: string;
}

export type SemanticTag = "food_area" | "water_area" | "exit_door" | "rest_area" | "play_area" | "litter_area" | "window" | "other";

export interface Zone {
  zone_id: string;
  label: string;
  semantic_tag: SemanticTag;
  polygon: [number, number][];
}

export interface ZoneConfig {
  schema: "petperception.zones/1";
  coordinate_space: "normalized_image";
  anchor: "bottom_center" | "center";
  zones: Zone[];
}

export interface Place {
  id: string;
  name: string;
  zones: ZoneConfig;
  created_at: string;
  updated_at: string;
}

export interface Upload {
  id: string;
  filename: string;
  size_bytes: number;
  received_bytes: number;
  status: "receiving" | "complete" | "consumed" | "aborted";
  max_chunk_bytes: number;
  expires_at: number;
}

export type AnalysisStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type SubjectMode = "unknown" | "single_known" | "multiple_known";

export interface AnalysisSummary {
  tracks: { track_id: string; species: string; first_seen_s: number; last_seen_s: number }[];
  sound_events: { label: string; t_start_s: number; t_end_s: number }[];
  hypotheses: { name: string; status: string }[];
  decision: { state: string; interpretation: string; summary: string };
  quality_flags: string[];
  modalities: { audio: string; video: string };
}

export interface Analysis {
  id: string;
  status: AnalysisStatus;
  stage: string | null;
  progress: number;
  attempts: number;
  max_attempts: number;
  subject_mode: SubjectMode;
  pet_ids: string[];
  place_id: string | null;
  modalities: string[];
  original_filename: string;
  duration_s: number | null;
  note: string;
  decision_state: string | null;
  summary: AnalysisSummary | null;
  error: { code: string; message: string; action: string | null } | null;
  cancel_requested: boolean;
  media_available: boolean;
  media_retained_until: number | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  finished_at: string | null;
  result?: AnalysisResult | null;
}

export interface Feedback {
  id: string;
  analysis_id: string;
  target_type: "overall" | "hypothesis" | "observation" | "sound_event" | "track";
  target_id: string | null;
  verdict: "agree" | "disagree" | "unsure";
  actual_behavior: string | null;
  note: string;
  consent_research_use: boolean;
  created_at: string;
}

// ---- Analysis result (subset of schema 1.x used by the UI) ----
export interface EvidenceItem {
  evidence_id: string;
  kind: string;
  t_start_s: number;
  t_end_s: number;
  summary: string;
  track_id: string | null;
  reliability: "high" | "medium" | "low";
}

export interface Detection {
  t_s: number;
  box_xyxy_norm: [number, number, number, number];
  species_scores: Record<string, number>;
  flags: string[];
}

export interface AnimalTrack {
  track_id: string;
  identity_note: string;
  species: "cat" | "dog";
  first_seen_s: number;
  last_seen_s: number;
  duration_s: number;
  n_detections: number;
  detections: Detection[];
  movement: { available: boolean; units: string; reason?: string | null; median_speed?: number | null; moving_fraction?: number | null; steps?: number | null; reliability: string };
  zones: { zone_id: string; semantic_tag: string; label: string; intervals_s: [number, number][]; dwell_s: number }[];
  pose_features: { name: string; available: boolean; value?: number | boolean | null; units?: string | null; reason?: string | null }[];
  quality_flags: string[];
  evidence_ids: string[];
  behavior_segments?: BehaviorSegment[];
  posture?: PostureEstimate[];
  keypoints?: { t_s: number; points: Record<string, [number, number, number]>; provider: string }[];
  verification?: { verifier_id: string; checked_detections: number; median_score: number | null; threshold: number; accepted: boolean } | null;
}

export interface BehaviorSegment {
  segment_id: string;
  kind: string;
  t_start_s: number;
  t_end_s: number;
  reliability: string;
  detail?: string | null;
  track_id?: string | null;
  measurements?: Record<string, number>;
  method?: string;
}

export interface PostureEstimate {
  t_s: number;
  label: string;
  score: number;
  score_kind: string;
}

export interface SoundEvent {
  event_id: string;
  label: string;
  t_start_s: number;
  t_end_s: number;
  peak_raw_score: number;
  threshold: number;
  subtypes: Record<string, number>;
  /** Per-subtype validation status (result schema 1.4.0); only "passed" subtypes are shown. */
  subtype_validation?: Record<string, string>;
  attribution: { status: string; candidate_track_ids: string[]; note: string };
  evidence_id: string;
}

export interface Hypothesis {
  hypothesis_id: string;
  name: string;
  semantic_type: string;
  status: "experimental_hypothesis" | "insufficient_evidence" | "abstained";
  track_id: string | null;
  t_start_s: number | null;
  t_end_s: number | null;
  supporting_evidence_ids: string[];
  conflicting_evidence_ids: string[];
  score: number | null;
  score_kind: string;
  calibration_status: string;
  calibrated_confidence: number | null;
  reason_codes: string[];
  explanation: string;
}

export interface IntentResultView {
  version: string;
  species: string;
  species_source: string;
  missing_inputs: Record<string, string>;
  readings: { channel: string; top: string; top_score: number; threshold: number; confident: boolean; evidence_gate_passed: boolean; note: string }[];
  situation: { label: string; score: number; status: string; branches_used: string[]; qualifier: string | null } | null;
  qualifiers: string[];
  hypotheses: { name: string; status: string; support: number | null; supporting: string[]; qualifying: string[]; contradicting: string[]; explanation: string }[];
  conflicts: string[];
  status: string;
  summary: string;
  caveats: string[];
}

export interface AnalysisResult {
  schema_version: string;
  analysis_id: string;
  media: { duration_s: number | null; audio: unknown | null; video: { width: number; height: number; is_still_image: boolean } | null };
  modalities: { requested: string[]; available: string[]; audio_state: string; video_state: string };
  components: { name: string; state: string; reason: string | null }[];
  observations: {
    animal_tracks: AnimalTrack[];
    sound_events: SoundEvent[];
    context_sounds: SoundEvent[];
    evidence: EvidenceItem[];
    sampled_frames: number;
    behavior_events?: BehaviorSegment[];
  };
  quality_flags: { code: string; severity: string; t_start_s: number | null; t_end_s: number | null; detail: string | null }[];
  hypotheses: Hypothesis[];
  decision: { state: string; overall_interpretation: string; reason_codes: string[]; summary: string };
  suggestions: { suggestion_id: string; text: string; based_on_evidence_ids: string[]; kind: string }[];
  limitations: string[];
  fusion: { mode: string; method: string; validated: boolean; multimodal_accuracy: string };
  /** Trained behaviour/intent channels (result schema 1.5.0; absent unless the intent stage ran). */
  intent?: IntentResultView | null;
}

export interface Capabilities {
  upload: { max_bytes: number; max_chunk_bytes: number; allowed_extensions: string[]; max_duration_s: number };
  retention: { max_retain_media_days: number };
  registration: string;
  components: { name: string; state: string; reason: string | null }[];
}
