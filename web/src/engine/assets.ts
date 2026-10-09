// Types for engine_assets.json (written by petperception.ondevice.bundle).

export interface AudioAssets {
  model_id: string;
  model_file: string;
  sample_rate: number;
  window_samples: number;
  hop_s: number;
  labels: string[];
  groups: Record<string, number[]>;
  subtypes: Record<string, number>;
  context_tags: Record<string, number>;
  thresholds: Record<string, number>;
  threshold_source: Record<string, string>;
  context_tag_threshold: number;
  min_event_gap_s: number;
  silence_dbfs: number;
  clipping_fraction: number;
}

export interface DetectorAssets {
  model_id: string;
  model_file: string;
  input_hw: [number, number];
  norm_mean: number;
  norm_std: number;
  labels: string[];
  output_order: ("scores" | "boxes")[];
  decoding: {
    anchors_b64: string; num_boxes: number; num_classes: number;
    x_scale: number; y_scale: number; w_scale: number; h_scale: number;
    apply_exponential_on_box_size: boolean; sigmoid_score: boolean;
  };
  score_threshold: number;
  nms_iou: number;
  max_detections_per_frame: number;
  target_classes: string[];
}

export interface VerifierClass {
  feature_set: string; mean: number[]; scale: number[]; coef: number[]; intercept: number; tau: number;
}

export interface VerifierAssets {
  params: { schema: string; detector_id: string; classes: Record<string, VerifierClass> };
  groups: string[];
  ranges: Record<string, [number, number]>;
  teddy_index: number;
  animal_end: number;
  margin: number;
  eps: number;
  max_checks_per_frame: number;
}

export interface PostureAssets {
  mean_b64: string; scale_b64: string; coef_b64: string; coef_shape: [number, number]; intercept: number[];
  gate: number; version: string; labels: string[]; min_detection_score: number; max_per_frame: number;
}

export interface BehaviorParams {
  enabled: boolean;
  moving_speed: number; active_speed: number; min_stationary_s: number; min_moving_s: number; min_active_s: number;
  pacing_min_amplitude: number; pacing_min_reversals: number; pacing_max_leg_s: number; pacing_min_span_s: number;
  approach_min_drop: number; approach_min_s: number; approach_lookback_s: number; view_edge_margin_s: number;
  vocal_max_gap_s: number; startle_max_lag_s: number; startle_baseline_s: number; startle_ratio: number;
}

export interface FusionConfig {
  method: "rule_baseline" | "weighted_average" | "product_of_experts";
  weights: Record<string, number>;
  temperature: number;
  smoothing_epsilon: number;
  prior_logit: number;
  max_evidence_age_s: number;
  co_occurrence_window_s: number;
  min_zone_dwell_s: number;
  abstain_below_score: number;
}

export interface EngineAssets {
  schema: string;
  engine_version: string;
  release_profile: string;
  audio?: AudioAssets;
  detector?: DetectorAssets;
  backbone?: { model_file: string; norm_mean: number; norm_std: number; input_hw: [number, number]; outputs: string[] };
  verifier?: VerifierAssets;
  posture?: PostureAssets;
  vision: {
    sample_fps: number; max_side: number;
    quality: { low_light_luma: number; blur_laplacian_var: number; tiny_area_fraction: number; camera_motion_threshold: number; frame_gap_factor: number };
    tracker: { method: "iou" | "kalman"; iou_threshold: number; max_gap_s: number; min_hits: number };
    camera_motion_compensation: boolean;
  };
  behavior: BehaviorParams;
  fusion: FusionConfig;
  limits: { max_duration_s: number; max_video_frames: number; max_file_bytes: number };
  notes: string[];
}

export function f32FromB64(b64: string): Float32Array {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}
