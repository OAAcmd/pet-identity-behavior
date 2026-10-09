"""Readable Markdown report rendered deterministically from an AnalysisResult."""

from __future__ import annotations

from ..schemas.analysis import AnalysisResult


def render_markdown(r: AnalysisResult) -> str:
    L: list[str] = []
    L.append(f"# Analysis report `{r.analysis_id}`")
    L.append("")
    L.append(f"*Schema {r.schema_version} - engine {r.engine_version} - profile `{r.config_profile}` - research use only.*")
    L.append("")
    L.append("> Observations first; hypotheses are experimental and uncalibrated. This is not a translation of animal "
             "language, an emotion reading, or a medical assessment.")
    L.append("")
    m = r.media
    L.append("## Input")
    L.append(f"- File: `{m.analysis_filename}` ({m.size_bytes} bytes, sha256 `{m.sha256[:16]}...`), container `{m.container}`, duration {m.duration_s}")
    L.append(f"- Audio: {r.modalities.audio_state}" + (f" ({m.audio.codec}, {m.audio.sample_rate_hz} Hz, {m.audio.channels} ch)" if m.audio else ""))
    L.append(f"- Video: {r.modalities.video_state}" + (f" ({m.video.codec}, {m.video.width}x{m.video.height})" if m.video else ""))
    L.append("")
    L.append("## Decision")
    L.append(f"- State: **{r.decision.state}** ({r.decision.overall_interpretation})")
    L.append(f"- {r.decision.summary}")
    if r.decision.reason_codes:
        L.append(f"- Reason codes: {', '.join(r.decision.reason_codes)}")
    L.append("")
    L.append("## Observations")
    o = r.observations
    if not o.animal_tracks and not o.sound_events:
        L.append("- No cat or dog observations.")
    for t in o.animal_tracks:
        mv = t.movement
        mvtxt = (f"median apparent speed {mv.median_speed:.3f} {mv.units}, moving fraction {mv.moving_fraction:.2f} (reliability {mv.reliability})"
                 if mv.available else f"movement unavailable ({mv.reason})")
        L.append(f"- **{t.track_id}** {t.species}: {t.first_seen_s:.2f}-{t.last_seen_s:.2f} s, {t.n_detections} detections; {mvtxt}"
                 + (f"; flags: {', '.join(t.quality_flags)}" if t.quality_flags else ""))
        for z in t.zones:
            L.append(f"  - in zone `{z.zone_id}` ({z.semantic_tag}) for {z.dwell_s:.1f} s")
        unavailable = [p.name for p in t.pose_features if not p.available]
        if unavailable:
            L.append(f"  - pose features unavailable: {', '.join(unavailable)}")
    for e in o.sound_events:
        L.append(f"- Sound `{e.label}` {e.t_start_s:.2f}-{e.t_end_s:.2f} s, peak raw score {e.peak_raw_score:.2f} (threshold {e.threshold:.2f}, uncalibrated); "
                 f"attribution: {e.attribution.status} - {e.attribution.note}")
    for e in o.context_sounds:
        L.append(f"- Context sound `{e.label}` {e.t_start_s:.2f}-{e.t_end_s:.2f} s (raw {e.peak_raw_score:.2f})")
    L.append("")
    L.append("## Hypotheses (experimental)")
    if not r.hypotheses:
        L.append("- None considered.")
    for h in r.hypotheses:
        L.append(f"- `{h.name}` [{h.semantic_type}] -> **{h.status}**: {h.explanation}")
    L.append("")
    if r.suggestions:
        L.append("## Suggestions")
        for sgg in r.suggestions:
            L.append(f"- {sgg.text}")
        L.append("")
    if r.quality_flags:
        L.append("## Quality flags")
        for q in r.quality_flags:
            span = f" [{q.t_start_s:.2f}-{q.t_end_s:.2f} s]" if q.t_start_s is not None and q.t_end_s is not None else ""
            L.append(f"- `{q.code}` ({q.severity}){span}: {q.detail or ''}")
        L.append("")
    L.append("## Limitations")
    for lim in r.limitations:
        L.append(f"- {lim}")
    L.append("")
    L.append("## Timings (ms)")
    L.append(", ".join(f"{k}: {v:.1f}" for k, v in r.timings_ms.items()))
    L.append("")
    return "\n".join(L)
