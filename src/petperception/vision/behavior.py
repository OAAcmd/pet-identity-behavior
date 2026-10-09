"""Temporal behavior segments derived from tracks, zones and sound events.

Every segment is an *operational* observation with an explicit, measurable
definition (speeds in image diagonals per second, durations in seconds,
reversal counts). None of them names a mental state. Thresholds are set a
priori in configuration; they were checked only on synthetic and
semi-synthetic sequences (E8), never on labelled household behavior, so
segments carry reliability ``medium`` at best and ``low`` for candidates.

Kinds
-----
Per track: ``stationary``, ``moving``, ``active_bout``, ``pacing_candidate``,
``approach_zone``, ``enter_zone``, ``dwell_zone``, ``leave_zone``,
``entered_view``, ``left_view``.
Per analysis: ``vocal_bout`` (sound events only) and ``startle_candidate``
(sudden movement shortly after a sound onset; temporal coincidence only).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..context.zones import ZoneConfig, anchor_point


@dataclass
class BehaviorParams:
    moving_speed: float = 0.05  # diag/s; same as movement_summary's moving threshold
    active_speed: float = 0.25  # diag/s
    min_stationary_s: float = 2.0
    min_moving_s: float = 1.0
    min_active_s: float = 1.0
    pacing_min_amplitude: float = 0.08  # diag units per leg
    pacing_min_reversals: int = 3
    pacing_max_leg_s: float = 10.0
    pacing_min_span_s: float = 4.0
    approach_min_drop: float = 0.05  # diag units of distance to the zone
    approach_min_s: float = 1.0
    approach_lookback_s: float = 10.0
    view_edge_margin_s: float = 1.0
    vocal_max_gap_s: float = 2.0
    startle_max_lag_s: float = 1.5
    startle_baseline_s: float = 3.0
    startle_ratio: float = 3.0


@dataclass
class Segment:
    kind: str
    t_start_s: float
    t_end_s: float
    reliability: str
    detail: str
    track_id: str | None = None
    measurements: dict[str, float] = field(default_factory=dict)
    method: str = ""
    validation: str = "synthetic_and_semi_synthetic_only"


@dataclass
class StepSeries:
    """Camera-compensated per-step speeds of one track."""

    t0: np.ndarray  # step start times
    t1: np.ndarray  # step end times
    speed: np.ndarray  # diag/s (median-of-3 smoothed)
    raw_speed: np.ndarray
    pos: np.ndarray  # (n_dets, 2) centers in diag units, camera-compensated cumulatively
    times: np.ndarray  # detection times


def step_series(times: list[float], boxes: list[np.ndarray], frame_w: int, frame_h: int,
                camera: dict[float, tuple[float, float]] | None, max_gap_s: float) -> StepSeries:
    diag = float(np.hypot(frame_w, frame_h))
    sx, sy = frame_w / diag, frame_h / diag
    t = np.asarray(times, float)
    c = np.array([[(b[0] + b[2]) / 2 * sx, (b[1] + b[3]) / 2 * sy] for b in boxes]) if boxes else np.zeros((0, 2))
    # Undo accumulated camera translation so positions live in a scene-fixed frame.
    pos = c.copy()
    acc = np.zeros(2)
    for i in range(1, len(t)):
        if camera is not None and t[i] in camera:
            acc += np.array([camera[t[i]][0] * sx, camera[t[i]][1] * sy])
        pos[i] = c[i] - acc
    t0, t1, sp = [], [], []
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0 or dt > max_gap_s:
            continue
        t0.append(t[i])
        t1.append(t[i + 1])
        sp.append(float(np.hypot(*(pos[i + 1] - pos[i]))) / dt)
    raw = np.array(sp)
    smooth = raw.copy()
    if raw.size >= 3:
        smooth = np.array([np.median(raw[max(0, k - 1): k + 2]) for k in range(raw.size)])
    return StepSeries(np.array(t0), np.array(t1), smooth, raw, pos, t)


def _runs(mask: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> list[tuple[float, float, list[int]]]:
    """Contiguous runs of True steps (steps separated by a time gap break a run)."""
    out: list[tuple[float, float, list[int]]] = []
    cur: list[int] = []
    for k in range(mask.size):
        if mask[k] and cur and abs(t0[k] - t1[cur[-1]]) > 1e-6:
            out.append((t0[cur[0]], t1[cur[-1]], cur))
            cur = []
        if mask[k]:
            cur.append(k)
        elif cur:
            out.append((t0[cur[0]], t1[cur[-1]], cur))
            cur = []
    if cur:
        out.append((t0[cur[0]], t1[cur[-1]], cur))
    return out


def _mode3(states: np.ndarray) -> np.ndarray:
    """Remove isolated one-step flips (majority of each 3-step window)."""
    if states.size < 3:
        return states
    out = states.copy()
    for k in range(1, states.size - 1):
        if states[k - 1] == states[k + 1] != states[k]:
            out[k] = states[k - 1]
    return out


def motion_segments(ss: StepSeries, p: BehaviorParams, track_id: str, rel: str) -> list[Segment]:
    if ss.speed.size == 0:
        return []
    state = np.where(ss.speed >= p.active_speed, 2, np.where(ss.speed >= p.moving_speed, 1, 0))
    state = _mode3(state)
    segs: list[Segment] = []
    for a, b, ix in _runs(state == 0, ss.t0, ss.t1):
        if b - a >= p.min_stationary_s:
            segs.append(Segment("stationary", a, b, rel, f"median speed {np.median(ss.speed[ix]):.3f} diag/s for {b - a:.1f} s", track_id,
                                {"median_speed": float(np.median(ss.speed[ix])), "duration_s": b - a}, "speed below moving threshold"))
    for a, b, ix in _runs(state >= 1, ss.t0, ss.t1):
        if b - a >= p.min_moving_s:
            path = float(np.sum(ss.raw_speed[ix] * (ss.t1[ix] - ss.t0[ix])))
            segs.append(Segment("moving", a, b, rel, f"moved {path:.2f} image diagonals in {b - a:.1f} s", track_id,
                                {"median_speed": float(np.median(ss.speed[ix])), "path_diagonals": path, "duration_s": b - a},
                                "speed at or above moving threshold"))
    for a, b, ix in _runs(state == 2, ss.t0, ss.t1):
        if b - a >= p.min_active_s:
            segs.append(Segment("active_bout", a, b, rel, f"fast movement (median {np.median(ss.speed[ix]):.2f} diag/s) for {b - a:.1f} s", track_id,
                                {"median_speed": float(np.median(ss.speed[ix])), "duration_s": b - a}, "speed at or above active threshold"))
    return segs


def _zigzag(x: np.ndarray, t: np.ndarray, thr: float) -> list[tuple[float, float]]:
    """Turning points of a 1-D signal with hysteresis ``thr`` (ZigZag indicator)."""
    n = x.size
    if n < 2:
        return []
    idx: list[int] = []
    direction, ext = 0, 0
    lo = hi = 0
    for i in range(1, n):
        if direction == 0:
            lo = i if x[i] < x[lo] else lo
            hi = i if x[i] > x[hi] else hi
            if x[i] - x[lo] >= thr:
                idx.append(lo)
                direction, ext = 1, i
            elif x[hi] - x[i] >= thr:
                idx.append(hi)
                direction, ext = -1, i
        elif direction == 1:
            if x[i] > x[ext]:
                ext = i
            elif x[ext] - x[i] >= thr:
                idx.append(ext)
                direction, ext = -1, i
        else:
            if x[i] < x[ext]:
                ext = i
            elif x[i] - x[ext] >= thr:
                idx.append(ext)
                direction, ext = 1, i
    if direction != 0:
        idx.append(ext)
    return [(float(t[i]), float(x[i])) for i in idx]


def pacing_segments(ss: StepSeries, p: BehaviorParams, track_id: str) -> list[Segment]:
    """Back-and-forth movement along the dominant axis (candidate; unvalidated on real behavior)."""
    if ss.pos.shape[0] < 2 * p.pacing_min_reversals:
        return []
    centered = ss.pos - ss.pos.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    x = centered @ vt[0]
    pts = _zigzag(x, ss.times, p.pacing_min_amplitude)
    legs = [(pts[i][0], pts[i + 1][0], abs(pts[i + 1][1] - pts[i][1])) for i in range(len(pts) - 1)]
    runs: list[list[tuple[float, float, float]]] = []
    cur: list[tuple[float, float, float]] = []
    for leg in legs:
        if leg[1] - leg[0] <= p.pacing_max_leg_s:
            cur.append(leg)
        else:
            runs.append(cur)
            cur = []
    runs.append(cur)
    segs: list[Segment] = []
    for run in runs:
        reversals = len(run) - 1
        if reversals >= p.pacing_min_reversals and run[-1][1] - run[0][0] >= p.pacing_min_span_s:
            amp = float(np.median([leg[2] for leg in run]))
            segs.append(Segment("pacing_candidate", run[0][0], run[-1][1], "low",
                                f"{reversals} direction reversals along one axis, median leg {amp:.2f} image diagonals", track_id,
                                {"reversals": float(reversals), "median_leg_diagonals": amp, "duration_s": run[-1][1] - run[0][0]},
                                "zigzag turning points on the principal movement axis"))
    return segs


def _zone_distance(pt: tuple[float, float], poly: list[tuple[float, float]], sx: float, sy: float) -> float:
    cnt = np.array([[x * sx, y * sy] for x, y in poly], dtype=np.float32)
    d = cv2.pointPolygonTest(cnt, (pt[0] * sx, pt[1] * sy), True)
    return max(0.0, -float(d))  # inside -> 0


def zone_segments(times: list[float], boxes: list[np.ndarray], zones: ZoneConfig | None, frame_w: int, frame_h: int,
                  p: BehaviorParams, max_gap_s: float, track_id: str, rel: str) -> list[Segment]:
    if not zones or not times:
        return []
    diag = float(np.hypot(frame_w, frame_h))
    sx, sy = frame_w / diag, frame_h / diag
    t = np.asarray(times, float)
    segs: list[Segment] = []
    for z in zones.zones:
        anchors = [anchor_point(b, zones.anchor) for b in boxes]
        dist = np.array([_zone_distance(a, z.polygon, sx, sy) for a in anchors])
        inside = dist <= 0.0
        k = 0
        while k < t.size:
            if not inside[k]:
                k += 1
                continue
            a = k
            while k + 1 < t.size and inside[k + 1] and t[k + 1] - t[k] <= max_gap_s:
                k += 1
            b = k
            name = f"'{z.label}' ({z.semantic_tag})"
            came_from_outside = a > 0 and not inside[a - 1] and t[a] - t[a - 1] <= max_gap_s
            left_to_outside = b + 1 < t.size and not inside[b + 1] and t[b + 1] - t[b] <= max_gap_s
            if came_from_outside:
                segs.append(Segment("enter_zone", t[a], t[a], rel, f"entered {name}", track_id, {}, f"anchor {zones.anchor} crossed into the owner polygon"))
                # Approach: longest run before entry with non-increasing distance.
                s = a - 1
                while s - 1 >= 0 and t[a] - t[s - 1] <= p.approach_lookback_s and t[s] - t[s - 1] <= max_gap_s and dist[s - 1] >= dist[s] - 0.01:
                    s -= 1
                drop = float(dist[s] - 0.0)
                if t[a] - t[s] >= p.approach_min_s and drop >= p.approach_min_drop:
                    segs.append(Segment("approach_zone", t[s], t[a], rel, f"distance to {name} fell by {drop:.2f} image diagonals before entering",
                                        track_id, {"distance_drop_diagonals": drop, "duration_s": float(t[a] - t[s])}, "monotone distance decrease"))
            segs.append(Segment("dwell_zone", t[a], t[b], rel, f"in {name} for {t[b] - t[a]:.1f} s (sampled)", track_id,
                                {"dwell_s": float(t[b] - t[a])}, "consecutive sampled detections inside the polygon"))
            if left_to_outside:
                segs.append(Segment("leave_zone", t[b + 1], t[b + 1], rel, f"left {name}", track_id, {}, f"anchor {zones.anchor} left the owner polygon"))
            k += 1
    return segs


def view_segments(times: list[float], boxes: list[np.ndarray], media_start: float, media_end: float | None,
                  p: BehaviorParams, track_id: str, rel: str) -> list[Segment]:
    segs: list[Segment] = []
    if not times:
        return segs

    def near_edge(b: np.ndarray) -> bool:
        return bool(min(b[0], b[1], 1 - b[2], 1 - b[3]) <= 0.03)
    if times[0] - media_start > p.view_edge_margin_s:
        segs.append(Segment("entered_view", times[0], times[0], rel, "first seen after the recording started" + (" at the frame edge" if near_edge(boxes[0]) else ""),
                            track_id, {}, "first sampled detection time"))
    if media_end is not None and media_end - times[-1] > p.view_edge_margin_s:
        segs.append(Segment("left_view", times[-1], times[-1], rel, "not seen again before the recording ended" + (" (last seen at the frame edge)" if near_edge(boxes[-1]) else ""),
                            track_id, {}, "last sampled detection time"))
    return segs


def vocal_bouts(events: list[tuple[str, float, float]], p: BehaviorParams) -> list[Segment]:
    """Group vocalization events of the same label separated by <= vocal_max_gap_s."""
    segs: list[Segment] = []
    by: dict[str, list[tuple[float, float]]] = {}
    for label, a, b in events:
        by.setdefault(label, []).append((a, b))
    for label, ivs in by.items():
        ivs.sort()
        cur = [ivs[0]]
        for iv in ivs[1:] + [(np.inf, np.inf)]:
            if iv[0] - cur[-1][1] <= p.vocal_max_gap_s:
                cur.append(iv)
                continue
            if len(cur) >= 2:
                a, b = cur[0][0], cur[-1][1]
                segs.append(Segment("vocal_bout", a, b, "medium", f"{len(cur)} {label.replace('_', ' ')} events within {b - a:.1f} s", None,
                                    {"events": float(len(cur)), "duration_s": b - a}, f"sound events merged across gaps <= {p.vocal_max_gap_s} s"))
            cur = [iv]
    return segs


def startle_candidates(series: dict[str, StepSeries], onsets: list[tuple[str, float]], p: BehaviorParams) -> list[Segment]:
    """Sudden speed increase shortly after a sound onset (coincidence only, not causality)."""
    segs: list[Segment] = []
    for label, t_on in onsets:
        for tid, ss in series.items():
            if ss.speed.size == 0:
                continue
            base = ss.raw_speed[(ss.t1 <= t_on) & (ss.t0 >= t_on - p.startle_baseline_s)]
            if base.size == 0:
                continue
            mid = (ss.t0 + ss.t1) / 2
            after = np.where((mid > t_on) & (mid <= t_on + p.startle_max_lag_s))[0]
            if after.size == 0:
                continue
            k = int(after[np.argmax(ss.raw_speed[after])])
            peak, b = float(ss.raw_speed[k]), float(np.median(base))
            if peak >= p.active_speed and peak >= p.startle_ratio * max(b, p.moving_speed / 2):
                segs.append(Segment("startle_candidate", float(ss.t0[k]), float(ss.t1[k]), "low",
                                    f"speed rose from {b:.2f} to {peak:.2f} diag/s within {p.startle_max_lag_s:.1f} s after {label.replace('_', ' ')}",
                                    tid, {"baseline_speed": b, "peak_speed": peak, "lag_s": float(mid[k] - t_on)},
                                    "speed ratio after a sound onset; temporal coincidence only"))
    return segs


def interaction_segments(times: list[float], scores: list[float | None], threshold: float, min_frames: int, max_gap_s: float,
                         track_id: str) -> list[Segment]:
    """Runs of sampled frames whose person-pet interaction score (E13) is >= threshold.

    Frames without a detected person score None and end a run. The per-frame model was
    evaluated on public still images (E13); grouping frames over time is not validated."""
    segs: list[Segment] = []
    run: list[int] = []

    def close() -> None:
        if len(run) >= min_frames:
            sc = [scores[i] for i in run]
            segs.append(Segment("person_interaction_candidate", times[run[0]], times[run[-1]], "low",
                                f"pattern of a person holding or interacting with the animal in {len(run)} sampled frames (max score {max(sc):.2f})",
                                track_id, {"frames": float(len(run)), "max_score": float(max(sc)), "threshold": float(threshold)},
                                f"E13 still-image pair model on detection boxes, runs of >= {min_frames} sampled frames",
                                "per_frame_model_public_still_images"))

    for i, (t, v) in enumerate(zip(times, scores)):
        if v is not None and v >= threshold and (not run or t - times[run[-1]] <= max_gap_s):
            run.append(i)
            continue
        close()
        run = [i] if v is not None and v >= threshold else []
    close()
    return segs
