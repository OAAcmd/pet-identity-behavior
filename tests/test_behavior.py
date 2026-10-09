"""Behavior segmentation on SYNTHETIC trajectories with known ground truth.

These check the operational definitions and edge cases. They say nothing
about how often real pets produce these patterns or how well the segments
match human behavior labels (no such labels are available).
"""

import numpy as np

from petperception.context.zones import ZoneConfig
from petperception.vision import behavior as B

W, H = 640, 480
P = B.BehaviorParams()


def box_at(cx: float, cy: float, s: float = 0.2) -> np.ndarray:
    return np.array([cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2])


def series(times, centers, camera=None):
    boxes = [box_at(x, y) for x, y in centers]
    return times, boxes, B.step_series(times, boxes, W, H, camera, max_gap_s=1.5)


def kinds(segs):
    return [s.kind for s in segs]


def test_still_then_moving_then_fast():
    # 2 fps: still 5 s, slow walk (0.064 diag/s) 2 s, fast run (0.32 diag/s) 3.5 s.
    xs = [0.1] * 11 + [0.1 + 0.04 * k for k in range(1, 5)] + [0.46, 0.66, 0.86, 0.66, 0.46, 0.26, 0.06]
    t = [i * 0.5 for i in range(len(xs))]
    _, _, ss = series(t, [(x, 0.5) for x in xs])
    segs = B.motion_segments(ss, P, "track-1", "medium")
    still = [s for s in segs if s.kind == "stationary"]
    assert still and still[0].t_start_s == 0.0 and 4.5 <= still[0].t_end_s <= 5.5
    assert any(s.kind == "moving" and s.t_start_s >= 4.5 for s in segs)
    assert any(s.kind == "active_bout" for s in segs)


def test_jitter_is_not_movement():
    rng = np.random.default_rng(0)
    t = [i * 0.5 for i in range(20)]
    _, _, ss = series(t, [(0.5 + rng.normal(0, 0.004), 0.5 + rng.normal(0, 0.004)) for _ in t])
    segs = B.motion_segments(ss, P, "track-1", "medium")
    assert kinds(segs) == ["stationary"]


def test_camera_pan_alone_is_stationary():
    t = [i * 0.5 for i in range(12)]
    # The animal is fixed in the scene; the camera pans so the box drifts right 0.05/frame.
    centers = [(0.2 + 0.05 * i, 0.5) for i in range(12)]
    camera = {t[i]: (0.05, 0.0) for i in range(1, 12)}
    _, _, ss = series(t, centers, camera)
    assert kinds(B.motion_segments(ss, P, "track-1", "medium")) == ["stationary"]


def test_pacing_detected_and_straight_walk_not():
    t = [i * 0.5 for i in range(33)]  # 16 s
    xs = [0.5 + 0.2 * np.sin(2 * np.pi * ti / 4.0) for ti in t]  # 4 s period, 0.4 peak-to-peak
    _, _, ss = series(t, [(x, 0.5) for x in xs])
    pac = B.pacing_segments(ss, P, "track-1")
    assert len(pac) == 1 and pac[0].measurements["reversals"] >= 3 and pac[0].reliability == "low"
    _, _, walk = series(t, [(0.1 + 0.025 * i, 0.5) for i in range(33)])
    assert B.pacing_segments(walk, P, "track-1") == []


def test_zone_approach_enter_dwell_leave():
    zones = ZoneConfig.model_validate({"schema": "petperception.zones/1", "anchor": "center", "zones": [
        {"zone_id": "z1", "semantic_tag": "food_area", "label": "Bowl", "polygon": [[0.7, 0.3], [0.95, 0.3], [0.95, 0.7], [0.7, 0.7]]}]})
    t = [i * 0.5 for i in range(20)]
    xs = [0.2 + 0.08 * i for i in range(8)] + [0.8] * 8 + [0.6, 0.5, 0.4, 0.3]
    times, boxes, _ = series(t, [(x, 0.5) for x in xs])
    segs = B.zone_segments(times, boxes, zones, W, H, P, 1.5, "track-1", "medium")
    k = kinds(segs)
    assert k == ["enter_zone", "approach_zone", "dwell_zone", "leave_zone"]
    dwell = segs[2]
    assert dwell.t_start_s == 3.5 and dwell.t_end_s == 7.5


def test_already_in_zone_has_no_enter_event():
    zones = ZoneConfig.model_validate({"schema": "petperception.zones/1", "anchor": "center", "zones": [
        {"zone_id": "z1", "semantic_tag": "rest_area", "label": "Bed", "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]}]})
    t = [i * 0.5 for i in range(6)]
    times, boxes, _ = series(t, [(0.5, 0.5)] * 6)
    assert kinds(B.zone_segments(times, boxes, zones, W, H, P, 1.5, "track-1", "medium")) == ["dwell_zone"]


def test_view_entry_and_exit():
    segs = B.view_segments([3.0, 3.5, 4.0], [box_at(0.1, 0.5)] * 3, 0.0, 8.0, P, "track-1", "medium")
    assert kinds(segs) == ["entered_view", "left_view"]
    assert B.view_segments([0.0, 7.5], [box_at(0.5, 0.5)] * 2, 0.0, 8.0, P, "track-1", "medium") == []


def test_vocal_bouts_group_by_gap():
    ev = [("cat_vocalization", 1.0, 1.5), ("cat_vocalization", 2.5, 3.0), ("cat_vocalization", 4.0, 4.4), ("cat_vocalization", 9.0, 9.5),
          ("dog_vocalization", 2.0, 2.5)]
    segs = B.vocal_bouts(ev, P)
    assert len(segs) == 1 and segs[0].t_start_s == 1.0 and segs[0].t_end_s == 4.4 and segs[0].measurements["events"] == 3


def test_startle_candidate_needs_sudden_change_after_onset():
    t = [i * 0.5 for i in range(16)]
    xs = [0.3] * 9 + [0.5, 0.7, 0.85, 0.85, 0.85, 0.85, 0.85]
    _, _, ss = series(t, [(x, 0.5) for x in xs])
    hit = B.startle_candidates({"track-1": ss}, [("vacuum", 4.1)], P)
    assert len(hit) == 1 and hit[0].reliability == "low" and 0 <= hit[0].measurements["lag_s"] <= P.startle_max_lag_s
    assert B.startle_candidates({"track-1": ss}, [("vacuum", 1.0)], P) == []  # nothing happens within the lag window
    _, _, moving = series(t, [(0.1 + 0.05 * i, 0.5) for i in range(16)])
    assert B.startle_candidates({"track-1": moving}, [("vacuum", 4.2)], P) == []  # already moving: no sudden change
