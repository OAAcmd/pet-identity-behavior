"""E36 helpers on synthetic data: pose and X-CLIP window features, thresholds, bouts, group merging, scoring."""

from __future__ import annotations

import numpy as np

from petperception.eval import e36_evaluate as E


def test_pose_windows_curl_and_rotation():
    t = np.arange(0, 4, 0.5, dtype=np.float32)
    n = len(t)
    ang = np.linspace(0, 2 * np.pi, n)  # body axis turns once in 4 s
    wi = np.stack([0.5 + 0.0 * ang, 0.5 + 0.0 * ang], 1)
    ts = wi + 0.2 * np.stack([np.cos(ang), np.sin(ang)], 1)
    nose = ts + 0.01  # nose at the tail base: curled
    te = ts + 0.02
    xy = np.stack([nose, ts, te, wi], 1).astype(np.float32)
    pz = {"t": t, "xy": xy, "vis": np.ones((n, 4), np.float32), "box": np.tile([0.2, 0.2, 0.8, 0.8], (n, 1)).astype(np.float32)}
    w = E.pose_windows(pz, np.array([0.0, 2.0], np.float32))
    assert w.shape == (2, 7) and (w[:, 0] < 0.3).all() and (w[:, 4] > 0.5).all() and np.allclose(w[:, 6], 1.0)
    assert np.allclose(E.pose_windows(None, np.array([0.0])), 0.0)


def test_xclip_windows_take_the_tail_prompts():
    xz = {"t": np.array([0.0, 2.0], np.float32), "p": np.array([[0.3, 0.2, 0.1, 0.4], [0.0, 0.0, 0.1, 0.9]], np.float32), "n_tail": np.array(3)}
    w = E.xclip_windows(xz, np.array([0.0, 2.5, 9.0], np.float32))
    assert np.allclose(w, [0.6, 0.1, 0.0])


def test_threshold_bouts_overlap_and_groups():
    y, s = np.array([1, 1, 0, 0]), np.array([0.9, 0.6, 0.7, 0.1])
    assert E.best_f1_threshold(y, s) == 0.6
    b = E.bouts_from(np.array([0.0, 0.5, 5.0]), np.array([0.9, 0.8, 0.9]), 0.5)
    assert b == [[0.0, 2.5], [5.0, 7.0]] and E.overlap_s(b, [{"t0": 1.0, "t1": 6.0}]) == 2.5
    rows = [{"clip": "a", "group": "g1"}, {"clip": "b", "group": "g2"}, {"clip": "c", "group": "g3"}]
    g = E._groups(rows, [["a", "b", 0.98]])
    assert g["a"] == g["b"] != g["c"]


def test_summarize_counts_and_coverage():
    def d(label, bouts=()):
        return {"row": {"label": label, "clip": label, "bouts": [{"t0": a, "t1": b} for a, b in bouts]}, "starts": np.array([0.0, 2.0, 4.0])}

    data = [d("positive", [(0, 2)]), d("positive", [(4, 6)]), d("hard_negative"), d("negative")]
    scores, thr = np.array([0.9, 0.2, 0.8, 0.1]), np.full(4, 0.5)
    win = [np.array([0.9, 0.1, 0.1]), np.array([0.1, 0.1, 0.2]), None, None]
    r = E.summarize("m", data, scores, thr, win, [0.5] * 4)
    assert (r["tp"], r["fp"], r["fn"], r["fp_hard_negatives"]) == (1, 1, 1, 1)
    assert r["usable_answer_coverage_positives"] == {"k": 1, "n": 2} and r["temporal_on_detected_positives"][0]["overlap_s"] == 2.0
    assert r["reliable"] is False
