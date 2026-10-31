"""E36 evaluation (protocol in ``e36_tail_chasing_scenario.py``): leave-one-group-out over every usable real clip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Settings
from ..utils import utc_now, write_json
from .e16_breed_robustness import wilson

WIN_S = 2.0
C_LR = 0.1


def _groups(rows: list[dict], near: list[list]) -> dict[str, str]:
    """Union of source groups joined by cross-group near-duplicate clips."""
    parent = {r["group"]: r["group"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    g = {r["clip"]: r["group"] for r in rows}
    for a, b, _ in near:
        if a in g and b in g:
            parent[find(g[a])] = find(g[b])
    return {k: find(v) for k, v in g.items()}


def pose_windows(pz, starts: np.ndarray) -> np.ndarray:
    """Per window: mean / min nose-tail_end and mean nose-tail_start distance (body lengths), tail-end visibility, body-axis
    rotation (rad per s), curl index, fraction of frames with pose."""
    out = np.zeros((len(starts), 7), np.float32)
    if pz is None or not len(pz["t"]):
        return out
    t, xy, vis, box = pz["t"], pz["xy"].astype(np.float64), pz["vis"], pz["box"]
    nose, ts, te, wi = xy[:, 0], xy[:, 1], xy[:, 2], xy[:, 3]
    diag = np.hypot(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]) + 1e-6
    L = np.maximum(np.hypot(*(wi - ts).T), 0.2 * diag)
    d_end = np.hypot(*(nose - te).T) / L
    d_start = np.hypot(*(nose - ts).T) / L
    curl = np.hypot(*(nose - ts).T) / (np.hypot(*(nose - wi).T) + np.hypot(*(wi - ts).T) + 1e-6)
    ang = np.unwrap(np.arctan2(*(ts - wi).T[::-1]))
    for i, s in enumerate(starts):
        m = (t >= s) & (t < s + WIN_S)
        if m.sum() == 0:
            continue
        rot = float(abs(ang[m][-1] - ang[m][0]) / WIN_S) if m.sum() > 1 else 0.0
        out[i] = [d_end[m].mean(), d_end[m].min(), d_start[m].mean(), vis[m, 2].mean(), rot, curl[m].mean(), m.sum() / (WIN_S * 2)]
    return out


def xclip_windows(xz, starts: np.ndarray) -> np.ndarray:
    out = np.zeros(len(starts), np.float32)
    if xz is None or not len(xz["t"]):
        return out
    pt = xz["p"][:, : int(xz["n_tail"])].sum(1)
    for i, s in enumerate(starts):
        m = (xz["t"] < s + WIN_S) & (xz["t"] + WIN_S > s)
        if m.any():
            out[i] = pt[m].max()
    return out


def clip_score(p: np.ndarray) -> float:
    return float(np.sort(p)[-max(1, len(p) // 4):].mean()) if len(p) else 0.0


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    best, thr = -1.0, float(np.max(s)) + 1e-6
    for t in np.unique(s):
        pred = s >= t
        tp = float(np.sum(pred & (y == 1)))
        f1 = 2 * tp / max(1.0, float(pred.sum() + y.sum()))
        if f1 > best:
            best, thr = f1, float(t)
    return thr


def bouts_from(starts: np.ndarray, p: np.ndarray, thr: float) -> list[list[float]]:
    out: list[list[float]] = []
    for s, v in zip(starts, p):
        if v < thr:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = float(s) + WIN_S
        else:
            out.append([float(s), float(s) + WIN_S])
    return out


def overlap_s(a: list[list[float]], b: list[dict]) -> float:
    return sum(max(0.0, min(x[1], r["t1"]) - max(x[0], r["t0"])) for x in a for r in b)


def load(settings: Settings):
    from ..understand import recognize as R
    from ..understand import temporal as TM
    from ..understand.features import SiglipEncoder
    from .e32_behavior_v2 import zs_frames
    from .e36_tail_chasing_scenario import SCENARIO

    sc = json.loads(SCENARIO.read_text())
    rows = [r for r in sc["clips"] if r["label"] != "excluded"]
    grp = _groups(rows, sc["near_duplicate_pairs"])
    root = settings.paths.resolve("data_root").parent / "understand"
    zsm = R.ZeroShot(SiglipEncoder(threads=settings.runtime.num_threads))
    data = []
    for r in rows:
        z = np.load(root / "features_v2" / f"{r['clip']}.npz", allow_pickle=False)
        pz = np.load(root / "pose" / f"{r['clip']}.npz") if (root / "pose" / f"{r['clip']}.npz").exists() else None
        xz = np.load(root / "xclip" / f"{r['clip']}.npz") if (root / "xclip" / f"{r['clip']}.npz").exists() else None
        zf = zs_frames(z, r["species"], zsm)
        st, Xw = TM.window_features(z, zf)
        Xp, Xx = pose_windows(pz, st), xclip_windows(xz, st)
        y = np.zeros(len(st), int)
        if r["label"] == "positive":
            for i, s in enumerate(st):
                ov = sum(max(0.0, min(s + WIN_S, b["t1"]) - max(s, b["t0"])) for b in r["bouts"]) / WIN_S
                y[i] = 1 if ov >= 0.5 else (0 if ov == 0 else -1)
        zs_clip = R.zero_shot_scores(z, r["species"], zsm).get("tail_chasing", 0.0)
        xc = clip_score(Xx) if xz is not None else None
        data.append({"row": r, "group": grp[r["clip"]], "starts": st, "track": Xw[:, :11], "zs": Xw[:, 11:], "pose": Xp, "xclip": Xx[:, None],
                     "y": y, "zs_clip": float(zs_clip), "xclip_clip": xc, "zs_frames": zf, "has_xclip": xz is not None, "has_pose": pz is not None})
    return sc, data


FEATURESETS = {"A1_track_siglip": ("track", "zs"), "P_track_pose": ("track", "pose"), "F_track_pose_siglip_xclip": ("track", "pose", "zs", "xclip")}


def _X(d: dict, fs: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([d[k] for k in fs], axis=1) if len(d["starts"]) else np.zeros((0, sum(d[k].shape[1] for k in fs)))


def fit(Xs, ys):
    from sklearn.linear_model import LogisticRegression

    X, y = np.concatenate(Xs), np.concatenate(ys)
    keep = y >= 0
    X, y = X[keep], y[keep]
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    clf = LogisticRegression(C=C_LR, class_weight="balanced", max_iter=3000).fit((X - mu) / sd, y)
    return lambda Z: clf.predict_proba((Z - mu) / sd)[:, 1] if len(Z) else np.zeros(0)


def summarize(name: str, data: list[dict], scores: np.ndarray, thr: np.ndarray, win: list[np.ndarray | None], win_thr: list[float]) -> dict:
    from sklearn.metrics import average_precision_score

    lab = np.array([d["row"]["label"] for d in data])
    y = (lab == "positive").astype(int)
    pred = scores >= thr
    tp, fp, fn = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 1)))
    hard = lab == "hard_negative"
    pos_idx = np.where(y == 1)[0]
    temporal, covered = [], 0
    for i in pos_idx:
        if not pred[i] or win[i] is None:
            continue
        b = bouts_from(data[i]["starts"], win[i], win_thr[i])
        ref = data[i]["row"]["bouts"]
        ref_s = sum(r["t1"] - r["t0"] for r in ref)
        pred_s = sum(x[1] - x[0] for x in b)
        ov = overlap_s(b, ref)
        temporal.append({"clip": data[i]["row"]["clip"], "reference_s": ref_s, "predicted_s": round(pred_s, 1), "overlap_s": round(ov, 1),
                         "recall_s": round(ov / ref_s, 2) if ref_s else None, "precision_s": round(min(1.0, ov / pred_s), 2) if pred_s else None})
        covered += ov > 0
    neg = y == 0
    return {"method": name, "ap": float(average_precision_score(y, scores)), "positives": int(y.sum()), "negatives": int(neg.sum()),
            "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp), "precision_ci95": wilson(tp, tp + fp),
            "recall": tp / max(1, y.sum()), "recall_ci95": wilson(tp, int(y.sum())),
            "fp_hard_negatives": int(np.sum(pred & hard)), "hard_negatives": int(hard.sum()),
            "fp_clips": [data[i]["row"]["clip"] for i in np.where(pred & neg)[0]],
            "missed_positives": [data[i]["row"]["clip"] for i in np.where(~pred & (y == 1))[0]],
            "usable_answer_coverage_positives": {"k": covered, "n": int(y.sum())},
            "no_false_claim_negatives": {"k": int(np.sum(~pred & neg)), "n": int(neg.sum())},
            "temporal_on_detected_positives": temporal,
            "reliable": bool(y.sum() >= 10 and tp / max(1, y.sum()) >= 0.8 and tp / max(1, tp + fp) >= 0.8)}


def evaluate(settings: Settings, out_dir: Path) -> dict:
    sc, data = load(settings)
    groups = sorted({d["group"] for d in data})
    n = len(data)
    res = {"experiment": "E36", "created_at": utc_now().isoformat(timespec="seconds"), "clips": n, "groups": len(groups),
           "coverage": {"pose": sum(d["has_pose"] for d in data), "xclip": sum(d["has_xclip"] for d in data)}, "methods": {}}
    y = np.array([d["row"]["label"] == "positive" for d in data], int)
    gidx = {g: [i for i, d in enumerate(data) if d["group"] == g] for g in groups}
    # pretrained baselines: fixed scores, threshold chosen on the training groups
    for name, key in (("B1_siglip_zero_shot", "zs_clip"), ("B2_xclip_zero_shot", "xclip_clip")):
        s = np.array([d[key] if d[key] is not None else 0.0 for d in data])
        thr = np.zeros(n)
        for g in groups:
            tr = np.array([i for i in range(n) if data[i]["group"] != g])
            thr[gidx[g]] = best_f1_threshold(y[tr], s[tr])
        win = []
        for d in data:
            if key == "zs_clip":
                zt, zc, _ = d["zs_frames"]
                win.append(np.array([zc[(zt >= st) & (zt < st + WIN_S)].max() if np.any((zt >= st) & (zt < st + WIN_S)) else 0.0 for st in d["starts"]]))
            else:
                win.append(d["xclip"][:, 0])
        res["methods"][name] = summarize(name, data, s, thr, win, list(thr))
    # adapted models, leave one group out
    for name, fs in FEATURESETS.items():
        Xs = [_X(d, fs) for d in data]
        s, thr = np.zeros(n), np.zeros(n)
        win: list = [None] * n
        for g in groups:
            tr = [i for i in range(n) if data[i]["group"] != g]
            f = fit([Xs[i] for i in tr], [data[i]["y"] for i in tr])
            tr_scores = np.array([clip_score(f(Xs[i])) for i in tr])
            t = best_f1_threshold(y[tr], tr_scores)
            for i in gidx[g]:
                win[i] = f(Xs[i])
                s[i], thr[i] = clip_score(win[i]), t
        res["methods"][name] = summarize(name, data, s, thr, win, [0.5] * n)
    res["summary"] = {k: {x: (round(v[x], 3) if isinstance(v[x], float) else v[x]) for x in ("ap", "tp", "fp", "fn", "precision", "recall", "fp_hard_negatives", "reliable")}
                      for k, v in res["methods"].items()}
    write_json(out_dir / "metrics.json", res)
    return res
