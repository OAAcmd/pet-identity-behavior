"""Frozen YAMNet audio-event backend (MediaPipe TFLite build).

Verified interface (see manifests/resources.json):
* input: 15600 float32 samples of 16 kHz mono audio in [-1, 1] (0.975 s);
  the log-mel frontend is inside the graph;
* output: 521 independent sigmoid scores (in-graph LOGISTIC). These are raw,
  uncalibrated multi-label scores; softmax is never applied;
* internal activations are int8, so scores are multiples of 1/256.

Windowing follows the original YAMNet framing: 0.96 s patches with a
configurable hop (default 0.48 s); the waveform is zero-padded at the end so
that the final partial window is scored, and clips shorter than one window are
zero-padded to 0.975 s (reported as ``padded``).
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..ingest.media import AudioTrack, resample

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 15600
EMBEDDING_TENSOR_NAME = "tower0/network/layer28/reduce_mean"


@dataclass
class WindowScores:
    t_start: np.ndarray  # window start times on the media timeline (s)
    t_end: np.ndarray
    scores: np.ndarray  # [N, 521] raw sigmoid scores
    embeddings: np.ndarray | None = None  # [N, 1024] dequantized int8 activations
    padded_windows: int = 0
    native_sample_rate: int = 0
    effective_bandwidth_hz: float = 0.0
    notes: list[str] = field(default_factory=list)


class YamnetBackend:
    name = "yamnet"

    def __init__(self, model_dir: Path, num_threads: int = 2, with_embeddings: bool = False):
        from ai_edge_litert.interpreter import Interpreter  # local import keeps CLI startup light

        self.model_path = model_dir / "yamnet.tflite"
        self.with_embeddings = with_embeddings
        kwargs = {"model_path": str(self.model_path), "num_threads": num_threads}
        if with_embeddings:
            kwargs["experimental_preserve_all_tensors"] = True
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.interp = Interpreter(**kwargs)
        self.interp.allocate_tensors()
        self.in_idx = self.interp.get_input_details()[0]["index"]
        self.out_idx = self.interp.get_output_details()[0]["index"]
        self.emb_idx = None
        self.emb_quant = (1.0, 0)
        if with_embeddings:
            for d in self.interp.get_tensor_details():
                if d["name"] == EMBEDDING_TENSOR_NAME:
                    self.emb_idx = d["index"]
                    self.emb_quant = d["quantization"]
            if self.emb_idx is None:
                raise RuntimeError("embedding tensor not found in YAMNet graph")
        # Label map: embedded list defines output order; class map adds AudioSet MIDs.
        embedded = zipfile.ZipFile(io.BytesIO(self.model_path.read_bytes())).read("yamnet_label_list.txt").decode().splitlines()
        with open(model_dir / "yamnet_class_map.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.labels = [r["display_name"] for r in rows]
        self.mids = [r["mid"] for r in rows]
        if self.labels != [s.strip() for s in embedded]:
            raise RuntimeError("YAMNet embedded label order differs from yamnet_class_map.csv")
        self.mid_to_index = {m: i for i, m in enumerate(self.mids)}
        with open(model_dir / "audioset_ontology.json", encoding="utf-8") as fh:
            self.ontology = {n["id"]: n for n in json.load(fh)}

    def descendants(self, mid: str) -> list[str]:
        out = [mid]
        for c in self.ontology.get(mid, {}).get("child_ids", []):
            out.extend(self.descendants(c))
        return list(dict.fromkeys(out))

    def score_waveform16k(self, y: np.ndarray, start_s: float, hop_s: float) -> WindowScores:
        hop = int(round(hop_s * SAMPLE_RATE))
        n = len(y)
        padded = 0
        if n < WINDOW_SAMPLES:
            y = np.concatenate([y, np.zeros(WINDOW_SAMPLES - n, np.float32)])
            padded = 1
            n_win = 1
        else:
            n_win = 1 + int(np.ceil(max(0, n - WINDOW_SAMPLES) / hop))
            need = (n_win - 1) * hop + WINDOW_SAMPLES
            if need > n:
                y = np.concatenate([y, np.zeros(need - n, np.float32)])
                padded = 1
        scores = np.zeros((n_win, len(self.labels)), np.float32)
        embs = np.zeros((n_win, 1024), np.float32) if self.with_embeddings else None
        sc, zp = self.emb_quant
        for k in range(n_win):
            seg = np.ascontiguousarray(y[k * hop: k * hop + WINDOW_SAMPLES], dtype=np.float32)
            self.interp.set_tensor(self.in_idx, seg)
            self.interp.invoke()
            scores[k] = self.interp.get_tensor(self.out_idx)[0]
            if embs is not None:
                raw = self.interp.get_tensor(self.emb_idx).astype(np.float32).reshape(-1)
                embs[k] = (raw - zp) * sc if sc else raw
        t0 = start_s + np.arange(n_win) * hop / SAMPLE_RATE
        return WindowScores(t0, t0 + WINDOW_SAMPLES / SAMPLE_RATE, scores, embs, padded)

    def score_track(self, track: AudioTrack, hop_s: float = 0.48) -> WindowScores:
        if track.samples.size == 0:
            return WindowScores(np.zeros(0), np.zeros(0), np.zeros((0, len(self.labels)), np.float32))
        x = np.clip(track.samples, -1.0, 1.0)
        y = resample(x, track.sample_rate_hz, SAMPLE_RATE)
        ws = self.score_waveform16k(y, track.start_s, hop_s)
        ws.native_sample_rate = track.sample_rate_hz
        ws.effective_bandwidth_hz = min(track.sample_rate_hz, SAMPLE_RATE) / 2.0
        if track.sample_rate_hz < SAMPLE_RATE:
            ws.notes.append(
                f"input sampled at {track.sample_rate_hz} Hz: content above {track.sample_rate_hz / 2:.0f} Hz is absent; "
                "upsampling to 16 kHz does not recreate it (out-of-domain for YAMNet's 125-7500 Hz mel range)"
            )
        return ws
