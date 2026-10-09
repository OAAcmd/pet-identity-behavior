"""Person-pet interaction runtime pieces (E13): geometry and frame-run segments.
The same expected values are asserted in web/tests/unit/interaction.test.ts."""

from pathlib import Path

import numpy as np
import pytest

from petperception.vision.behavior import interaction_segments
from petperception.vision.interaction import InteractionModel, geometry

HELD = ([0.2, 0.1, 0.6, 0.9], [0.3, 0.4, 0.5, 0.7])
APART = ([0.1, 0.1, 0.3, 0.5], [0.6, 0.5, 0.9, 0.9])


def test_geometry_values():
    g = geometry(np.array(HELD[0]), np.array(HELD[1]), (480, 640))
    assert np.allclose(g, [0.1875, 1.0, 0.1875, 0.0, 0.0625, -1.673976, -0.405465, -0.117783, 0.0, 1.0, 0.32, 0.06], atol=1e-6)
    g = geometry(np.array(APART[0]), np.array(APART[1]), (480, 640))
    assert np.allclose(g, [0.0, 0.0, 0.0, 2.75, 1.0, 0.405465, -0.405465, 0.0, 1.06066, 0.0, 0.08, 0.12], atol=1e-6)


def test_segments_need_consecutive_frames_with_a_person():
    t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    s = [0.9, 0.8, None, 0.9, 0.2, 0.9, 0.95]
    segs = interaction_segments(t, s, 0.7, 2, 1.5, "t1")
    assert [(g.t_start_s, g.t_end_s, g.measurements["frames"]) for g in segs] == [(0.0, 0.5, 2.0), (2.5, 3.0, 2.0)]
    assert all(g.kind == "person_interaction_candidate" and g.reliability == "low" for g in segs)
    assert interaction_segments([0.0, 5.0], [0.9, 0.9], 0.7, 2, 1.5, "t1") == []  # gap longer than the tracker gap


MODEL = Path(__file__).resolve().parents[1] / "artifacts/interaction/person_pet_interaction_v1.npz"


@pytest.mark.skipif(not MODEL.exists(), reason="run petperception evaluate --experiment e13")
def test_model_orders_held_above_apart():
    m = InteractionModel.load(MODEL)
    held = m.score(np.array(HELD[0]), np.array(HELD[1]), (480, 640), "dog")
    apart = m.score(np.array(APART[0]), np.array(APART[1]), (480, 640), "dog")
    assert held > m.threshold > apart
