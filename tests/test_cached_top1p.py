"""Check scoring floor semantics and invariance to tied boundary ordering."""
import runpy
from pathlib import Path

import numpy as np


aggregate_scores = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/evaluate_cached_top1p.py"))["aggregate_scores"]


def test_top_percent_uses_floor_and_float64_mean():
    maximum, mean, count = aggregate_scores(np.arange(199, dtype=np.float32))
    assert (maximum, mean, count) == (198.0, 198.0, 1)
    maximum, mean, count = aggregate_scores(np.arange(65536, dtype=np.float32))
    assert count == 655
    assert maximum == 65535.0
    assert mean == 65208.0


def test_boundary_ties_do_not_depend_on_pixel_order():
    values = np.full(65536, 2.0, dtype=np.float32)
    values[:327] = 5.0
    expected = (327 * 5.0 + (655 - 327) * 2.0) / 655
    assert aggregate_scores(values) == (5.0, expected, 655)
    assert aggregate_scores(values[::-1]) == (5.0, expected, 655)
