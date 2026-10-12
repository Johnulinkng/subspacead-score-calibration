from __future__ import annotations

import numpy as np
import pytest

from industrial_anomaly_lab.metrics import evaluate_category
from industrial_anomaly_lab.quick_metrics import evaluate_from_files


def _save(tmp_path, maps, masks):
    map_files, mask_files = [], []
    for index, (score, mask) in enumerate(zip(maps, masks, strict=True)):
        map_file, mask_file = tmp_path / f"map-{index}.npy", tmp_path / f"mask-{index}.npy"
        np.save(map_file, score)
        np.save(mask_file, mask)
        map_files.append(map_file)
        mask_files.append(mask_file)
    return map_files, mask_files


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 11, 262144])
@pytest.mark.parametrize("fpr_limit", [0.01, 0.30, 1.0])
@pytest.mark.parametrize("tied", [False, True])
def test_random_metrics_equal_existing_exact_definition(tmp_path, chunk_size, fpr_limit, tied):
    rng = np.random.default_rng(81733)
    maps = rng.normal(size=(4, 9, 11)).astype(np.float32)
    if tied:
        maps = np.round(maps, decimals=0)
    masks = (rng.uniform(size=maps.shape) < 0.22).astype(np.uint8)
    masks[0] = 0
    masks[1, 2:5, 3:8] = 1
    labels = [0, 1, 1, 1]
    files = _save(tmp_path, maps, masks)
    expected = evaluate_category(maps, masks, labels, fpr_limit=fpr_limit).to_dict()
    actual = evaluate_from_files(*files, labels, chunk_size=chunk_size, fpr_limit=fpr_limit)
    assert actual == pytest.approx(expected, abs=1e-12)
    assert not list(tmp_path.glob(".quick-metrics-*"))


@pytest.mark.parametrize("kind", ["perfect", "reversed", "all_tied", "unequal_regions"])
def test_boundary_and_equal_region_fixtures(tmp_path, kind):
    masks = np.zeros((2, 5, 5), dtype=np.uint8)
    masks[1, 0, 0] = 1
    masks[1, 1, 1] = 1  # Eight-connected to (0, 0).
    masks[1, 4, 1:5] = 1
    maps = masks.astype(np.float64)
    if kind == "reversed":
        maps = 1.0 - maps
    elif kind == "all_tied":
        maps[:] = 1.0
    elif kind == "unequal_regions":
        maps[1, 4, 1:5] = -1.0
    files = _save(tmp_path, maps, masks)
    expected = evaluate_category(maps, masks, [0, 1]).to_dict()
    assert evaluate_from_files(*files, [0, 1], chunk_size=1) == pytest.approx(expected, abs=1e-12)


def test_one_tied_group_crosses_fpr_boundary(tmp_path):
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[1, 0, :3] = True
    maps = np.zeros(masks.shape, dtype=np.float32)
    maps[1, 0, :] = 0.5
    maps[0, :3, :] = 0.5  # Tied group crosses the 0.30 FPR limit.
    files = _save(tmp_path, maps, masks)
    expected = evaluate_category(maps, masks, [0, 1]).to_dict()
    assert evaluate_from_files(*files, [0, 1], chunk_size=2) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("invalid", ["nonbinary", "nonfinite", "mismatch", "no_foreground", "bad_labels"])
def test_invalid_inputs_fail_and_remove_temporary_files(tmp_path, invalid):
    masks = np.zeros((2, 3, 3), dtype=np.uint8)
    masks[1, 1, 1] = 1
    maps = masks.astype(np.float32)
    labels = [0, 1]
    if invalid == "nonbinary":
        masks[1, 1, 1] = 255
    elif invalid == "nonfinite":
        maps[0, 0, 0] = np.nan
    elif invalid == "no_foreground":
        masks[:] = 0
    elif invalid == "bad_labels":
        labels = [0, 0]
    files = _save(tmp_path, maps, masks)
    if invalid == "mismatch":
        np.save(files[1][0], np.zeros((2, 2)))
    with pytest.raises(ValueError):
        evaluate_from_files(*files, labels, chunk_size=2)
    assert not list(tmp_path.glob(".quick-metrics-*"))
