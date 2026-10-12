from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA

from industrial_anomaly_lab.subspace_features import (
    StreamingPCA,
    image_aggregates,
    load_subspace_model,
    render_subspace_map,
    save_subspace_model,
)


@pytest.mark.parametrize("batch_size", [1, 7, 64, 501])
def test_streaming_pca_matches_full_svd_and_training_mean(batch_size: int) -> None:
    rng = np.random.default_rng(992)
    transform, _ = np.linalg.qr(rng.normal(size=(9, 9)))
    samples = rng.normal(size=(501, 9)) @ (np.diag([9, 4, 2, 1, 0.7, 0.3, 0.15, 0.03, 0.01]) @ transform)
    samples += np.arange(9) * 1e4
    accumulator = StreamingPCA(9)
    for start in range(0, len(samples), batch_size):
        accumulator.update(samples[start : start + batch_size])
    actual = accumulator.finalize(0.99)
    expected = PCA(n_components=0.99, svd_solver="full").fit(samples)
    assert actual.count == len(samples)
    assert actual.k == expected.n_components_
    np.testing.assert_allclose(actual.mean, expected.mean_, rtol=0, atol=3e-10)
    np.testing.assert_allclose(actual.eigenvalues, expected.explained_variance_, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        actual.components @ actual.components.T,
        expected.components_.T @ expected.components_, rtol=3e-10, atol=3e-10,
    )
    query = rng.normal(size=(23, 9)) + np.arange(9) * 1e4
    reference = np.sum((query - expected.inverse_transform(expected.transform(query))) ** 2, axis=1)
    np.testing.assert_allclose(actual.reconstruction_scores(query), reference, rtol=2e-9, atol=2e-9)


def test_float32_scores_match_official_reconstruction_and_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    # Execute the pinned author's function, rather than a mirrored formula.
    vendor_source = Path(__file__).resolve().parents[1] / "vendor" / "SubspaceAD" / "src"
    monkeypatch.syspath_prepend(str(vendor_source))
    from subspacead.post_process.scoring import calculate_anomaly_scores

    rng = np.random.default_rng(712)
    normal = rng.normal(size=(100, 8)).astype(np.float32) * np.arange(1, 9)
    accumulator = StreamingPCA(8)
    accumulator.update(normal)
    model = accumulator.finalize(0.8)
    query = rng.normal(size=(19, 8)).astype(np.float32)
    expected = calculate_anomaly_scores(
        query,
        {"mu": model.mean, "components": model.components, "eigvals": model.eigenvalues,
         "k": model.k, "eps": model.eps, "whiten": False},
        "reconstruction", drop_k=0,
    )
    for chunk in (1, 3, 19, 99):
        actual = model.reconstruction_scores(query, chunk_size=chunk)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


def test_rank_threshold_is_left_search_and_not_intermediate_truncation() -> None:
    # Orthogonal centered rows give covariance with exact eigenvalues [3,1].
    normal = np.asarray([[3, 0], [-3, 0], [0, np.sqrt(3)], [0, -np.sqrt(3)]], dtype=np.float64)
    accumulator = StreamingPCA(2)
    accumulator.update(normal[:2])
    accumulator.update(normal[2:])
    model = accumulator.finalize(0.75)
    assert model.k == 1
    assert model.retained_variance == pytest.approx(0.75)
    assert accumulator.finalize(0.75001).k == 2
    assert accumulator.finalize(1).k == 2


def test_constant_training_definition_and_validation() -> None:
    accumulator = StreamingPCA(3)
    with pytest.raises(ValueError, match="at least two"):
        accumulator.finalize()
    accumulator.update(np.full((8, 3), 2.0))
    model = accumulator.finalize()
    assert model.total_variance == 0
    assert model.k == 1
    assert model.retained_variance == 1
    np.testing.assert_array_equal(model.reconstruction_scores(np.full((1, 3), 2.0)), [0])
    assert not model.components.flags.writeable
    with pytest.raises(ValueError, match="shape"):
        accumulator.update(np.ones((2, 4)))
    with pytest.raises(ValueError, match="finite"):
        accumulator.update(np.full((2, 3), np.nan))
    with pytest.raises(ValueError, match="empty"):
        accumulator.update(np.empty((0, 3)))
    with pytest.raises(ValueError, match="float32 or float64"):
        accumulator.update(np.ones((2, 3), dtype=int))
    with pytest.raises(ValueError, match="variance"):
        accumulator.finalize(0)
    with pytest.raises(ValueError, match="orthonormal"):
        replace(model, components=np.zeros_like(model.components))


def test_serialized_model_replays_exactly_and_refuses_overwrite(tmp_path: Path) -> None:
    rng = np.random.default_rng(15)
    accumulator = StreamingPCA(5)
    accumulator.update(rng.normal(size=(53, 5)))
    model = accumulator.finalize(0.7)
    path = tmp_path / "pca.npz"
    save_subspace_model(model, path, {"training_split": "normal_train_only"})
    replay = load_subspace_model(path)
    assert replay.metadata["training_split"] == "normal_train_only"
    np.testing.assert_array_equal(replay.mean, model.mean)
    np.testing.assert_array_equal(replay.components, model.components)
    query = rng.normal(size=(4, 5)).astype(np.float32)
    np.testing.assert_array_equal(replay.reconstruction_scores(query), model.reconstruction_scores(query))
    with pytest.raises(FileExistsError):
        save_subspace_model(model, path)


def test_postprocess_is_official_opencv_path_followed_common_resize() -> None:
    cv2 = pytest.importorskip("cv2")
    grid = np.asarray([[1, 3, 7], [2, 6, 0]], dtype=np.float64)
    expected = cv2.resize(grid.astype(np.float32), (28, 28), interpolation=cv2.INTER_LINEAR)
    expected = cv2.GaussianBlur(expected, (3, 3), 4.0)
    expected = cv2.resize(expected, (16, 16), interpolation=cv2.INTER_LINEAR)
    actual = render_subspace_map(grid, model_size=28, output_size=16)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert actual.min() >= 0


def test_aggregation_is_fixed_top_floor_with_ties_and_raw_scale() -> None:
    raw = np.zeros((256, 256), dtype=np.float32)
    raw.ravel()[:700] = 7
    raw.ravel()[:13] = 20
    scores = image_aggregates(raw)
    assert scores["top_count"] == 655
    assert scores["max"] == 20
    assert scores["top1p"] == pytest.approx((13 * 20 + (655 - 13) * 7) / 655)
    scaled = image_aggregates(raw * 4)
    assert scaled["top1p"] == pytest.approx(scores["top1p"] * 4)
    assert scaled["max"] == 80
    assert image_aggregates(np.ones((2, 2))) == {"max": 1.0, "top1p": 1.0, "top_count": 1}
