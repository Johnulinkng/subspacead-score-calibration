from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from industrial_anomaly_lab.subspace_controls import (
    OFFICIAL_PCA_SHA256,
    EquivalenceTolerances,
    TokenMemmap,
    compare_subspace_models,
    fit_current_cpu_one_pass,
    fit_numpy_cpu_two_pass,
    fit_official_cpu_two_pass,
    load_official_pca,
    write_token_memmap,
)


FITTERS = (fit_official_cpu_two_pass, fit_numpy_cpu_two_pass, fit_current_cpu_one_pass)


def normal_features() -> np.ndarray:
    rng = np.random.default_rng(2801)
    rotation, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    values = rng.normal(size=(317, 7)) @ (np.diag([5, 3, 1, .6, .2, .07, .01]) @ rotation)
    return (values + np.arange(7) * 3.25).astype(np.float32)


def stream(values: np.ndarray, batch_size: int = 23):
    def replay():
        for start in range(0, len(values), batch_size):
            yield values[start : start + batch_size].copy()
    return replay


def fit(fitter, values: np.ndarray, batch_size: int = 23):
    return fitter(
        stream(values, batch_size), feature_dim=values.shape[1],
        total_tokens=len(values), num_batches=(len(values) + batch_size - 1) // batch_size,
        explained_variance=.97,
    )


@pytest.mark.parametrize("batch_size", [1, 23, 317])
def test_paired_fits_execute_pinned_author_and_preserve_subspace(batch_size: int) -> None:
    values = normal_features()
    official, numpy_two_pass, current = [fit(fitter, values, batch_size) for fitter in FITTERS]
    assert official.source_sha256 == OFFICIAL_PCA_SHA256
    assert official.passes == numpy_two_pass.passes == 2
    assert current.passes == 1
    assert official.model.k == numpy_two_pass.model.k == current.model.k
    assert official.input_sha256 == numpy_two_pass.input_sha256 == current.input_sha256
    query = np.random.default_rng(1920).normal(size=(43, 7)).astype(np.float32) + 8
    for result in (numpy_two_pass, current):
        comparison = compare_subspace_models(official, result, stream(query, 13))
        assert comparison["passed"], comparison
        assert comparison["checks"]["reconstruction_scores"]["token_count"] == 43
        assert comparison["detection_gain_claim"] is False
        assert result.model.count == len(values)
        assert result.model.metadata["covariance_denominator"] == "normal_patch_token_count-1"
        assert result.diagnostics["fit_epochs"] is None
        assert result.diagnostics["fit_device"] == "cpu"
        assert result.diagnostics["fit_dtype"] == "float64"
        assert result.elapsed_seconds > 0


def test_all_controls_match_full_covariance_n_minus_one() -> None:
    values = normal_features()
    covariance = np.cov(values.astype(np.float64), rowvar=False, ddof=1)
    expected = np.linalg.eigvalsh(covariance)[::-1]
    for fitter in FITTERS:
        result = fit(fitter, values)
        np.testing.assert_allclose(result.model.mean, values.mean(axis=0, dtype=np.float64), rtol=0, atol=1e-12)
        np.testing.assert_allclose(result.model.eigenvalues, expected[:result.model.k], rtol=1e-12, atol=1e-12)
        assert result.model.total_variance == pytest.approx(float(np.trace(covariance)), rel=1e-12)


def test_adapter_reconstruction_scores_match_author_function(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "vendor" / "SubspaceAD" / "src"))
    from subspacead.post_process.scoring import calculate_anomaly_scores

    result = fit(fit_official_cpu_two_pass, normal_features())
    query = np.random.default_rng(7).normal(size=(29, 7)).astype(np.float32)
    model = result.model
    params = {
        "mu": model.mean, "components": model.components,
        "eigvals": model.eigenvalues, "k": model.k,
        "eps": model.eps, "whiten": False,
    }
    expected = calculate_anomaly_scores(query, params, "reconstruction", drop_k=0)
    np.testing.assert_allclose(model.reconstruction_scores(query, chunk_size=11), expected, rtol=3e-6, atol=3e-6)


def test_comparison_uses_rotation_invariant_projector() -> None:
    result = fit(fit_current_cpu_one_pass, normal_features())
    rng = np.random.default_rng(441)
    rotation, _ = np.linalg.qr(rng.normal(size=(result.model.k, result.model.k)))
    rotated = replace(result.model, components=result.model.components @ rotation)
    query = rng.normal(size=(40, 7)).astype(np.float32)
    assert compare_subspace_models(result.model, rotated, [query])["passed"]
    shifted = replace(result.model, mean=result.model.mean + .1)
    report = compare_subspace_models(result.model, shifted, [query])
    assert report["passed"] is False
    assert report["checks"]["mean"]["passed"] is False


def test_rank_difference_and_changed_training_inputs_are_reported() -> None:
    values = normal_features()
    result = fit(fit_current_cpu_one_pass, values)
    lower_rank = replace(
        result.model, components=result.model.components[:, :-1],
        eigenvalues=result.model.eigenvalues[:-1],
    )
    report = compare_subspace_models(result.model, lower_rank)
    assert report["passed"] is False
    assert report["checks"]["rank"]["passed"] is False
    assert report["checks"]["eigenvalues"]["max_absolute_error"] is None
    changed = fit(fit_current_cpu_one_pass, values + np.float32(.2))
    assert compare_subspace_models(result, changed)["checks"]["identical_fit_input"]["passed"] is False


@pytest.mark.parametrize("fitter", FITTERS[:2])
def test_two_pass_replay_rejects_changed_features(fitter) -> None:
    values = normal_features()
    calls = 0

    def changing():
        nonlocal calls
        calls += 1
        yield values + np.float32(calls)

    with pytest.raises(ValueError, match="changed between PCA passes"):
        fitter(changing, feature_dim=7, total_tokens=len(values), num_batches=1)


@pytest.mark.parametrize("fitter", FITTERS)
def test_fit_refuses_count_dtype_and_nonfinite_inputs(fitter) -> None:
    values = normal_features()
    with pytest.raises(ValueError, match="count"):
        fitter(lambda: [values], feature_dim=7, total_tokens=len(values) + 1, num_batches=1)
    with pytest.raises(ValueError, match="float32"):
        fitter(lambda: [values.astype(np.float64)], feature_dim=7, total_tokens=len(values), num_batches=1)
    nonfinite = values.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fitter(lambda: [nonfinite], feature_dim=7, total_tokens=len(values), num_batches=1)
    with pytest.raises(ValueError, match="total_tokens"):
        fitter(lambda: [values], feature_dim=7, total_tokens=1, num_batches=1)


def test_official_source_is_authenticated_before_execution(tmp_path: Path) -> None:
    module, digest = load_official_pca()
    assert module.PCAModel.__name__ == "PCAModel"
    assert digest == OFFICIAL_PCA_SHA256
    tampered = tmp_path / "pca.py"
    tampered.write_text('raise RuntimeError("unauthenticated source was executed")', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_official_pca(tampered)


def test_token_cache_is_bounded_replayable_and_exclusive(tmp_path: Path) -> None:
    values = normal_features()
    path = tmp_path / "normal.npy"
    cache = write_token_memmap(path, stream(values, 91), total_tokens=len(values), feature_dim=7, batch_tokens=31)
    assert cache.shape == values.shape
    assert cache.num_batches == 11
    for _ in range(2):
        blocks = list(cache())
        assert all(len(block) <= 31 for block in blocks)
        assert all(not isinstance(block, np.memmap) and block.flags.writeable for block in blocks)
        np.testing.assert_array_equal(np.concatenate(blocks), values)
    np.testing.assert_array_equal(cache.read_tokens(13, 19), values[13:19])
    result = fit_current_cpu_one_pass(cache, feature_dim=7, total_tokens=len(values), num_batches=cache.num_batches, explained_variance=.97)
    expected = fit(fit_current_cpu_one_pass, values, 31)
    assert compare_subspace_models(result, expected)["passed"]
    with pytest.raises(FileExistsError):
        write_token_memmap(path, [values], total_tokens=len(values), feature_dim=7)
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), values)
    with pytest.raises(ValueError, match="dimension"):
        TokenMemmap(path, feature_dim=8)
    with pytest.raises(ValueError, match="bounds"):
        cache.read_tokens(0, len(values) + 1)
    # No retained OS mapping prevents per-category cache deletion on Windows.
    path.unlink()
    assert not path.exists()


def test_failed_cache_write_removes_only_owned_incomplete_file(tmp_path: Path) -> None:
    path = tmp_path / "partial.npy"
    with pytest.raises(ValueError, match="declared token count"):
        write_token_memmap(path, [normal_features()], total_tokens=400, feature_dim=7)
    assert not path.exists()
    protected = tmp_path / "existing.npy"
    protected.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        write_token_memmap(protected, [normal_features()], total_tokens=317, feature_dim=7)
    assert protected.read_bytes() == b"preserve"


def test_tolerance_and_empty_query_validation() -> None:
    result = fit(fit_current_cpu_one_pass, normal_features())
    with pytest.raises(ValueError, match="tolerances"):
        EquivalenceTolerances(projector_atol=-1)
    with pytest.raises(ValueError, match="at least one"):
        compare_subspace_models(result, result, [])
