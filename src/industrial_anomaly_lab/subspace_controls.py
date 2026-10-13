"""Paired PCA controls for the existing DINO / SubspaceAD experiment.

The author's pinned PCAModel already streams its exact covariance in two
passes. These controls isolate its CPU Torch implementation, a CPU NumPy
two-pass implementation, and this project's existing CPU NumPy Chan
one-pass implementation. None changes the frozen feature extractor, PCA
variance target, reconstruction score, or anomaly-map post-processing.

All controls consume the same restartable float32 token stream and fit in
float64, using every normal-training token and the N-1 denominator. Content
digests are collected inside the measured fit; no full feature bank is held
in RAM. The optional .npy cache maps and closes one bounded block at a time.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from .subspace_features import StreamingPCA, SubspaceModel, UPSTREAM_COMMIT


OFFICIAL_PCA_SHA256 = "cb81701be6ba2f32d629dca821862a5d83f7ffedd071522ed683ddf990ff1fcd"
PCA_VARIANTS = (
    "official_cpu_two_pass", "numpy_cpu_two_pass", "current_cpu_one_pass",
)
TokenGenerator = Callable[[], Iterable[np.ndarray]]


def _positive_int(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _batch(values: np.ndarray, feature_dim: int) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != feature_dim or values.shape[0] == 0:
        raise ValueError(f"token batches must have nonempty shape [N,{feature_dim}]")
    if values.dtype != np.dtype("float32"):
        raise ValueError("paired controls require exactly float32 input tokens")
    if not np.isfinite(values).all():
        raise ValueError("token batches must be finite")
    # The official torch.from_numpy path expects writable, contiguous arrays.
    if not values.flags.c_contiguous or not values.flags.writeable:
        values = np.array(values, dtype=np.float32, order="C", copy=True)
    return values


def _validate_fit(
    feature_generator: TokenGenerator, feature_dim: int, total_tokens: int,
    num_batches: int, explained_variance: float, eps: float,
) -> None:
    if not callable(feature_generator):
        raise TypeError("feature_generator must be a restartable callable")
    _positive_int(feature_dim, "feature_dim")
    _positive_int(total_tokens, "total_tokens", 2)
    _positive_int(num_batches, "num_batches")
    if not np.isfinite(explained_variance) or not 0 < explained_variance <= 1:
        raise ValueError("explained_variance must be within (0,1]")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")


class _CheckedReplay:
    """Validate counts and prove both official passes see identical bytes."""

    def __init__(
        self, generator: TokenGenerator, feature_dim: int, total_tokens: int,
        num_batches: int,
    ) -> None:
        self.generator = generator
        self.feature_dim = feature_dim
        self.total_tokens = total_tokens
        self.num_batches = num_batches
        self.passes: list[dict[str, Any]] = []

    def __call__(self) -> Iterator[np.ndarray]:
        digest = hashlib.sha256()
        partition = hashlib.sha256()
        count = batch_count = 0
        for raw in self.generator():
            values = _batch(raw, self.feature_dim)
            count += len(values)
            batch_count += 1
            if count > self.total_tokens or batch_count > self.num_batches:
                raise ValueError("token stream exceeds declared tokens or batches")
            digest.update(memoryview(values).cast("B"))
            partition.update(len(values).to_bytes(8, "little"))
            yield values
        if count != self.total_tokens or batch_count != self.num_batches:
            raise ValueError("token stream count does not match declared tokens and batches")
        observed = {
            "token_count": count, "batch_count": batch_count,
            "input_sha256": digest.hexdigest(),
            "batch_partition_sha256": partition.hexdigest(),
        }
        if self.passes and observed != self.passes[0]:
            raise ValueError("restartable token stream changed between PCA passes")
        self.passes.append(observed)


def load_official_pca(pca_source: str | Path | None = None) -> tuple[Any, str]:
    """Execute only the exact pinned author source, imported by file path."""
    source = Path(pca_source) if pca_source is not None else (
        Path(__file__).resolve().parents[2] / "vendor" / "SubspaceAD" /
        "src" / "subspacead" / "core" / "pca.py"
    )
    source = source.resolve(strict=True)
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != OFFICIAL_PCA_SHA256:
        raise ValueError("official PCA source SHA256 does not match the pinned commit")
    spec = importlib.util.spec_from_file_location("_subspacead_pca_pinned_control", source)
    if spec is None or spec.loader is None:
        raise ImportError("cannot import pinned official PCA source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, actual


@dataclass
class FitResult:
    model: SubspaceModel
    variant: str
    elapsed_seconds: float
    diagnostics: dict[str, Any]
    source_sha256: str | None = None

    @property
    def passes(self) -> int:
        return int(self.diagnostics["pass_count"])

    @property
    def input_sha256(self) -> str:
        return str(self.diagnostics["input_sha256"])

    def metadata(self) -> dict[str, Any]:
        return {
            "variant": self.variant, "elapsed_seconds": self.elapsed_seconds,
            "source_sha256": self.source_sha256, **self.diagnostics,
        }


def _result(
    model: SubspaceModel, variant: str, elapsed_seconds: float,
    replay: _CheckedReplay, *, backend: str, source_sha256: str | None = None,
) -> FitResult:
    diagnostics = {
        **replay.passes[0], "pass_count": len(replay.passes),
        "feature_dim": model.feature_dim, "rank": model.k,
        "variance_target": model.variance_target,
        "retained_variance": model.retained_variance,
        "total_variance": model.total_variance,
        "fit_device": "cpu", "fit_dtype": "float64",
        "input_dtype": "float32", "linear_algebra_backend": backend,
        "covariance_denominator": "normal_patch_token_count-1",
        "covariance_rank_truncation": "none_before_final_rank_selection",
        "fit_epochs": None, "backpropagation": False,
        "timing_scope": "validated_token_stream_statistics_and_eigendecomposition",
        "timing_includes_content_digest": True,
        "timing_excludes_official_source_authentication_and_import": True,
        "subspacead_commit": UPSTREAM_COMMIT,
    }
    model.metadata.update({
        **diagnostics, "control_variant": variant,
        "official_pca_source_sha256": source_sha256,
        "paired_control": True,
    })
    return FitResult(model, variant, elapsed_seconds, diagnostics, source_sha256)


def fit_official_cpu_two_pass(
    feature_generator: TokenGenerator, *, feature_dim: int, total_tokens: int,
    num_batches: int, explained_variance: float = 0.99, eps: float = 1e-6,
    pca_source: str | Path | None = None,
) -> FitResult:
    """Run the unchanged author PCAModel with an explicit CPU device."""
    _validate_fit(feature_generator, feature_dim, total_tokens, num_batches, explained_variance, eps)
    module, source_sha = load_official_pca(pca_source)
    import torch

    official = module.PCAModel(ev=explained_variance, whiten=False, eps=eps)
    official.device = torch.device("cpu")
    official.dtype = torch.float64
    replay = _CheckedReplay(feature_generator, feature_dim, total_tokens, num_batches)
    started = time.perf_counter()
    params = official.fit(replay, feature_dim, total_tokens, num_batches)
    elapsed = time.perf_counter() - started
    full_eigenvalues = official.explained_variance_.detach().cpu().numpy()
    tolerance = max(1.0, float(np.max(np.abs(full_eigenvalues)))) * np.finfo(np.float64).eps * feature_dim * 20
    if float(full_eigenvalues.min()) < -tolerance:
        raise ValueError("official training covariance is not positive semidefinite")
    # Only roundoff-level negative eigenvalues are clipped in the adapter;
    # original eigenvectors and author-selected rank remain untouched.
    full_nonnegative = np.maximum(full_eigenvalues, 0)
    total_variance = float(full_nonnegative.sum())
    k = int(params["k"])
    retained = float(full_nonnegative[:k].sum() / total_variance) if total_variance else 1.0
    model = SubspaceModel(
        mean=params["mu"], components=params["components"],
        eigenvalues=np.maximum(params["eigvals"], 0), count=total_tokens,
        variance_target=float(explained_variance), retained_variance=retained,
        total_variance=total_variance, eps=float(eps),
        metadata={
            "fit_algorithm": "pinned_author_two_pass_full_covariance",
            "token_normalization": "none", "aggregation": "mean_before_pca",
            "scoring": "squared_reconstruction_residual", "whiten": False,
            "drop_k": 0,
            "adapter_clips_roundoff_negative_eigenvalues": True,
        },
    )
    return _result(model, "official_cpu_two_pass", elapsed, replay, backend="torch", source_sha256=source_sha)


def fit_numpy_cpu_two_pass(
    feature_generator: TokenGenerator, *, feature_dim: int, total_tokens: int,
    num_batches: int, explained_variance: float = 0.99, eps: float = 1e-6,
    pca_source: str | Path | None = None,
) -> FitResult:
    """NumPy isolation control: two passes, same finalizer as the Chan fit.

    This is a backend control implemented here, not the author code. The
    unused pca_source keyword keeps the three fit signatures interchangeable.
    """
    _validate_fit(feature_generator, feature_dim, total_tokens, num_batches, explained_variance, eps)
    replay = _CheckedReplay(feature_generator, feature_dim, total_tokens, num_batches)
    started = time.perf_counter()
    accumulator = StreamingPCA(feature_dim)
    for batch in replay():
        accumulator.mean += np.sum(batch, axis=0, dtype=np.float64)
    accumulator.mean /= total_tokens
    for batch in replay():
        centered = np.asarray(batch, dtype=np.float64) - accumulator.mean
        accumulator.scatter += centered.T @ centered
    accumulator.count = total_tokens
    model = accumulator.finalize(explained_variance, eps)
    elapsed = time.perf_counter() - started
    model.metadata["fit_algorithm"] = "numpy_two_pass_full_covariance_backend_control"
    return _result(model, "numpy_cpu_two_pass", elapsed, replay, backend="numpy")


def fit_current_cpu_one_pass(
    feature_generator: TokenGenerator, *, feature_dim: int, total_tokens: int,
    num_batches: int, explained_variance: float = 0.99, eps: float = 1e-6,
    pca_source: str | Path | None = None,
) -> FitResult:
    """Fit the existing, unchanged StreamingPCA with one checked pass."""
    _validate_fit(feature_generator, feature_dim, total_tokens, num_batches, explained_variance, eps)
    replay = _CheckedReplay(feature_generator, feature_dim, total_tokens, num_batches)
    started = time.perf_counter()
    accumulator = StreamingPCA(feature_dim)
    for batch in replay():
        accumulator.update(batch)
    model = accumulator.finalize(explained_variance, eps)
    elapsed = time.perf_counter() - started
    return _result(model, "current_cpu_one_pass", elapsed, replay, backend="numpy")


@dataclass(frozen=True)
class EquivalenceTolerances:
    mean_atol: float = 1e-9
    mean_rtol: float = 1e-10
    eigenvalue_atol: float = 1e-9
    eigenvalue_rtol: float = 1e-9
    projector_atol: float = 1e-9
    projector_rtol: float = 1e-9
    score_atol: float = 1e-4
    score_rtol: float = 3e-5

    def __post_init__(self) -> None:
        if any(not np.isfinite(value) or value < 0 for value in asdict(self).values()):
            raise ValueError("equivalence tolerances must be finite and nonnegative")


def _array_comparison(reference: np.ndarray, candidate: np.ndarray, *, atol: float, rtol: float) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "passed": False, "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape), "max_absolute_error": None,
            "max_relative_error": None,
        }
    absolute = np.abs(np.asarray(reference, dtype=np.float64) - candidate)
    relative = absolute / np.maximum(np.abs(reference), 1e-12)
    return {
        "passed": bool(np.allclose(reference, candidate, atol=atol, rtol=rtol)),
        "max_absolute_error": float(absolute.max(initial=0)),
        "max_relative_error": float(relative.max(initial=0)),
        "atol": atol, "rtol": rtol,
    }


def compare_subspace_models(
    reference: SubspaceModel | FitResult, candidate: SubspaceModel | FitResult,
    query_batches: Iterable[np.ndarray] | TokenGenerator | None = None, *,
    tolerances: EquivalenceTolerances | None = None,
) -> dict[str, Any]:
    """Compare rotation-invariant fitted states and bounded query scores.

    Vector signs and within-subspace eigenbasis rotations are immaterial;
    compare C C.T rather than C. Query features are never used for fitting.
    A passing numerical check is not a measured detection-quality gain.
    """
    tol = tolerances or EquivalenceTolerances()
    ref_fit = reference if isinstance(reference, FitResult) else None
    cand_fit = candidate if isinstance(candidate, FitResult) else None
    ref = reference.model if ref_fit else reference
    cand = candidate.model if cand_fit else candidate
    if ref.feature_dim != cand.feature_dim:
        raise ValueError("compared models must have the same feature dimension")
    checks: dict[str, Any] = {
        "rank": {"passed": ref.k == cand.k, "reference": ref.k, "candidate": cand.k},
        "count": {"passed": ref.count == cand.count, "reference": ref.count, "candidate": cand.count},
        "variance_target": {"passed": ref.variance_target == cand.variance_target},
        "mean": _array_comparison(ref.mean, cand.mean, atol=tol.mean_atol, rtol=tol.mean_rtol),
        "eigenvalues": _array_comparison(ref.eigenvalues, cand.eigenvalues, atol=tol.eigenvalue_atol, rtol=tol.eigenvalue_rtol),
        "projector": _array_comparison(
            ref.components @ ref.components.T, cand.components @ cand.components.T,
            atol=tol.projector_atol, rtol=tol.projector_rtol,
        ),
        "total_variance": _array_comparison(
            np.asarray(ref.total_variance), np.asarray(cand.total_variance),
            atol=tol.eigenvalue_atol, rtol=tol.eigenvalue_rtol,
        ),
    }
    if ref_fit and cand_fit:
        checks["identical_fit_input"] = {
            "passed": (
                ref_fit.input_sha256 == cand_fit.input_sha256 and
                ref_fit.diagnostics["batch_partition_sha256"] == cand_fit.diagnostics["batch_partition_sha256"]
            ),
            "reference_sha256": ref_fit.input_sha256,
            "candidate_sha256": cand_fit.input_sha256,
        }
    if query_batches is not None:
        batches = query_batches() if callable(query_batches) else query_batches
        passed = True
        count = batch_count = 0
        max_absolute = max_relative = 0.0
        digest = hashlib.sha256()
        for raw in batches:
            tokens = _batch(raw, ref.feature_dim)
            digest.update(memoryview(tokens).cast("B"))
            values = _array_comparison(
                ref.reconstruction_scores(tokens), cand.reconstruction_scores(tokens),
                atol=tol.score_atol, rtol=tol.score_rtol,
            )
            passed = passed and values["passed"]
            max_absolute = max(max_absolute, values["max_absolute_error"])
            max_relative = max(max_relative, values["max_relative_error"])
            count += len(tokens)
            batch_count += 1
        if not count:
            raise ValueError("query score comparison requires at least one nonempty batch")
        checks["reconstruction_scores"] = {
            "passed": passed, "token_count": count, "batch_count": batch_count,
            "max_absolute_error": max_absolute, "max_relative_error": max_relative,
            "atol": tol.score_atol, "rtol": tol.score_rtol,
            "query_input_sha256": digest.hexdigest(),
        }
    return {
        "passed": all(value["passed"] for value in checks.values()),
        "checks": checks, "tolerances": asdict(tol),
        "basis_comparison": "rotation_invariant_components_times_transpose",
        "detection_gain_claim": False,
    }


def _close_memmap(values: np.memmap) -> None:
    mapping = getattr(values, "_mmap", None)
    if mapping is not None:
        mapping.close()


class TokenMemmap:
    """Restartable .npy stream; each copied block is unmapped before yield."""

    def __init__(
        self, path: str | Path, *, total_tokens: int | None = None,
        feature_dim: int | None = None, batch_tokens: int = 1024,
    ) -> None:
        self.path = Path(path).resolve(strict=True)
        self.batch_tokens = _positive_int(batch_tokens, "batch_tokens")
        values = np.load(self.path, mmap_mode="r", allow_pickle=False)
        try:
            if not isinstance(values, np.memmap) or values.dtype != np.dtype("float32"):
                raise ValueError("token cache must be an uncompressed float32 .npy")
            if values.ndim != 2 or 0 in values.shape or not values.flags.c_contiguous:
                raise ValueError("token cache must be a nonempty C-order [N,D] matrix")
            self.total_tokens, self.feature_dim = map(int, values.shape)
            self.data_offset = int(values.offset)
            if total_tokens is not None and total_tokens != self.total_tokens:
                raise ValueError("token cache token count mismatch")
            if feature_dim is not None and feature_dim != self.feature_dim:
                raise ValueError("token cache feature dimension mismatch")
        finally:
            if isinstance(values, np.memmap):
                _close_memmap(values)
        self.num_batches = math.ceil(self.total_tokens / self.batch_tokens)

    @property
    def shape(self) -> tuple[int, int]:
        return self.total_tokens, self.feature_dim

    def read_tokens(self, start: int, stop: int) -> np.ndarray:
        if not 0 <= start < stop <= self.total_tokens:
            raise ValueError("token cache slice is out of bounds")
        values = np.memmap(
            self.path, mode="r", dtype=np.float32,
            offset=self.data_offset + start * self.feature_dim * 4,
            shape=(stop - start, self.feature_dim), order="C",
        )
        try:
            return np.array(values, dtype=np.float32, order="C", copy=True)
        finally:
            _close_memmap(values)

    def __call__(self) -> Iterator[np.ndarray]:
        for start in range(0, self.total_tokens, self.batch_tokens):
            yield self.read_tokens(start, min(start + self.batch_tokens, self.total_tokens))


def write_token_memmap(
    destination: str | Path, feature_batches: Iterable[np.ndarray] | TokenGenerator,
    *, total_tokens: int, feature_dim: int, batch_tokens: int = 1024,
) -> TokenMemmap:
    """Exclusively create a bounded .npy cache, flushing/closing each block.

    The caller provides normal-training features only and owns lifecycle of
    this per-category cache. Existing paths are never overwritten. Failed
    writes remove only the newly created incomplete cache.
    """
    _positive_int(total_tokens, "total_tokens")
    _positive_int(feature_dim, "feature_dim")
    _positive_int(batch_tokens, "batch_tokens")
    path = Path(destination)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            np.lib.format.write_array_header_2_0(stream, {
                "descr": np.dtype(np.float32).str,
                "fortran_order": False, "shape": (total_tokens, feature_dim),
            })
            data_offset = stream.tell()
            stream.truncate(data_offset + total_tokens * feature_dim * 4)
        batches = feature_batches() if callable(feature_batches) else feature_batches
        position = 0
        for raw in batches:
            values = _batch(raw, feature_dim)
            if position + len(values) > total_tokens:
                raise ValueError("token cache write exceeds declared token count")
            for start in range(0, len(values), batch_tokens):
                block = values[start : start + batch_tokens]
                mapping = np.memmap(
                    path, mode="r+", dtype=np.float32,
                    offset=data_offset + position * feature_dim * 4,
                    shape=block.shape, order="C",
                )
                try:
                    mapping[:] = block
                    mapping.flush()
                finally:
                    _close_memmap(mapping)
                position += len(block)
        if position != total_tokens:
            raise ValueError("token cache write does not match declared token count")
        return TokenMemmap(path, total_tokens=total_tokens, feature_dim=feature_dim, batch_tokens=batch_tokens)
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise
