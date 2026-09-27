"""Strict aggregation and paired statistics for frozen experiment matrices."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .formal_provenance import (
    DEFAULT_PROJECT_CONFIG,
    DEFAULT_REGISTRATION,
    validate_embedded_formal_result,
)
from .formal_test_inventory import (
    validate_formal_prediction_rows,
    validate_formal_test_inventory_evidence,
)
from .metrics import (
    COMMON_IMAGE_AUROC_METRIC,
    COMMON_IMAGE_SCORE_FIELD,
    COMMON_IMAGE_SCORE_SEMANTICS,
    EVALUATION_PROTOCOL,
    METHOD_NATIVE_IMAGE_AUROC_METRIC,
    METHOD_NATIVE_IMAGE_SCORE_FIELD,
    METRICS_SCHEMA_VERSION,
    evaluate_category,
    image_auroc_from_scores,
)
from .run_artifacts import sha256_file


METRIC_NAMES = ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30")


def _validated_metric_number(value: object, name: str, root: Path) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError(f"missing or non-finite {name}: {root}")
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"metric outside [0, 1], {name}={value}: {root}")
    return resolved


def _validate_image_score_contract(
    root: Path,
    config: Mapping[str, object],
    metrics_payload: Mapping[str, object],
    metric_values: Mapping[str, object],
) -> list[Mapping[str, object]]:
    """Recompute both image AUROCs from the hash-bound prediction index."""

    if metrics_payload.get("schema_version") != METRICS_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported or legacy metrics schema; expected "
            f"schema_version={METRICS_SCHEMA_VERSION}: {root}"
        )
    contract = metrics_payload.get("image_score_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"missing image_score_contract: {root}")
    required_contract = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "common_image_auroc_metric": COMMON_IMAGE_AUROC_METRIC,
        "common_image_score_field": COMMON_IMAGE_SCORE_FIELD,
        "common_image_score_semantics": COMMON_IMAGE_SCORE_SEMANTICS,
        "method_native_image_auroc_metric": METHOD_NATIVE_IMAGE_AUROC_METRIC,
        "method_native_image_score_field": METHOD_NATIVE_IMAGE_SCORE_FIELD,
        "raw_map_field": "raw_map",
        "visualization_source_field": "raw_map",
    }
    for field, expected in required_contract.items():
        if contract.get(field) != expected:
            raise ValueError(
                f"image score contract mismatch for {field}: "
                f"expected={expected!r}, observed={contract.get(field)!r}: {root}"
            )
    native_semantics = contract.get("method_native_image_score_semantics")
    if not isinstance(native_semantics, str) or not native_semantics:
        raise ValueError(f"missing method-native score semantics: {root}")

    prediction_contract = config.get("prediction_contract")
    if not isinstance(prediction_contract, Mapping):
        raise ValueError(f"missing prediction_contract: {root}")
    if prediction_contract.get("image_scores") != contract:
        raise ValueError(f"run and metrics image-score contracts disagree: {root}")

    index_path = root / "predictions" / "index.jsonl"
    rows: list[Mapping[str, object]] = []
    with index_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid prediction index JSON on line {line_number}: {root}"
                ) from error
            if not isinstance(row, Mapping):
                raise ValueError(f"prediction row must be an object: {root}")
            rows.append(row)
    if not rows:
        raise ValueError(f"prediction index is empty: {root}")
    if metrics_payload.get("num_images") != len(rows):
        raise ValueError(f"metrics num_images does not match prediction index: {root}")

    labels: list[int] = []
    common_scores: list[float] = []
    native_scores: list[float] = []
    persisted_maps: list[np.ndarray] = []
    persisted_masks: list[np.ndarray] = []
    resolved_root = root.resolve()
    for position, row in enumerate(rows):
        if row.get("index") != position:
            raise ValueError(f"prediction index is not contiguous at row {position}: {root}")
        if "image_score" in row:
            raise ValueError(
                f"legacy generic image_score is forbidden by metrics schema v2: {root}"
            )
        label = row.get("label")
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"invalid prediction label at row {position}: {root}")
        labels.append(label)

        raw_reference = row.get("raw_map")
        if not isinstance(raw_reference, str) or not raw_reference:
            raise ValueError(f"missing raw_map at row {position}: {root}")
        relative = Path(raw_reference)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe raw_map path at row {position}: {root}")
        raw_path = (root / relative).resolve()
        try:
            raw_path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"raw_map escapes run directory at row {position}: {root}") from error
        expected_sha = row.get("raw_map_sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"missing raw_map_sha256 at row {position}: {root}")
        if not raw_path.is_file() or sha256_file(raw_path) != expected_sha:
            raise ValueError(f"raw_map hash mismatch at row {position}: {root}")
        raw_map = np.load(raw_path, allow_pickle=False)
        if raw_map.ndim != 2 or raw_map.size == 0 or not np.isfinite(raw_map).all():
            raise ValueError(f"invalid raw_map array at row {position}: {root}")
        recomputed_common = float(np.max(raw_map))

        mask_reference = row.get("evaluation_mask")
        if not isinstance(mask_reference, str) or not mask_reference:
            raise ValueError(f"missing evaluation_mask at row {position}: {root}")
        mask_relative = Path(mask_reference)
        if mask_relative.is_absolute() or ".." in mask_relative.parts:
            raise ValueError(f"unsafe evaluation_mask path at row {position}: {root}")
        mask_path = (root / mask_relative).resolve()
        try:
            mask_path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"evaluation_mask escapes run directory at row {position}: {root}"
            ) from error
        expected_mask_sha = row.get("evaluation_mask_sha256")
        if not isinstance(expected_mask_sha, str) or len(expected_mask_sha) != 64:
            raise ValueError(
                f"missing evaluation_mask_sha256 at row {position}: {root}"
            )
        if not mask_path.is_file() or sha256_file(mask_path) != expected_mask_sha:
            raise ValueError(f"evaluation_mask hash mismatch at row {position}: {root}")
        evaluation_mask = np.load(mask_path, allow_pickle=False)
        if (
            evaluation_mask.ndim != 2
            or evaluation_mask.shape != raw_map.shape
            or evaluation_mask.size == 0
            or not np.isin(evaluation_mask, (0, 1, False, True)).all()
        ):
            raise ValueError(f"invalid evaluation_mask array at row {position}: {root}")
        persisted_maps.append(raw_map)
        persisted_masks.append(evaluation_mask)

        common_value = row.get(COMMON_IMAGE_SCORE_FIELD)
        if (
            isinstance(common_value, (bool, np.bool_))
            or not isinstance(common_value, (int, float))
            or not np.isfinite(common_value)
            or float(common_value) != recomputed_common
        ):
            raise ValueError(
                f"raw_map_max_score does not match persisted raw_map at row {position}: {root}"
            )
        native_value = row.get(METHOD_NATIVE_IMAGE_SCORE_FIELD)
        if (
            isinstance(native_value, (bool, np.bool_))
            or not isinstance(native_value, (int, float))
            or not np.isfinite(native_value)
        ):
            raise ValueError(
                f"missing or non-finite method_native_image_score at row {position}: {root}"
            )
        common_scores.append(recomputed_common)
        native_scores.append(float(native_value))

    label_array = np.asarray(labels, dtype=np.int64)
    recomputed_common_auroc = image_auroc_from_scores(label_array, common_scores)
    persisted_common_auroc = _validated_metric_number(
        metric_values.get(COMMON_IMAGE_AUROC_METRIC),
        COMMON_IMAGE_AUROC_METRIC,
        root,
    )
    if not np.isclose(recomputed_common_auroc, persisted_common_auroc, rtol=0.0, atol=1e-12):
        raise ValueError(f"image_auroc does not match persisted raw-map maxima: {root}")
    recomputed_native_auroc = image_auroc_from_scores(label_array, native_scores)
    persisted_native_auroc = _validated_metric_number(
        metric_values.get(METHOD_NATIVE_IMAGE_AUROC_METRIC),
        METHOD_NATIVE_IMAGE_AUROC_METRIC,
        root,
    )
    if not np.isclose(recomputed_native_auroc, persisted_native_auroc, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"image_auroc_method_native does not match prediction index: {root}"
        )
    try:
        recomputed_metrics = evaluate_category(
            np.stack(persisted_maps),
            np.stack(persisted_masks),
            label_array,
        ).to_dict()
    except ValueError as error:
        raise ValueError(
            f"persisted raw_map/evaluation_mask inputs are not evaluable: {root}"
        ) from error
    for name in METRIC_NAMES:
        persisted_value = _validated_metric_number(metric_values.get(name), name, root)
        if not np.isclose(
            recomputed_metrics[name], persisted_value, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"{name} does not match persisted raw_map/evaluation_mask files: {root}"
            )
    return rows


def validate_registered_invocation(
    registration: Mapping[str, object],
    *,
    methods: Sequence[str],
    seeds: Sequence[int],
    baseline: str | None,
    candidate: str | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    """Reject a formal aggregation command that differs from its registration."""

    registered_seeds = registration.get("model_seeds")
    if not isinstance(registered_seeds, list) or not all(
        isinstance(seed, int) and not isinstance(seed, bool)
        for seed in registered_seeds
    ):
        raise ValueError("registration is missing model_seeds")
    if list(seeds) != registered_seeds:
        raise ValueError(
            f"formal seeds differ from registration: expected={registered_seeds}, "
            f"observed={list(seeds)}"
        )

    statistics = registration.get("formal_statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("registration is missing formal_statistics")
    registered_samples = statistics.get("bootstrap_samples")
    registered_bootstrap_seed = statistics.get("bootstrap_seed")
    if bootstrap_samples != registered_samples or bootstrap_seed != registered_bootstrap_seed:
        raise ValueError(
            "bootstrap settings differ from registration: "
            f"expected=({registered_samples}, {registered_bootstrap_seed}), "
            f"observed=({bootstrap_samples}, {bootstrap_seed})"
        )

    registry = registration.get("formal_method_registry")
    if not isinstance(registry, Mapping):
        raise ValueError("registration is missing formal_method_registry")
    if registry.get("status") != "frozen_before_formal_evaluation":
        raise ValueError("formal method registry is not frozen")
    registered_methods = registry.get("methods")
    if not isinstance(registered_methods, list) or not registered_methods or not all(
        isinstance(method, str) and method for method in registered_methods
    ):
        raise ValueError("formal method registry has no valid methods")
    if list(methods) != registered_methods:
        raise ValueError(
            f"formal methods differ from registration: expected={registered_methods}, "
            f"observed={list(methods)}"
        )
    if baseline != registry.get("baseline") or candidate != registry.get("candidate"):
        raise ValueError(
            "formal comparison pair differs from registration: "
            f"expected=({registry.get('baseline')}, {registry.get('candidate')}), "
            f"observed=({baseline}, {candidate})"
        )


@dataclass(frozen=True)
class ResultRow:
    method: str
    dataset: str
    category: str
    seed: int
    run_id: str
    image_auroc: float
    pixel_auroc: float
    pixel_ap: float
    aupro_0_30: float

    def metric(self, name: str) -> float:
        if name not in METRIC_NAMES:
            raise ValueError(f"unknown metric: {name}")
        return float(getattr(self, name))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_completed_run(
    run_directory: str | Path,
    *,
    project_config_path: str | Path = DEFAULT_PROJECT_CONFIG,
    registration_path: str | Path = DEFAULT_REGISTRATION,
) -> ResultRow:
    """Load one completed full run from its persisted metric artifacts."""

    root = Path(run_directory)
    with (root / "run_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    with (root / "run_state.json").open(encoding="utf-8") as handle:
        state = json.load(handle)
    with (root / "metrics.json").open(encoding="utf-8") as handle:
        metrics_payload = json.load(handle)

    if state.get("status") != "completed":
        raise ValueError(f"run is not completed: {root}")
    execution_scope = config.get("execution_scope")
    if execution_scope is None:
        raise ValueError(f"missing execution_scope; refusing formal aggregation: {root}")
    if execution_scope != "full":
        raise ValueError(f"debug-limited run cannot enter formal aggregation: {root}")
    if config.get("run_purpose") != "formal_full":
        raise ValueError(f"run_purpose is not formal_full: {root}")
    if config.get("claim_scope") != "formal_matrix_candidate":
        raise ValueError(f"claim_scope is not formal_matrix_candidate: {root}")
    if config.get("formal_matrix_eligible") is not True:
        raise ValueError(f"run is not marked formal-matrix eligible: {root}")
    for key, expected in (
        ("execution_scope", "full"),
        ("run_purpose", "formal_full"),
        ("claim_scope", "formal_matrix_candidate"),
        ("formal_matrix_eligible", True),
    ):
        if state.get(key) != expected:
            raise ValueError(f"run state {key} mismatch: {root}")
    bound_files = {
        "run_config_sha256": root / "run_config.json",
        "metrics_sha256": root / "metrics.json",
        "predictions_index_sha256": root / "predictions" / "index.jsonl",
    }
    for field, path in bound_files.items():
        expected_hash = state.get(field)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"run state is missing {field}: {root}")
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"run artifact hash mismatch for {field}: {root}")
    if metrics_payload.get("protocol") != EVALUATION_PROTOCOL:
        raise ValueError(f"unexpected evaluation protocol: {root}")

    settings = config.get("settings", {})
    if not isinstance(settings, Mapping):
        raise ValueError(f"missing settings: {root}")
    metric_values = metrics_payload.get("metrics", {})
    if not isinstance(metric_values, Mapping):
        raise ValueError(f"missing metrics mapping: {root}")
    method = config.get("method")
    dataset = metrics_payload.get("dataset")
    category = metrics_payload.get("category")
    seed = settings.get("seed")
    if not isinstance(method, str) or not method:
        raise ValueError(f"missing method: {root}")
    if not isinstance(dataset, str) or not isinstance(category, str):
        raise ValueError(f"missing dataset/category: {root}")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"missing or invalid seed: {root}")
    if settings.get("dataset") != dataset or settings.get("category") != category:
        raise ValueError(f"settings and metrics dataset/category mismatch: {root}")
    formal_provenance = config.get("formal_provenance")
    if not isinstance(formal_provenance, Mapping):
        raise ValueError(f"missing formal provenance: {root}")
    validate_embedded_formal_result(
        formal_provenance,
        method=method,
        dataset=dataset,
        category=category,
        seed=seed,
        project_config_path=project_config_path,
        registration_path=registration_path,
    )
    inventory_evidence = formal_provenance.get("formal_test_inventory")
    if not isinstance(inventory_evidence, Mapping):
        raise ValueError(f"formal result is missing test inventory evidence: {root}")
    selection = validate_formal_test_inventory_evidence(
        inventory_evidence,
        dataset=dataset,
        category=category,
        project_config_path=project_config_path,
    )
    for payload, label in (
        (config, "run config"),
        (state, "run state"),
        (metrics_payload, "metrics"),
    ):
        if payload.get("formal_test_inventory") != inventory_evidence:
            raise ValueError(f"{label} formal test inventory binding mismatch: {root}")
    prediction_rows = _validate_image_score_contract(
        root, config, metrics_payload, metric_values
    )
    validate_formal_prediction_rows(prediction_rows, selection)

    resolved_metrics: dict[str, float] = {}
    for name in METRIC_NAMES:
        resolved_metrics[name] = _validated_metric_number(
            metric_values.get(name), name, root
        )

    return ResultRow(
        method=method,
        dataset=dataset,
        category=category,
        seed=seed,
        run_id=root.resolve().as_posix(),
        **resolved_metrics,
    )


def validate_complete_matrix(
    rows: Sequence[ResultRow],
    *,
    methods: Sequence[str],
    dataset_categories: Mapping[str, Sequence[str]],
    seeds: Sequence[int],
) -> None:
    """Reject duplicates, omissions and unregistered formal result rows."""

    expected = {
        (method, dataset, category, seed)
        for method in methods
        for dataset, categories in dataset_categories.items()
        for category in categories
        for seed in seeds
    }
    observed: set[tuple[str, str, str, int]] = set()
    duplicates: list[tuple[str, str, str, int]] = []
    for row in rows:
        key = (row.method, row.dataset, row.category, row.seed)
        if key in observed:
            duplicates.append(key)
        observed.add(key)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if duplicates or missing or extra:
        raise ValueError(
            "formal matrix mismatch: "
            f"duplicates={duplicates[:5]}, missing={missing[:5]} ({len(missing)} total), "
            f"extra={extra[:5]} ({len(extra)} total)"
        )


def macro_by_seed(rows: Iterable[ResultRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[ResultRow]] = {}
    for row in rows:
        grouped.setdefault((row.method, row.dataset, row.seed), []).append(row)
    output: list[dict[str, object]] = []
    for (method, dataset, seed), group in sorted(grouped.items()):
        item: dict[str, object] = {
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "category_count": len(group),
        }
        for metric in METRIC_NAMES:
            item[metric] = float(np.mean([row.metric(metric) for row in group]))
        output.append(item)
    return output


def method_summary(rows: Iterable[ResultRow]) -> list[dict[str, object]]:
    seed_rows = macro_by_seed(rows)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in seed_rows:
        grouped.setdefault((str(row["method"]), str(row["dataset"])), []).append(row)
    output: list[dict[str, object]] = []
    for (method, dataset), group in sorted(grouped.items()):
        item: dict[str, object] = {
            "method": method,
            "dataset": dataset,
            "seed_count": len(group),
            "category_count": group[0]["category_count"],
        }
        for metric in METRIC_NAMES:
            values = np.asarray([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def _category_paired_differences(
    rows: Sequence[ResultRow], baseline: str, candidate: str, dataset: str, metric: str
) -> tuple[list[str], np.ndarray]:
    lookup = {
        (row.method, row.category, row.seed): row.metric(metric)
        for row in rows
        if row.dataset == dataset
    }
    categories = sorted({row.category for row in rows if row.dataset == dataset})
    seeds = sorted({row.seed for row in rows if row.dataset == dataset})
    differences: list[float] = []
    for category in categories:
        per_seed = []
        for seed in seeds:
            baseline_key = (baseline, category, seed)
            candidate_key = (candidate, category, seed)
            if baseline_key not in lookup or candidate_key not in lookup:
                raise ValueError(f"unpaired result for {dataset}/{category}/seed{seed}")
            per_seed.append(lookup[candidate_key] - lookup[baseline_key])
        differences.append(float(np.mean(per_seed)))
    return categories, np.asarray(differences, dtype=np.float64)


def _exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    if differences.size == 0:
        raise ValueError("paired comparison requires at least one category")
    observed = abs(float(differences.mean()))
    extreme = 0
    total = 2 ** differences.size
    for signs in itertools.product((-1.0, 1.0), repeat=differences.size):
        value = abs(float(np.mean(differences * np.asarray(signs))))
        if value >= observed - 1e-15:
            extreme += 1
    return extreme / total


def paired_comparison(
    rows: Sequence[ResultRow],
    *,
    baseline: str,
    candidate: str,
    dataset: str,
    metric: str,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260927,
) -> dict[str, object]:
    """Compare category-level seed means with bootstrap CI and sign-flip test."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    categories, differences = _category_paired_differences(
        rows, baseline, candidate, dataset, metric
    )
    rng = np.random.default_rng(bootstrap_seed)
    sampled_indices = rng.integers(
        0, len(differences), size=(bootstrap_samples, len(differences))
    )
    bootstrap_means = differences[sampled_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "baseline": baseline,
        "candidate": candidate,
        "dataset": dataset,
        "metric": metric,
        "unit": "category mean across registered seeds",
        "category_count": len(categories),
        "categories": categories,
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "improved_category_fraction": float(np.mean(differences > 0.0)),
        "nonnegative_category_fraction": float(np.mean(differences >= 0.0)),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "ci95_percentile": [float(lower), float(upper)],
        "two_sided_exact_sign_flip_p": _exact_sign_flip_pvalue(differences),
        "category_differences": {
            category: float(value) for category, value in zip(categories, differences, strict=True)
        },
    }


def adjudicate_registered_claim(
    comparisons: Sequence[Mapping[str, object]],
    *,
    datasets: Sequence[str],
    primary_metric: str = "aupro_0_30",
    average_gain_min: float = 0.01,
    nonnegative_fraction_min: float = 0.60,
    secondary_regression_max: float = 0.005,
) -> dict[str, object]:
    """Apply the pre-registered cross-dataset improvement gates.

    All thresholds use the native [0, 1] metric scale.  A failed gate is
    preserved in the returned payload rather than being hidden by a pooled
    average.
    """

    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("datasets must be a non-empty unique sequence")
    if primary_metric not in METRIC_NAMES:
        raise ValueError(f"unknown primary metric: {primary_metric}")
    if average_gain_min < 0 or secondary_regression_max < 0:
        raise ValueError("gain and regression thresholds must be non-negative")
    if not 0.0 <= nonnegative_fraction_min <= 1.0:
        raise ValueError("nonnegative_fraction_min must be within [0, 1]")

    lookup: dict[tuple[str, str], Mapping[str, object]] = {}
    for item in comparisons:
        dataset = item.get("dataset")
        metric = item.get("metric")
        if not isinstance(dataset, str) or not isinstance(metric, str):
            raise ValueError("comparison is missing dataset or metric")
        key = (dataset, metric)
        if key in lookup:
            raise ValueError(f"duplicate comparison: {dataset}/{metric}")
        lookup[key] = item

    primary: dict[str, dict[str, object]] = {}
    secondary: dict[str, dict[str, float]] = {}
    for dataset in datasets:
        key = (dataset, primary_metric)
        if key not in lookup:
            raise ValueError(f"missing primary comparison: {dataset}/{primary_metric}")
        item = lookup[key]
        mean = float(item["mean_difference"])
        nonnegative = float(item["nonnegative_category_fraction"])
        interval = item.get("ci95_percentile")
        if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)) or len(interval) != 2:
            raise ValueError(f"invalid confidence interval: {dataset}/{primary_metric}")
        lower, upper = (float(interval[0]), float(interval[1]))
        values = (mean, nonnegative, lower, upper)
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"non-finite claim statistic: {dataset}/{primary_metric}")
        primary[dataset] = {
            "mean_difference": mean,
            "nonnegative_category_fraction": nonnegative,
            "ci95_percentile": [lower, upper],
        }

        secondary[dataset] = {}
        for metric in METRIC_NAMES:
            if metric == primary_metric:
                continue
            secondary_key = (dataset, metric)
            if secondary_key not in lookup:
                raise ValueError(f"missing secondary comparison: {dataset}/{metric}")
            difference = float(lookup[secondary_key]["mean_difference"])
            if not np.isfinite(difference):
                raise ValueError(f"non-finite secondary statistic: {dataset}/{metric}")
            secondary[dataset][metric] = difference

    primary_positive_each = all(
        float(primary[dataset]["mean_difference"]) > 0.0 for dataset in datasets
    )
    equal_dataset_average = float(
        np.mean([float(primary[dataset]["mean_difference"]) for dataset in datasets])
    )
    nonnegative_each = all(
        float(primary[dataset]["nonnegative_category_fraction"])
        >= nonnegative_fraction_min
        for dataset in datasets
    )
    ci_positive_datasets = [
        dataset
        for dataset in datasets
        if float(primary[dataset]["ci95_percentile"][0]) > 0.0  # type: ignore[index]
    ]
    ci_rule = bool(ci_positive_datasets) and primary_positive_each
    secondary_within_limit = all(
        difference >= -secondary_regression_max
        for dataset_metrics in secondary.values()
        for difference in dataset_metrics.values()
    )
    gates = {
        "primary_positive_each_dataset": primary_positive_each,
        "equal_dataset_average_gain": equal_dataset_average >= average_gain_min,
        "nonnegative_category_fraction_each_dataset": nonnegative_each,
        "bootstrap_ci_rule": ci_rule,
        "secondary_metric_regression_limit": secondary_within_limit,
    }
    return {
        "schema_version": 1,
        "decision": "pass" if all(gates.values()) else "fail",
        "primary_metric": primary_metric,
        "datasets": list(datasets),
        "thresholds": {
            "equal_dataset_average_gain": average_gain_min,
            "nonnegative_category_fraction_each_dataset": nonnegative_fraction_min,
            "secondary_metric_regression_max_each_dataset": secondary_regression_max,
        },
        "observed": {
            "primary": primary,
            "equal_dataset_average_gain": equal_dataset_average,
            "ci_lower_positive_datasets": ci_positive_datasets,
            "secondary_mean_differences": secondary,
        },
        "gates": gates,
    }
