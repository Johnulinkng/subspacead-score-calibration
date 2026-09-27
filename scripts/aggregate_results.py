"""Aggregate a complete frozen experiment matrix into auditable tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_anomaly_lab.result_aggregation import (  # noqa: E402
    METRIC_NAMES,
    adjudicate_registered_claim,
    load_completed_run,
    macro_by_seed,
    method_summary,
    paired_comparison,
    validate_complete_matrix,
    validate_registered_invocation,
)
from industrial_anomaly_lab.run_artifacts import (  # noqa: E402
    prepare_output_directory,
    sha256_file,
    utc_now,
    write_json,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260927)
    args = parser.parse_args()
    if (args.baseline is None) != (args.candidate is None):
        raise ValueError("--baseline and --candidate must be provided together")

    with (PROJECT_ROOT / "configs" / "project.json").open(encoding="utf-8") as handle:
        project = json.load(handle)
    registration_path = PROJECT_ROOT / "configs" / "preregistered_experiments.json"
    with registration_path.open(encoding="utf-8") as handle:
        registration = json.load(handle)
    claim_rules = registration["formal_claim_rules"]
    if claim_rules["primary_metric"] != "AUPRO@0.30":
        raise ValueError("registered primary metric is not supported by this aggregator")
    categories = {
        dataset: payload["categories"] for dataset, payload in project["datasets"].items()
    }
    validate_registered_invocation(
        registration,
        methods=args.methods,
        seeds=args.seeds,
        baseline=args.baseline,
        candidate=args.candidate,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    metric_paths = sorted(args.runs_root.resolve().rglob("metrics.json"))
    if not metric_paths:
        raise ValueError(f"no metrics.json files below {args.runs_root}")
    rows = [load_completed_run(path.parent) for path in metric_paths]
    validate_complete_matrix(
        rows,
        methods=args.methods,
        dataset_categories=categories,
        seeds=args.seeds,
    )

    output = prepare_output_directory(args.output_dir)
    detail_rows = [row.to_dict() for row in rows]
    seed_rows = macro_by_seed(rows)
    summary_rows = method_summary(rows)
    _write_csv(output / "per_category_seed.csv", detail_rows)
    _write_csv(output / "macro_by_seed.csv", seed_rows)
    _write_csv(output / "method_summary.csv", summary_rows)

    comparisons: list[dict[str, object]] = []
    claim_adjudication = None
    if args.baseline is not None:
        for dataset in categories:
            for metric in METRIC_NAMES:
                comparisons.append(
                    paired_comparison(
                        rows,
                        baseline=args.baseline,
                        candidate=args.candidate,
                        dataset=dataset,
                        metric=metric,
                        bootstrap_samples=args.bootstrap_samples,
                        bootstrap_seed=args.bootstrap_seed,
                    )
                )
        claim_adjudication = adjudicate_registered_claim(
            comparisons,
            datasets=list(categories),
            primary_metric="aupro_0_30",
            average_gain_min=float(claim_rules["equal_dataset_average_gain_pp_min"]) / 100.0,
            nonnegative_fraction_min=float(
                claim_rules["category_nonnegative_fraction_min_each_dataset"]
            ),
            secondary_regression_max=float(
                claim_rules["secondary_metric_macro_regression_pp_max_each_dataset"]
            )
            / 100.0,
        )
    payload = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "runs_root": str(args.runs_root.resolve()),
        "methods": args.methods,
        "seeds": args.seeds,
        "dataset_categories": categories,
        "registration": {
            "path": registration_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(registration_path),
            "registration_id": registration["registration_id"],
        },
        "matrix_validation": "passed",
        "row_count": len(rows),
        "comparisons": comparisons,
        "claim_adjudication": claim_adjudication,
    }
    write_json(output / "aggregation_manifest.json", payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "comparisons"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
