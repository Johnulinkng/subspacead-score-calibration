"""Paired category inference for a complete, already observed calibration run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np


def paired_summary(delta, seed=20260927, draws=10000):
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Finite paired category differences required")
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    # All sign assignments, including the observed assignment: an exact test.
    signs = 1 - 2 * ((np.arange(2 ** len(delta))[:, None] >> np.arange(len(delta))) & 1)
    null = (signs * delta).mean(axis=1)
    observed = float(delta.mean())
    p = float(np.mean(np.abs(null) >= abs(observed) - 1e-14))
    return {"categories": len(delta), "mean_delta_pp": observed * 100,
            "median_delta_pp": float(np.median(delta)) * 100,
            "bootstrap_95_ci_pp": (np.quantile(means, [.025, .975]) * 100).tolist(),
            "exact_two_sided_sign_flip_p": p,
            "positive_categories": int((delta > 1e-12).sum()),
            "negative_categories": int((delta < -1e-12).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError("Fresh analysis output required")
    out.mkdir(parents=True)
    metrics = ("aupro_0_30", "pixel_ap", "pixel_auroc", "image_auroc")
    with source.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    lookup = {(r["dataset"], r["category"], r["variant"]): r for r in rows}
    summaries, differences = [], []
    for dataset, expected in (("mvtec_ad", 15), ("visa", 12)):
        cats = sorted({r["category"] for r in rows if r["dataset"] == dataset})
        if len(cats) != expected:
            raise ValueError("Incomplete category matrix")
        for reference in ("baseline", "conditioning_removed"):
            for metric in metrics:
                delta = [float(lookup[dataset,c,"conditional_tail"][metric]) -
                         float(lookup[dataset,c,reference][metric]) for c in cats]
                result = {"dataset": dataset, "candidate": "conditional_tail",
                          "reference": reference, "metric": metric, **paired_summary(delta)}
                summaries.append(result)
                differences.extend({"dataset": dataset, "category": c, "reference": reference,
                                    "metric": metric, "delta_pp": d * 100} for c, d in zip(cats, delta))
    # Holm correction across the entire reported family; no best-metric selection.
    ordered = sorted(range(len(summaries)), key=lambda i: summaries[i]["exact_two_sided_sign_flip_p"])
    previous = 0.
    for rank, i in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered)-rank) * summaries[i]["exact_two_sided_sign_flip_p"]))
        summaries[i]["holm_familywise_p"] = previous
    report = {"scope": "Retrospective single-fit category inference; not repeated training or independent test validation",
              "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "bootstrap_draws": 10000, "bootstrap_seed": 20260927,
              "paired_unit": "category", "results": summaries}
    (out / "statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    for name, records in (("statistics.csv", summaries), ("category_differences.csv", differences)):
        with (out/name).open("w", newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    for r in summaries:
        if r["metric"] in ("aupro_0_30", "pixel_ap"):
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
