"""Cross-check the project AUPRO implementation against MVTec's official code.

This script deliberately imports the untouched code extracted from the official
``mvtec_ad_evaluation.tar.xz`` archive.  It exercises fixed edge cases and a
deterministic randomized corpus, then writes a machine-readable report under
``artifacts/validation``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OFFICIAL_DIR = ROOT / "third_party" / "mvtec_ad_evaluation"
ARCHIVE = ROOT / "data" / "archives" / "mvtec_ad_evaluation.tar.xz"
REPORT = ROOT / "artifacts" / "validation" / "aupro_official_crosscheck.json"
OFFICIAL_URL = (
    "https://www.mydrive.ch/shares/150450/"
    "bb24b914a28ddd2b5e35bd53d23177cd/download/"
    "439517473-1665675012/mvtec_ad_evaluation.tar.xz"
)

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(OFFICIAL_DIR))

from industrial_anomaly_lab.metrics import aupro as project_aupro  # noqa: E402
from generic_util import trapezoid as official_trapezoid  # noqa: E402
from pro_curve_util import compute_pro as official_compute_pro  # noqa: E402


@dataclass(frozen=True)
class Comparison:
    name: str
    fpr_limit: float
    official: float
    project: float
    absolute_difference: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_aupro(scores: np.ndarray, masks: np.ndarray, limit: float) -> float:
    # The official helper emits progress text on every call.  Silence it so
    # large deterministic fuzz runs leave a concise audit log.
    with contextlib.redirect_stdout(io.StringIO()):
        fpr, pro = official_compute_pro(list(scores), list(masks))
    return float(official_trapezoid(fpr, pro, x_max=limit) / limit)


def compare(
    name: str, scores: np.ndarray, masks: np.ndarray, limit: float
) -> Comparison:
    official = official_aupro(scores, masks, limit)
    project = project_aupro(scores, masks, fpr_limit=limit)
    return Comparison(
        name=name,
        fpr_limit=limit,
        official=official,
        project=project,
        absolute_difference=abs(official - project),
    )


def fixed_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    cases: list[tuple[str, np.ndarray, np.ndarray]] = []

    mask = np.zeros((2, 4, 4), dtype=np.uint8)
    mask[1, 1:3, 1:3] = 1
    cases.append(("perfect_with_normal_image", mask.astype(np.float64), mask))
    cases.append(("reverse_ranking", 1.0 - mask.astype(np.float64), mask))
    cases.append(("all_scores_tied", np.ones_like(mask, dtype=np.float64), mask))

    mask = np.zeros((1, 5, 6), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[0, 2, 2:5] = 1
    score = np.zeros_like(mask, dtype=np.float64)
    score[0, 0, 0] = 2.0
    score[0, 2, 2:5] = -1.0
    cases.append(("unequal_region_sizes", score, mask))

    mask = np.zeros((1, 5, 5), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[0, 1, 1] = 1
    mask[0, 4, 4] = 1
    score = np.zeros_like(mask, dtype=np.float64)
    score[0, 0, 0] = 2.0
    score[0, 1, 1] = -1.0
    score[0, 4, 4] = 2.0
    cases.append(("eight_connected_diagonal", score, mask))

    mask = np.zeros((2, 3, 5), dtype=np.uint8)
    mask[0, 1, 1] = 1
    score = np.zeros_like(mask, dtype=np.float64)
    score[0, 1, 1] = 1.0
    score[1] = 2.0
    cases.append(("normal_image_background", score, mask))

    mask = np.zeros((1, 2, 6), dtype=np.uint8)
    mask[0, 0, 0:2] = 1
    score = np.array(
        [[[9, 6, 8, 7, 6, 6], [5, 4, 3, 2, 1, 0]]], dtype=np.float64
    )
    cases.append(("exact_fpr_boundary_and_ties", score, mask))

    return cases


def randomized_cases(seed: int = 20260927, count: int = 500) -> list[Comparison]:
    rng = np.random.default_rng(seed)
    results: list[Comparison] = []
    limits = (0.01, 0.05, 0.10, 0.30, 1.00)

    for index in range(count):
        image_count = int(rng.integers(1, 5))
        height = int(rng.integers(2, 9))
        width = int(rng.integers(2, 9))
        masks = (rng.random((image_count, height, width)) < 0.22).astype(np.uint8)
        if masks.all() or not masks.any():
            masks.flat[0] = 1
            masks.flat[-1] = 0

        # Alternate continuous scores and heavily tied scores.  This targets
        # the threshold-grouping semantics that commonly vary across AUPRO
        # implementations.
        if index % 2:
            scores = rng.integers(-2, 4, size=masks.shape).astype(np.float64)
        else:
            scores = rng.normal(size=masks.shape)

        for limit in limits:
            results.append(compare(f"random_{index:04d}", scores, masks, limit))

    return results


def main() -> None:
    expected_archive_sha256 = (
        "dfcda7d67eee25316ec6ae5042c0b1684a4cabf33b2346be351e2ce36013f220"
    )
    archive_sha256 = sha256(ARCHIVE)
    if archive_sha256 != expected_archive_sha256:
        raise RuntimeError(
            f"official archive SHA-256 mismatch: {archive_sha256}"
        )

    fixed: list[Comparison] = []
    for name, scores, masks in fixed_cases():
        for limit in (0.01, 0.05, 0.10, 0.30, 1.00):
            fixed.append(compare(name, scores, masks, limit))

    randomized = randomized_cases()
    all_results = fixed + randomized
    max_difference = max(item.absolute_difference for item in all_results)
    tolerance = 1e-6
    failures = [
        asdict(item) for item in all_results if item.absolute_difference > tolerance
    ]

    payload = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "official_source": {
            "page": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
            "archive_url": OFFICIAL_URL,
            "archive_path": str(ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
            "archive_size_bytes": ARCHIVE.stat().st_size,
            "archive_sha256": archive_sha256,
            "license": "BSD-3-Clause; see extracted LICENSE.txt",
            "pro_curve_util_sha256": sha256(OFFICIAL_DIR / "pro_curve_util.py"),
            "generic_util_sha256": sha256(OFFICIAL_DIR / "generic_util.py"),
        },
        "protocol": {
            "connectivity": 8,
            "region_weighting": "equal connected-component weight",
            "background": "all non-anomalous pixels, including normal images",
            "tie_handling": "one curve point after the complete tied-score group",
            "integration": "linear interpolation at limit, trapezoidal area / limit",
        },
        "comparison": {
            "fixed_case_count": len(fixed_cases()),
            "fixed_comparison_count": len(fixed),
            "random_seed": 20260927,
            "random_case_count": 500,
            "random_comparison_count": len(randomized),
            "fpr_limits": [0.01, 0.05, 0.10, 0.30, 1.00],
            "tolerance": tolerance,
            "max_absolute_difference": max_difference,
            "failures": failures[:20],
            "fixed_results": [asdict(item) for item in fixed],
        },
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "comparisons": len(all_results),
        "max_absolute_difference": max_difference,
        "report": str(REPORT),
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
