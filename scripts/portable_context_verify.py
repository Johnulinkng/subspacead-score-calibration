"""Offline full-matrix verification for the delivered experiment folder."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / "code/src"))
import numpy as np
from industrial_anomaly_lab.context_calibration import ContextCalibration, fit_context_from_cache
from industrial_anomaly_lab.quick_metrics import evaluate_from_files
from industrial_anomaly_lab.subspace_features import render_subspace_map


def replay_map(model, raw, labels, variant):
    if variant == "baseline":
        return render_subspace_map(raw)
    if variant == "global":
        return (render_subspace_map(raw).astype(np.float64) - model.global_center) / model.global_scale
    envelope = model.tails[labels].reshape(32, 32) if variant == "conditional_tail" else model.global_tail
    return render_subspace_map(raw.astype(np.float64) * np.minimum(1., raw / envelope))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    manifest = read(PACKAGE / "manifest.json")
    for rel, expected in manifest["sha256"].items():
        path = (PACKAGE / rel).resolve(strict=True)
        if not path.is_relative_to(PACKAGE) or sha(path) != expected:
            raise ValueError(f"Delivered input changed: {rel}")
    expected_rows = read(PACKAGE / "category_metrics.json")
    index = read(PACKAGE / "sample_index.json")
    cfg = read(PACKAGE / "config.json")
    temporary_root = PACKAGE / ".verify_tmp"
    temporary_root.mkdir(exist_ok=True)
    replay = []
    max_error = 0.
    for d, c in manifest["tasks"]:
        cat = PACKAGE / "data" / d / c
        with np.load(cat / "calibration.npz", allow_pickle=False) as a:
            model = ContextCalibration(a["centers"].copy(), a["tails"].copy(), float(a["global_tail"]), float(a["global_center"]), float(a["global_scale"]))
        with np.load(cat / "normal_context.npz", allow_pickle=False) as a:
            cache = {k: a[k].copy() for k in a.files}
        fitted, _ = fit_context_from_cache(cache, cfg["minimum_group_tokens"], cfg["tail_quantile"])
        for k in ("centers", "tails", "global_tail", "global_center", "global_scale"):
            np.testing.assert_array_equal(getattr(model, k), getattr(fitted, k))
        samples = [r for r in index if (r["dataset"], r["category"]) == (d, c)]
        with TemporaryDirectory(prefix="metric-", dir=temporary_root) as directory:
            tmp = Path(directory).resolve()
            if not tmp.is_relative_to(temporary_root.resolve()):
                raise ValueError("Temporary path escapes package")
            masks = []
            for j, sample in enumerate(samples):
                with np.load(PACKAGE / sample["mask"], allow_pickle=False) as a:
                    mask = a["mask"]
                outfile = tmp / f"mask_{j:06d}.npy"
                np.save(outfile, mask, allow_pickle=False)
                masks.append(outfile)
            baseline_metric = None
            for variant in cfg["variants"]:
                expected = next(r for r in expected_rows if (r["dataset"], r["category"], r["variant"]) == (d, c, variant))
                threshold_maps = np.stack([model.apply(raw, z, variant) for raw, z in zip(cache["residual_grid"][~cache["statistics_partition"]], cache["latent"][~cache["statistics_partition"]], strict=True)])
                pixel_cut = float(np.quantile(threshold_maps, .99, method="higher"))
                image_cut = float(np.quantile(threshold_maps.max(axis=(1, 2)), .95, method="higher"))
                if pixel_cut != expected["pixel_threshold"] or image_cut != expected["image_threshold"]:
                    raise ValueError("Normal operating threshold differs")
                counts = {key: 0 for key in expected["raw_counts"]}
                score_files = []
                for j, sample in enumerate(samples):
                    with np.load(PACKAGE / sample["frame"], allow_pickle=False) as a:
                        scores = replay_map(model, a["residual_grid"], a["cluster_labels"], variant)
                    if hashlib.sha256(np.ascontiguousarray(scores).tobytes()).hexdigest() != sample["map_content_sha256"][variant]:
                        raise ValueError("Reconstructed raw map differs from packaged experiment")
                    mask = np.load(masks[j], allow_pickle=False).astype(bool)
                    counts["background_pixels"] += int((~mask).sum())
                    counts["background_fp"] += int(np.count_nonzero(scores[~mask] > pixel_cut))
                    counts["foreground_pixels"] += int(mask.sum())
                    counts["foreground_tp"] += int(np.count_nonzero(scores[mask] > pixel_cut))
                    if sample["label"] == 0:
                        counts["normal_pixels"] += scores.size
                        counts["normal_fp"] += int(np.count_nonzero(scores > pixel_cut))
                        counts["normal_images"] += 1
                        counts["normal_image_fp"] += int(scores.max() > image_cut)
                    else:
                        counts["anomaly_images"] += 1
                        counts["anomaly_image_tp"] += int(scores.max() > image_cut)
                    # Global affine preserves all ranks. It is still applied
                    # to every image above and its operating counts checked;
                    # exact pixel metrics reuse freshly recomputed baseline.
                    if variant != "global":
                        if shutil.disk_usage(PACKAGE).free < cfg["resource_policy"]["minimum_disk_gib"] * 1024**3:
                            raise RuntimeError("Frozen disk reserve reached")
                        outfile = tmp / f"{variant}_{j:06d}.npy"
                        np.save(outfile, scores, allow_pickle=False)
                        score_files.append(outfile)
                if variant == "global":
                    metric = baseline_metric.copy()
                else:
                    metric = evaluate_from_files(score_files, masks, [s["label"] for s in samples], temp_root=tmp)
                    if variant == "baseline":
                        baseline_metric = metric.copy()
                error = max(abs(metric[k] - expected[k]) for k in metric)
                max_error = max(max_error, error)
                if error != 0 or counts != expected["raw_counts"]:
                    raise ValueError(f"Offline replay differs: {d}/{c}/{variant}")
                replay.append({"dataset": d, "category": c, "variant": variant, **metric})
                for path in score_files:
                    if path.resolve().parent != tmp:
                        raise ValueError("Unexpected generated output path")
                    path.unlink()
        print(f"verified {d}/{c}", flush=True)
    macros = read(PACKAGE / "dataset_macro_metrics.json")
    for row in macros:
        subset = [r for r in replay if (r["dataset"], r["variant"]) == (row["dataset"], row["variant"])]
        for k in ("image_auroc", "pixel_auroc", "pixel_ap", "aupro_0_30"):
            if float(np.mean([r[k] for r in subset])) != row[k]:
                raise ValueError("Macro replay differs")
    receipt = {"status": "passed", "categories": 27, "test_images": len(index), "four_group_rows": len(replay),
        "max_metric_error": max_error, "normal_parameters_and_operating_counts": "passed",
        "method": "Reconstruct maps from actual residual grids and verified content assignments, checking every raw-map digest and original GT masks; independently recompute baseline/candidate/ablation exact metrics, use proved global-affine rank invariance for global metrics. Full float16 Test descriptors remain in the source experiment; this is offline metric replay, not a new backbone run."}
    (PACKAGE / "复算结果.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
