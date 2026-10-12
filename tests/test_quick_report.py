"""Integrity checks for quick report/ZIP delivery, using synthetic fixtures."""

import csv
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from industrial_anomaly_lab.quick_report import build_report, package_delivery


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def experiment(tmp_path):
    project, run = tmp_path / "project", tmp_path / "run"
    variants = ["baseline", "shrinkage", "baseline_unsmoothed", "shrinkage_unsmoothed"]
    for name in ["pyproject.toml", "uv.lock", "quickimpl.py", "scripts/package_quick_delivery.py", "scripts/verify_quick_delivery.py", "src/industrial_anomaly_lab/quick_report.py"]:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    license_path = project / ".venv/Lib/site-packages/torchvision-fixture.dist-info/LICENSE"
    license_path.parent.mkdir(parents=True)
    license_path.write_text("BSD 3-Clause License\nCopyright fixture\n", encoding="utf-8")
    weight = project / "assets/resnet18-fixture.pth"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fixture tensor bytes")
    config = {
        "expected_matrix": {"datasets": {"visa": ["candle", "capsules"]}, "variants": variants, "seeds": [7]},
        "source_files": ["quickimpl.py"],
        "backbone": {"architecture": "resnet18", "weight_path": "assets/resnet18-fixture.pth", "sha256": hashlib.sha256(weight.read_bytes()).hexdigest()},
        "reproduction_command": "python quickimpl.py --config run_config.json",
    }
    _write(run / "run_config.json", config)
    _write(run / "run_state.json", {"status": "completed"})
    rows, samples = [], []
    for category_index, category in enumerate(["candle", "capsules"]):
        for variant_index, variant in enumerate(variants):
            rows.append({
                "dataset": "visa", "category": category, "variant": variant, "seed": 7, "status": "completed",
                "image_auroc": 0.8 + category_index * 0.1,
                "pixel_auroc": 0.75 + variant_index * 0.01,
                "pixel_ap": 0.5 - variant_index * 0.01,
                "aupro_0_30": 0.6 + variant_index * 0.005,
            })
        model_dir = run / "models/visa" / category
        model_dir.mkdir(parents=True)
        np.savez_compressed(model_dir / "statistics.npz", mean=np.zeros((2, 2)))
        for label in [0, 1]:
            image = project / "data" / f"{category}-{label}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20), "gray").save(image)
            mask = run / "predictions/visa" / category / "masks" / f"{label}.npy"
            mask.parent.mkdir(parents=True, exist_ok=True)
            np.save(mask, np.eye(20, dtype=bool) if label else np.zeros((20, 20), dtype=bool))
            maps = {}
            for variant_index, variant in enumerate(variants):
                path = run / "predictions/visa" / category / variant / f"{label}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.arange(400).reshape(20, 20).astype(np.float32) * (variant_index + 1) + label)
                maps[variant] = path.relative_to(run).as_posix()
            samples.append({"dataset": "visa", "category": category, "sample_id": f"{category}/{label}", "seed": 7, "label": label, "image_path": image.relative_to(project).as_posix(), "mask_path": mask.relative_to(run).as_posix(), "raw_maps": maps})
    (run / "results").mkdir()
    (run / "results/category_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (run / "results/samples.jsonl").write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    return project, run, rows, samples


def test_incomplete_state_and_partial_matrix_refuse_report(experiment):
    project, run, rows, _ = experiment
    _write(run / "run_state.json", {"status": "running"})
    with pytest.raises(ValueError, match="Run is incomplete"):
        build_report(run, project)
    assert not (run / "report").exists()
    _write(run / "run_state.json", {"status": "completed"})
    (run / "results/category_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete result coverage"):
        build_report(run, project)


def test_report_keeps_metrics_regressions_and_shared_map_scale(experiment):
    project, run, rows, samples = experiment
    original_hash = hashlib.sha256((run / samples[0]["raw_maps"]["baseline"]).read_bytes()).hexdigest()
    result = build_report(run, project)
    assert result["result_count"] == 8
    assert result["figure_count"] == 4
    macro = json.loads((run / "report/macro_metrics.json").read_text(encoding="utf-8"))["rows"]
    baseline = next(row for row in macro if row["variant"] == "baseline")
    assert baseline["image_auroc"] == pytest.approx(0.85)
    with (run / "report/category_metrics.csv").open(encoding="utf-8-sig") as stream:
        saved = list(csv.DictReader(stream))
    assert float(saved[0]["image_auroc"]) == rows[0]["image_auroc"]
    effect = json.loads((run / "report/effect_summary.json").read_text(encoding="utf-8"))
    assert effect["dataset_effects"][0]["delta"]["pixel_ap"] < 0
    assert effect["scientific_improvement_confirmed"] is False
    visuals = json.loads((run / "report/visualization_manifest.json").read_text(encoding="utf-8"))["samples"]
    assert visuals[0]["scale"] == visuals[1]["scale"]
    assert visuals[0]["raw_maps"]["baseline"]["sha256"] == original_hash
    assert visuals[0]["scale"]["upper"] > 1000


def test_zip_verifies_every_member_preserves_origins_and_excludes_dataset(experiment, tmp_path):
    project, run, _, _ = experiment
    output = tmp_path / "delivery.zip"
    receipt = package_delivery(run, output, project)
    assert receipt["all_member_hashes_verified"]
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "assets/weights/resnet18-fixture.pth" in names
        assert "models/visa/candle/statistics.npz" in names
        assert "licenses/torchvision-LICENSE.txt" in names
        assert not any(name.startswith("data/") or name.startswith("predictions/") for name in names)
        sources = json.loads(archive.read("report/source_provenance.json"))
        assert sources["paper"]["url"] == "https://arxiv.org/abs/2011.08785"
        assert sources["original_method_claimed"] is False
        manifest = json.loads(archive.read("DELIVERY-MANIFEST.json"))
        for entry in manifest["files"]:
            data = archive.read(entry["path"])
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            assert len(data) == entry["size_bytes"]
        extracted = tmp_path / "extracted"
        archive.extractall(extracted)
    script = Path(__file__).resolve().parents[1] / "scripts/verify_quick_delivery.py"
    spec = importlib.util.spec_from_file_location("quick_verifier_fixture", script)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    assert verifier.verify_manifest(extracted)["all_member_hashes_verified"]
    assert verifier.verify_inference(extracted, 0) == []
    (extracted / "models/visa/candle/statistics.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="File integrity mismatch"):
        verifier.verify_manifest(extracted)
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        package_delivery(run, output, project)


def test_changed_results_reject_existing_report(experiment, tmp_path):
    project, run, rows, _ = experiment
    build_report(run, project)
    rows[0]["image_auroc"] = 0.99
    (run / "results/category_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Report no longer matches"):
        package_delivery(run, tmp_path / "delivery.zip", project)


def test_nonfinite_metrics_are_never_packaged(experiment, tmp_path):
    project, run, rows, _ = experiment
    rows[0]["pixel_auroc"] = float("nan")
    (run / "results/category_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Metric must be finite"):
        package_delivery(run, tmp_path / "delivery.zip", project)
    assert not (tmp_path / "delivery.zip").exists()
