"""Record successful route-1 acceptance, then leave final seals to the caller.

Stdlib and metadata only. Never compute metrics, inference, PCA, or query/start
processes. The earlier pending integration receipt remains unchanged history.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from finish_route1_delivery import EXTENSION, METHODS, METRICS, ROOT_DOCS, copy_checked, digest, member, read, require


PENDING = "审计核验模型、全部raw map、GT和图像AUROC；本入口整合时，完整像素指标复算和新的CPU原图推理验收尚待执行，最终状态与回执随后保存到验收记录。"
COMPLETED = "独立语义审计核验模型、全部raw map、GT和图像AUROC；新增离线验收已完整复算27类162行四项指标，并对固定2张原图的三种方法完成6张CPU分数图核对。完整指标与CPU推理回执已保存在同骨干对照目录的验收记录中；原整合pending回执保留较早阶段状态。"
REPRO_PENDING = "独立完整运行审计只复核保存的像素指标一致性，完整像素重算与新CPU前向需要上述验收回执单独证明。"
REPRO_COMPLETED = "独立语义审计保留其原有范围：未重算像素指标，也未执行新前向。新增离线验收已复算27类162行四项指标，并核对固定2张原图×3方法的6张CPU分数图；实际回执见验收记录/完整指标与CPU推理验收.json。再次验收仍使用上述命令，输出置于交付文件夹外。"
FULL_RECEIPT = "验收记录/完整指标与CPU推理验收.json"
AUDIT_RECEIPT = "验收记录/独立完整运行审计.json"
FINAL_SUMMARY = "验收记录/最终验收摘要.json"


def replace_once(raw: bytes, old: str, new: str) -> bytes:
    before, after = old.encode("utf-8"), new.encode("utf-8")
    require(raw.count(before) == 1, "Expected exactly one pending acceptance sentence")
    return raw.replace(before, after, 1)


def finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def validate_verifier(receipt: dict[str, Any], tasks: list[tuple[str, str]], selected: list[dict[str, Any]], config: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    require(receipt.get("status") == "verified", "Successful new delivery verifier receipt required")
    require(all(receipt.get(key) is False for key in ("network_access_required", "gradient_training_performed", "normal_training_fit_performed", "gpu_used")) and receipt.get("cpu_threads") == 4, "Acceptance computation/network/device boundary differs")
    require(receipt["integrity"]["status"] == "verified" and receipt["integrity"]["extension_members"] == len(info["files"]), "Acceptance receipt does not cover the current sealed extension membership")
    require(set(receipt["integrity"]["root_documents_archived_before_change"]) == set(ROOT_DOCS), "Acceptance receipt did not authenticate the three exact historical root archives")
    sources = receipt["sources"]
    require(sources["status"] == "verified" and sources["same_historical_dino_checkpoint"] is True and sources["source_files"] == len(config["source_sha256"]) and sources["vendor_files"] == len(config.get("vendor_source_sha256", {})), "Acceptance source/backbone scope differs")
    cache = receipt["cache"]
    coverage = {"status": "verified", "categories": 27, "normal_train_images": 12288, "test_images": 3887, "raw_maps": 11661, "evaluation_masks": 3887, "pca_models": 81, "metric_rows": 162, "all_maps_masks_and_models_authenticated": True}
    require(all(cache.get(key) == value for key, value in coverage.items()), "Acceptance does not authenticate all 27 classes/81 models/3887 inputs/162 rows")
    matrix = {(d, c, method + "_" + score) for d, c in tasks for method in METHODS for score in ("max", "top1p")}
    cached_rows = cache["cached_image_auroc_replays"]
    require(len(cached_rows) == 162 and {(row["dataset"], row["category"], row["variant"]) for row in cached_rows} == matrix and all(finite_nonnegative(row["absolute_error"]) and row["absolute_error"] <= 1e-12 for row in cached_rows), "Cached image AUROC acceptance scope differs")
    metrics = receipt["metric_replay"]
    require(len(metrics) == 162 and {(row["dataset"], row["category"], row["variant"]) for row in metrics} == matrix and all(row["status"] == "verified" and set(row["absolute_errors"]) == set(METRICS) and all(finite_nonnegative(error) and error <= 1e-12 for error in row["absolute_errors"].values()) for row in metrics), "Full 27-category four-metric replay is missing, duplicated, or failed")
    require(len(selected) >= 2 and len({(row["dataset"], row["category"], row["sample_id"]) for row in selected[:2]}) == 2, "The two fixed original inputs are missing")
    expected_inference = {(row["dataset"], row["category"], row["sample_id"], method) for row in selected[:2] for method in METHODS}
    inference = receipt["fresh_cpu_inference"]
    require(len(inference) == 6 and {(row["dataset"], row["category"], row["sample_id"], row["method"]) for row in inference} == expected_inference, "Fresh inference must cover exactly the first two fixed inputs and all three methods")
    for row in inference:
        require(row["status"] == "verified" and row["device"] == "cpu" and all(finite_nonnegative(row[key]) for key in ("maximum_absolute_error", "mean_absolute_error", "atol", "rtol")) and row["mean_absolute_error"] <= row["maximum_absolute_error"] and row["atol"] <= 1e-4 and row["rtol"] <= 1e-4, "Fresh CPU inference failed or changed the documented tolerance boundary")
    resources = receipt["resources"]
    resource_coverage = {"status": "verified", "trials": 117, "failed_or_blocked_final_trials": 0, "full_category_fit_timings_authenticated": 81, "successful_registered_trials_protocol_authenticated": 36, "all_resource_matrix_failures_preserved": True, "source_measurement_scopes_retained": True}
    require(all(resources.get(key) == value for key, value in resource_coverage.items()) and resources["failed_retry_attempts_retained"] == info["resource_failed_retry_attempts_retained"], "Acceptance resource or retained failure scope differs")
    return {"categories": 27, "normal_train_images": 12288, "test_images": 3887, "pca_models": 81, "raw_maps": 11661, "metric_rows_replayed": 162, "metrics_categories_replayed": 27, "fresh_original_inputs": 2, "fresh_method_maps": 6, "resource_fresh_trials": 36, "full_category_fit_timings": 81, "failed_resource_retries_retained": resources["failed_retry_attempts_retained"], "maximum_replayed_metric_absolute_error": max(error for row in metrics for error in row["absolute_errors"].values()), "fresh_inference_tolerances": sorted({(row["atol"], row["rtol"]) for row in inference}), "cpu_used": True, "gpu_used": False, "new_training_or_pca_fit_performed": False}


def sealed_extension(package: Path, extension: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    info = read(member(extension, "extension_manifest.json"))
    expected = {"status": "completed", "categories": 27, "normal_train_images": 12288, "test_images": 3887, "metric_rows": 162, "pca_models": 81, "raw_maps": 11661, "evaluation_masks": 3887, "resource_trials": 117, "scientific_claim_eligible": False}
    require(all(info.get(key) == value for key, value in expected.items()), "Complete route-1 extension required")
    declared = set()
    for row in info["files"]:
        path = member(extension, row["path"])
        require(row["path"] not in declared and not path.is_symlink() and path.stat().st_size == row["size_bytes"] and digest(path) == row["sha256"], "Extension membership or original seal changed")
        declared.add(row["path"])
    actual = {path.relative_to(extension).as_posix() for path in extension.rglob("*") if path.is_file()}
    require(actual == declared | {"extension_manifest.json"}, "Extension was modified after the acceptance seal")
    historical = member(extension, "history/DELIVERY-MANIFEST.json")
    require(digest(historical) == info["previous_delivery_manifest_sha256"], "Original delivery manifest snapshot changed")
    original = read(historical)
    for row in original["files"]:
        current = member(package, row["path"])
        if row["path"] in ROOT_DOCS:
            current = member(extension, "history/" + row["path"])
        require(current.stat().st_size == row["size_bytes"] and digest(current) == row["sha256"], "Historical evidence, source, license, or exact root archive changed")
    integrated = read(member(extension, "整合回执.json"))
    require(integrated["status"] == "integrated_pending_main_verification_and_seals" and integrated["historical_sources_models_and_metrics_replaced"] is False, "Expected historical pending integration receipt")
    for name in ROOT_DOCS:
        require(digest(member(package, name)) == integrated["root_integrated_sha256"][name] and digest(member(extension, "history/" + name)) == integrated["root_original_sha256"][name], "Root documentation changed since its recorded integration")
    return info, integrated


def prepare(project: Path, package: Path, verifier_path: Path, auditor_path: Path) -> dict[str, Any]:
    require(package.is_relative_to(project / "artifacts/deliveries"), "Explicit project delivery directory required")
    for path in (verifier_path, auditor_path):
        require(path.is_file() and not path.is_symlink() and path.is_relative_to(project) and not path.is_relative_to(package), "Acceptance receipts must be plain external project files")
    extension = package / EXTENSION
    require(not (extension / FINAL_SUMMARY).exists(), "Final acceptance is already recorded; refusing replacement")
    info, integrated = sealed_extension(package, extension)
    require(digest(Path(member.__code__.co_filename).resolve()) == digest(member(extension, "code/scripts/finish_route1_delivery.py")), "Acceptance utility's imported integration helper differs from its delivered source")
    config, state = read(member(extension, "run/run_config.json")), read(member(extension, "run/run_state.json"))
    require(all(state.get(key) == value for key, value in {"status": "completed", "completed_categories": 27, "training_images": 12288, "test_images": 3887, "metric_rows": 162, "current_task": None}.items()), "Copied precision run did not finish")
    tasks = [tuple(pair) for pair in config["expected_matrix"]["categories"]]
    require(len(tasks) == len(set(tasks)) == 27 and {dataset: sum(d == dataset for d, _ in tasks) for dataset, _ in tasks} == {"mvtec_ad": 15, "visa": 12} and tuple(config["expected_matrix"]["methods"]) == METHODS, "Frozen category/method matrix differs")
    for section in ("source_sha256", "environment_sha256"):
        for relative, expected in config[section].items():
            require(digest(member(extension / "code", relative)) == expected, "Delivered executed source/environment changed")
    auditor = read(auditor_path)
    audit_coverage = {"status": "passed", "categories": 27, "metric_rows": 162, "normal_training_images": 12288, "test_images": 3887, "raw_maps": 11661, "bool_masks": 3887, "selected_samples": 54, "scientific_claim_eligible": False, "forward_or_training_executed": False, "pixel_metrics_recomputed": False}
    require(all(auditor.get(key) == value for key, value in audit_coverage.items()) and tuple(auditor["methods"]) == METHODS and Path(auditor["run"]).resolve() == Path(info["source_run"]).resolve(), "Independent auditor receipt belongs to a different run or scope")
    require(len(auditor["category_audits"]) == 27 and {(row["dataset"], row["category"]) for row in auditor["category_audits"]} == set(tasks) and all(set(row["models"]) == set(METHODS) for row in auditor["category_audits"]), "Auditor did not authenticate all 81 PCA models")
    for filename, field in (("run_config.json", "run_config_sha256"), ("run_state.json", "run_state_sha256"), ("results/category_metrics.jsonl", "category_metrics_sha256"), ("results/samples.jsonl", "samples_sha256")):
        require(digest(member(extension / "run", filename)) == auditor[field], "Auditor is not bound to the accepted frozen run bytes")
    auditor_sha = digest(auditor_path)
    require(digest(member(extension, "code/scripts/audit_route1_run.py")) == auditor["auditor_source_sha256"] and auditor_sha == integrated["independent_audit_sha256"], "Auditor source or prior integration binding differs")
    old_auditor = extension / AUDIT_RECEIPT
    require(not old_auditor.exists() or digest(old_auditor) == auditor_sha, "Different independent audit receipt is already copied")
    verifier = read(verifier_path)
    identity = verifier["verification_identity"]
    identity_fields = {"package_root": str(package), "extension_manifest_sha256": digest(member(extension, "extension_manifest.json")),
                       "run_config_sha256": digest(member(extension, "run/run_config.json")), "run_state_sha256": digest(member(extension, "run/run_state.json")),
                       "category_metrics_sha256": digest(member(extension, "run/results/category_metrics.jsonl")), "samples_sha256": digest(member(extension, "run/results/samples.jsonl")),
                       "verifier_source_sha256": digest(member(extension, "code/scripts/verify_route1_delivery.py")),
                       "requested_metrics_categories": 27, "requested_inference_samples": 2, "requested_inference_atol": 1e-4, "requested_inference_rtol": 1e-4}
    require(all(identity.get(key) == value for key, value in identity_fields.items()) and isinstance(identity.get("requested_verify_master"), bool) and identity["requested_verify_master"] == verifier["integrity"]["master_manifest_verified"], "Full acceptance receipt is not bound to this exact package/seal/run/source/default call scope")
    require(all(row["atol"] == identity["requested_inference_atol"] and row["rtol"] == identity["requested_inference_rtol"] for row in verifier["fresh_cpu_inference"]), "Fresh inference tolerances differ from the actual recorded invocation")
    selected = read(member(extension, "selected_samples.json"))["samples"]
    scope = validate_verifier(verifier, tasks, selected, config, info)
    require(verifier["integrity"]["historical_members_preserved"] == len(read(member(extension, "history/DELIVERY-MANIFEST.json"))["files"]), "Verifier historical preservation scope differs")
    full_destination = extension / FULL_RECEIPT
    require(not full_destination.exists(), "Refusing to replace a prior full acceptance receipt")
    completed_html = html.escape(COMPLETED) + f" <a href='{EXTENSION}/{FULL_RECEIPT}'>完整指标与CPU推理回执</a>。"
    modifications = []
    for name in ROOT_DOCS:
        path = member(package, name)
        raw = path.read_bytes()
        updated = replace_once(raw, PENDING, completed_html if name.endswith(".html") else COMPLETED)
        modifications.append((path, raw, updated))
    colleague = member(extension, "给同事的说明.txt")
    raw = colleague.read_bytes()
    modifications.append((colleague, raw, replace_once(raw, PENDING, COMPLETED)))
    reproduction = member(extension, "复现说明.txt")
    raw = reproduction.read_bytes()
    modifications.append((reproduction, raw, replace_once(raw, REPRO_PENDING, REPRO_COMPLETED)))
    return {"extension": extension, "info": info, "scope": scope, "modifications": modifications, "verifier_sha256": digest(verifier_path), "auditor_sha256": auditor_sha, "integration_receipt_sha256": digest(member(extension, "整合回执.json")), "accepted_extension_manifest_sha256": digest(member(extension, "extension_manifest.json")), "historical_manifest_sha256": info["previous_delivery_manifest_sha256"], "verifier_source_sha256": digest(member(extension, "code/scripts/verify_route1_delivery.py"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--verifier-receipt", type=Path, required=True)
    parser.add_argument("--audit-receipt", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        project, package, verifier, auditor = (path.resolve(strict=True) for path in (args.project_root, args.package_root, args.verifier_receipt, args.audit_receipt))
        prepared = prepare(project, package, verifier, auditor)
        if args.check_only:
            print(json.dumps({"status": "ready_to_finalize_acceptance", "delivery_modified": False, "scope": prepared["scope"], "documentation_files": [path.relative_to(package).as_posix() for path, _, _ in prepared["modifications"]]}, ensure_ascii=False, indent=2))
            return 0
        extension = prepared["extension"]
        copied = [copy_checked(verifier, extension / FULL_RECEIPT, prepared["verifier_sha256"]), copy_checked(auditor, extension / AUDIT_RECEIPT, prepared["auditor_sha256"])]
        for path, original, _ in prepared["modifications"]:
            require(path.read_bytes() == original, "Acceptance documentation changed during evidence copy")
        updated_documents = []
        for path, original, updated in prepared["modifications"]:
            path.write_bytes(updated)
            updated_documents.append({"path": path.relative_to(package).as_posix(), "before_sha256": hashlib.sha256(original).hexdigest(), "after_sha256": digest(path), "only_the_unique_pending_sentence_replaced": True})
        require(digest(member(extension, "整合回执.json")) == prepared["integration_receipt_sha256"], "Historical pending integration receipt was changed")
        summary = {"schema_version": 1, "status": "completed", "scope": "complete cached metrics and fixed two-input CPU inference acceptance; file seals follow this write", "accepted_at_utc": datetime.now(timezone.utc).isoformat(), **prepared["scope"], "scientific_claim_eligible": False, "primary_image_score": "max", "auxiliary_image_score": "top1p", "algorithm_component_added": False, "source_verifier_sha256": prepared["verifier_source_sha256"], "acceptance_utility_sha256": digest(Path(__file__).resolve()), "full_acceptance_receipt": FULL_RECEIPT, "full_acceptance_receipt_sha256": prepared["verifier_sha256"], "independent_auditor_receipt": AUDIT_RECEIPT, "independent_auditor_receipt_sha256": prepared["auditor_sha256"], "accepted_pre_update_extension_manifest_sha256": prepared["accepted_extension_manifest_sha256"], "historical_delivery_manifest_sha256": prepared["historical_manifest_sha256"], "pending_integration_receipt_preserved_as_history": True, "pending_integration_receipt_sha256": prepared["integration_receipt_sha256"], "updated_documents": updated_documents, "copied_receipts": copied, "metrics_or_inference_computed_by_this_utility": False, "extension_manifest_updated_by_this_utility": False, "master_manifest_updated_by_this_utility": False, "sealing_required_after_this_write": True, "final_master_hash_check_performed_by_this_utility": False, "next_steps": ["copy this utility source into extension/code if not already present", "refresh extension manifest with builder --seal-only", "refresh master DELIVERY-MANIFEST.json", "perform final master and historical/current hash verification with receipt outside the delivery"], "self_hash_scope": "this summary and the final manifests are not assigned their own hashes; recorded extension SHA identifies the pre-update acceptance seal"}
        path = extension / FINAL_SUMMARY
        with path.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "acceptance_finalization_failed", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
