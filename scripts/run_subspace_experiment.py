"""Frozen DINOv2-S/SubspaceAD full-normal experiment with bounded resources."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(tmp, path)


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")


def checked(root, path, expected):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or digest(path) != expected:
        raise ValueError(f"Input path/content drift: {path}")
    return path


def process_memory():
    if os.name != "nt":
        return {}
    class Counters(ctypes.Structure):
        _fields_ = [("cb",ctypes.c_ulong),("faults",ctypes.c_ulong),
            ("peak",ctypes.c_size_t),("current",ctypes.c_size_t),
            ("pp_peak",ctypes.c_size_t),("pp",ctypes.c_size_t),("np_peak",ctypes.c_size_t),
            ("np",ctypes.c_size_t),("commit",ctypes.c_size_t),("commit_peak",ctypes.c_size_t)]
    c = Counters()
    c.cb = ctypes.sizeof(c)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = ctypes.c_void_p
    info = ctypes.windll.psapi.GetProcessMemoryInfo
    info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not info(current(), ctypes.byref(c), c.cb):
        raise OSError("Memory counters unavailable")
    return {"working_set_bytes": int(c.current), "peak_working_set_bytes": int(c.peak)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/subspace_dinos_20261012.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-train-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = args.output_dir.resolve()
    if not run.is_relative_to(ROOT/"runs/quick") or run.exists() != args.resume:
        raise ValueError("Use a fresh runs/quick output, or explicitly resume an existing one")
    if args.resume and args.benchmark_train_only:
        raise ValueError("Benchmark cannot resume")
    run.mkdir(parents=True, exist_ok=args.resume)
    (run/"results").mkdir(exist_ok=args.resume)
    protocol = read(args.config)
    if (protocol["image_res"] != 448 or protocol["layers"] != [-4,-5] or protocol["explained_variance"] != .99
        or protocol["output_size"] != 256 or protocol["feature_dim"] != 384 or protocol["batch_size"] != 1):
        raise ValueError("Fixed method geometry and PCA settings changed")
    original_root = (ROOT/protocol["baseline_run"]).resolve()
    original = read(original_root/"run_config.json")
    if read(original_root/"run_state.json")["status"] != "completed":
        raise ValueError("Original control must be completed")
    tasks = [(d,c) for d,v in original["protocol"]["datasets"].items() for c in v["categories"]]
    if len(tasks) != 27 or original["protocol"]["seeds"] != [protocol["seed"]]:
        raise ValueError("Control identities mismatch")
    source_paths = [Path(__file__), ROOT/"src/industrial_anomaly_lab/subspace_features.py", ROOT/"src/industrial_anomaly_lab/quick_metrics.py"]
    source_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in source_paths}
    if source_hashes["src/industrial_anomaly_lab/quick_metrics.py"] != original["source_sha256"]["src/industrial_anomaly_lab/quick_metrics.py"]:
        raise ValueError("Common evaluator drift")
    metadata = [original_root/p for p in ("run_config.json","run_state.json","results/category_metrics.jsonl","results/samples.jsonl")]
    for d,c in tasks:
        metadata.extend(original_root/"models"/d/c/p for p in ("train_manifest.json","test_inventory.json"))
    metadata_hashes = {p.relative_to(original_root).as_posix(): digest(p) for p in metadata}
    vendor = ROOT/protocol["vendor_root"]
    if (vendor/".git").exists():
        commit = subprocess.check_output(["git","-C",str(vendor),"rev-parse","HEAD"],text=True).strip()
        if commit != protocol["vendor_commit"] or subprocess.check_output(["git","-C",str(vendor),"status","--porcelain"],text=True).strip():
            raise ValueError("Official reference commit/tree drift")
    else:
        reference = read(ROOT/"docs/subspace_reference.json")
        commit = reference["commit"]
        if commit != protocol["vendor_commit"]:
            raise ValueError("Sealed reference identity mismatch")
        for relative,h in reference["source_files_sha256"].items():
            checked(vendor,vendor/relative,h)
    refs = [vendor/"LICENSE",vendor/"README.md",vendor/"main.py",vendor/"scripts/backbone_ablation.sh",vendor/"scripts/benchmark_full_shot.sh"]
    refs.extend(sorted((vendor/"src").rglob("*.py")))
    vendor_hashes = {p.relative_to(vendor).as_posix():digest(p) for p in refs}
    if not (vendor/".git").exists() and vendor_hashes != reference["source_files_sha256"]:
        raise ValueError("Sealed reference file coverage changed")
    weights = ROOT/protocol["weights_dir"]
    for filename,h in protocol["weight_files_sha256"].items():
        checked(weights,weights/filename,h)
    config = {"schema_version":1,"protocol":protocol,"protocol_sha256":digest(args.config),
        "source_sha256":source_hashes,"input_metadata_sha256":metadata_hashes,"baseline_root":str(original_root),
        "vendor_source_sha256":vendor_hashes,"project_root":str(ROOT),"started_at_utc":now(),
        "environment_sha256":{p:digest(ROOT/p) for p in ("pyproject.toml","uv.lock")},
        "backbone":{"architecture":"dinov2-small","weight_dir":protocol["weights_dir"],
            "checkpoint_sha256":protocol["checkpoint_sha256"],"revision":protocol["weight_revision"],"files_sha256":protocol["weight_files_sha256"]},
        "benchmark_train_only":args.benchmark_train_only,
        "expected_matrix":{"categories":tasks,"variants":["subspace_max","subspace_top1p"]}}
    state = {"status":"initializing","started_at_utc":now(),"pid":os.getpid(),"completed_categories":0,
        "test_content_accessed":False,"scientific_claim_eligible":False}
    completed = set()
    prior_elapsed = 0.0
    if args.resume:
        previous = read(run/"run_config.json")
        old_state = read(run/"run_state.json")
        if previous["benchmark_train_only"] or old_state["status"] == "completed":
            raise ValueError("Completed run/benchmark cannot resume")
        for name in ("protocol_sha256","source_sha256","input_metadata_sha256","vendor_source_sha256","environment_sha256"):
            if config[name] != previous[name]:
                raise ValueError(f"Frozen execution drift: {name}")
        config = previous
        state.update(started_at_utc=old_state["started_at_utc"],test_content_accessed=old_state["test_content_accessed"])
        prior_elapsed = float(old_state.get("elapsed_seconds",0))
        history = run/"execution_history"/time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())
        history.mkdir(parents=True)
        shutil.copy2(run/"run_state.json",history/"run_state.json")
        for d,c in tasks:
            cat = run/"categories"/d/c
            if (cat/"completion.json").exists():
                for relative,h in read(cat/"completion.json")["sha256"].items():
                    checked(run,run/relative,h)
                completed.add((d,c))
            else:
                for partial in (cat,run/"predictions"/d/c):
                    if partial.exists():
                        destination = history/partial.relative_to(run)
                        if not partial.resolve().is_relative_to(run) or not destination.resolve().is_relative_to(run):
                            raise ValueError("Partial archive escapes run")
                        destination.parent.mkdir(parents=True,exist_ok=True)
                        partial.rename(destination)
    write(run/"run_config.json",config)
    state["completed_categories"] = len(completed)
    write(run/"run_state.json",state)
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        import cv2
        from PIL import Image
        from sklearn.metrics import roc_auc_score
        from threadpoolctl import threadpool_limits
        from industrial_anomaly_lab.subspace_features import (LocalDinoExtractor,StreamingPCA,save_subspace_model,render_subspace_map,image_aggregates)
        from industrial_anomaly_lab.quick_metrics import evaluate_from_files
        threadpool_limits(limits=protocol["cpu_threads"])
        cv2.setNumThreads(protocol["cpu_threads"])
        torch.manual_seed(protocol["seed"])
        extractor = LocalDinoExtractor(weights,expected_checkpoint_sha256=protocol["checkpoint_sha256"],
            expected_artifact_sha256=protocol["weight_files_sha256"],image_res=448,layers=(-4,-5),device=protocol["device"],
            cpu_threads=protocol["cpu_threads"],max_cuda_memory_gib=protocol["max_cuda_memory_gib"])
        config["extractor"] = extractor.metadata
        config["opencv_threads"] = cv2.getNumThreads()
        write(run/"run_config.json",config)
        old_samples = {(r["dataset"],r["category"],r["sample_id"]):r for r in read_lines(original_root/"results/samples.jsonl")}
        def gpu_memory():
            if extractor.device.type != "cuda":
                return {"gpu_used":False}
            return {"gpu_used":True,"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(extractor.device),
                "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(extractor.device)}
        for d,c in tasks:
            if (d,c) in completed:
                continue
            if digest(args.config) != config["protocol_sha256"] or any(digest(ROOT/p)!=h for p,h in source_hashes.items()):
                raise ValueError("Frozen method/source changed")
            category_started = time.monotonic()
            source_model = original_root/"models"/d/c
            for filename in ("train_manifest.json","test_inventory.json"):
                checked(original_root,source_model/filename,metadata_hashes[(source_model/filename).relative_to(original_root).as_posix()])
            train = read(source_model/"train_manifest.json")
            if args.benchmark_train_only:
                train = train[:16]
            cat = run/"categories"/d/c
            cat.mkdir(parents=True)
            write(cat/"train_manifest.json",train)
            state.update(status="running",current_task={"dataset":d,"category":c,"phase":"fit","images":0},updated_at_utc=now())
            write(run/"run_state.json",state)
            estimator = StreamingPCA(feature_dim=384)
            fit_start = time.monotonic()
            for index,item in enumerate(train):
                feature = extractor(checked(ROOT,ROOT/item["path"],item["sha256"]))
                if feature.shape != (32,32,384) or feature.dtype != np.float32:
                    raise ValueError("DINO feature geometry/dtype drift")
                estimator.update(feature.reshape(-1,384))
                if (index+1)%25 == 0 or index+1 == len(train):
                    state.update(current_task={"dataset":d,"category":c,"phase":"fit","images":index+1},updated_at_utc=now(),elapsed_seconds=prior_elapsed+time.monotonic()-started)
                    write(run/"run_state.json",state)
                    append(run/"results/events.jsonl",{"phase":"fit","dataset":d,"category":c,"images":index+1,"time":now()})
            model = estimator.finalize(explained_variance=.99,eps=1e-6)
            if model.count != len(train)*1024:
                raise ValueError("Patch count differs from authenticated normal-train inputs")
            model_meta = {"dataset":d,"category":c,"seed":protocol["seed"],"train_images":len(train),
                "train_manifest_sha256":digest(cat/"train_manifest.json"),"checkpoint_sha256":protocol["checkpoint_sha256"],
                "protocol_sha256":config["protocol_sha256"],"image_res":448,"feature_shape":[32,32,384],"layers":[-4,-5],
                "official_commit":commit,"estimation":"exact centered streaming float64 mean and M2/eigh; no rank truncation"}
            save_subspace_model(model,cat/"model.npz",model_meta)
            write(cat/"model_integrity.json",{"sha256":digest(cat/"model.npz"),"size_bytes":(cat/"model.npz").stat().st_size,"sealed_at_utc":now()})
            fit_seconds = time.monotonic()-fit_start
            write(cat/"metadata.json",{**model_meta,"patch_count":model.count,"k":model.k,"retained_variance":model.retained_variance,
                "variance_target":.99,"fit_seconds":fit_seconds,"extractor":extractor.metadata})
            del estimator
            gc.collect()
            if args.benchmark_train_only:
                replay_feature = extractor(checked(ROOT,ROOT/train[0]["path"],train[0]["sha256"]))
                raw = render_subspace_map(model.reconstruction_scores(replay_feature.reshape(-1,384)).reshape(32,32),model_size=448,output_size=256)
                write(run/"benchmark.json",{"normal_train_images":len(train),"patch_count":model.count,"fit_seconds":fit_seconds,
                    "feature_and_fit_seconds_per_image":fit_seconds/len(train),"pca_rank":model.k,"map_finite":bool(np.isfinite(raw).all()),
                    "elapsed_seconds":time.monotonic()-started,"test_content_accessed":False,"cpu_threads":protocol["cpu_threads"],
                    **process_memory(),**gpu_memory()})
                state.update(status="benchmark_completed",elapsed_seconds=time.monotonic()-started,updated_at_utc=now())
                write(run/"run_state.json",state)
                return 0
            inventory = read(source_model/"test_inventory.json")
            write(cat/"test_inventory.json",inventory)
            pred = run/"predictions"/d/c
            for name in ("subspace","masks"):
                (pred/name).mkdir(parents=True)
            paths,masks,labels,scores = [],[],[],[]
            state.update(test_content_accessed=True,current_task={"dataset":d,"category":c,"phase":"predict","images":0},updated_at_utc=now())
            write(run/"run_state.json",state)
            dataset_root = ROOT/original["protocol"]["datasets"][d]["root"]
            for index,sample in enumerate(inventory):
                old = old_samples[(d,c,sample["sample_id"])]
                image = checked(dataset_root,sample["image_path"],sample["image_sha256"])
                feature = extractor(image)
                raw = render_subspace_map(model.reconstruction_scores(feature.reshape(-1,384)).reshape(32,32),model_size=448,output_size=256)
                values = image_aggregates(raw)
                if values["top_count"] != 655:
                    raise ValueError("Image aggregation geometry changed")
                scores.append((values["max"],values["top1p"]))
                if sample["mask_path"] is None:
                    mask = np.zeros((256,256),dtype=bool)
                else:
                    mask_source = checked(dataset_root,sample["mask_path"],sample["mask_sha256"])
                    with Image.open(mask_source) as handle:
                        mask = np.asarray(handle.convert("L").resize((256,256),Image.Resampling.NEAREST)) > 0
                if not np.array_equal(mask,np.load(original_root/old["mask_path"],allow_pickle=False)):
                    raise ValueError("Authenticated GT differs from the control")
                map_path,mask_path = pred/"subspace"/f"{index:06d}.npy",pred/"masks"/f"{index:06d}.npy"
                np.save(map_path,raw,allow_pickle=False)
                np.save(mask_path,mask,allow_pickle=False)
                paths.append(map_path); masks.append(mask_path); labels.append(sample["label"])
                append(cat/"samples.jsonl",{"dataset":d,"category":c,"seed":protocol["seed"],"sample_id":sample["sample_id"],
                    "label":sample["label"],"image_path":str(image),"image_sha256":sample["image_sha256"],
                    "raw_maps":{"subspace":map_path.relative_to(run).as_posix()},"raw_map_sha256":{"subspace":digest(map_path)},
                    "mask_path":mask_path.relative_to(run).as_posix(),"mask_sha256":digest(mask_path),"selected":old["selected"],
                    "image_scores":{"subspace":{"max":values["max"],"top1p":values["top1p"]}}})
                if (index+1)%25 == 0 or index+1 == len(inventory):
                    state.update(current_task={"dataset":d,"category":c,"phase":"predict","images":index+1},updated_at_utc=now())
                    write(run/"run_state.json",state)
            state.update(current_task={"dataset":d,"category":c,"phase":"metrics"},updated_at_utc=now())
            write(run/"run_state.json",state)
            measured = evaluate_from_files(paths,masks,labels,temp_root=run/"metric_tmp")
            aurocs = [float(roc_auc_score(labels,np.asarray(scores)[:,i])) for i in (0,1)]
            if abs(measured["image_auroc"]-aurocs[0]) > 1e-12:
                raise ValueError("Max score metric mismatch")
            rows = [{"dataset":d,"category":c,"variant":"subspace_"+score,"seed":protocol["seed"],"status":"completed",
                "train_images":len(train),"test_images":len(inventory),"patch_count":model.count,"pca_rank":model.k,
                **measured,"image_auroc":aurocs[i],"image_auroc_raw_max":aurocs[0],"image_auroc_top1p_mean":aurocs[1],
                "score_definition":protocol["score_definitions"][score]} for i,score in enumerate(("max","top1p"))]
            write(cat/"metrics.json",rows)
            files = [p for directory in (cat,pred) for p in directory.rglob("*") if p.is_file()]
            write(cat/"completion.json",{"status":"completed","time":now(),"sha256":{p.relative_to(run).as_posix():digest(p) for p in files}})
            state["completed_categories"] += 1
            state.update(elapsed_seconds=prior_elapsed+time.monotonic()-started,updated_at_utc=now(),**process_memory(),**gpu_memory())
            write(run/"run_state.json",state)
            print(json.dumps({"completed":state["completed_categories"],"dataset":d,"category":c,"fit_seconds":fit_seconds,
                "category_seconds":time.monotonic()-category_started,"metrics":rows}),flush=True)
            del model,feature,raw
            gc.collect()
        rows = [r for d,c in tasks for r in read(run/"categories"/d/c/"metrics.json")]
        samples = [r for d,c in tasks for r in read_lines(run/"categories"/d/c/"samples.jsonl")]
        if len(rows)!=54 or len(samples)!=3887 or sum(r["train_images"] for r in rows if r["variant"]=="subspace_max")!=12288:
            raise ValueError("Incomplete full-normal evaluation matrix")
        (run/"results/category_metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
        (run/"results/samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in samples),encoding="utf-8")
        state.update(status="completed",current_task=None,completed_categories=27,training_images=12288,test_images=3887,metric_rows=54,
            elapsed_seconds=prior_elapsed+time.monotonic()-started,finished_at_utc=now(),**process_memory(),**gpu_memory())
        write(run/"run_state.json",state)
        return 0
    except BaseException as error:
        state.update(status="failed",error=f"{type(error).__name__}: {error}",elapsed_seconds=prior_elapsed+time.monotonic()-started,updated_at_utc=now())
        write(run/"run_state.json",state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
