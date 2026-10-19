"""Resume the registered Q3 work after an explicit user request, preserving attempts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from audit_q3_recovery import ROOT, RUN, now, read, sha, write
from run_route1_resource_trial import Monitor

GIB = 1024 ** 3


def preserve(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.exists():
        return
    if not source.resolve().is_relative_to(RUN.resolve()):
        raise ValueError("Refuse to move an artifact outside the research run")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.resolve().is_relative_to(RUN.resolve()) or destination.exists():
        raise ValueError("Unsafe or occupied history destination")
    source.rename(destination)


def prepare(recovery):
    audit = read(recovery / "preflight-audit.json")
    if audit["status"] != "passed":
        raise ValueError("Recovery audit has not passed")
    original_queue = RUN / "baseline-queue-v2.json"
    if sha(original_queue) != audit["queue_sha256"]:
        raise ValueError("Queue changed after audit")
    history = recovery / "prior-attempt"
    history.mkdir()
    for name in ("trainonly-development", "selected-candidate", "final-analysis", "prediction-comparisons"):
        preserve(RUN / name, history / name)
    independent = RUN / "independent-partitions"
    for filename in ("run_state.json", "failure.json", "resources.json"):
        preserve(independent / filename, history / "independent-partitions-state" / filename)
    partial = []
    for marker in list((independent / "categories").rglob("partitions.json")):
        if not (marker.parent / "metrics.json").exists():
            destination = history / "incomplete-independent-units" / marker.parent.relative_to(independent / "categories")
            partial.append({"source": str(marker.parent), "preserved_as": str(destination)})
            preserve(marker.parent, destination)
    queue = read(original_queue)
    first = queue["training_jobs"][0]
    if first["id"] != "RD4AD/mvtec_ad/bottle/1887537248":
        raise ValueError("Unexpected first interrupted comparator unit")
    first_base = RUN / "baselines-restart-20261018/RD4AD/mvtec_ad/bottle/1887537248"
    if first_base.exists():
        raise ValueError("RD4AD recovery must use a fresh training attempt")
    first["receipt_dir"] = str(first_base / "execution")
    arguments = first["arguments"]
    arguments[arguments.index("--output-dir") + 1] = str(first_base / "model")
    queue["recovery_revision"] = {
        "registered_at": now(), "prior_queue_sha256": sha(original_queue),
        "reason": "User explicitly requested continuation after reboot; interrupted RD4AD has no checkpoint",
        "scientific_parameters_changed": False, "frozen_source_changed": False,
        "preserved_interrupted_attempt": str(RUN / "baselines/RD4AD/mvtec_ad/bottle/1887537248"),
        "fresh_restarted_attempt": str(first_base)}
    queue_path = RUN / "baseline-queue-recovery-20261018.json"
    if queue_path.exists():
        raise ValueError("Recovery queue already exists; do not silently replace it")
    write(queue_path, queue)
    queue_state = RUN / "baseline_queue_state-recovery-20261018.json"
    write(RUN / "active_baseline_queue.json", {"queue": str(queue_path), "state": str(queue_state),
                                               "queue_sha256": sha(queue_path), "updated_at": now()})
    preserve(RUN / "PAUSED_BY_USER.json", history / "PAUSED_BY_USER.json")
    record = {"status": "resumed_by_user", "user_request": "请你继续", "resumed_at": now(),
              "preflight_audit_sha256": sha(recovery / "preflight-audit.json"),
              "history": str(history), "partial_units_preserved": partial,
              "reused_completed_independent_units": audit["completed_independent_units"],
              "reused_feature_images": audit["feature_files"],
              "baseline_queue": str(queue_path), "supervisor_sha256": sha(Path(__file__)),
              "scientific_gates_changed": False, "q3_goal_achieved": False}
    write(RUN / "RESUMED_BY_USER.json", record)
    write(recovery / "preparation.json", record)
    return queue_path, queue_state


def eligible_stage(stage, completed, sample, gpu_free_mib):
    free_ram = sample["system_free_ram_bytes"] / GIB
    free_commit = sample["system_free_commit_bytes"] / GIB
    if stage == "independent-partitions":
        return free_ram >= 4.0 and free_commit >= 6.0
    if stage == "baselines":
        return free_ram >= 4.5 and free_commit >= 14.0 and gpu_free_mib >= 6912
    if stage == "trainonly-development":
        return "independent-partitions" in completed and free_ram >= 3.5 and free_commit >= 6.0 and gpu_free_mib >= 1792
    if stage == "selected-candidate":
        return "trainonly-development" in completed and free_ram >= 3.25 and free_commit >= 4.0
    if stage == "prediction-comparisons":
        return "selected-candidate" in completed and free_ram >= 3.0 and free_commit >= 4.0
    if stage == "final-analysis":
        return {"baselines", "independent-partitions", "selected-candidate"}.issubset(completed) and free_ram >= 2.5
    raise ValueError(stage)


def gpu_free():
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True, timeout=10)
    return float(result.strip().splitlines()[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-dir", type=Path, required=True)
    args = parser.parse_args()
    recovery = args.recovery_dir.resolve()
    if not recovery.is_relative_to(RUN) or (recovery / "supervisor.lock").exists():
        raise ValueError("Use a fresh, unique recovery supervisor")
    write(recovery / "supervisor.lock", {"pid": os.getpid(), "started_at": now()})
    while True:
        audit = recovery / "preflight-audit.json"
        status = read(audit).get("status") if audit.exists() else None
        if status == "passed":
            break
        if status == "failed":
            raise ValueError("Saved-result audit failed; computation remains paused")
        write(recovery / "run_state.json", {"status": "waiting_for_recovery_audit", "pid": os.getpid(), "updated_at": now()})
        time.sleep(5)
    queue_path, queue_state = prepare(recovery)
    commands = {
        "independent-partitions": ["scripts/run_q3_independent_partitions.py", "--output", str(RUN / "independent-partitions"), "--resume"],
        "baselines": ["scripts/run_q3_baseline_queue.py", "--queue", str(queue_path)],
        "trainonly-development": ["scripts/run_q3_trainonly_development.py", "--output", str(RUN / "trainonly-development")],
        "selected-candidate": ["scripts/evaluate_q3_selected_candidate.py", "--output", str(RUN / "selected-candidate")],
        "prediction-comparisons": ["scripts/render_q3_prediction_comparisons.py"],
        "final-analysis": ["scripts/aggregate_q3_followup.py"],
    }
    environment = dict(os.environ)
    for key, relative in {"TEMP": "tmp", "TMP": "tmp", "TORCH_HOME": ".cache/torch", "HF_HOME": ".cache/huggingface",
                          "XDG_CACHE_HOME": ".cache", "MPLCONFIGDIR": ".cache/matplotlib", "PYTHONPYCACHEPREFIX": ".cache/pycache"}.items():
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        environment[key] = str(path)
    environment.update(PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3")
    gate = Monitor({"reserve_free_ram_gib": 1.5, "reserve_free_commit_gib": 2.0, "max_working_set_gib": 1.0})
    active, receipts, completed, failed = {}, {}, set(), set()
    write(recovery / "supervisor.json", {"pid": os.getpid(), "started_at": now(), "commands": commands,
                                         "source_sha256": {command[0]: sha(ROOT / command[0]) for command in commands.values()},
                                         "supervisor_sha256": sha(Path(__file__)), "q3_goal_achieved": False})
    while True:
        if (RUN / "PAUSED_BY_USER.json").exists():
            for stage, child in active.items():
                if child.poll() is None:
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
            write(recovery / "run_state.json", {"status": "paused_by_user", "pid": os.getpid(), "updated_at": now(),
                                                "checkpoint_notice": "Interrupted comparator training may need a new full attempt"})
            return
        for stage, child in list(active.items()):
            code = child.poll()
            if code is None:
                continue
            receipt = receipts[stage]
            receipt.update(exit_code=code, finished_at=now(), status="completed" if code == 0 else "failed")
            write(recovery / (stage + ".json"), receipt)
            if code == 0:
                completed.add(stage)
            else:
                failed.add(stage)
                output = RUN / stage
                if stage != "baselines" and not (output / "failure.json").exists():
                    write(output / "failure.json", {"error": "Recovery child exited unsuccessfully", "exit_code": code,
                                                    "recovery_receipt": str(recovery / (stage + ".json")), "at": now()})
            del active[stage]
        sample = gate.sample()
        gate.samples.clear()
        free_mib = gpu_free()
        # Small CPU fitting and a GPU comparator may overlap. Waiting stages do not load a model.
        for stage, command in commands.items():
            if stage in active or stage in completed or stage in failed:
                continue
            dependencies = {"trainonly-development": {"independent-partitions"}, "selected-candidate": {"trainonly-development"},
                            "prediction-comparisons": {"selected-candidate"}, "final-analysis": {"baselines", "independent-partitions", "selected-candidate"}}
            if dependencies.get(stage, set()) & failed:
                failed.add(stage)
                continue
            if eligible_stage(stage, completed, sample, free_mib):
                with (recovery / (stage + ".stdout.log")).open("wb") as stdout, (recovery / (stage + ".stderr.log")).open("wb") as stderr:
                    child = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                active[stage] = child
                receipts[stage] = {"status": "running", "launcher_pid": child.pid, "supervisor_pid": os.getpid(),
                                   "command": [sys.executable, *command], "started_at": now()}
                write(recovery / (stage + ".json"), receipts[stage])
                print("STARTED " + stage + " pid=" + str(child.pid), flush=True)
                # Wait for the new worker's real allocation before admitting another stage.
                break
        waiting = [stage for stage in commands if stage not in active and stage not in completed and stage not in failed]
        finished = len(completed | failed) == len(commands)
        status = "finished_with_failures" if finished and failed else "completed_pipeline" if finished else "running"
        state = {"status": status, "pid": os.getpid(), "updated_at": now(), "active": {k: p.pid for k, p in active.items()},
                 "completed_stages": sorted(completed), "failed_stages": sorted(failed), "waiting_stages": waiting,
                 "free_ram_gib": sample["system_free_ram_bytes"] / GIB, "free_commit_gib": sample["system_free_commit_bytes"] / GIB,
                 "gpu_free_mib": free_mib, "E_free_gib": shutil.disk_usage(ROOT).free / GIB,
                 "D_free_gib": shutil.disk_usage("D:/").free / GIB, "baseline_state": read(queue_state) if queue_state.exists() else None,
                 "q3_goal_achieved": False, "scope": "Live execution state; completion does not establish journal acceptance"}
        write(recovery / "run_state.json", state)
        write(RUN / "active_recovery.json", {"recovery_dir": str(recovery), "state": str(recovery / "run_state.json"),
                                            "supervisor_pid": os.getpid(), "updated_at": now()})
        if finished:
            return
        time.sleep(15)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        # Keep the launch log diagnostic; never turn a prerequisite or resource error into scientific success.
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise
