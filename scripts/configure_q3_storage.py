"""Assign only newly generated DRAEM follow-up artifacts to a fresh D: root."""
from pathlib import Path
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_score_calibration import read,write,now,sha


def main():
    run=ROOT/"runs/research-q3-20261016";queue=run/"baseline-queue-v2.json";cfg=read(queue)
    destination=Path("D:/3k-research-20261016").resolve()
    if destination!=Path("D:/3k-research-20261016"):raise ValueError("Unexpected research storage root")
    if shutil.disk_usage(destination.anchor).free<40*1024**3:raise RuntimeError("New D: output assignment needs 40 GiB free")
    if destination.exists() and not (destination/"storage_owner.json").exists():raise ValueError("Refuse a pre-existing unowned directory")
    destination.mkdir(exist_ok=True)
    write(destination/"storage_owner.json",{"project_root":str(ROOT),"task_run":str(run),"purpose":"new DRAEM checkpoints/predictions only; existing D: files are untouched","created_at":now()})
    backup=run/"baseline-queue-v2.before-storage.json"
    if backup.exists():raise ValueError("Storage allocation already registered")
    shutil.copy2(queue,backup);prior_hash=sha(queue)
    for job in cfg["training_jobs"]:
        if job["method"]!="DRAEM":continue
        base=destination/"baselines/DRAEM"/job["dataset"]/job["category"]/str(job["seed"])
        job["receipt_dir"]=str(base/"execution")
        index=job["arguments"].index("--output-dir");job["arguments"][index+1]=str(base/"model")
    cfg["storage_revision"]={"registered_at":now(),"prior_queue_sha256":prior_hash,"new_artifact_root":str(destination),"scientific_parameters_changed":False,"existing_files_moved_or_deleted":False,"E_estimated_total_new_gib":145,"D_estimated_total_new_gib":34,"minimum_free_gib_each_drive":8}
    write(queue,cfg);print(json.dumps(cfg["storage_revision"]))


if __name__=="__main__":main()
