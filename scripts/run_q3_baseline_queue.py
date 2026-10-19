"""Resource-admitted, resumable full comparator training and inference queue."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from run_score_calibration import read,write,sha,now
from run_route1_resource_trial import Monitor


def main():
    p=argparse.ArgumentParser();p.add_argument("--queue",type=Path,default=ROOT/"runs/research-q3-20261016/baseline-queue.json");a=p.parse_args()
    cfg=read(a.queue);run=a.queue.parent;source=Path(cfg["source_snapshot"])
    def respect_user_pause():
        pause=run/"PAUSED_BY_USER.json"
        if pause.exists():
            print("Queue is paused by user: "+str(pause),flush=True)
            raise SystemExit(0)
    respect_user_pause()
    state_path=run/(a.queue.stem.replace("baseline-queue","baseline_queue_state")+".json")
    write(run/(a.queue.stem+"-controller.json"),{"controller_path":str(Path(__file__)),"controller_sha256":sha(Path(__file__)),"queue_sha256":sha(a.queue),"started_at":now()})
    for rel,digest in cfg["source_sha256"].items():
        if sha(source/rel)!=digest:raise ValueError("Frozen source mismatch: "+rel)
    if subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True).strip():raise ValueError("Frozen source snapshot not clean")
    env=dict(os.environ)
    for key,sub in {"TEMP":"tmp","TMP":"tmp","TORCH_HOME":".cache/torch","HF_HOME":".cache/huggingface","XDG_CACHE_HOME":".cache","MPLCONFIGDIR":".cache/matplotlib","PYTHONPYCACHEPREFIX":".cache/pycache"}.items():
        d=source/sub;d.mkdir(parents=True,exist_ok=True);env[key]=str(d)
    env.update(OMP_NUM_THREADS="3",MKL_NUM_THREADS="3",OPENBLAS_NUM_THREADS="3",HF_HUB_OFFLINE="1",PYTHONNOUSERSITE="1",PYTHONUNBUFFERED="1")
    done=[];failed=[];started=time.monotonic()
    policy={"reserve_free_ram_gib":1.5,"reserve_free_commit_gib":2.,"max_working_set_gib":4.5}
    gate=Monitor(policy)
    def execute(job):
        respect_user_pause()
        receipt=Path(job["receipt_dir"]);receipt.mkdir(parents=True,exist_ok=True)
        final=receipt/"receipt.json"
        if final.exists():
            if read(final)["status"]=="completed":done.append(job["id"]);return True
            failed.append(job["id"]);return False
        while True:
            respect_user_pause()
            sample=gate.sample()
            free_gpu_mib=float(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True,timeout=10).strip().splitlines()[0])
            sample["gpu_free_bytes"]=int(free_gpu_mib*1024**2)
            if sample["system_free_ram_bytes"]>=3.25*1024**3 and sample["system_free_commit_bytes"]>=12*1024**3 and sample["gpu_free_bytes"]>=6.75*1024**3 and min(shutil.disk_usage(run).free,shutil.disk_usage(receipt).free)>=8*1024**3:break
            write(state_path,{"status":"waiting_for_resources","pid":os.getpid(),"job":job["id"],"sample":sample,"completed":len(done),"failed":failed,"updated_at":now()})
            time.sleep(10)
        jobfile=receipt/"job.json";write(jobfile,job)
        respect_user_pause()
        with (receipt/"stdout.log").open("wb") as out,(receipt/"stderr.log").open("wb") as err:
            cmd=[sys.executable,str(source/"scripts/run_q3_baseline_job.py"),"--job",str(jobfile)]
            child=subprocess.Popen(cmd,cwd=source,env=env,stdout=out,stderr=err)
            write(state_path,{"status":"running","pid":os.getpid(),"child_pid":child.pid,"job":job["id"],"completed":len(done),"failed":failed,"updated_at":now()})
            code=child.wait()
        if code!=0 or not final.exists() or read(final)["status"]!="completed":
            failed.append(job["id"]);print("FAILED "+job["id"],flush=True);return False
        done.append(job["id"]);print("COMPLETED "+job["id"],flush=True);return True
    for method in ("RD4AD","PatchCore","DRAEM"):
        smokes=[j for j in cfg["resource_smokes"] if j["method"]==method]
        if not all([execute(j) for j in smokes]):
            print("Method held at its unchanged resource configuration: "+method,flush=True);continue
        for job in [j for j in cfg["training_jobs"] if j["method"]==method]:
            if not execute(job):
                receipt=Path(job["receipt_dir"])
                text=(receipt/"stderr.log").read_text(encoding="utf-8",errors="replace") if (receipt/"stderr.log").exists() else ""
                if any(term in text for term in ("out of memory","ResourceViolation","MemoryError","reserve does not fit")):
                    print("Resource failure holds remaining tasks for "+method,flush=True);break
                continue
            train_args=job["arguments"];base=Path(job["receipt_dir"]).parent
            common=["--dataset",job["dataset"],"--category",job["category"],"--dataset-root",train_args[train_args.index("--dataset-root")+1],"--seed",str(job["seed"]),"--device","cuda:0","--output-dir",str(base/"inference"),"--run-purpose","author_reproduction"]
            if method=="RD4AD":
                script="infer_rd4ad.py";common += ["--teacher-checkpoint",train_args[train_args.index("--teacher-checkpoint")+1],"--student-checkpoint",str(base/"model/checkpoints/rd4ad_final.pt"),"--batch-size","1"]
            elif method=="PatchCore":
                script="infer_patchcore.py";common += ["--fitted-model-dir",str(base/"model/model"),"--execution-scope","full","--batch-size","2","--faiss-workers","3","--backbone-checkpoint",train_args[train_args.index("--backbone-checkpoint")+1]]
            else:
                script="infer_draem.py";common += ["--checkpoint-dir",str(base/"model/checkpoints"),"--batch-size","8"]
            infer={**job,"id":job["id"]+"/inference","kind":"full_test_inference_previously_accessed_benchmarks","script":script,"arguments":common,"receipt_dir":str(base/"inference-execution")}
            execute(infer)
    write(state_path,{"status":"completed" if not failed else "completed_with_failures","completed":done,"failed":failed,"elapsed_seconds":time.monotonic()-started,"finished_at":now(),"q3_goal_achieved":False})


if __name__=="__main__":main()
