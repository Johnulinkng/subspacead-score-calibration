"""Measured child for one full baseline or a clearly marked resource smoke."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import runpy
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from run_score_calibration import read,write,now,sha
from run_route1_resource_trial import Monitor


def main():
    p=argparse.ArgumentParser();p.add_argument("--job",type=Path,required=True);a=p.parse_args()
    job=read(a.job);out=Path(job["receipt_dir"]);out.mkdir(parents=True,exist_ok=True)
    policy={"initial_free_ram_gib":3.25,"initial_free_commit_gib":12.,"reserve_free_ram_gib":1.5,
            "reserve_free_commit_gib":2.,"max_working_set_gib":4.5}
    monitor=Monitor(policy);sample=monitor.sample()
    if sample["system_free_ram_bytes"]<policy["initial_free_ram_gib"]*1024**3 or sample["system_free_commit_bytes"]<policy["initial_free_commit_gib"]*1024**3:
        raise RuntimeError("Baseline admission needs 3.25 GiB RAM and 12 GiB commit")
    monitor.thread.start();started=time.perf_counter();status="failed";error=None
    try:
        import torch
        torch.set_num_threads(3)
        if torch.cuda.is_available():
            free,total=torch.cuda.mem_get_info();cap=min(6.5*1024**3,free-.75*1024**3)
            if cap<3*1024**3:raise RuntimeError("GPU admission reserve failed")
            torch.cuda.set_per_process_memory_fraction(cap/total)
            torch.cuda.reset_peak_memory_stats()
        script=ROOT/"scripts"/job["script"]
        sys.argv=[str(script)]+job["arguments"]
        write(out/"active.json",{"pid":os.getpid(),"status":"running","job":job,"started_at":now(),"script_sha256":sha(script)})
        try:runpy.run_path(str(script),run_name="__main__")
        except SystemExit as e:
            if e.code not in (None,0):raise
        monitor.check();status="completed"
    except BaseException as e:
        error={"error":repr(e),"traceback":traceback.format_exc()};raise
    finally:
        resources=monitor.close()
        if "torch" in sys.modules and torch.cuda.is_available():
            resources.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(out/"receipt.json",{"status":status,"kind":job["kind"],"job":job,"elapsed_seconds":time.perf_counter()-started,"finished_at":now(),"resources":resources,"failure":error})


if __name__=="__main__":main()
