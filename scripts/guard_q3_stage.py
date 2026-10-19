"""Persist a stage's actual child exit code, including errors before its main loop."""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_score_calibration import read,write,now,sha


def main():
    p=argparse.ArgumentParser();p.add_argument("--name",required=True);p.add_argument("--script");p.add_argument("--output");p.add_argument("--adopt-pid",type=int);a=p.parse_args()
    run=ROOT/"runs/research-q3-20261016";guards=run/"guards";guards.mkdir(exist_ok=True)
    receipt=guards/(a.name+".json");stage=Path(a.output).resolve() if a.output else run/a.name
    if not stage.is_relative_to(run):raise ValueError("Guard output must stay in the current research run")
    if a.adopt_pid is None:
        script=ROOT/"scripts"/str(a.script)
        if not script.is_relative_to(ROOT/"scripts") or not script.is_file():raise ValueError("Unexpected child script")
        args=[sys.executable,str(script)]+(["--output",str(stage)] if a.output else [])
        with (guards/(a.name+".stdout.log")).open("wb") as stdout,(guards/(a.name+".stderr.log")).open("wb") as stderr:
            child=subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr)
            write(receipt,{"status":"running","guard_pid":os.getpid(),"child_pid":child.pid,"script_sha256":sha(script),"started_at":now()})
            code=child.wait()
    else:
        # Keep an OS handle to this exact process, so PID reuse cannot mimic liveness.
        kernel=ctypes.WinDLL("kernel32",use_last_error=True);kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(0x100000|0x1000,False,a.adopt_pid)
        if not handle:raise OSError(ctypes.get_last_error(),"Cannot adopt existing research process")
        kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        write(receipt,{"status":"observing_existing_process","guard_pid":os.getpid(),"child_pid":a.adopt_pid,"started_at":now()})
        try:
            while kernel.WaitForSingleObject(handle,2000)==0x102:pass
            result=ctypes.c_ulong();kernel.GetExitCodeProcess(handle,ctypes.byref(result));code=int(result.value)
        finally:kernel.CloseHandle(handle)
    if code:
        stage.mkdir(parents=True,exist_ok=True)
        if not (stage/"failure.json").exists():write(stage/"failure.json",{"error":"Child process exited unsuccessfully","exit_code":code,"guard":str(receipt),"at":now()})
    old=read(receipt);write(receipt,{**old,"status":"completed" if code==0 else "failed","exit_code":code,"finished_at":now()})
    return code


if __name__=="__main__":raise SystemExit(main())
