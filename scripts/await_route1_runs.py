"""Complete the frozen runs when real memory permits; touch only owned jobs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_route1_controls import read, write, memory


def retryable_resource_failure(log_text):
    """Retry only an explicit resource failure without an integrity error."""
    lowered=str(log_text).casefold()
    if 'drift' in lowered or 'valueerror' in lowered:
        return False
    evidence=(
        'resourceviolation', 'resource reserve boundary', 'below reserve',
        'ram fell below reserve', 'commit fell below reserve',
        'cpu admission needs', 'initial free ram below admission',
        'initial free commit below admission', 'initial disk free space below admission',
    )
    return any(marker in lowered for marker in evidence)


def failure_evidence(log_path, phase, full_output):
    """Include the owned precision monitor's redirected error stream."""
    evidence=log_path.read_text(encoding='utf-8',errors='replace')
    if phase!='precision':return evidence
    decoder=json.JSONDecoder();position=0
    while True:
        start=evidence.find('{',position)
        if start<0:break
        try:value,end=decoder.raw_decode(evidence[start:]);position=start+end
        except json.JSONDecodeError:position=start+1;continue
        if not isinstance(value,dict) or not value.get('monitor'):continue
        monitor=Path(value['monitor']).resolve()
        if not monitor.is_relative_to(ROOT/'runs/quick') or not monitor.name.startswith(full_output.name+'-monitor-'):
            raise ValueError('Precision failure monitor escapes its registered output')
        for name in ('summary.json','stderr.log'):
            path=monitor/name
            if path.is_file():evidence+='\n'+path.read_text(encoding='utf-8',errors='replace')
        break
    return evidence


def process_info(pid):
    command="[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); Get-CimInstance Win32_Process -Filter 'ProcessId = %d' | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress" % int(pid)
    output=subprocess.check_output(['powershell.exe','-NoProfile','-Command',command],creationflags=0x08000000)
    return json.loads(output.decode('utf-8-sig')) if output.strip() else None


def state_process_active(saved,phase,output):
    pid=saved.get('pid')
    if not pid:return False
    info=process_info(pid)
    if not info:return False
    executable=str(info.get('ExecutablePath','')).replace('\\','/').casefold()
    command=str(info.get('CommandLine','')).replace('\\','/').casefold()
    entries=('run_route1_controls.py','resume_route1_controls.py') if phase=='precision' else ('run_route1_resource_matrix.py','resume_route1_resource_matrix.py')
    directory=output.relative_to(ROOT).as_posix().casefold()
    return executable.startswith(ROOT.as_posix().casefold()+'/') and any(entry in command for entry in entries) and directory in command


def main():
    ap=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    fixed_output=(ROOT/'runs/waiters/route1-completion-20261014').resolve()
    ap.add_argument('--output-dir',type=Path,default=fixed_output)
    args=ap.parse_args();out=args.output_dir.resolve()
    if out!=fixed_output:raise ValueError('Only the fixed route1 waiter output directory is allowed')
    out.mkdir(parents=True,exist_ok=True)
    lock=out/'waiter.lock.json';token=uuid.uuid4().hex
    if lock.exists():
        previous=read(lock);info=process_info(previous['pid'])
        if info and 'await_route1_runs.py' in str(info.get('CommandLine','')):raise RuntimeError('An existing route1 waiter is active')
        archive=out/'stale_locks'/('waiter-'+previous['token']+'.json');archive.parent.mkdir(exist_ok=True)
        if not lock.resolve().is_relative_to(out) or not archive.resolve().is_relative_to(out):raise ValueError('Unsafe lock archive')
        os.replace(lock,archive)
    with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'token':token},f)
    full=ROOT/'runs/quick/route1-controls-20261014'
    resources=ROOT/'runs/quick/route1-resources-20261014'
    state={'status':'waiting_resources','pid':os.getpid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
           'scope':'finite registered route1 matrix only; no training restart and no foreign process changes'}
    jobs={};handles=[];stable=0;attempt={'precision':0,'resources':0};job_logs={};needs_attention=None
    try:
        while True:
            precision=read(full/'run_state.json');resource=read(resources/'matrix_state.json')
            for name,job in list(jobs.items()):
                if job.poll() is not None:
                    state[name+'_last_exit_code']=job.returncode;del jobs[name]
                    if job.returncode:
                        text=failure_evidence(job_logs[name],name,full)
                        if not retryable_resource_failure(text):
                            needs_attention={'phase':name,'log':str(job_logs[name]),'reason':'non-resource failure requires inspection'}
            if needs_attention:
                state.update(status='needs_attention',attention=needs_attention);write(out/'state.json',state)
                print(json.dumps(state),flush=True);return 2
            if precision['status']=='completed' and resource['status']=='completed':
                state.update(status='completed',finished_at_utc=datetime.now(timezone.utc).isoformat(),
                             precision_completed=precision['completed_categories'],resource_completed=resource['completed_trials']);write(out/'state.json',state)
                print(json.dumps(state),flush=True);return 0
            sample=memory()
            external_active={
                'precision':state_process_active(precision,'precision',full) if 'precision' not in jobs else False,
                'resources':state_process_active(resource,'resources',resources) if 'resources' not in jobs else False,
            }
            occupied=bool(jobs) or any(external_active.values())
            stable=stable+1 if sample['free_ram_gib']>=3.5 and sample['free_commit_gib']>=3.5 else 0
            pending=None
            if stable>=3:
                if precision['status']!='completed' and 'precision' not in jobs and not external_active['precision'] and not occupied:pending='precision'
                elif resource['status']!='completed' and 'resources' not in jobs and not external_active['resources']:
                    # Measured workers each peak near one GiB. Allow overlap
                    # only with additional real headroom, not a relaxed gate.
                    if not occupied or sample['free_ram_gib']>=5.25 and sample['free_commit_gib']>=5.25:pending='resources'
            if pending:
                attempt[pending]+=1
                log=out/f'{pending}-attempt-{attempt[pending]:03d}-{datetime.now(timezone.utc).strftime("%H%M%S")}.log'
                handle=log.open('wb');handles.append(handle)
                job_logs[pending]=log
                if pending=='precision':
                    command=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/Invoke-Route1Controls.ps1'),
                             '-OutputDirectory','runs/quick/route1-controls-20261014','-Resume']
                else:
                    command=[str(ROOT/'.venv/Scripts/python.exe'),'-B',str(ROOT/'scripts/resume_route1_resource_matrix.py'),
                             '--output-dir','runs/quick/route1-resources-20261014']
                env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PYTHONUNBUFFERED']='1';env['PYTHONDONTWRITEBYTECODE']='1'
                jobs[pending]=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=env,
                                                creationflags=0x08000000 if os.name=='nt' else 0)
                stable=0
                print(json.dumps({'event':'launched','phase':pending,'pid':jobs[pending].pid,'free_ram_gib':sample['free_ram_gib']}),flush=True)
            state.update(status='running' if jobs or any(external_active.values()) else 'waiting_resources',updated_at_utc=datetime.now(timezone.utc).isoformat(),
                         precision_completed=precision.get('completed_categories',0),resource_completed=resource.get('completed_trials',0),
                         owned_children={name:job.pid for name,job in jobs.items()},existing_owned_runs_active=external_active,launch_attempts=attempt,**sample)
            write(out/'state.json',state)
            time.sleep(10)
    finally:
        # Do not blindly terminate descendants; their own monitors enforce
        # resource reserves and record durable state if this waiter exits.
        for handle in handles:handle.close()
        if lock.exists() and read(lock).get('token')==token:lock.unlink()


if __name__=='__main__':raise SystemExit(main())
