# Monitored experiment launcher

`scripts/Invoke-MonitoredExperiment.ps1` is the outer resource gate and evidence recorder for long-running training and inference commands. It never changes model code and it does not provide a general preflight bypass.

Experiment mode accepts `debug_limited`, `development_full`, `author_reproduction`, and `formal_full`. The same fixed preflight and monitoring apply to every purpose; the model entry point marks `debug_limited` artifacts ineligible for performance claims.

## Formal or development run

Every experiment invocation performs these steps in order:

1. resolves the output path and requires a new or empty child directory under `runs/monitored/`;
2. accepts only `E:\3k\.venv\Scripts\python.exe` running an existing, non-reparse-point `.py` file inside `E:\3k`;
3. invokes the fixed `scripts/Test-TrainingPreflight.ps1` with its fixed 8 GiB RAM and 40 GiB disk defaults in a hidden process;
4. proceeds only when the emitted report says `ready_for_training_launch=true` and the preflight process exits successfully;
5. atomically creates `runs/.training.lock.json` with `FileMode.CreateNew`;
6. starts the experiment with `Start-Process -WindowStyle Hidden` and redirects stdout and stderr;
7. samples the child working set, available host RAM, GPU memory/utilization, and GPU compute processes into JSONL;
8. records the child exit code and releases the lock only when its random ownership token still matches.

Example (this command is illustrative; run it only after the registered experiment configuration is ready):

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
& .\scripts\Invoke-MonitoredExperiment.ps1 `
  -OutputDirectory runs/monitored/draem-visa-candle-seed1887537248 `
  -ScriptPath scripts/train_draem.py `
  -RunPurpose development_full `
  -ScriptArgument @('--help')
```

The launcher refuses path escape, a non-empty output directory, a project path containing a reparse point, a script outside the project, a non-Python command, a failed fixed preflight, and any lock collision. It does not delete a stale or foreign lock automatically. Rejections before a safe output directory can be claimed are recorded under `artifacts/monitor_rejections/`.

Each accepted output directory contains:

- `command.json`: exact executable, script, arguments, and SHA-256 identities;
- `preflight/` plus `preflight.stdout.log` and `preflight.stderr.log` for non-test runs;
- `stdout.log` and `stderr.log` from the child;
- `resource_samples.jsonl`, one compact JSON object per sample;
- `summary.json`, including status, timestamps, exit code, lock ownership/release state, evidence paths, and SHA-256 hashes for stable evidence files. If an interrupted launcher observes a child still running, it omits hashes for the still-mutable child logs and records that fact.

## Registered DRAEM S0 smoke

The S0 integrity smoke has one project wrapper so the resource probe cannot
silently fall back to the training CLI's CPU default. It fixes VisA `candle`,
seed `1887537248`, 256 px, author batch 8, zero DataLoader workers, disabled
pinned host memory, `cuda:0`, and at most two optimizer steps. The result is
always `debug_limited` and cannot enter a performance table.

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
& .\scripts\Invoke-DraemS0Smoke.ps1
```

The wrapper still calls the fixed monitored launcher, so it cannot bypass the
8 GiB host-RAM gate, an existing GPU compute process, WSL, disk reserve, or the
training lock.

When another project owns the GPU, `Wait-For-DraemS0Window.ps1` may wait for a
safe window without stopping that process. It records each poll below
`runs/waiters`, owns a separate create-new waiter lock, and calls the same S0
wrapper only after RAM, disk, GPU-process, WSL and project-lock checks pass.
The monitored launcher repeats the authoritative preflight immediately before
creating the training lock.

Synthetic self-test artifacts say `claim_eligible=false`. Experiment artifacts leave `claim_eligible` null because resource monitoring cannot adjudicate a scientific claim. Formal result eligibility remains governed by the registered protocol, checkpoint provenance, prediction index, and result aggregation gates.

## Safe self-test

The only mode that omits the training preflight is `-SelfTest`. PowerShell parameter sets prevent it from accepting a caller-selected script or arguments. It runs only `scripts/monitored_synthetic_child.py`, forces `CUDA_VISIBLE_DEVICES=-1`, lasts two seconds, and records `claim_eligible=false`.

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
& .\scripts\Invoke-MonitoredExperiment.ps1 `
  -SelfTest `
  -OutputDirectory runs/monitored/launcher-selftest `
  -SampleIntervalSeconds 1
```

Use a new empty directory for every invocation. The launcher never kills a child process. If its own host is interrupted while a child remains alive, the summary records that condition and the launcher performs its token-checked lock cleanup.
