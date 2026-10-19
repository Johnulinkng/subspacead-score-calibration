param([string]$ProjectRoot='E:\3k')
$ErrorActionPreference='Stop'
$runRoot=Join-Path $ProjectRoot 'runs/research-q3-20261016'
$pausePath=Join-Path $runRoot 'PAUSED_BY_USER.json'
if (Test-Path -LiteralPath $pausePath) { Write-Output "Queue is paused by user: $pausePath"; exit 0 }
$statePath=Join-Path $runRoot 'baseline_queue_state.json'
$receiptPath=Join-Path $runRoot 'queue-handoff.json'
$first=Join-Path $runRoot 'baselines/RD4AD/mvtec_ad/bottle/1887537248'
while ($true) {
    if (Test-Path -LiteralPath $pausePath) { Write-Output "Queue is paused by user: $pausePath"; exit 0 }
    $fit=Join-Path $first 'execution/receipt.json'
    $inference=Join-Path $first 'inference-execution/receipt.json'
    if ((Test-Path -LiteralPath $fit) -and (Test-Path -LiteralPath $inference)) {
        $fr=Get-Content -LiteralPath $fit -Raw | ConvertFrom-Json
        $ir=Get-Content -LiteralPath $inference -Raw | ConvertFrom-Json
        if ($fr.status -eq 'completed' -and $ir.status -eq 'completed') { break }
    }
    Start-Sleep -Seconds 10
}
$state=Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$queueProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$($state.pid)"
if ($null -ne $queueProcess) {
    if ($queueProcess.CommandLine -notlike '*scripts/run_q3_baseline_queue.py*' -or $queueProcess.CommandLine -like '*--queue*') {throw 'Unexpected original queue process'}
    $all=@(Get-CimInstance Win32_Process)
    $children=@($all | Where-Object {$_.ParentProcessId -eq $state.pid})
    $descendants=@($children)
    foreach($child in $children) {$descendants += @($all | Where-Object {$_.ParentProcessId -eq $child.ProcessId})}
    foreach($process in $descendants) {
        if ($process.CommandLine -notlike '*baseline-source*run_q3_baseline_job.py*') {throw 'Unexpected descendant of original queue'}
    }
    $record=@{at=(Get-Date).ToUniversalTime().ToString('o');reason='Switch future jobs to disk-backed exact PatchCore collection; preserve completed full RD4AD run';originalQueuePid=$state.pid;interruptedJob=$state.job;descendants=@($descendants | Select-Object ProcessId,ParentProcessId,CommandLine)}
    $record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding utf8
    Stop-Process -Id $state.pid -ErrorAction SilentlyContinue
    foreach($process in $descendants){Stop-Process -Id $process.ProcessId -ErrorAction SilentlyContinue}
}
$python=Join-Path $ProjectRoot '.venv/Scripts/python.exe'
$nextQueue=Join-Path $runRoot 'baseline-queue-v2.json'
if (Test-Path -LiteralPath $pausePath) { Write-Output "Queue is paused by user: $pausePath"; exit 0 }
$p=Start-Process -FilePath $python -ArgumentList @('scripts/run_q3_baseline_queue.py','--queue',$nextQueue) -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runRoot 'baseline-queue-v2.stdout.log') -RedirectStandardError (Join-Path $runRoot 'baseline-queue-v2.stderr.log') -PassThru
@{status='handed_over';launcherPid=$p.Id;at=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'queue-handoff-completed.json') -Encoding utf8
