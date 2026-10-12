[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$OutputDirectory, [switch]$BenchmarkTrainOnly, [switch]$Resume)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$subspaceScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$subspaceProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $subspaceScriptRoot '..'))
Set-Location -LiteralPath $subspaceProjectRoot
. (Join-Path $subspaceScriptRoot 'Set-ProjectEnvironment.ps1') | Out-Null
foreach ($subspaceVar in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')) { [Environment]::SetEnvironmentVariable($subspaceVar,'4','Process') }
$env:PYTHONUNBUFFERED = '1'
function Get-SubspaceGpuSample {
    $subspaceGpuLine = @(& nvidia-smi -i 0 --query-gpu=memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0 -or $subspaceGpuLine.Count -ne 1) { throw 'GPU resource counters unavailable.' }
    $subspaceGpuFields = $subspaceGpuLine[0].Split(',')
    $subspaceGpuTotal = [double]::Parse($subspaceGpuFields[0].Trim(),[System.Globalization.CultureInfo]::InvariantCulture)
    $subspaceGpuUsed = [double]::Parse($subspaceGpuFields[1].Trim(),[System.Globalization.CultureInfo]::InvariantCulture)
    return [pscustomobject]@{used_gib=$subspaceGpuUsed/1024;free_gib=($subspaceGpuTotal-$subspaceGpuUsed)/1024;utilization=[int]$subspaceGpuFields[2].Trim()}
}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class SubspaceOwnedProcessMemory {
    [DllImport("psapi.dll", SetLastError=true)] public static extern bool EmptyWorkingSet(IntPtr process);
}
'@
$subspaceOutput = [System.IO.Path]::GetFullPath((Join-Path $subspaceProjectRoot $OutputDirectory))
if (-not $subspaceOutput.StartsWith((Join-Path $subspaceProjectRoot 'runs\quick').TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Output outside runs/quick.' }
if ((Test-Path -LiteralPath $subspaceOutput) -and -not $Resume) { throw 'Refusing to overwrite Subspace run.' }
$subspaceMonitor = $subspaceOutput + '-monitor-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
New-Item -ItemType Directory -Path $subspaceMonitor | Out-Null
$subspaceOutcome = [ordered]@{status='initializing';started_at_utc=(Get-Date).ToUniversalTime().ToString('o');output=$subspaceOutput;monitor=$subspaceMonitor;launcher_pid=$PID;child_pid=$null;error=$null;claim_eligible=$false}
$subspaceLock = Join-Path $subspaceProjectRoot 'runs\.quick-subspace.lock.json'
$subspaceToken = [guid]::NewGuid().ToString('N')
$subspaceLockOwned = $false
$subspaceChild = $null
try {
    $subspaceOs = Get-CimInstance Win32_OperatingSystem
    if ($subspaceOs.FreePhysicalMemory*1KB/1GB -lt 3.25) { throw 'Admission: need 3.25 GiB free RAM (existing measured process peak + reserve and uncertainty).' }
    if ($subspaceOs.FreeVirtualMemory*1KB/1GB -lt 3.25) { throw 'Admission: need 3.25 GiB free committed-memory capacity.' }
    $subspaceGpu = Get-SubspaceGpuSample
    if ($subspaceGpu.free_gib -lt 4) { throw 'Admission: need 4 GiB free GPU memory for the bounded inference profile and reserve.' }
    if ((Get-PSDrive E).Free/1GB -lt 10) { throw 'Admission: need 10 GiB available on E.' }
    $subspaceStream = [System.IO.File]::Open($subspaceLock,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
    $subspaceWriter = [System.IO.StreamWriter]::new($subspaceStream,[System.Text.UTF8Encoding]::new($false))
    try { $subspaceWriter.Write(([ordered]@{pid=$PID;token=$subspaceToken;output=$subspaceOutput;time=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json)) } finally { $subspaceWriter.Dispose() }
    $subspaceLockOwned = $true
    $subspaceArgs = @('scripts/run_subspace_experiment.py','--output-dir',$OutputDirectory)
    if ($BenchmarkTrainOnly) { $subspaceArgs += '--benchmark-train-only' }
    if ($Resume) { $subspaceArgs += '--resume' }
    $subspaceLauncher = Start-Process -FilePath (Join-Path $subspaceProjectRoot '.venv\Scripts\python.exe') -ArgumentList $subspaceArgs -WorkingDirectory $subspaceProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $subspaceMonitor 'stdout.log') -RedirectStandardError (Join-Path $subspaceMonitor 'stderr.log') -PassThru
    $subspaceChild = $subspaceLauncher
    Start-Sleep -Milliseconds 300
    $subspaceDesc = @(Get-CimInstance Win32_Process -Filter ("ParentProcessId = {0}" -f $subspaceLauncher.Id) | Where-Object { $_.ExecutablePath -like "$subspaceProjectRoot\*" -and $_.Name -eq 'python.exe' })
    if ($subspaceDesc.Count -eq 1) { $subspaceChild = Get-Process -Id $subspaceDesc[0].ProcessId }
    $subspaceChild.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Normal
    $subspaceOutcome.child_pid = $subspaceChild.Id
    $subspaceOutcome.status = 'running'
    while (-not $subspaceLauncher.HasExited) {
        $subspaceChild.Refresh()
        if ($subspaceChild.HasExited) { break }
        $subspaceOs = Get-CimInstance Win32_OperatingSystem
        $subspaceGpu = Get-SubspaceGpuSample
        try {
            $subspaceCpuTime = $subspaceChild.TotalProcessorTime
            if ($null -eq $subspaceCpuTime) {
                $subspaceChild.Refresh()
                if ($subspaceChild.HasExited) { break }
                throw 'Owned process CPU counter unavailable while the child remains active.'
            }
            $subspaceSample = [ordered]@{time=(Get-Date).ToUniversalTime().ToString('o');pid=$subspaceChild.Id;working_set_gib=$subspaceChild.WorkingSet64/1GB;peak_working_set_gib=$subspaceChild.PeakWorkingSet64/1GB;free_ram_gib=$subspaceOs.FreePhysicalMemory*1KB/1GB;free_commit_gib=$subspaceOs.FreeVirtualMemory*1KB/1GB;cpu_seconds=$subspaceCpuTime.TotalSeconds}
        } catch {
            $subspaceChild.Refresh()
            if ($subspaceChild.HasExited) { break }
            throw
        }
        $subspaceSample['gpu_used_gib'] = $subspaceGpu.used_gib
        $subspaceSample['gpu_free_gib'] = $subspaceGpu.free_gib
        $subspaceSample['gpu_utilization'] = $subspaceGpu.utilization
        [System.IO.File]::AppendAllText((Join-Path $subspaceMonitor 'resources.jsonl'),($subspaceSample|ConvertTo-Json -Compress)+[Environment]::NewLine,[System.Text.UTF8Encoding]::new($false))
        if (-not $subspaceChild.HasExited -and $subspaceSample.free_ram_gib -lt 2.0 -and $subspaceSample.free_commit_gib -ge 2) {
            $subspaceTrimmed = [SubspaceOwnedProcessMemory]::EmptyWorkingSet($subspaceChild.Handle)
            Start-Sleep -Seconds 1
            $subspaceChild.Refresh()
            $subspaceOs = Get-CimInstance Win32_OperatingSystem
            $subspaceSample.working_set_gib = $subspaceChild.WorkingSet64/1GB
            $subspaceSample.free_ram_gib = $subspaceOs.FreePhysicalMemory*1KB/1GB
            $subspaceSample.free_commit_gib = $subspaceOs.FreeVirtualMemory*1KB/1GB
            $subspaceSample['event'] = 'owned_working_set_trim'
            $subspaceSample['trim_succeeded'] = $subspaceTrimmed
            [System.IO.File]::AppendAllText((Join-Path $subspaceMonitor 'resources.jsonl'),($subspaceSample|ConvertTo-Json -Compress)+[Environment]::NewLine,[System.Text.UTF8Encoding]::new($false))
        }
        if (-not $subspaceChild.HasExited -and ($subspaceSample.working_set_gib -gt 2.5 -or $subspaceSample.free_ram_gib -lt 1.5 -or $subspaceSample.free_commit_gib -lt 2 -or $subspaceSample.gpu_free_gib -lt 2)) {
            $subspaceChild.Kill()
            $subspaceChild.WaitForExit(10000) | Out-Null
            throw 'Owned Subspace process stopped at measured resource-reserve boundary; partial evidence retained.'
        }
        Start-Sleep -Seconds 5
        $subspaceLauncher.Refresh()
    }
    $subspaceLauncher.WaitForExit()
    $subspaceOutcome['exit_code'] = $subspaceLauncher.ExitCode
    $subspaceOutcome.status = if ($subspaceLauncher.ExitCode -eq 0) { 'completed' } else { 'failed' }
} catch { $subspaceOutcome.status='failed'; $subspaceOutcome.error=$_.Exception.Message }
finally {
    if ($subspaceOutcome.status -ne 'completed' -and $null -ne $subspaceChild) {
        $subspaceChild.Refresh()
        if (-not $subspaceChild.HasExited) { $subspaceChild.Kill(); $subspaceChild.WaitForExit(10000) | Out-Null }
    }
    $subspaceOutcome['finished_at_utc']=(Get-Date).ToUniversalTime().ToString('o')
    [System.IO.File]::WriteAllText((Join-Path $subspaceMonitor 'summary.json'),($subspaceOutcome|ConvertTo-Json -Depth 5),[System.Text.UTF8Encoding]::new($false))
    if ($subspaceLockOwned -and (Test-Path -LiteralPath $subspaceLock)) {
        $subspaceExisting = Get-Content -LiteralPath $subspaceLock -Raw | ConvertFrom-Json
        if ($subspaceExisting.token -ceq $subspaceToken) { [System.IO.File]::Delete($subspaceLock) }
    }
}
$subspaceOutcome | ConvertTo-Json -Depth 5
if ($subspaceOutcome.status -ne 'completed') { exit 1 }
