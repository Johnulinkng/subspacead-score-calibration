[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$OutputDirectory, [switch]$BenchmarkTrainOnly, [switch]$Resume)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$oasScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$oasProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $oasScriptRoot '..'))
Set-Location -LiteralPath $oasProjectRoot
. (Join-Path $oasScriptRoot 'Set-ProjectEnvironment.ps1') | Out-Null
foreach ($oasVar in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')) { [Environment]::SetEnvironmentVariable($oasVar,'2','Process') }
$env:PYTHONUNBUFFERED = '1'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class OASOwnedProcessMemory {
    [DllImport("psapi.dll", SetLastError=true)] public static extern bool EmptyWorkingSet(IntPtr process);
}
'@
$oasOutput = [System.IO.Path]::GetFullPath((Join-Path $oasProjectRoot $OutputDirectory))
if (-not $oasOutput.StartsWith((Join-Path $oasProjectRoot 'runs\quick').TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Output outside runs/quick.' }
if ((Test-Path -LiteralPath $oasOutput) -and -not $Resume) { throw 'Refusing to overwrite OAS run.' }
$oasMonitor = $oasOutput + '-monitor-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
New-Item -ItemType Directory -Path $oasMonitor | Out-Null
$oasOutcome = [ordered]@{status='initializing';started_at_utc=(Get-Date).ToUniversalTime().ToString('o');output=$oasOutput;monitor=$oasMonitor;launcher_pid=$PID;child_pid=$null;error=$null;claim_eligible=$false}
$oasLock = Join-Path $oasProjectRoot 'runs\.quick-oas.lock.json'
$oasToken = [guid]::NewGuid().ToString('N')
$oasLockOwned = $false
$oasChild = $null
try {
    $oasOs = Get-CimInstance Win32_OperatingSystem
    if ($oasOs.FreePhysicalMemory*1KB/1GB -lt 3.25) { throw 'Admission: need 3.25 GiB free RAM (existing measured process peak + reserve and uncertainty).' }
    if ($oasOs.FreeVirtualMemory*1KB/1GB -lt 3.25) { throw 'Admission: need 3.25 GiB free committed-memory capacity.' }
    if ((Get-PSDrive E).Free/1GB -lt 10) { throw 'Admission: need 10 GiB available on E.' }
    $oasStream = [System.IO.File]::Open($oasLock,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
    $oasWriter = [System.IO.StreamWriter]::new($oasStream,[System.Text.UTF8Encoding]::new($false))
    try { $oasWriter.Write(([ordered]@{pid=$PID;token=$oasToken;output=$oasOutput;time=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json)) } finally { $oasWriter.Dispose() }
    $oasLockOwned = $true
    $oasArgs = @('scripts/run_oas_experiment.py','--output-dir',$OutputDirectory)
    if ($BenchmarkTrainOnly) { $oasArgs += '--benchmark-train-only' }
    if ($Resume) { $oasArgs += '--resume' }
    $oasLauncher = Start-Process -FilePath (Join-Path $oasProjectRoot '.venv\Scripts\python.exe') -ArgumentList $oasArgs -WorkingDirectory $oasProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $oasMonitor 'stdout.log') -RedirectStandardError (Join-Path $oasMonitor 'stderr.log') -PassThru
    $oasChild = $oasLauncher
    Start-Sleep -Milliseconds 300
    $oasDesc = @(Get-CimInstance Win32_Process -Filter ("ParentProcessId = {0}" -f $oasLauncher.Id) | Where-Object { $_.ExecutablePath -like "$oasProjectRoot\*" -and $_.Name -eq 'python.exe' })
    if ($oasDesc.Count -eq 1) { $oasChild = Get-Process -Id $oasDesc[0].ProcessId }
    $oasChild.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
    $oasOutcome.child_pid = $oasChild.Id
    $oasOutcome.status = 'running'
    while (-not $oasLauncher.HasExited) {
        $oasChild.Refresh()
        $oasOs = Get-CimInstance Win32_OperatingSystem
        $oasSample = [ordered]@{time=(Get-Date).ToUniversalTime().ToString('o');pid=$oasChild.Id;working_set_gib=$oasChild.WorkingSet64/1GB;peak_working_set_gib=$oasChild.PeakWorkingSet64/1GB;free_ram_gib=$oasOs.FreePhysicalMemory*1KB/1GB;free_commit_gib=$oasOs.FreeVirtualMemory*1KB/1GB;cpu_seconds=$oasChild.TotalProcessorTime.TotalSeconds}
        [System.IO.File]::AppendAllText((Join-Path $oasMonitor 'resources.jsonl'),($oasSample|ConvertTo-Json -Compress)+[Environment]::NewLine,[System.Text.UTF8Encoding]::new($false))
        if (-not $oasChild.HasExited -and $oasSample.free_ram_gib -lt 2.0 -and $oasSample.free_commit_gib -ge 2) {
            $oasTrimmed = [OASOwnedProcessMemory]::EmptyWorkingSet($oasChild.Handle)
            Start-Sleep -Seconds 1
            $oasChild.Refresh()
            $oasOs = Get-CimInstance Win32_OperatingSystem
            $oasSample.working_set_gib = $oasChild.WorkingSet64/1GB
            $oasSample.free_ram_gib = $oasOs.FreePhysicalMemory*1KB/1GB
            $oasSample.free_commit_gib = $oasOs.FreeVirtualMemory*1KB/1GB
            $oasSample['event'] = 'owned_working_set_trim'
            $oasSample['trim_succeeded'] = $oasTrimmed
            [System.IO.File]::AppendAllText((Join-Path $oasMonitor 'resources.jsonl'),($oasSample|ConvertTo-Json -Compress)+[Environment]::NewLine,[System.Text.UTF8Encoding]::new($false))
        }
        if (-not $oasChild.HasExited -and ($oasSample.working_set_gib -gt 2.5 -or $oasSample.free_ram_gib -lt 1.5 -or $oasSample.free_commit_gib -lt 2)) {
            $oasChild.Kill()
            $oasChild.WaitForExit(10000) | Out-Null
            throw 'Owned OAS process stopped at measured resource-reserve boundary; partial evidence retained.'
        }
        Start-Sleep -Seconds 5
        $oasLauncher.Refresh()
    }
    $oasLauncher.WaitForExit()
    $oasOutcome['exit_code'] = $oasLauncher.ExitCode
    $oasOutcome.status = if ($oasLauncher.ExitCode -eq 0) { 'completed' } else { 'failed' }
} catch { $oasOutcome.status='failed'; $oasOutcome.error=$_.Exception.Message }
finally {
    $oasOutcome['finished_at_utc']=(Get-Date).ToUniversalTime().ToString('o')
    [System.IO.File]::WriteAllText((Join-Path $oasMonitor 'summary.json'),($oasOutcome|ConvertTo-Json -Depth 5),[System.Text.UTF8Encoding]::new($false))
    if ($oasLockOwned -and (Test-Path -LiteralPath $oasLock)) {
        $oasExisting = Get-Content -LiteralPath $oasLock -Raw | ConvertFrom-Json
        if ($oasExisting.token -ceq $oasToken) { [System.IO.File]::Delete($oasLock) }
    }
}
$oasOutcome | ConvertTo-Json -Depth 5
if ($oasOutcome.status -ne 'completed') { exit 1 }
