[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$OutputDirectory, [switch]$Resume)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot
. (Join-Path $PSScriptRoot 'Set-ProjectEnvironment.ps1') | Out-Null
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:NUMEXPR_NUM_THREADS = '2'
$env:PYTHONUNBUFFERED = '1'
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
if (-not $outputRoot.StartsWith((Join-Path $projectRoot 'runs\quick').TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Quick output outside its run directory.' }
if ((Test-Path -LiteralPath $outputRoot) -and -not $Resume) { throw 'Refusing to overwrite a quick run.' }
$monitorRoot = if ($Resume) { $outputRoot + '-monitor-resume-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') } else { $outputRoot + '-monitor' }
New-Item -ItemType Directory -Path $monitorRoot -ErrorAction Stop | Out-Null
$os = Get-CimInstance Win32_OperatingSystem
if ($os.FreePhysicalMemory * 1KB / 1GB -lt 3.5) { throw 'Quick experiment needs 3.5 GiB initial free RAM.' }
if ((Get-PSDrive E).Free / 1GB -lt 20) { throw 'Quick experiment needs 20 GiB available disk space.' }
$lockPath = Join-Path $projectRoot 'runs\.quick-padim.lock.json'
$token = [guid]::NewGuid().ToString('N')
$lockStream = [System.IO.File]::Open($lockPath,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
$lockWriter = [System.IO.StreamWriter]::new($lockStream,[System.Text.UTF8Encoding]::new($false))
try { $lockWriter.Write(([ordered]@{pid=$PID;token=$token;output=$outputRoot;created_at_utc=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json)) } finally { $lockWriter.Dispose() }
$processLauncher = $null
$child = $null
$outcome = [ordered]@{status='initializing';started_at_utc=(Get-Date).ToUniversalTime().ToString('o');output=$outputRoot;monitor=$monitorRoot;launcher_pid=$PID;child_pid=$null;error=$null;claim_eligible=$false}
try {
    $arguments = @('scripts/run_quick_experiment.py','--output-dir',$OutputDirectory)
    if ($Resume) { $arguments += '--resume' }
    $processLauncher = Start-Process -FilePath (Join-Path $projectRoot '.venv\Scripts\python.exe') -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $monitorRoot 'stdout.log') -RedirectStandardError (Join-Path $monitorRoot 'stderr.log') -PassThru
    $child = $processLauncher
    Start-Sleep -Milliseconds 300
    $descendants = @(Get-CimInstance Win32_Process -Filter ("ParentProcessId = {0}" -f $processLauncher.Id) | Where-Object { $_.ExecutablePath -like "$projectRoot\*" -and $_.Name -eq 'python.exe' })
    if ($descendants.Count -eq 1) { $child = Get-Process -Id $descendants[0].ProcessId }
    $child.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
    $outcome.child_pid = $child.Id
    $outcome.status = 'running'
    while (-not $processLauncher.HasExited) {
        $child.Refresh()
        $os = Get-CimInstance Win32_OperatingSystem
        $sample = [ordered]@{time=(Get-Date).ToUniversalTime().ToString('o');pid=$child.Id;alive=(-not $child.HasExited);working_set_gib=$child.WorkingSet64/1GB;peak_working_set_gib=$child.PeakWorkingSet64/1GB;free_ram_gib=$os.FreePhysicalMemory*1KB/1GB;free_commit_gib=$os.FreeVirtualMemory*1KB/1GB;cpu_seconds=$child.TotalProcessorTime.TotalSeconds}
        [System.IO.File]::AppendAllText((Join-Path $monitorRoot 'resources.jsonl'), ($sample | ConvertTo-Json -Compress) + [Environment]::NewLine,[System.Text.UTF8Encoding]::new($false))
        if (-not $child.HasExited -and ($sample.working_set_gib -gt 2.5 -or $sample.free_ram_gib -lt 1.5 -or $sample.free_commit_gib -lt 2)) {
            $child.Kill()
            $child.WaitForExit(10000) | Out-Null
            throw 'Owned quick process stopped because its resource reserve was exceeded.'
        }
        Start-Sleep -Seconds 10
        $processLauncher.Refresh()
    }
    $processLauncher.WaitForExit()
    $outcome.exit_code = $processLauncher.ExitCode
    $outcome.status = if ($processLauncher.ExitCode -eq 0) { 'completed' } else { 'failed' }
} catch { $outcome.status='failed'; $outcome.error=$_.Exception.Message }
finally {
    $outcome.finished_at_utc=(Get-Date).ToUniversalTime().ToString('o')
    [System.IO.File]::WriteAllText((Join-Path $monitorRoot 'summary.json'), ($outcome | ConvertTo-Json -Depth 5),[System.Text.UTF8Encoding]::new($false))
    if (Test-Path -LiteralPath $lockPath) {
        $existing = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
        if ($existing.token -ceq $token) { [System.IO.File]::Delete($lockPath) }
    }
}
if ($outcome.status -ne 'completed') { exit 1 }
