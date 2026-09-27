[CmdletBinding()]
param(
    [double]$MinimumFreeRamGiB = 8,
    [double]$MinimumFreeDiskGiB = 40,
    [string]$OutputDirectory = "artifacts/preflight"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
}
$projectPrefix = $projectRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
if (-not $outputRoot.StartsWith($projectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Preflight output must remain inside the project: $outputRoot"
}
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$os = Get-CimInstance Win32_OperatingSystem
$freeRamGiB = [math]::Round([double]$os.FreePhysicalMemory * 1KB / 1GB, 3)
$totalRamGiB = [math]::Round([double]$os.TotalVisibleMemorySize * 1KB / 1GB, 3)
$projectDriveName = ([System.IO.Path]::GetPathRoot($projectRoot)).Substring(0, 1)
$projectDrive = Get-PSDrive -Name $projectDriveName
$freeDiskGiB = [math]::Round([double]$projectDrive.Free / 1GB, 3)

$gpuRows = @()
$gpuQueryError = $null
try {
    $gpuOutput = @(& nvidia-smi.exe --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>&1)
    if ($LASTEXITCODE -ne 0) { throw ($gpuOutput -join "`n") }
    foreach ($line in $gpuOutput) {
        $parts = @($line -split ',' | ForEach-Object { $_.Trim() })
        if ($parts.Count -ne 7) { throw "Unexpected nvidia-smi GPU row: $line" }
        $gpuRows += [ordered]@{
            index = [int]$parts[0]
            name = $parts[1]
            uuid = $parts[2]
            memory_total_mib = [int]$parts[3]
            memory_used_mib = [int]$parts[4]
            memory_free_mib = [int]$parts[5]
            utilization_percent = [int]$parts[6]
        }
    }
} catch {
    $gpuQueryError = $_.Exception.Message
}

$computeProcesses = @()
$computeQueryError = $null
try {
    $computeOutput = @(& nvidia-smi.exe --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1)
    if ($LASTEXITCODE -ne 0) { throw ($computeOutput -join "`n") }
    foreach ($line in $computeOutput) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $parts = @($line -split ',' | ForEach-Object { $_.Trim() })
        $computeProcesses += [ordered]@{
            pid = if ($parts[0] -match '^\d+$') { [int]$parts[0] } else { $parts[0] }
            process_name = if ($parts.Count -gt 1) { $parts[1] } else { $null }
            used_memory_mib = if ($parts.Count -gt 2 -and $parts[2] -match '^\d+$') { [int]$parts[2] } else { if ($parts.Count -gt 2) { $parts[2] } else { $null } }
        }
    }
} catch {
    $computeQueryError = $_.Exception.Message
}

$runningWsl = @()
$wslQueryError = $null
try {
    $wslOutput = @(& wsl.exe --list --running --quiet 2>&1)
    if ($LASTEXITCODE -ne 0) { throw ($wslOutput -join "`n") }
    $runningWsl = @($wslOutput | ForEach-Object { ($_ -replace "`0", '').Trim() } | Where-Object { $_ })
} catch {
    $wslQueryError = $_.Exception.Message
}

$lockPath = Join-Path $projectRoot "runs/.training.lock.json"
$lock = [ordered]@{ path = $lockPath; exists = (Test-Path -LiteralPath $lockPath); pid = $null; process_alive = $false; parse_error = $null }
if ($lock.exists) {
    try {
        $lockData = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
        if ($null -ne $lockData.pid) {
            $lock.pid = [int]$lockData.pid
            $lock.process_alive = $null -ne (Get-Process -Id $lock.pid -ErrorAction SilentlyContinue)
        }
    } catch {
        $lock.parse_error = $_.Exception.Message
    }
}

$checks = [ordered]@{
    free_ram = $freeRamGiB -ge $MinimumFreeRamGiB
    free_disk = $freeDiskGiB -ge $MinimumFreeDiskGiB
    gpu_query = $null -eq $gpuQueryError -and $gpuRows.Count -gt 0
    no_compute_processes = $null -eq $computeQueryError -and $computeProcesses.Count -eq 0
    no_running_wsl = $null -eq $wslQueryError -and $runningWsl.Count -eq 0
    no_active_project_lock = -not ($lock.exists -and ($lock.process_alive -or $null -ne $lock.parse_error))
}
$ready = -not ($checks.Values -contains $false)
$timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
$report = [ordered]@{
    schema_version = 1
    checked_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    project_root = $projectRoot
    ready_for_training_launch = $ready
    thresholds = [ordered]@{ minimum_free_ram_gib = $MinimumFreeRamGiB; minimum_free_disk_gib = $MinimumFreeDiskGiB }
    checks = $checks
    host = [ordered]@{ total_ram_gib = $totalRamGiB; free_ram_gib = $freeRamGiB; project_drive = $projectDriveName; free_disk_gib = $freeDiskGiB }
    gpus = $gpuRows
    gpu_query_error = $gpuQueryError
    compute_processes = $computeProcesses
    compute_query_error = $computeQueryError
    running_wsl_distributions = $runningWsl
    wsl_query_error = $wslQueryError
    project_lock = $lock
}
$reportPath = Join-Path $outputRoot ("preflight_" + $timestamp + ".json")
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
$report | ConvertTo-Json -Depth 8
Write-Host "Preflight report: $reportPath"
if (-not $ready) { exit 2 }
