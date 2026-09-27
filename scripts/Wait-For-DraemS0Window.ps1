[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [ValidateRange(10, 3600)]
    [int]$PollIntervalSeconds = 30,

    [ValidateRange(1, 1440)]
    [int]$TimeoutMinutes = 240
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
. (Join-Path $PSScriptRoot "Set-ProjectEnvironment.ps1") | Out-Null

$waiterRoot = Join-Path $projectRoot "runs/waiters"
$waiterLock = Join-Path $projectRoot "runs/.draem-s0-waiter.lock.json"
$outputFull = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
}
$outputPrefix = (Join-Path $projectRoot "runs/monitored").TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
if (-not $outputFull.StartsWith($outputPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "S0 output must remain below runs/monitored: $outputFull"
}
if (Test-Path -LiteralPath $outputFull) {
    throw "S0 output already exists: $outputFull"
}

New-Item -ItemType Directory -Path $waiterRoot -Force | Out-Null
$waiterId = [guid]::NewGuid().ToString("N")
$waiterLog = Join-Path $waiterRoot ("draem-s0-waiter-{0}.jsonl" -f $waiterId)
$startedAtUtc = (Get-Date).ToUniversalTime()
$deadline = $startedAtUtc.AddMinutes($TimeoutMinutes)

function Add-WaiterRecord {
    param([Parameter(Mandatory = $true)]$Value)
    $line = $Value | ConvertTo-Json -Compress -Depth 6
    [System.IO.File]::AppendAllText($waiterLog, $line + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
}

$lockPayload = [ordered]@{
    schema_version = 1
    token = $waiterId
    pid = $PID
    started_at_utc = $startedAtUtc.ToString("o")
    output_directory = $outputFull
    waiter_log = $waiterLog
}
$lockStream = [System.IO.File]::Open(
    $waiterLock,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::Read
)
try {
    $writer = [System.IO.StreamWriter]::new($lockStream, [System.Text.UTF8Encoding]::new($false))
    try { $writer.Write(($lockPayload | ConvertTo-Json -Depth 4)) } finally { $writer.Dispose() }
} finally {
    $lockStream.Dispose()
}

try {
    Add-WaiterRecord ([ordered]@{
        event = "started"
        recorded_at_utc = $startedAtUtc.ToString("o")
        output_directory = $outputFull
        poll_interval_seconds = $PollIntervalSeconds
        timeout_minutes = $TimeoutMinutes
    })

    while ((Get-Date).ToUniversalTime() -lt $deadline) {
        $checkedAt = (Get-Date).ToUniversalTime()
        $os = Get-CimInstance Win32_OperatingSystem
        $freeRamGiB = [math]::Round([double]$os.FreePhysicalMemory * 1KB / 1GB, 3)
        $driveName = ([System.IO.Path]::GetPathRoot($projectRoot)).Substring(0, 1)
        $freeDiskGiB = [math]::Round([double](Get-PSDrive -Name $driveName).Free / 1GB, 3)

        $computeQueryOk = $false
        $computePids = @()
        try {
            $computeRows = @(& nvidia-smi.exe --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)
            $computeQueryOk = $LASTEXITCODE -eq 0
            if ($computeQueryOk) {
                $computePids = @(
                    $computeRows |
                        ForEach-Object { $_.Trim() } |
                        Where-Object { $_ -match '^\d+$' } |
                        ForEach-Object { [int]$_ }
                )
            }
        } catch {
            $computeQueryOk = $false
        }

        $wslQueryOk = $false
        $runningWsl = @()
        try {
            $wslRows = @(& wsl.exe --list --running --quiet 2>&1)
            $wslQueryOk = $LASTEXITCODE -eq 0
            if ($wslQueryOk) {
                $runningWsl = @(
                    $wslRows |
                        ForEach-Object { ($_ -replace "`0", '').Trim() } |
                        Where-Object { $_ }
                )
            }
        } catch {
            $wslQueryOk = $false
        }

        $trainingLockExists = Test-Path -LiteralPath (Join-Path $projectRoot "runs/.training.lock.json")
        $readyToAttempt = (
            $freeRamGiB -ge 8.0 -and
            $freeDiskGiB -ge 40.0 -and
            $computeQueryOk -and
            $computePids.Count -eq 0 -and
            $wslQueryOk -and
            $runningWsl.Count -eq 0 -and
            -not $trainingLockExists
        )

        Add-WaiterRecord ([ordered]@{
            event = "poll"
            recorded_at_utc = $checkedAt.ToString("o")
            free_ram_gib = $freeRamGiB
            free_disk_gib = $freeDiskGiB
            compute_query_ok = $computeQueryOk
            compute_pids = $computePids
            wsl_query_ok = $wslQueryOk
            running_wsl = $runningWsl
            training_lock_exists = $trainingLockExists
            ready_to_attempt = $readyToAttempt
        })

        if ($readyToAttempt) {
            Add-WaiterRecord ([ordered]@{
                event = "launching"
                recorded_at_utc = (Get-Date).ToUniversalTime().ToString("o")
            })
            & (Join-Path $PSScriptRoot "Invoke-DraemS0Smoke.ps1") -OutputDirectory $OutputDirectory
            $launchExitCode = $LASTEXITCODE
            Add-WaiterRecord ([ordered]@{
                event = "launcher_finished"
                recorded_at_utc = (Get-Date).ToUniversalTime().ToString("o")
                exit_code = $launchExitCode
            })
            exit $launchExitCode
        }

        Start-Sleep -Seconds $PollIntervalSeconds
    }

    Add-WaiterRecord ([ordered]@{
        event = "timeout"
        recorded_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    })
    exit 3
} finally {
    if (Test-Path -LiteralPath $waiterLock) {
        try {
            $currentLock = Get-Content -LiteralPath $waiterLock -Raw | ConvertFrom-Json
            if ($currentLock.token -eq $waiterId) {
                Remove-Item -LiteralPath $waiterLock -Force
            }
        } catch {
            Add-WaiterRecord ([ordered]@{
                event = "lock_cleanup_failed"
                recorded_at_utc = (Get-Date).ToUniversalTime().ToString("o")
                error = $_.Exception.Message
            })
        }
    }
}
