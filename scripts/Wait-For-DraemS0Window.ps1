[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [string]$ResourceProfile,

    [ValidateRange(10, 3600)]
    [int]$PollIntervalSeconds = 30,

    [ValidateRange(1, 1440)]
    [int]$TimeoutMinutes = 240
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
. (Join-Path $PSScriptRoot "Set-ProjectEnvironment.ps1") | Out-Null
if ([string]::IsNullOrWhiteSpace($ResourceProfile)) {
    $resourceConfig = Get-Content -LiteralPath (Join-Path $projectRoot "configs/resource_budget.json") -Raw | ConvertFrom-Json
    $ResourceProfile = $resourceConfig.draem_s0.profile
}

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
        # Use the same authoritative budget as the launcher. Never duplicate
        # a fixed RAM threshold here, which could make waiting permanent.
        $pollOutput = "runs/waiters/preflight-$waiterId-$($checkedAt.ToString('yyyyMMddTHHmmssfffZ'))"
        $pollOutputFull = Join-Path $projectRoot $pollOutput
        $preflightScript = Join-Path $PSScriptRoot "Test-TrainingPreflight.ps1"
        $hostExecutable = (Get-Process -Id $PID).Path
        $pollProcess = Start-Process -FilePath $hostExecutable -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $preflightScript + '"'),
            '-ResourceProfile', ('"' + $ResourceProfile + '"'), '-OutputDirectory', ('"' + $pollOutput + '"')
        ) -WorkingDirectory $projectRoot -WindowStyle Hidden -Wait -PassThru
        $reports = @(Get-ChildItem -LiteralPath $pollOutputFull -Filter 'preflight_*.json' -File)
        if ($reports.Count -ne 1) { throw "Preflight produced no unambiguous waiter snapshot." }
        $report = Get-Content -LiteralPath $reports[0].FullName -Raw | ConvertFrom-Json
        $readyToAttempt = $pollProcess.ExitCode -eq 0 -and $report.ready_for_training_launch

        Add-WaiterRecord ([ordered]@{
            event = "poll"
            recorded_at_utc = $checkedAt.ToString("o")
            preflight_report = $reports[0].FullName
            preflight_exit_code = $pollProcess.ExitCode
            budget_basis = $report.budget_basis
            thresholds = $report.thresholds
            free_ram_gib = $report.host.free_ram_gib
            free_disk_gib = $report.host.free_disk_gib
            checks = $report.checks
            ready_to_attempt = $readyToAttempt
        })

        if ($readyToAttempt) {
            Add-WaiterRecord ([ordered]@{
                event = "launching"
                recorded_at_utc = (Get-Date).ToUniversalTime().ToString("o")
            })
            & (Join-Path $PSScriptRoot "Invoke-DraemS0Smoke.ps1") -OutputDirectory $OutputDirectory -ResourceProfile $ResourceProfile
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
