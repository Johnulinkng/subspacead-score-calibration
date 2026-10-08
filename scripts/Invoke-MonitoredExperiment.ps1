[CmdletBinding(DefaultParameterSetName = "Experiment")]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [Parameter(Mandatory = $true, ParameterSetName = "Experiment")]
    [string]$ScriptPath,

    [Parameter(ParameterSetName = "Experiment")]
    [string[]]$ScriptArgument = @(),

    [Parameter(Mandatory = $true, ParameterSetName = "Experiment")]
    [ValidateSet("debug_limited", "development_full", "author_reproduction", "formal_full")]
    [string]$RunPurpose,

    [Parameter(Mandatory = $true, ParameterSetName = "SelfTest")]
    [switch]$SelfTest,

    [Parameter(Mandatory = $true, ParameterSetName = "CapacityDiagnostic")]
    [switch]$CapacityDiagnostic,

    [Parameter(ParameterSetName = "CapacityDiagnostic")]
    [ValidateSet("original", "activation_checkpointing")]
    [string]$DiagnosticMemoryStrategy = "original",

    [Parameter(ParameterSetName = "Experiment")]
    [string]$ResourceProfile,

    [ValidateRange(1, 3600)]
    [int]$SampleIntervalSeconds = 30
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$monitoredRunsRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "runs/monitored"))
$monitoredRunsPrefix = $monitoredRunsRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
$projectPython = [System.IO.Path]::GetFullPath((Join-Path $projectRoot ".venv/Scripts/python.exe"))
$preflightScript = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "scripts/Test-TrainingPreflight.ps1"))
$selfTestScript = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "scripts/monitored_synthetic_child.py"))
$lockPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "runs/.training.lock.json"))
$fallbackEvidenceRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "artifacts/monitor_rejections"))
$launcherPath = [System.IO.Path]::GetFullPath($PSCommandPath)
$launcherParameterSetName = $PSCmdlet.ParameterSetName

$startedAtUtc = (Get-Date).ToUniversalTime()
$runToken = [guid]::NewGuid().ToString("N")
$outputRoot = $null
$summaryPath = $null
$samplesPath = $null
$stdoutPath = $null
$stderrPath = $null
$commandPath = $null
$preflightReportPath = $null
$preflightReady = $false
$preflightExitCode = $null
$lockCreated = $false
$lockReleased = $false
$lockReleaseStatus = "not_created"
$processLauncher = $null
$processLauncherPid = $null
$child = $null
$childPid = $null
$childExecutable = $null
$childExitCode = $null
$sampleCount = 0
$status = "initializing"
$failureKind = $null
$failureMessage = $null
$fallbackEvidencePath = $null
$interrupted = $false
$exitCode = 70
$launcherSha256 = $null
$preflightSha256 = $null

function Test-IsInsidePath {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Root,
        [switch]$AllowRoot
    )
    $candidateFull = [System.IO.Path]::GetFullPath($Candidate)
    $rootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    if ($AllowRoot -and $candidateFull.Equals($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    $prefix = $rootFull + [System.IO.Path]::DirectorySeparatorChar
    return $candidateFull.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
}

function Assert-NoReparsePoint {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Boundary
    )
    $candidateFull = [System.IO.Path]::GetFullPath($Candidate)
    $boundaryFull = [System.IO.Path]::GetFullPath($Boundary).TrimEnd('\', '/')
    if (-not (Test-IsInsidePath -Candidate $candidateFull -Root $boundaryFull -AllowRoot)) {
        throw "Path escapes its required boundary: $candidateFull"
    }
    $cursor = $candidateFull
    while ($true) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Reparse points are not allowed in monitored paths: $cursor"
            }
        }
        if ($cursor.Equals($boundaryFull, [System.StringComparison]::OrdinalIgnoreCase)) { break }
        $parent = [System.IO.Path]::GetDirectoryName($cursor.TrimEnd('\', '/'))
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent.Equals($cursor, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Could not walk monitored path to boundary: $candidateFull"
        }
        $cursor = $parent
    }
}

function Assert-ReparseTargetsInsideBoundary {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Boundary,
        [System.Collections.Generic.HashSet[string]]$Visited = $null
    )
    $candidateFull = [System.IO.Path]::GetFullPath($Candidate)
    $boundaryFull = [System.IO.Path]::GetFullPath($Boundary).TrimEnd('\', '/')
    if (-not (Test-IsInsidePath -Candidate $candidateFull -Root $boundaryFull -AllowRoot)) {
        throw "Path escapes its required boundary: $candidateFull"
    }
    if ($null -eq $Visited) {
        $Visited = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    }
    $cursor = $candidateFull
    while ($true) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                if ($Visited.Add($item.FullName)) {
                    foreach ($target in @($item.Target)) {
                        if ([string]::IsNullOrWhiteSpace([string]$target)) {
                            throw "A reparse point has no inspectable target: $($item.FullName)"
                        }
                        $targetFull = if ([System.IO.Path]::IsPathRooted([string]$target)) {
                            [System.IO.Path]::GetFullPath([string]$target)
                        } else {
                            [System.IO.Path]::GetFullPath((Join-Path $item.Parent.FullName ([string]$target)))
                        }
                        if (-not (Test-IsInsidePath -Candidate $targetFull -Root $boundaryFull -AllowRoot)) {
                            throw "A project path redirects outside the project: $($item.FullName) -> $targetFull"
                        }
                        Assert-ReparseTargetsInsideBoundary -Candidate $targetFull -Boundary $boundaryFull -Visited $Visited
                    }
                }
            }
        }
        if ($cursor.Equals($boundaryFull, [System.StringComparison]::OrdinalIgnoreCase)) { break }
        $parent = [System.IO.Path]::GetDirectoryName($cursor.TrimEnd('\', '/'))
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent.Equals($cursor, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Could not walk project path to boundary: $candidateFull"
        }
        $cursor = $parent
    }
}

function ConvertTo-WindowsCommandLineArgument {
    param([AllowEmptyString()][string]$Value)
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashCount = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq [char]'\') {
            $backslashCount += 1
            continue
        }
        if ($character -eq [char]'"') {
            if ($backslashCount -gt 0) {
                [void]$builder.Append((('\' * ($backslashCount * 2)) -join ''))
            }
            [void]$builder.Append('\')
            [void]$builder.Append('"')
        } else {
            if ($backslashCount -gt 0) {
                [void]$builder.Append((('\' * $backslashCount) -join ''))
            }
            [void]$builder.Append($character)
        }
        $backslashCount = 0
    }
    if ($backslashCount -gt 0) {
        [void]$builder.Append((('\' * ($backslashCount * 2)) -join ''))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Join-WindowsCommandLine {
    param([string[]]$Arguments)
    return (($Arguments | ForEach-Object { ConvertTo-WindowsCommandLineArgument -Value $_ }) -join ' ')
}

function Write-JsonCreateNew {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value,
        [int]$Depth = 10
    )
    $json = $Value | ConvertTo-Json -Depth $Depth
    $bytes = New-Object System.Text.UTF8Encoding($false)
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
    try {
        $writer = New-Object System.IO.StreamWriter($stream, $bytes)
        try { $writer.Write($json) } finally { $writer.Dispose() }
    } finally {
        if ($null -ne $stream) { $stream.Dispose() }
    }
}

function Get-FileSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $hash = $algorithm.ComputeHash($stream)
        return ([System.BitConverter]::ToString($hash)).Replace("-", "").ToLowerInvariant()
    } finally {
        $stream.Dispose()
        $algorithm.Dispose()
    }
}

function Ensure-EmptyLogFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
        $stream.Dispose()
    }
}

function Write-FallbackEvidence {
    param(
        [Parameter(Mandatory = $true)][string]$Kind,
        [Parameter(Mandatory = $true)][string]$Message
    )
    Assert-NoReparsePoint -Candidate $fallbackEvidenceRoot -Boundary $projectRoot
    New-Item -ItemType Directory -Path $fallbackEvidenceRoot -Force | Out-Null
    Assert-NoReparsePoint -Candidate $fallbackEvidenceRoot -Boundary $projectRoot
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
    $path = Join-Path $fallbackEvidenceRoot ("launcher_rejection_{0}_{1}.json" -f $stamp, $runToken)
    $record = [ordered]@{
        schema_version = 1
        recorded_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        status = "rejected"
        failure_kind = $Kind
        failure_message = $Message
        requested_output_directory = $OutputDirectory
        parameter_set = $launcherParameterSetName
        claim_eligible = $false
        project_root = $projectRoot
    }
    Write-JsonCreateNew -Path $path -Value $record
    return $path
}

function New-ResourceSample {
    param([System.Diagnostics.Process]$Process)
    $sampledAt = (Get-Date).ToUniversalTime().ToString("o")
    $processAlive = $false
    $workingSetBytes = $null
    $peakWorkingSetBytes = $null
    $peakPrivateCommitBytes = $null
    $processQueryError = $null
    if ($null -ne $Process) {
        try {
            $Process.Refresh()
            $processAlive = -not $Process.HasExited
            if ($processAlive) { $workingSetBytes = [int64]$Process.WorkingSet64 }
            if ($processAlive) { $peakWorkingSetBytes = [int64]$Process.PeakWorkingSet64; $peakPrivateCommitBytes = [int64]$Process.PeakPagedMemorySize64 }
        } catch {
            $processQueryError = $_.Exception.Message
        }
    }

    $freeRamGiB = $null
    $freeCommitGiB = $null
    $ramQueryError = $null
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $freeRamGiB = [math]::Round([double]$os.FreePhysicalMemory * 1KB / 1GB, 3)
        $freeCommitGiB = [math]::Round([double]$os.FreeVirtualMemory * 1KB / 1GB, 3)
    } catch {
        $ramQueryError = $_.Exception.Message
    }

    $gpuRows = @()
    $gpuQueryError = $null
    try {
        $gpuOutput = @(& nvidia-smi.exe --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>&1)
        if ($LASTEXITCODE -ne 0) { throw ($gpuOutput -join "`n") }
        foreach ($line in $gpuOutput) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
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

    return [ordered]@{
        schema_version = 1
        sampled_at_utc = $sampledAt
        child = [ordered]@{
            pid = $childPid
            alive = $processAlive
            working_set_bytes = $workingSetBytes
            peak_working_set_bytes = $peakWorkingSetBytes
            peak_private_commit_bytes = $peakPrivateCommitBytes
            query_error = $processQueryError
        }
        host = [ordered]@{
            free_ram_gib = $freeRamGiB
            free_commit_gib = $freeCommitGiB
            query_error = $ramQueryError
        }
        gpus = $gpuRows
        gpu_query_error = $gpuQueryError
        compute_processes = $computeProcesses
        compute_query_error = $computeQueryError
    }
}

function Add-ResourceSample {
    param([System.Diagnostics.Process]$Process)
    $sample = New-ResourceSample -Process $Process
    $line = $sample | ConvertTo-Json -Depth 8 -Compress
    $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::AppendAllText($samplesPath, $line + [Environment]::NewLine, $utf8WithoutBom)
    $script:sampleCount += 1
}

function Resolve-MonitoredChildProcess {
    param([Parameter(Mandatory = $true)][System.Diagnostics.Process]$LauncherProcess)
    $deadline = (Get-Date).ToUniversalTime().AddSeconds(5)
    while ((Get-Date).ToUniversalTime() -lt $deadline) {
        $descendants = @()
        try {
            $descendants = @(Get-CimInstance Win32_Process -Filter ("ParentProcessId = {0}" -f $LauncherProcess.Id))
        } catch {
            $descendants = @()
        }
        foreach ($candidate in $descendants) {
            if ($null -eq $candidate.ProcessId) { continue }
            $candidatePath = $null
            try { $candidatePath = [System.IO.Path]::GetFullPath([string]$candidate.ExecutablePath) } catch { continue }
            if ([string]::IsNullOrWhiteSpace($candidatePath)) { continue }
            if (-not (Test-IsInsidePath -Candidate $candidatePath -Root $projectRoot)) { continue }
            if (-not ([System.IO.Path]::GetFileName($candidatePath)).Equals("python.exe", [System.StringComparison]::OrdinalIgnoreCase)) { continue }
            try { Assert-ReparseTargetsInsideBoundary -Candidate $candidatePath -Boundary $projectRoot } catch { continue }
            $candidateProcess = Get-Process -Id ([int]$candidate.ProcessId) -ErrorAction SilentlyContinue
            if ($null -ne $candidateProcess) {
                return [ordered]@{ process = $candidateProcess; executable = $candidatePath; source = "project_python_descendant" }
            }
        }
        $LauncherProcess.Refresh()
        if ($LauncherProcess.HasExited) { break }
        Start-Sleep -Milliseconds 50
    }
    $LauncherProcess.Refresh()
    if (-not $LauncherProcess.HasExited) {
        return [ordered]@{ process = $LauncherProcess; executable = $projectPython; source = "start_process_handle" }
    }
    throw "The project Python launcher exited before a monitorable child process could be identified."
}

function Release-OwnedLock {
    if (-not $script:lockCreated) {
        $script:lockReleaseStatus = "not_created"
        return
    }
    if (-not (Test-Path -LiteralPath $lockPath)) {
        $script:lockReleaseStatus = "already_missing"
        return
    }
    try {
        $lockData = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
        if ($null -eq $lockData.token -or [string]$lockData.token -cne $runToken) {
            $script:lockReleaseStatus = "token_mismatch_preserved"
            return
        }
        [System.IO.File]::Delete($lockPath)
        $script:lockReleased = $true
        $script:lockReleaseStatus = "released"
    } catch {
        $script:lockReleaseStatus = "release_failed: " + $_.Exception.Message
    }
}

function Write-RunSummary {
    if ($null -eq $outputRoot -or -not (Test-Path -LiteralPath $outputRoot)) { return }
    Ensure-EmptyLogFile -Path $stdoutPath
    Ensure-EmptyLogFile -Path $stderrPath
    Ensure-EmptyLogFile -Path $samplesPath
    $finishedAt = (Get-Date).ToUniversalTime()
    $childStillRunning = $false
    if ($null -ne $child) {
        try { $child.Refresh(); $childStillRunning = -not $child.HasExited } catch { $childStillRunning = $false }
    } elseif ($null -ne $processLauncher) {
        try { $processLauncher.Refresh(); $childStillRunning = -not $processLauncher.HasExited } catch { $childStillRunning = $false }
    }
    $commandSha256 = if ($null -ne $commandPath -and (Test-Path -LiteralPath $commandPath -PathType Leaf)) { Get-FileSha256 -Path $commandPath } else { $null }
    $samplesSha256 = if ($null -ne $samplesPath -and (Test-Path -LiteralPath $samplesPath -PathType Leaf)) { Get-FileSha256 -Path $samplesPath } else { $null }
    $stdoutSha256 = if (-not $childStillRunning -and $null -ne $stdoutPath -and (Test-Path -LiteralPath $stdoutPath -PathType Leaf)) { Get-FileSha256 -Path $stdoutPath } else { $null }
    $stderrSha256 = if (-not $childStillRunning -and $null -ne $stderrPath -and (Test-Path -LiteralPath $stderrPath -PathType Leaf)) { Get-FileSha256 -Path $stderrPath } else { $null }
    $preflightReportSha256 = if ($null -ne $preflightReportPath -and (Test-Path -LiteralPath $preflightReportPath -PathType Leaf)) { Get-FileSha256 -Path $preflightReportPath } else { $null }
    $summary = [ordered]@{
        schema_version = 1
        launcher = "scripts/Invoke-MonitoredExperiment.ps1"
        launcher_sha256 = $launcherSha256
        started_at_utc = $startedAtUtc.ToString("o")
        finished_at_utc = $finishedAt.ToString("o")
        duration_seconds = [math]::Round(($finishedAt - $startedAtUtc).TotalSeconds, 3)
        status = $status
        failure_kind = $failureKind
        failure_message = $failureMessage
        mode = if ($SelfTest) { "self_test" } elseif ($CapacityDiagnostic) { "capacity_diagnostic" } else { "experiment" }
        run_purpose = if ($SelfTest) { "synthetic_self_test" } elseif ($CapacityDiagnostic) { "synthetic_capacity_diagnostic" } else { $RunPurpose }
        claim_eligible = if ($SelfTest -or $CapacityDiagnostic) { $false } else { $null }
        claim_eligibility_reason = if ($CapacityDiagnostic) { "Synthetic, zero-learning-rate allocation diagnostic; no dataset or checkpoint and no performance claim." } elseif ($SelfTest) { "Synthetic launcher self-test; no dataset, model training, or GPU work is permitted." } else { "Not adjudicated by the resource launcher; use the formal provenance and aggregation gates." }
        project_root = $projectRoot
        output_directory = $outputRoot
        command_record = $commandPath
        preflight = [ordered]@{
            required = -not $SelfTest
            fixed_script = $preflightScript
            report = $preflightReportPath
            ready = $preflightReady
            exit_code = $preflightExitCode
        }
        lock = [ordered]@{
            path = $lockPath
            token = $runToken
            created = $lockCreated
            released = $lockReleased
            release_status = $lockReleaseStatus
        }
        child = [ordered]@{
            pid = $childPid
            executable = $childExecutable
            process_launcher_pid = $processLauncherPid
            exit_code = $childExitCode
            still_running_at_launcher_exit = $childStillRunning
        }
        monitoring = [ordered]@{
            sample_interval_seconds = $SampleIntervalSeconds
            sample_count = $sampleCount
            samples_jsonl = $samplesPath
        }
        stdout_log = $stdoutPath
        stderr_log = $stderrPath
        evidence_sha256 = [ordered]@{
            command = $commandSha256
            resource_samples_jsonl = $samplesSha256
            stdout_log = $stdoutSha256
            stderr_log = $stderrSha256
            preflight_report = $preflightReportSha256
            mutable_child_logs_omitted = $childStillRunning
        }
        launcher_exit_code = $exitCode
        interrupted = $interrupted
    }
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        Write-JsonCreateNew -Path $summaryPath -Value $summary -Depth 10
    }
}

try {
    if (-not (Test-Path -LiteralPath $projectPython -PathType Leaf)) {
        throw "Project Python is missing: $projectPython"
    }
    if (-not (Test-Path -LiteralPath $preflightScript -PathType Leaf)) {
        throw "Fixed preflight script is missing: $preflightScript"
    }
    Assert-NoReparsePoint -Candidate $launcherPath -Boundary $projectRoot
    Assert-NoReparsePoint -Candidate $projectPython -Boundary $projectRoot
    Assert-NoReparsePoint -Candidate $preflightScript -Boundary $projectRoot
    $launcherSha256 = Get-FileSha256 -Path $launcherPath
    $preflightSha256 = Get-FileSha256 -Path $preflightScript

    $outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
        [System.IO.Path]::GetFullPath($OutputDirectory)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
    }
    if (-not $outputRoot.StartsWith($monitoredRunsPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "OutputDirectory must be a child of $monitoredRunsRoot; received $outputRoot"
    }
    Assert-NoReparsePoint -Candidate $outputRoot -Boundary $projectRoot
    if (Test-Path -LiteralPath $outputRoot) {
        $outputItem = Get-Item -LiteralPath $outputRoot -Force
        if (-not $outputItem.PSIsContainer) { throw "OutputDirectory is not a directory: $outputRoot" }
        if ($null -ne (Get-ChildItem -LiteralPath $outputRoot -Force | Select-Object -First 1)) {
            throw "Refusing to overwrite a non-empty output directory: $outputRoot"
        }
    } else {
        New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
    }
    Assert-NoReparsePoint -Candidate $outputRoot -Boundary $projectRoot

    $summaryPath = Join-Path $outputRoot "summary.json"
    $samplesPath = Join-Path $outputRoot "resource_samples.jsonl"
    $stdoutPath = Join-Path $outputRoot "stdout.log"
    $stderrPath = Join-Path $outputRoot "stderr.log"
    $commandPath = Join-Path $outputRoot "command.json"

    if ($CapacityDiagnostic) {
        $resolvedScript = Join-Path $projectRoot "scripts/profile_draem_resources.py"
        Assert-NoReparsePoint -Candidate $resolvedScript -Boundary $projectRoot
        $effectiveArguments = @("--output-dir", (Join-Path $outputRoot "capacity"))
        if ($DiagnosticMemoryStrategy -eq "activation_checkpointing") { $effectiveArguments += "--activation-checkpointing" }
        $effectivePurpose = "synthetic_capacity_diagnostic"
    } elseif ($SelfTest) {
        if (-not (Test-Path -LiteralPath $selfTestScript -PathType Leaf)) {
            throw "Synthetic self-test child is missing: $selfTestScript"
        }
        $resolvedScript = $selfTestScript
        $effectiveArguments = @(
            "--duration-seconds", "2",
            "--claim-eligible", "false",
            "--sentinel", 'space and "quote" ending\'
        )
        $effectivePurpose = "synthetic_self_test"
    } else {
        $resolvedScript = if ([System.IO.Path]::IsPathRooted($ScriptPath)) {
            [System.IO.Path]::GetFullPath($ScriptPath)
        } else {
            [System.IO.Path]::GetFullPath((Join-Path $projectRoot $ScriptPath))
        }
        if (-not (Test-IsInsidePath -Candidate $resolvedScript -Root $projectRoot)) {
            throw "Experiment script must remain inside the project: $resolvedScript"
        }
        Assert-NoReparsePoint -Candidate $resolvedScript -Boundary $projectRoot
        if (-not (Test-Path -LiteralPath $resolvedScript -PathType Leaf)) {
            throw "Experiment script does not exist: $resolvedScript"
        }
        if ([System.IO.Path]::GetExtension($resolvedScript) -cne ".py") {
            throw "Only project-local Python scripts are accepted: $resolvedScript"
        }
        $effectiveArguments = @($ScriptArgument)
        $effectivePurpose = $RunPurpose
        if (-not [string]::IsNullOrWhiteSpace($ResourceProfile)) {
            if ($resolvedScript -ne (Join-Path $projectRoot "scripts/train_draem.py")) {
                throw "A DRAEM resource profile can only authorize the matching DRAEM command."
            }
            $profilePath = if ([System.IO.Path]::IsPathRooted($ResourceProfile)) { [System.IO.Path]::GetFullPath($ResourceProfile) } else { [System.IO.Path]::GetFullPath((Join-Path $projectRoot $ResourceProfile)) }
            if (-not (Test-IsInsidePath -Candidate $profilePath -Root $projectRoot)) { throw "Resource profile must remain inside the project." }
            Assert-NoReparsePoint -Candidate $profilePath -Boundary $projectRoot
            $profile = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json
            if ($RunPurpose -ne "debug_limited" -and $profile.kind -ne "draem_real_s0_capacity_profile") {
                throw "A synthetic resource profile cannot authorize full experiments; confirm real train-only S0 first."
            }
            $expected = @{ "--image-size" = "256"; "--batch-size" = "8"; "--workers" = "0"; "--device" = "cuda:0"; "--run-purpose" = $RunPurpose }
            if ($RunPurpose -eq "debug_limited") { $expected["--max-steps"] = "2" }
            foreach ($flag in $expected.Keys) {
                $indices = @(for ($i = 0; $i -lt $effectiveArguments.Count; $i++) { if ($effectiveArguments[$i] -ceq $flag) { $i } })
                if ($indices.Count -ne 1 -or $indices[0] + 1 -ge $effectiveArguments.Count -or $effectiveArguments[$indices[0] + 1] -cne $expected[$flag]) {
                    throw "Measured profile requires one explicit matching argument: $flag"
                }
            }
            if (@($effectiveArguments | Where-Object { $_ -ceq "--no-pin-memory" }).Count -ne 1 -or @($effectiveArguments | Where-Object { $_ -like "--*=*" -or $_ -ceq "--pin-memory" }).Count -gt 0) {
                throw "Measured S0 requires --no-pin-memory and separate argument tokens."
            }
            if ($RunPurpose -ne "debug_limited" -and $effectiveArguments -ccontains "--max-steps") { throw "Full experiment cannot use truncated steps." }
            $checkpointFlags = @($effectiveArguments | Where-Object { $_ -ceq "--activation-checkpointing" }).Count
            if ($checkpointFlags -gt 1 -or [bool]$profile.recipe.activation_checkpointing -ne ($checkpointFlags -eq 1)) {
                throw "Command activation checkpointing does not match the measured profile."
            }
        }
    }

    $commandRecord = [ordered]@{
        schema_version = 1
        recorded_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        mode = if ($SelfTest) { "self_test" } elseif ($CapacityDiagnostic) { "capacity_diagnostic" } else { "experiment" }
        run_purpose = $effectivePurpose
        claim_eligible = if ($SelfTest -or $CapacityDiagnostic) { $false } else { $null }
        launcher = $launcherPath
        launcher_sha256 = $launcherSha256
        executable = $projectPython
        executable_sha256 = Get-FileSha256 -Path $projectPython
        script = $resolvedScript
        script_sha256 = Get-FileSha256 -Path $resolvedScript
        arguments = $effectiveArguments
        working_directory = $projectRoot
        preflight_required = -not $SelfTest
        preflight_script = $preflightScript
        preflight_sha256 = $preflightSha256
    }
    Write-JsonCreateNew -Path $commandPath -Value $commandRecord

    if (-not $SelfTest) {
        $status = "preflight"
        $preflightEvidenceDirectory = Join-Path $outputRoot "preflight"
        New-Item -ItemType Directory -Path $preflightEvidenceDirectory -Force | Out-Null
        $preflightStdout = Join-Path $outputRoot "preflight.stdout.log"
        $preflightStderr = Join-Path $outputRoot "preflight.stderr.log"
        $hostExecutable = (Get-Process -Id $PID).Path
        $preflightArguments = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", $preflightScript,
            "-OutputDirectory", $preflightEvidenceDirectory
        )
        if ($CapacityDiagnostic) { $preflightArguments += "-CapacityDiagnostic" }
        if (-not [string]::IsNullOrWhiteSpace($ResourceProfile)) { $preflightArguments += @("-ResourceProfile", $ResourceProfile) }
        $preflightArgumentLine = Join-WindowsCommandLine -Arguments $preflightArguments
        $preflightProcess = Start-Process -FilePath $hostExecutable -ArgumentList $preflightArgumentLine -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $preflightStdout -RedirectStandardError $preflightStderr -PassThru -Wait
        $preflightExitCode = [int]$preflightProcess.ExitCode
        $preflightReports = @(Get-ChildItem -LiteralPath $preflightEvidenceDirectory -Filter "preflight_*.json" -File)
        if ($preflightReports.Count -ne 1) {
            throw "Fixed preflight produced $($preflightReports.Count) JSON reports; expected exactly one."
        }
        $preflightReportPath = $preflightReports[0].FullName
        $preflightReport = Get-Content -LiteralPath $preflightReportPath -Raw | ConvertFrom-Json
        $preflightReady = if ($CapacityDiagnostic) { [bool]$preflightReport.ready_for_capacity_diagnostic } else { [bool]$preflightReport.ready_for_training_launch }
        if ($preflightExitCode -ne 0 -or -not $preflightReady) {
            $status = "preflight_rejected"
            $failureKind = "preflight_not_ready"
            $failureMessage = "Fixed training preflight rejected the launch."
            $exitCode = 2
        } else {
            New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($lockPath)) -Force | Out-Null
            $lockRecord = [ordered]@{
                schema_version = 1
                token = $runToken
                pid = $PID
                owner_kind = "monitored_experiment_launcher"
                created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
                run_purpose = $effectivePurpose
                output_directory = $outputRoot
            }
            Write-JsonCreateNew -Path $lockPath -Value $lockRecord
            $lockCreated = $true
            $lockReleaseStatus = "owned"
        }
    }

    if ($SelfTest -or $preflightReady) {
        $status = "launching"
        $argumentLine = Join-WindowsCommandLine -Arguments (@($resolvedScript) + @($effectiveArguments))
        $previousCudaVisibleDevices = $null
        $hadCudaVisibleDevices = Test-Path Env:CUDA_VISIBLE_DEVICES
        if ($hadCudaVisibleDevices) { $previousCudaVisibleDevices = $env:CUDA_VISIBLE_DEVICES }
        if ($SelfTest) { $env:CUDA_VISIBLE_DEVICES = "-1" }
        try {
            $processLauncher = Start-Process -FilePath $projectPython -ArgumentList $argumentLine -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
        } finally {
            if ($SelfTest) {
                if ($hadCudaVisibleDevices) { $env:CUDA_VISIBLE_DEVICES = $previousCudaVisibleDevices } else { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
            }
        }
        $processLauncherPid = [int]$processLauncher.Id
        $resolvedChild = Resolve-MonitoredChildProcess -LauncherProcess $processLauncher
        $child = $resolvedChild.process
        $childPid = [int]$child.Id
        $childExecutable = [string]$resolvedChild.executable
        $status = "running"
        Add-ResourceSample -Process $child
        $childStartedAt = (Get-Date).ToUniversalTime()
        while (-not $processLauncher.HasExited) {
            Start-Sleep -Seconds $SampleIntervalSeconds
            Add-ResourceSample -Process $child
            $processLauncher.Refresh()
            if ($CapacityDiagnostic -and -not $processLauncher.HasExited -and ((Get-Date).ToUniversalTime() - $childStartedAt).TotalSeconds -gt 180) {
                # Stop only the exact process handle owned by this launcher.
                $child.Refresh()
                if (-not $child.HasExited) { $child.Kill(); $child.WaitForExit(10000) | Out-Null }
                throw "Bounded capacity diagnostic exceeded 180 seconds."
            }
        }
        $processLauncher.WaitForExit()
        $childExitCode = [int]$processLauncher.ExitCode
        Add-ResourceSample -Process $child
        if ($childExitCode -eq 0) {
            $status = "completed"
            $exitCode = 0
        } else {
            $status = "child_failed"
            $failureKind = "child_nonzero_exit"
            $failureMessage = "Child process exited with code $childExitCode."
            $exitCode = $childExitCode
            if ($exitCode -eq 0) { $exitCode = 1 }
        }
    }
} catch [System.Management.Automation.PipelineStoppedException] {
    $interrupted = $true
    $status = "interrupted"
    $failureKind = "launcher_interrupted"
    $failureMessage = $_.Exception.Message
    $exitCode = 130
} catch {
    if ($status -eq "initializing") { $status = "rejected" } else { $status = "launcher_failed" }
    $failureKind = $_.Exception.GetType().FullName
    $failureMessage = $_.Exception.Message
    $exitCode = 70
} finally {
    Release-OwnedLock
    try {
        Write-RunSummary
    } catch {
        if ($null -eq $failureMessage) { $failureMessage = $_.Exception.Message }
    }
    if ($null -eq $summaryPath -or -not (Test-Path -LiteralPath $summaryPath)) {
        try { $fallbackEvidencePath = Write-FallbackEvidence -Kind ($(if ($null -ne $failureKind) { $failureKind } else { "summary_unavailable" })) -Message ($(if ($null -ne $failureMessage) { $failureMessage } else { "Launcher could not write its requested summary." })) } catch { }
    }
}

if ($null -ne $summaryPath -and (Test-Path -LiteralPath $summaryPath)) {
    Write-Host "Monitored experiment summary: $summaryPath"
} elseif ($null -ne $fallbackEvidencePath) {
    Write-Host "Launcher rejection evidence: $fallbackEvidencePath"
}
exit $exitCode
