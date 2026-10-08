[CmdletBinding()]
param(
    [ValidateRange(1, 1024)][double]$MinimumFreeRamGiB = 8,
    [double]$MinimumFreeDiskGiB = 40,
    [switch]$CapacityDiagnostic,
    [string]$ResourceProfile,
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
$freeCommitGiB = [double]$os.FreeVirtualMemory * 1KB / 1GB
$minimumFreeCommitGiB = $MinimumFreeRamGiB
$minimumFreeGpuMiB = 0
$profileEvidence = $null
if ($CapacityDiagnostic -and -not [string]::IsNullOrWhiteSpace($ResourceProfile)) {
    throw "A diagnostic uses the bootstrap budget; it cannot also use a measured profile."
}
if ($CapacityDiagnostic) {
    # Bounded, synthetic, zero-learning-rate allocation probe only. This is not
    # permission for an unprofiled training run to use a smaller reserve.
    $MinimumFreeRamGiB = 4
    $minimumFreeCommitGiB = 4
    $minimumFreeGpuMiB = 5888
} elseif (-not [string]::IsNullOrWhiteSpace($ResourceProfile)) {
    $profileFull = if ([System.IO.Path]::IsPathRooted($ResourceProfile)) { [System.IO.Path]::GetFullPath($ResourceProfile) } else { [System.IO.Path]::GetFullPath((Join-Path $projectRoot $ResourceProfile)) }
    if (-not $profileFull.StartsWith($projectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Resource profile must remain inside the project."
    }
    $profile = Get-Content -LiteralPath $profileFull -Raw | ConvertFrom-Json
    if ($profile.kind -notin @("draem_synthetic_capacity_diagnostic", "draem_real_s0_capacity_profile") -or $profile.status -ne "passed" -or $profile.completed_steps -ne 2) {
        throw "Only a successful, complete DRAEM capacity diagnostic can set the budget."
    }
    if ($null -ne $profile.error -or @($profile.sample_errors).Count -ne 0 -or $profile.claim_eligible -ne $false) {
        throw "Profile contains diagnostic errors or an invalid execution scope."
    }
    if ($profile.kind -eq "draem_synthetic_capacity_diagnostic" -and ($profile.dataset_accessed -ne $false -or $profile.checkpoint_saved -ne $false)) { throw "Invalid synthetic diagnostic scope." }
    if ($profile.kind -eq "draem_real_s0_capacity_profile") {
        $s0Path = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $profile.real_s0.path))
        $s0Prefix = (Join-Path $projectRoot "runs/monitored").TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
        if (-not $s0Path.StartsWith($s0Prefix, [System.StringComparison]::OrdinalIgnoreCase) -or $profile.dataset_accessed -ne $true -or $profile.checkpoint_saved -ne $true) { throw "Invalid real S0 resource scope." }
        $s0EvidenceKeys = @($profile.real_s0.evidence_sha256.PSObject.Properties | ForEach-Object { $_.Name })
        foreach ($required in @('summary.json','command.json','resource_samples.jsonl','stdout.log','stderr.log','model/run_config.json','model/run_state.json','model/checkpoints/manifest.json')) {
            if ($s0EvidenceKeys -cnotcontains $required) { throw "Real S0 resource evidence is incomplete." }
        }
        foreach ($property in $profile.real_s0.evidence_sha256.PSObject.Properties) {
            $path = [System.IO.Path]::GetFullPath((Join-Path $s0Path $property.Name))
            if (-not $path.StartsWith($s0Path + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $property.Value) { throw "Real S0 resource evidence drift." }
        }
        $s0State = Get-Content -LiteralPath (Join-Path $s0Path 'model/run_state.json') -Raw | ConvertFrom-Json
        if ($s0State.status -ne 'debug_completed' -or $s0State.global_step -ne 2) { throw "Real S0 has not completed." }
        $s0Config = Get-Content -LiteralPath (Join-Path $s0Path 'model/run_config.json') -Raw | ConvertFrom-Json
        $s0Summary = Get-Content -LiteralPath (Join-Path $s0Path 'summary.json') -Raw | ConvertFrom-Json
        if ($s0Summary.mode -ne 'experiment' -or $s0Summary.status -ne 'completed' -or $s0Summary.child.exit_code -ne 0 -or $s0Config.settings.device -ne 'cuda:0' -or $s0Config.settings.input_protocol -ne 'visa_train_dev_dtd_v1' -or $s0Config.settings.activation_checkpointing -ne $profile.recipe.activation_checkpointing) { throw "Real S0 resource execution scope does not match." }
    }
    if ($profile.recipe.image_size -ne 256 -or $profile.recipe.batch_size -ne 8 -or $profile.recipe.workers -ne 0 -or $profile.recipe.pin_memory -ne $false -or $profile.recipe.precision -ne "float32" -or $profile.recipe.device -ne "cuda:0") {
        throw "Profile does not cover the registered DRAEM runtime shape."
    }
    $requiredSources = @("pyproject.toml", "uv.lock", "scripts/profile_draem_resources.py", "src/industrial_anomaly_lab/draem_runtime.py", "src/industrial_anomaly_lab/draem_train.py", "scripts/train_draem.py", "src/industrial_anomaly_lab/draem_checkpointing.py", "third_party/DRAEM/model_unet.py", "third_party/DRAEM/loss.py")
    $profileSources = @($profile.source_sha256.PSObject.Properties | ForEach-Object { $_.Name })
    foreach ($source in $requiredSources) {
        if ($profileSources -cnotcontains $source) { throw "Resource profile lacks source evidence: $source" }
    }
    foreach ($property in $profile.source_sha256.PSObject.Properties) {
        $sourcePath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $property.Name))
        if (-not $sourcePath.StartsWith($projectPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $property.Value) {
            throw "Resource profile source drift: $($property.Name)"
        }
    }
    $installedTorchVersion = & (Join-Path $projectRoot ".venv/Scripts/python.exe") -c "from importlib.metadata import version; print(version('torch'))"
    if ($LASTEXITCODE -ne 0 -or $installedTorchVersion.Trim() -ne $profile.torch_version) { throw "Resource profile PyTorch runtime drift." }
    foreach ($number in @($profile.host_peak_working_set_gib, $profile.host_peak_private_commit_gib, $profile.gpu_peak_reserved_gib)) {
        if ([double]$number -le 0 -or [double]::IsNaN([double]$number) -or [double]::IsInfinity([double]$number)) { throw "Invalid profile peak." }
    }
    # Real loader/serialization allowance + 25% process margin + OS reserve.
    $MinimumFreeRamGiB = [math]::Round(2 + .5 + 1.25 * [double]$profile.host_peak_working_set_gib, 3)
    $minimumFreeCommitGiB = [math]::Round(2 + .5 + 1.25 * [double]$profile.host_peak_private_commit_gib, 3)
    $minimumFreeGpuMiB = [math]::Ceiling(1024 * ([double]$profile.gpu_peak_reserved_gib + .75))
    $profileEvidence = [ordered]@{ path = $profileFull; sha256 = (Get-FileHash -LiteralPath $profileFull -Algorithm SHA256).Hash.ToLowerInvariant(); gpu_name = $profile.gpu_name; gpu_total_gib = $profile.gpu_total_gib; source_sha256 = $profile.source_sha256; limitation = "Admission estimate only; S0 and long-run monitoring still required." }
}
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
        $uiProcess = $false
        if ($parts[0] -match '^\d+$' -and $parts.Count -gt 1) {
            $live = Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f $parts[0]) -ErrorAction SilentlyContinue
            $uiProcess = ($null -ne $live -and $live.ExecutablePath -eq $parts[1] -and $live.ExecutablePath -like "$env:ProgramFiles\WindowsApps\OpenAI.Codex_*\app\ChatGPT.exe")
        }
        $computeProcesses += [ordered]@{
            pid = if ($parts[0] -match '^\d+$') { [int]$parts[0] } else { $parts[0] }
            process_name = if ($parts.Count -gt 1) { $parts[1] } else { $null }
            used_memory_mib = if ($parts.Count -gt 2 -and $parts[2] -match '^\d+$') { [int]$parts[2] } else { if ($parts.Count -gt 2) { $parts[2] } else { $null } }
            verified_desktop_ui = $uiProcess
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
    free_commit = $freeCommitGiB -ge $minimumFreeCommitGiB
    free_disk = $freeDiskGiB -ge $MinimumFreeDiskGiB
    gpu_query = $null -eq $gpuQueryError -and $gpuRows.Count -gt 0
    no_conflicting_compute_processes = $null -eq $computeQueryError -and @($computeProcesses | Where-Object { -not $_.verified_desktop_ui }).Count -eq 0
    free_dedicated_vram = $gpuRows.Count -gt 0 -and $gpuRows[0].memory_free_mib -ge $minimumFreeGpuMiB
    profile_device_matches = $null -eq $profileEvidence -or ($gpuRows.Count -gt 0 -and $gpuRows[0].name -eq $profileEvidence.gpu_name -and [math]::Abs($gpuRows[0].memory_total_mib / 1024.0 - $profileEvidence.gpu_total_gib) -lt .01)
    no_project_lock = -not $lock.exists
}
$ready = -not ($checks.Values -contains $false)
$timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
$report = [ordered]@{
    schema_version = 2
    checked_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    project_root = $projectRoot
    ready_for_training_launch = $ready -and -not $CapacityDiagnostic
    ready_for_capacity_diagnostic = $ready -and $CapacityDiagnostic
    budget_basis = if ($CapacityDiagnostic) { "bounded_synthetic_bootstrap" } elseif ($null -ne $profileEvidence) { "measured_peak_plus_reserves" } else { "legacy_unprofiled_conservative_estimate_not_algorithm_minimum" }
    resource_profile = $profileEvidence
    thresholds = [ordered]@{ minimum_free_ram_gib = $MinimumFreeRamGiB; minimum_free_commit_gib = $minimumFreeCommitGiB; minimum_free_gpu_mib = $minimumFreeGpuMiB; minimum_free_disk_gib = $MinimumFreeDiskGiB }
    checks = $checks
    host = [ordered]@{ total_ram_gib = $totalRamGiB; free_ram_gib = $freeRamGiB; free_commit_gib = $freeCommitGiB; project_drive = $projectDriveName; free_disk_gib = $freeDiskGiB }
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
