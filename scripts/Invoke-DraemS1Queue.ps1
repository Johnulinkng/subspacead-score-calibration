[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$QueueDirectory,
    [Parameter(Mandatory = $true)][string]$ResourceProfile
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot
. (Join-Path $PSScriptRoot 'Set-ProjectEnvironment.ps1') | Out-Null
$queueRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $QueueDirectory))
$queuePrefix = (Join-Path $projectRoot 'runs\queues').TrimEnd('\') + '\'
if (-not $queueRoot.StartsWith($queuePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Queue records must remain below the project runs/queues directory.'
}
if (Test-Path -LiteralPath $queueRoot) { throw 'Refusing to replace an existing queue.' }
New-Item -ItemType Directory -Path $queueRoot | Out-Null
$profilePath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $ResourceProfile))
$profile = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json
if ($profile.kind -ne 'draem_real_s0_capacity_profile' -or $profile.status -ne 'passed') {
    throw 'Full development needs a completed real-data S0 capacity profile.'
}
$registrationPath = Join-Path $projectRoot 'configs\preregistered_experiments.json'
$developmentPath = Join-Path $projectRoot 'configs\development_protocol.json'
$registration = Get-Content -LiteralPath $registrationPath -Raw | ConvertFrom-Json
$development = Get-Content -LiteralPath $developmentPath -Raw | ConvertFrom-Json
$stage = @($registration.matrix | Where-Object { $_.stage -eq 'S1_baseline_development_reference' })
if ($stage.Count -ne 1 -or $development.visa.categories.Count -ne 12 -or $stage[0].seeds.Count -ne 3) {
    throw 'The complete registered S1 matrix is required.'
}
$queueId = Split-Path -Leaf $queueRoot
$tasks = @(foreach ($seed in $stage[0].seeds) {
    foreach ($category in $development.visa.categories) {
        [ordered]@{
            category = [string]$category; seed = [long]$seed; status = 'pending'
            output = "runs/monitored/$queueId-$category-seed$seed"
        }
    }
})
$frozenFiles = @($registrationPath, $developmentPath, $profilePath)
$hashes = @{}
foreach ($path in $frozenFiles) { $hashes[$path] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
$state = [ordered]@{
    schema_version = 1; queue_id = $queueId; owner_pid = $PID
    started_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    authorization = 'User explicitly requested start and advance today on 2026-10-09; earlier serial sequencing is superseded.'
    status = 'running'; completed_tasks = 0; current_task_index = $null
    stage = 'S1_baseline_development_reference'; dataset = 'visa'; method = 'DRAEM'
    epochs = 700; claim_eligible = $false; resource_profile = $profilePath
    frozen_file_sha256 = $hashes; tasks = $tasks; failure = $null
}
function Save-QueueState {
    $state.updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    $temporary = Join-Path $queueRoot '.state.json.tmp'
    [System.IO.File]::WriteAllText($temporary, ($state | ConvertTo-Json -Depth 12) + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination (Join-Path $queueRoot 'state.json') -Force
}
try {
    Save-QueueState
    for ($index = 0; $index -lt $tasks.Count; $index++) {
        foreach ($path in $frozenFiles) {
            if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -cne $hashes[$path]) {
                throw "Queue configuration changed: $path"
            }
        }
        $task = $tasks[$index]
        $state.current_task_index = $index
        $task.status = 'running'
        Save-QueueState
        $arguments = @(
            '--dataset', 'visa', '--category', $task.category, '--seed', [string]$task.seed,
            '--run-purpose', 'development_full', '--input-protocol', 'visa_train_dev_dtd_v1',
            '--development-protocol-config', 'configs/development_protocol.json',
            '--image-size', '256', '--batch-size', '8', '--epochs', '700',
            '--learning-rate', '0.0001', '--workers', '0', '--no-pin-memory',
            '--device', 'cuda:0', '--activation-checkpointing',
            '--output-dir', (Join-Path $task.output 'model')
        )
        & (Join-Path $PSScriptRoot 'Invoke-MonitoredExperiment.ps1') `
            -OutputDirectory $task.output -ScriptPath 'scripts/train_draem.py' `
            -RunPurpose 'development_full' -ResourceProfile $ResourceProfile `
            -ScriptArgument $arguments -SampleIntervalSeconds 30
        $runExit = $LASTEXITCODE
        $summary = Get-Content -LiteralPath (Join-Path $task.output 'summary.json') -Raw | ConvertFrom-Json
        if ($runExit -ne 0 -or $summary.status -ne 'completed' -or $summary.child.exit_code -ne 0) {
            throw "S1 task did not complete: $($task.category), seed=$($task.seed), exit=$runExit"
        }
        $result = Get-Content -LiteralPath (Join-Path $task.output 'model\run_state.json') -Raw | ConvertFrom-Json
        if ($result.status -ne 'completed' -or $result.epoch -ne 700 -or $result.run_purpose -ne 'development_full') {
            throw 'The complete 700-epoch development schedule was not completed.'
        }
        $task.status = 'completed'
        $state.completed_tasks += 1
        Save-QueueState
    }
    $state.status = 'training_matrix_completed_evaluation_pending'
    $state.current_task_index = $null
    Save-QueueState
} catch {
    $state.status = 'failed'
    $state.failure = $_.Exception.Message
    if ($null -ne $state.current_task_index) { $tasks[$state.current_task_index].status = 'failed' }
    Save-QueueState
    throw
}
