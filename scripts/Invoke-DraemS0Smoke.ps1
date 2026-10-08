[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [string]$ResourceProfile
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
. (Join-Path $PSScriptRoot "Set-ProjectEnvironment.ps1") | Out-Null

if ([string]::IsNullOrWhiteSpace($ResourceProfile)) {
    $resourceConfig = Get-Content -LiteralPath (Join-Path $projectRoot "configs/resource_budget.json") -Raw | ConvertFrom-Json
    $ResourceProfile = $resourceConfig.draem_s0.profile
}
$profilePath = if ([System.IO.Path]::IsPathRooted($ResourceProfile)) { $ResourceProfile } else { Join-Path $projectRoot $ResourceProfile }
$profile = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $OutputDirectory = "runs/monitored/draem-s0-visa-candle-seed1887537248-$stamp"
}

$modelOutput = Join-Path $OutputDirectory "model"
$arguments = @(
    "--dataset", "visa",
    "--category", "candle",
    "--seed", "1887537248",
    "--run-purpose", "debug_limited",
    "--input-protocol", "visa_train_dev_dtd_v1",
    "--development-protocol-config", "configs/development_protocol.json",
    "--image-size", "256",
    "--batch-size", "8",
    "--workers", "0",
    "--no-pin-memory",
    "--device", "cuda:0",
    "--max-steps", "2",
    "--output-dir", $modelOutput
)
if ($profile.recipe.activation_checkpointing) { $arguments += "--activation-checkpointing" }

& (Join-Path $PSScriptRoot "Invoke-MonitoredExperiment.ps1") `
    -OutputDirectory $OutputDirectory `
    -ScriptPath "scripts/train_draem.py" `
    -RunPurpose "debug_limited" `
    -ResourceProfile $ResourceProfile `
    -ScriptArgument $arguments `
    -SampleIntervalSeconds 5

exit $LASTEXITCODE
