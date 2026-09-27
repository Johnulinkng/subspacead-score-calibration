[CmdletBinding()]
param(
    [int]$MaxAttempts = 30,
    [int]$RetryDelaySeconds = 10
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
. (Join-Path $PSScriptRoot "Set-ProjectEnvironment.ps1")
$python = Join-Path $projectRoot ".venv/Scripts/python.exe"
$outputDirectory = Join-Path $projectRoot "artifacts/weights"
$output = Join-Path $outputDirectory "DRAEM_checkpoints.zip"
$fileId = "1eOE8wXNihjsiDvDANHFbg_mQkLesDrs1"
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $partial = @(Get-ChildItem -LiteralPath $outputDirectory -Filter "DRAEM_checkpoints.zip*.part" -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
    $currentBytes = if ($partial.Count -eq 1) { $partial[0].Length } elseif (Test-Path -LiteralPath $output) { (Get-Item -LiteralPath $output).Length } else { 0L }
    Write-Host "Attempt $attempt/$MaxAttempts; current bytes=$currentBytes"
    & $python -m gdown --continue "https://drive.google.com/uc?id=$fileId" -O $output
    if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $output)) {
        Write-Host "DRAEM weight archive download complete: $((Get-Item -LiteralPath $output).Length) bytes"
        exit 0
    }
    if ($attempt -lt $MaxAttempts) { Start-Sleep -Seconds $RetryDelaySeconds }
}

throw "DRAEM weight archive did not complete after $MaxAttempts attempts"
