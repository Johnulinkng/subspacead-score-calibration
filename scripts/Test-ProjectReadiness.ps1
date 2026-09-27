# Read-only preparation check. This does not establish training readiness.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$config = Get-Content -LiteralPath (Join-Path $projectRoot 'configs/project.json') -Raw | ConvertFrom-Json
$sourcePath = Join-Path $projectRoot $config.baseline.source_directory
$actualCommit = (& git -C $sourcePath rev-parse HEAD 2>$null)
if ($LASTEXITCODE -ne 0) { throw 'Baseline repository is unavailable.' }
$pythonPath = Join-Path $projectRoot $config.paths.python
$datasetPath = Join-Path $projectRoot $config.paths.mvtec
$visaPath = Join-Path $projectRoot $config.paths.visa
$dtdPath = Join-Path $projectRoot $config.paths.dtd
$missingCategories = @($config.datasets.mvtec_ad.categories | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $datasetPath ($_ + '/train/good')))
})
$missingVisaCategories = @($config.datasets.visa.categories | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $visaPath ($_ + '/train/good')))
})
$projectDrive = (Get-Item -LiteralPath $projectRoot).PSDrive
$freeGiB = [math]::Round($projectDrive.Free / 1GB, 2)
[PSCustomObject]@{
    project_root = $projectRoot
    baseline_commit = $actualCommit
    baseline_commit_matches = ($actualCommit -eq $config.baseline.commit)
    free_gib = $freeGiB
    preparation_space_available = ($freeGiB -ge $config.minimum_preparation_free_gib)
    project_python_file_present = (Test-Path -LiteralPath $pythonPath)
    missing_mvtec_train_category_directories = $missingCategories
    missing_visa_train_category_directories = $missingVisaCategories
    dtd_directory_present = (Test-Path -LiteralPath $dtdPath)
    mvtec_manifest_verified = $config.datasets.mvtec_ad.manifest_verified
    visa_manifest_verified = $config.datasets.visa.manifest_verified
    formal_protocol_frozen = $config.evaluation.formal_protocol_frozen
    experiment_started = $false
    gpu_launch_readiness = 'not_assessed_by_this_check'
    checked_at_utc = (Get-Date).ToUniversalTime().ToString('o')
} | ConvertTo-Json -Depth 5
