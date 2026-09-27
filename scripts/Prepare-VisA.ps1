[CmdletBinding()]
param(
    [string]$RawRoot = "data/visa/raw/VisA",
    [string]$OutputRoot = "data/visa/1cls",
    [string]$ManifestPath = "manifests/visa_prepared.json"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Join-Path $projectRoot ".venv/Scripts/python.exe"
$vendorRoot = Join-Path $projectRoot "third_party/spot-diff"
$splitFile = Join-Path $vendorRoot "split_csv/1cls.csv"
$prepareScript = Join-Path $vendorRoot "utils/prepare_data.py"
$expectedCommit = "2a692ab575001cbde74d402d897a7286086c6199"
$expectedSplitSha256 = "a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995"
$expectedPrepareSha256 = "6e70f97b16b589dc3cf6eab55e104bf13bbd7949909dc2279ebad5cb9f4c1c40"

function Resolve-ProjectPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $PathValue))
}

function Assert-InProject([string]$PathValue) {
    $normalizedProject = [System.IO.Path]::GetFullPath($projectRoot).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    $normalizedPath = [System.IO.Path]::GetFullPath($PathValue)
    $projectPrefix = $normalizedProject + [System.IO.Path]::DirectorySeparatorChar
    $isProjectRoot = [string]::Equals(
        $normalizedPath,
        $normalizedProject,
        [System.StringComparison]::OrdinalIgnoreCase
    )
    $isProjectChild = $normalizedPath.StartsWith(
        $projectPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )
    if (-not $isProjectRoot -and -not $isProjectChild) {
        throw "Path must remain inside project root: $PathValue"
    }
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $vendorRoot -PathType Container)) {
    throw "Official spot-diff checkout is missing: $vendorRoot"
}

$actualCommit = (& git -C $vendorRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $expectedCommit) {
    throw "spot-diff commit mismatch: expected $expectedCommit, got $actualCommit"
}
$actualSplitSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $splitFile).Hash.ToLowerInvariant()
$actualPrepareSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $prepareScript).Hash.ToLowerInvariant()
if ($actualSplitSha256 -ne $expectedSplitSha256) {
    throw "VisA split hash mismatch: $actualSplitSha256"
}
if ($actualPrepareSha256 -ne $expectedPrepareSha256) {
    throw "VisA preparation script hash mismatch: $actualPrepareSha256"
}

$raw = Resolve-ProjectPath $RawRoot
$output = Resolve-ProjectPath $OutputRoot
$manifest = Resolve-ProjectPath $ManifestPath
Assert-InProject $raw
Assert-InProject $output
Assert-InProject $manifest
if (-not (Test-Path -LiteralPath $raw -PathType Container)) {
    throw "Extracted VisA root is missing: $raw"
}
if (Test-Path -LiteralPath $output) {
    throw "Output already exists; refusing to overwrite: $output"
}

$outputParent = Split-Path -Parent $output
New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
$stagingRoot = Join-Path $outputParent (".visa-prepare-" + [guid]::NewGuid().ToString("N"))
Assert-InProject $stagingRoot

try {
    & $python (Join-Path $projectRoot "scripts/audit_visa.py") --split-file $splitFile --raw-root $raw
    if ($LASTEXITCODE -ne 0) { throw "Raw VisA audit failed" }

    & $python $prepareScript --split-type 1cls --data-folder $raw --save-folder $stagingRoot --split-file $splitFile
    if ($LASTEXITCODE -ne 0) { throw "Official VisA preparation failed" }

    $stagedPrepared = Join-Path $stagingRoot "1cls"
    & $python (Join-Path $projectRoot "scripts/audit_visa.py") --split-file $splitFile --prepared-root $stagedPrepared
    if ($LASTEXITCODE -ne 0) { throw "Prepared VisA audit failed" }

    Move-Item -LiteralPath $stagedPrepared -Destination $output
    & $python (Join-Path $projectRoot "scripts/audit_visa.py") --split-file $splitFile --raw-root $raw --prepared-root $output --json-output $manifest
    if ($LASTEXITCODE -ne 0) { throw "Final VisA audit failed" }
}
finally {
    if (Test-Path -LiteralPath $stagingRoot) {
        $resolvedStaging = [System.IO.Path]::GetFullPath($stagingRoot)
        Assert-InProject $resolvedStaging
        if ((Split-Path -Parent $resolvedStaging) -ne [System.IO.Path]::GetFullPath($outputParent) -or
            -not (Split-Path -Leaf $resolvedStaging).StartsWith(".visa-prepare-")) {
            throw "Refusing to clean unexpected staging path: $resolvedStaging"
        }
        Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
    }
}

Write-Host "Prepared and audited VisA 1-class dataset at $output"
Write-Host "Manifest: $manifest"
