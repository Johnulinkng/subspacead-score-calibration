[CmdletBinding()]
param(
    [string]$PackageRoot = $PSScriptRoot,
    [string]$PythonPath,
    [ValidateRange(0, 54)][int]$InferenceSamples = 2
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$taskPackageRoot = [System.IO.Path]::GetFullPath($PackageRoot)
$taskManifestPath = Join-Path $taskPackageRoot 'DELIVERY-MANIFEST.json'
if (-not (Test-Path -LiteralPath $taskManifestPath -PathType Leaf)) {
    throw 'Run this entry point from the extracted package, or pass -PackageRoot with its absolute path.'
}
$taskVerifierPath = Join-Path $taskPackageRoot 'code\scripts\verify_quick_delivery.py'
if (-not (Test-Path -LiteralPath $taskVerifierPath -PathType Leaf)) { throw 'The package is missing its verification script.' }
$taskPythonCandidates = [System.Collections.Generic.List[string]]::new()
if ($PythonPath) { $taskPythonCandidates.Add($PythonPath) }
$taskLocalPython = Join-Path $taskPackageRoot 'code\.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $taskLocalPython -PathType Leaf) { $taskPythonCandidates.Add($taskLocalPython) }
$taskConfig = Get-Content -LiteralPath (Join-Path $taskPackageRoot 'run_config.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($taskConfig.PSObject.Properties.Name -contains 'project_root') {
    $taskPriorPython = Join-Path ([string]$taskConfig.project_root) '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $taskPriorPython -PathType Leaf) { $taskPythonCandidates.Add($taskPriorPython) }
}
$taskPythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($taskPythonCommand -and $taskPythonCommand.Source -notlike '*\WindowsApps\python*.exe') { $taskPythonCandidates.Add($taskPythonCommand.Source) }
$taskSelectedPython = $null
foreach ($taskCandidate in $taskPythonCandidates) {
    if (-not (Test-Path -LiteralPath $taskCandidate -PathType Leaf)) { continue }
    $taskVersion = & $taskCandidate -c 'import sys; print(str(sys.version_info.major) + "." + str(sys.version_info.minor))' 2>$null
    if ($LASTEXITCODE -eq 0 -and $taskVersion -eq '3.11') { $taskSelectedPython = $taskCandidate; break }
}
if (-not $taskSelectedPython) { throw 'Python 3.11 was not found. Supply -PythonPath <python.exe>. See report/复现说明.md for the pinned environment.' }
Write-Output ('Verifying the complete file manifest and {0} saved inference samples with {1}' -f $InferenceSamples, $taskSelectedPython)
& $taskSelectedPython $taskVerifierPath --package-root $taskPackageRoot --inference-samples $InferenceSamples
if ($LASTEXITCODE -ne 0) { throw 'Package verification did not pass. Check the verifier output and the pinned dependencies in report/复现说明.md.' }
