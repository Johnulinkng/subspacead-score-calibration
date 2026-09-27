$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'Set-ProjectEnvironment.ps1')
$uvExecutable = Join-Path $projectRoot '.runtime/uv/uv.exe'
if (-not (Test-Path -LiteralPath $uvExecutable)) {
    throw "Project-local uv is missing: $uvExecutable"
}

& $uvExecutable sync --frozen --python (Join-Path $projectRoot '.venv/Scripts/python.exe')
if ($LASTEXITCODE -ne 0) {
    throw 'Environment synchronization failed.'
}
