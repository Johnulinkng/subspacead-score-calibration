$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectCachePaths = @{
    TEMP = 'tmp'
    TMP = 'tmp'
    PIP_CACHE_DIR = '.cache/pip'
    HF_HOME = '.cache/huggingface'
    TORCH_HOME = '.cache/torch'
    UV_CACHE_DIR = '.cache/uv'
    UV_PYTHON_INSTALL_DIR = '.runtime/python'
    UV_PYTHON_BIN_DIR = '.runtime/bin'
    CONDA_PKGS_DIRS = '.cache/conda'
    XDG_CACHE_HOME = '.cache'
    MPLCONFIGDIR = '.cache/matplotlib'
    NUMBA_CACHE_DIR = '.cache/numba'
    PYTHONPYCACHEPREFIX = '.cache/pycache'
    WANDB_DIR = 'runs/wandb'
}
foreach ($entry in $projectCachePaths.GetEnumerator()) {
    $cachePath = Join-Path $projectRoot $entry.Value
    New-Item -ItemType Directory -Path $cachePath -Force | Out-Null
    [Environment]::SetEnvironmentVariable($entry.Key, $cachePath, 'Process')
}
$env:PYTHONNOUSERSITE = '1'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $projectRoot '.venv'
Write-Output "Project cache and runtime paths configured for this shell: $projectRoot"
