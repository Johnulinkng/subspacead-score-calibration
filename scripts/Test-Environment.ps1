$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'Set-ProjectEnvironment.ps1')
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python is missing: $python"
}

$environmentProbe = @'
import json
import cv2
import imgaug
import numpy
import sklearn
import torch
import torchvision

payload = {
    "python": __import__("sys").version,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": numpy.__version__,
    "opencv": cv2.__version__,
    "imgaug": imgaug.__version__,
    "sklearn": sklearn.__version__,
    "cuda_available": torch.cuda.is_available(),
    "torch_cuda": torch.version.cuda,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(payload, indent=2))
'@
$environmentProbe | & $python -B -
if ($LASTEXITCODE -ne 0) {
    throw 'Dependency import or CUDA verification failed.'
}

& $python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    throw 'Project tests failed.'
}
