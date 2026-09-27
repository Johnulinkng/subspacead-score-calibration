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
import faiss
import imgaug
import numpy
import sklearn
import timm
import torch
import torchvision

faiss_probe = faiss.IndexFlatL2(2)
faiss_probe.add(numpy.array([[0.0, 1.0]], dtype=numpy.float32))

payload = {
    "python": __import__("sys").version,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": numpy.__version__,
    "opencv": cv2.__version__,
    "faiss": faiss.__version__,
    "faiss_index_ntotal": faiss_probe.ntotal,
    "imgaug": imgaug.__version__,
    "sklearn": sklearn.__version__,
    "timm": timm.__version__,
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
