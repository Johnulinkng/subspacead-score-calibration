# Normal-Content Score Calibration for Industrial Anomaly Detection

An experimental score-calibration adapter built on frozen DINOv2-S features and SubspaceAD-style PCA residuals. It includes matched baseline comparisons, ablations, numerical checks, and evaluation on MVTec AD and VisA.

This is an independent reproduction and adaptation project, not the official SubspaceAD implementation. Its main contribution is an optional scoring adjustment and bounded-memory execution tooling. The experiments do not establish a new state-of-the-art detector or publication-level novelty.

## What changes

The backbone and PCA residual detector stay fixed. Normal training features define eight content prototypes and per-prototype residual quantiles. A residual `r` with assigned normal upper bound `T` is calibrated as:

```text
score(r, T) = r * min(1, r / T)
```

The implementation includes a global-tail ablation, normal-only parameter fitting, PCA coordinate alignment, and bounded-memory prototype assignment. Single-pass mean/scatter accumulation and lazy feature loading are implementation optimizations; streaming PCA itself is established prior work, and upstream SubspaceAD already implements two-pass streaming PCA.

## Measured results

Full normal-training data, DINOv2-S, 448-pixel backbone input, 256-pixel evaluation, equal category weighting. These are this project's settings, not the official paper's few-shot configuration. Image scores use each variant's own maximum anomaly score.

| Dataset | Baseline pixel AP (%) | Calibrated pixel AP (%) | AP gain (percentage points) | AUPRO change (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| MVTec AD | 58.9369 | 59.2524 | +0.3155 | +0.0761 |
| VisA | 35.4937 | 35.7976 | +0.3039 | -0.1624 |

Conditioning adds approximately 0.1426 percentage points of pixel AP over the global-tail ablation on MVTec AD and only 0.0063 on VisA. False-positive rates do not improve consistently. Test data had been inspected in earlier experiments; the original delivery's PCA had seen both normal calibration partitions. Treat these results as exploratory comparisons, not independent blind validation or conformal guarantees.

Later retrospective refits, independent normal partitions, development controls and partial stronger-baseline runs are retained in the research-records release. A completed experimental artifact is not evidence that the broader paper-level research objective was achieved. Missing and partial runs must remain missing and partial.

## Code and dependencies

```bash
git clone --recurse-submodules https://github.com/Johnulinkng/subspacead-score-calibration.git
cd subspacead-score-calibration
uv sync --frozen
uv run pytest tests/test_context_calibration.py tests/test_tail_calibration.py tests/test_subspace_features.py tests/test_subspace_controls.py
```

Python 3.11 and the pinned environment are specified in `pyproject.toml` and `uv.lock`. The historical lock uses the PyTorch CUDA 12.8 index. Use a suitable PyTorch build for other hardware and document environment differences. Historical experiment configurations contain original Windows paths; update paths before a new run and do not alter archived experiment records.

The scoring module is `src/industrial_anomaly_lab/context_calibration.py`. PCA/extraction and numerical controls are in `subspace_features.py` and `subspace_controls.py`. Full scoring and audit entry points are `scripts/run_context_single_pass.py` and `scripts/audit_context_single_pass.py`; their historical inputs include normal-context and PCA-control artifacts.

## Download and replay the recorded experiment

The [archive release](https://github.com/Johnulinkng/subspacead-score-calibration/releases/tag/archive-20261019) contains the complete portable delivery, archived research records, fitted models and available self-trained checkpoints. See `docs/ARCHIVE.md` for coverage and checksums.

```bash
gh release download archive-20261019 --repo Johnulinkng/subspacead-score-calibration --pattern context-calibration-delivery.zip
unzip context-calibration-delivery.zip
cd context-calibration-delivery
python -m pip install -r requirements-offline.txt
python -B 复算指标.py
```

Offline replay checks cached residuals, assignments, masks, per-map hashes and exact metrics without DINO weights or a GPU. It requires temporary disk space and preserves the recorded 8 GiB free-space reserve. It is not a fresh backbone evaluation.

For inference on an original image, install the full dependencies and obtain the pinned public DINOv2-S weights identified by `backbone.json`:

```bash
python -B 原图推理.py --dataset mvtec_ad --category bottle --image /path/to/image.png --weights /path/to/dinov2-small --output ../prediction.npz --device cpu
```

Raw datasets, downloaded upstream pretrained checkpoints, package environments and large regenerable feature/map caches are not distributed as repository content. Download identities and hashes are retained in `manifests/` and the archive inventory. The release does retain the fitted models and exact compact replay data for the final delivery.

## Sources and licenses

- [SubspaceAD](https://github.com/CLendering/SubspaceAD), pinned reference commit `ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7`, Apache-2.0.
- [DINOv2](https://github.com/facebookresearch/dinov2); the pinned Hugging Face checkpoint revision is recorded in the delivery.
- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), MVTec Software GmbH, CC BY-NC-SA 4.0. Dataset-derived masks and illustrations retain these restrictions.
- [VisA](https://github.com/amazon-science/spot-diff), Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer, CC BY 4.0.
- scikit-learn MiniBatchKMeans and established normal-score calibration ideas are used; see `docs/q3_novelty_check_20261018.txt` and release provenance for related work and limitations.
- Historical DRAEM, PatchCore, RD4AD and Anomalib integrations retain their upstream references and licenses. The MVTec evaluation implementation retains its BSD-style license.

See `NOTICE` and the individual third-party license files. The root code license does not replace upstream software, model or dataset licenses.
