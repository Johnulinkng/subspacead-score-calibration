# Archive and local retirement

Publication date: 2026-10-19. Repository:
https://github.com/Johnulinkng/subspacead-score-calibration

## Release assets

The `archive-20261019` release contains the final portable experiment and
historical result reports. `archive-inventory.json` identifies the assets
and historical report paths, sizes and SHA-256 hashes. `SHA256SUMS.txt`
checks the downloaded assets. Source and its normal Git history are on
the repository's `main` branch.

- `context-calibration-delivery.zip`: complete portable final delivery,
  including fitted PCA/calibration parameters, compact exact replay inputs,
  masks, metrics, 54 visual comparisons, source snapshots and licenses.
  Extracts to `context-calibration-delivery/`.
- `research-results.zip`: 1,348 historical scalar report, metrics, status,
  registration and audit files, including failure and partial-run evidence.
  Extracts under `runs/`; historical HTML may refer to omitted images/caches.
- `archive-inventory.json` and `SHA256SUMS.txt`: final inventory and checksums.

The owner explicitly selected keeping source, result reports and the complete
final portable delivery, while deleting other old models and caches directly.
This supersedes the earlier 8.4 GB model-archive proposal. Draft partial model
uploads were cancelled and removed; those archives are not release assets.

## Intentionally not backed up as bulk binaries

This is not a byte-for-byte backup of the working machine. Old fitted models
and self-trained checkpoints outside the final portable delivery, downloaded
raw datasets and source archives, upstream pretrained weights, Python/package
environments and caches, transient pytest fixtures, and intermediate bulk
NPY/per-frame NPZ feature/map caches are omitted. Download identities and
hashes remain in the source manifests. Intermediate experiments require
regeneration for replay; old training states are intentionally not retained.
The final portable delivery retains its models and exact compact replay data.

MVTec-derived masks and figures remain CC BY-NC-SA 4.0; VisA-derived data
remain CC BY 4.0. Model/third-party licenses are separate from the root code
license. No private customer messages or payment discussions are needed to
reproduce this project.

## Authorized local cleanup scope

The user requested removing this project's local code, weights, datasets,
results and environments after publication. The identified roots are:

- `E:\3k`: main project, approximately 173,860,428,749 logical bytes before
  publication staging (hard-linked files can make actual reclaimed space
  smaller than the logical total).
- `D:\3k`: legacy directory with two small upstream metadata files.
- `D:\3k-research-20261016`: separate partial baseline output directory.
- The temporary publication/remote-verification checkout, if created.

Before deletion, verify the pushed Git commit and release asset sizes/hashes.
Remove reparse links themselves without following their targets. Preserve
unrelated projects, shared runtimes, application/chat history and unrelated
paused automations. No training is resumed by this operation.

The cleanup receipt in `manifests/local-retirement.json`, when present,
records actual deletion and post-checks; this document alone does not prove
cleanup completion.
