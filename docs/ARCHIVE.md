# Archive and local retirement

Publication date: 2026-10-19. Repository:
https://github.com/Johnulinkng/subspacead-score-calibration

## Release assets

The `archive-20261019` release separates usable source from large binary
artifacts. `archive-inventory.json` identifies every archived historical
record/model/checkpoint and its original relative path, size and SHA-256.
`SHA256SUMS.txt` checks the downloaded assets. Each numbered `.tar.gz` is an
independent archive, not a byte-split stream; extract all desired parts into
the same directory. They contain `project/` and legacy D-drive record paths.

- `context-calibration-delivery.zip`: complete portable final delivery,
  including fitted PCA/calibration parameters, compact exact replay inputs,
  masks, metrics, 54 visual comparisons, source snapshots and licenses.
  Extracts to `context-calibration-delivery/`.
- `research-records-*.tar.gz`: historical run registrations, metrics, logs,
  source snapshots, figures, audits, failure and partial-run records.
- `fitted-models-*.tar.gz`: non-frame fitted NPZ models and normal calibration
  state, including earlier PaDiM and PCA experiments.
- `trained-checkpoints-*.tar.gz`: available local training checkpoints,
  including smoke/debug outputs. A stored checkpoint is not proof that a
  complete baseline matrix or registered final training was finished.
- `source-history.bundle`: all original local Git refs after publication;
  the primary source history is also on the repository's `main` branch.

## Intentionally not backed up as bulk binaries

This is not a byte-for-byte backup of the working machine. Downloaded raw
datasets and source archives, upstream pretrained weights, Python/package
environments and caches, transient pytest fixtures, and intermediate bulk
NPY/per-frame NPZ feature/map caches are omitted. Download identities and
hashes remain in the source manifests. Intermediate runs whose raw caches
are omitted require regeneration for a full cache replay. The final portable
delivery retains its exact compact replay data.

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
