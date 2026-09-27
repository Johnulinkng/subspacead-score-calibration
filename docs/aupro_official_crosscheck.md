# AUPRO official-code cross-check

## Conclusion

`src/industrial_anomaly_lab/metrics.py::aupro` matches the MVTec AD official
evaluation package for valid binary masks. No change to the metric
implementation is required.

The deterministic audit ran 2,535 comparisons:

- seven fixed edge-case datasets;
- 500 randomized datasets, alternating continuous and tied scores;
- FPR integration limits 0.01, 0.05, 0.10, 0.30, and 1.00.

All comparisons passed at an absolute tolerance of `1e-6`. The largest
absolute difference was `3.193105974075294e-08`. This is explained by the
official implementation converting cumulative FPR values to `float32`, while
the project implementation retains `float64` precision.

## Official source provenance

- Landing page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>
- File linked by the page: `mvtec_ad_evaluation.tar.xz`
- Local archive: `data/archives/mvtec_ad_evaluation.tar.xz`
- Archive size: 11,056 bytes
- SHA-256: `dfcda7d67eee25316ec6ae5042c0b1684a4cabf33b2346be351e2ce36013f220`
- Extracted code: `third_party/mvtec_ad_evaluation`
- Code license: BSD 3-Clause, as stated in the extracted `LICENSE.txt`

The extracted source is kept unchanged. The project does not rely on a
third-party reimplementation for this audit.

## Semantics confirmed

The cross-check confirms the following details:

1. Ground-truth regions use 8-connectivity.
2. Every connected component has equal weight, regardless of its area.
3. FPR uses all background pixels, including pixels from normal test images.
4. Pixels with the same anomaly score enter the curve as one threshold group.
5. The curve is linearly interpolated at the FPR limit.
6. Trapezoidal area is divided by the FPR limit; the formal result is therefore
   normalized to the interval from zero to one.

## Contract differences

These differences affect invalid or unnormalized inputs, not the metric value
under the frozen experiment protocol:

- The official code accepts any nonzero mask value as foreground. The project
  API deliberately requires binary `0/1` or Boolean masks. Dataset adapters
  must convert raw `0/255` PNG masks before calling it.
- The project API rejects NaN and infinite scores, missing anomalous regions,
  and missing background pixels with explicit errors. The official code does
  not consistently validate these cases.
- The project computes cumulative FPR in `float64`; the official code casts it
  to `float32`. This produces only the numerical difference reported above.

## Reproduction

From the project root, with the locked environment already synchronized:

```powershell
.\.venv\Scripts\python.exe scripts\validate_aupro_against_official.py
```

The script verifies the archive hash before importing the official code and
writes the full machine-readable result to
`artifacts/validation/aupro_official_crosscheck.json`.
