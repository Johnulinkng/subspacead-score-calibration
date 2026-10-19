# Candidate upstream contributions

No upstream pull request has been opened by this archive operation.

The most direct target is CLendering/SubspaceAD. The existing upstream PCA
already streams statistics in two passes, so a proposal must not claim to
introduce streaming PCA. A small optional one-pass mean/scatter backend could
avoid repeating feature extraction while keeping final covariance, rank
selection and residual scores numerically equivalent. Such a PR should keep
the current default, include adversarial numerical tests and compare the
same extraction/loading settings. Historical combined lazy-loading and
single-pass timings cannot be attributed to the accumulator alone.

Optional lazy input loading and avoiding unused attention outputs are also
reasonable implementation proposals, subject to verifying upstream's current
feature/masking needs. An attention-output change must not break saliency or
background masking paths.

The content-conditioned tail scorer is better presented initially as an
optional experimental extension with matched ablations and all regressions.
The VisA conditioning-only AP benefit is about 0.0063 percentage points and
AUPRO decreases, so it should not replace upstream defaults on the current
evidence. Discussing an extension with maintainers is preferable to proposing
a new state-of-the-art detector.

Anomalib could be a later integration target after the adapter is packaged
against its current model/postprocessing interfaces. The current project is
not a drop-in Anomalib plugin. DINOv2 is not the natural target because no
backbone algorithm change is demonstrated here.

An engineering or reproducibility contribution can be useful and accepted
upstream even when it does not constitute a paper's main scientific novelty.
Acceptance remains the maintainers' decision.
