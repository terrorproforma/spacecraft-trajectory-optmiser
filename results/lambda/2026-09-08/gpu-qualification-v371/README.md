# Difficult-QP diagnostics: rejected local candidates

These RTX 5090 experiments use the unchanged captured GTOC12 QP
`14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080`.
They are numerical diagnostics, not new fleet searches or Lambda measurements.
The production baseline is the final v358 SOC-step correction.

| Candidate | Qualified / 32 | Matched baseline / 32 | Decision |
|---|---:|---:|---|
| v369: rounded-endpoint guard plus compensated NT normalization | 28 | 31 | Excluded; more iterations |
| v371: four GPU symmetric KKT equilibration passes per factorization | 16 | 28 | Excluded; more refinement iterations |

Each comparison runs baseline/candidate/candidate/baseline, 16 cold replays per
process. `summary.json` reports the individual groups. Their elapsed times include
process execution and the independent audit; they are not pure GPU timings.
The v369 comparison is the completed `v369b` experiment. An earlier missing-library
attempt is excluded. Prepared-source build logs retain the corrected include-order
failure before any candidate execution.

External acceptance remains status 1 or 2, normalized primal/dual/gap at most
1e-9, and cone violation at most 1e-8. Internal tolerances remain 1e-11. No
candidate changes the original trajectory equations or acceptance thresholds.

## Division trace v370

An observation-only build captures 68 cone-division calls from one host-dispatched
GPU QP solve. Synchronous downloads change execution scheduling, so its runtime
is not a benchmark. A 100-digit Decimal oracle evaluates the original binary64
inputs exactly. Of 15,912 captured cone pairs:

- 12,168 have valid finite interior inputs. Their worst relative output error is
  7.4991e-16; this trace does not justify replacing the division arithmetic.
- 3,696 already have nonfinite right-hand sides; another 48 have nonfinite lambda
  inputs. The first invalid call is `division-0052.bin`. These observations locate
  the failure upstream of division, without establishing a universal root cause.

The compact trace report records the oracle formula's worst case and individual
snapshot hashes. `snapshots.tar.gz` preserves all original inputs and outputs.

## KKT transformation check

The candidate keeps raw K separate and factors D K D. Both initial solves and
refinement corrections transform the RHS by D and recover the solution by D.
All residuals are still calculated against the original unregularized equations.
Persistent device buffers and graph kernels implement the transformations.

The isolated GPU probe checks a captured 17,767-row, 53,005-entry matrix, then
changes its numerical values and repeats using the same buffers. The independent
array calculation verifies scaling, matrix entries, RHS/solution transformations,
and byte-identical preservation of the raw matrix. Both checks pass. Maximum
matrix magnitude falls from 6.90655e14 to approximately 1, but complete-QP
qualification worsens. Better-scaled entries alone do not establish a better solve.

## Reproduction and scope

`recipes/` contains the exact preparation, compilation, execution and audit scripts;
paths identify the frozen local CUDA 12.8 / cuDSS 0.8 environment. The
`frozen-v*/source.tar.gz` archives contain the full prepared sources, with per-file
SHA-256 manifests and build/library provenance. Rebuild with the flags recorded
in the recipes, adapting environment paths. Large raw replay logs and probe input
files use lossless gzip; decompress before passing them to the original recipes.

The QP input and original-equation audit are the same as the previously published
[SOC-step evidence](../gpu-soc-step-v359/README.md). These candidates are not
enabled in production. The verified fleet score remains 12,805.194 weighted kg.
