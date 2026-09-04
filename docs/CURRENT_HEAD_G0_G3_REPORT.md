# Current-head G0-G3 regression report

## Decision

G0, G1, G2, and G3 **PASS** for tested source commit
`b6afb49d7fc7da5ed1ac9003c3bcae5d35506026`, tree
`a91553cd646393e343dabc332ce4921e753ca219`, on branch
`integration/single-gpu-v1`.

This authorises Gate G4 scientifically. It does not make a G4 campaign launch-ready: the existing
`/home/angus/g4-executor-capability-b0cd570.json` is pinned to the original head and was not used
as G3 authority. A new official capability must be generated from the final clean executable
before starting the `single-gpu-v1` claim core. No G4 scheduler, session, claim-core, or campaign
worker was launched in this regression task.

All archives are local-only. There are no immutable URIs, and none are fabricated.

## G0

- Ruff lint/format: PASS.
- Python: 314 passed, one explicit skip for the optional native CUDA session executable.
- Top-level CTest: 45/45 in RelWithDebInfo, Debug, and ASan+UBSan Werror builds.
- Standalone native inventory: 8/8 in all three configurations.
- Fresh isolated wheel consumer and CMake consumer: PASS.
- Wheel SHA-256:
  `fc150b87fd5a78e175a457fe7063b586b51e37925594df2f14920d4e43ec2b55`.
- sdist SHA-256:
  `f3aa933cf2056de1decfa35ba49f9836f15356e971588935d46d05b51358c9b4`.
- Archive:
  `results/gpu/current-head-b0cd570/seals/g0-b6afb49d7fc7.tar.gz`,
  SHA-256 `ded721d6e0b682691cbbc5d3171b0d6927f7b10b76c614c2581e6f009483b813`.

## G1

- Pinned PDHCG commit/tree:
  `167c8b72b4b96d2f94d405b8763e485514192b81` /
  `62b05e6c1bedd385f6c267af3645ae4aae0421b4`.
- Automated box/SOC exact-optimum gate: 15/15 cases.
- Declared expansion: 96/96 solver cases and 16/16 repeated-identical/value-update cases;
  48 box and 48 SOC solver cases across all four tolerances and all three starts.
- Overall maximum independent natural residual: `1.0668997501934837e-2` at the `1e-3` tier.
  At `1e-8`, the maximum natural residual is `1.0988043409355941e-7`; all tier-specific
  acceptance rules pass.
- No-device control: expected `cudaErrorNoDevice`, PASS.
- Archive:
  `results/gpu/current-head-b0cd570/seals/g1-b6afb49d7fc7.tar.gz`,
  SHA-256 `b30cff0bb7e81f36acf8d2718e8c4aa28f384199acac2e8ce8b39a0189dcbeb6`.

## G2

- Debug and RelWithDebInfo CUDA/native CTest: 62/62 each.
- Ten-update QP maximum CPU error `3.23909889e-7`, pinned one-shot error
  `3.90241894e-7`, natural residual `3.09112063e-7`.
- Managed SOCP cone distance and natural residual: `6.45576925e-11`.
- CuPy/PyTorch/JAX producer maximum solution error: `7.113320221741048e-8`.
- Four warm modes, refresh/reuse scaling, checkpoint/restore, pointer/topology/copy/allocation,
  default/non-default stream, cancellation/error/destruction, producer lifetime and premature
  release checks: PASS with zero post-create allocation delta and no fallback.
- Five retained sanitizer logs are clean. The first stream-lifetime racecheck was retained as a
  hung instrumentation attempt because it places a 100-billion-iteration cancellation kernel
  under racecheck. The complete persistent kernel racecheck passed on `persistent_cw_test`;
  stream cancellation/destruction passed natively. No criterion or threshold was weakened.
- Archive:
  `results/gpu/current-head-b0cd570/seals/g2-b6afb49d7fc7.tar.gz`,
  SHA-256 `2326a2a521e3816f1b35f3e6c643a8eea80d5bef1da867d7ab47c4144f8e6cb4`.

## G3

- Debug and Release CUDA/native CTest: 62/62 each.
- Tight canonical residuals: HCW `9.69295039e-7`, P1-C `1.42322019e-10`,
  P1-E `4.58086731e-7`, P1-D `1.15292919e-8`; maximum `9.69295039e-7`.
- Displaced HCW accepted three nonzero steps, retained change `1.1845741e-1`, and terminal
  residual `2.89659204e-8`.
- Direct non-campaign pure-QOCO warmup regressions accepted 2/24/2 steps for P1-C/P1-D/P1-E.
  Their canonical residuals are `7.248916111459482e-11`,
  `1.0665246463759104e-11`, and `1.7462298274040222e-10`; maximum terminal residual
  is `1.2271604820637482e-11`.
- Fixed-tight PDHCG P1-C/P1-D/P1-E warmup representatives each reached the retained 150-second
  timeout with zero accepted steps and remain honest negatives.
- Maximum CPU/GPU coefficient difference `2.7599450502791001e-13`; production maximum nonlinear
  residual `2.89659204e-8`; trajectory difference, post-create topology allocation/copy deltas,
  and fallback count are all zero.
- H1: supported from 20 intervals; six sizes, seven measured repeats each, zero censoring, and
  bootstrap interval `[0, 0]`.
- Sixteen memcheck/racecheck/initcheck/synccheck logs are clean. The recovery racecheck completed
  in 54 minutes rather than being inferred or truncated.
- Nsight Systems 2024.6.2 under WSL records CUDA API activity but no CUDA kernel or GPU-memory
  records. The raw report and corrected forced SQLite export are retained; no timeline residency
  claim is made.
- Archive:
  `results/gpu/current-head-b0cd570/seals/g3-b6afb49d7fc7.tar.gz`,
  SHA-256 `08ccc71bec848cea24c64b1efbf97e50962a2db14a4d6a2099b7750451756810`.

The ignored evidence root contains 722 indexed artifacts. Every gate archive contains its own
`evidence-index.json`, exact commands, environment/source references, failure retention, and
per-artifact SHA-256 records. The root index is
`results/gpu/current-head-b0cd570/evidence-index.json`.
