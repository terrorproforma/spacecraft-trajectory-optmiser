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

## G2/G3 reseal on main `8cb3759` (RTX 5090, 2026-09-05)

G2 and G3 **PASS** for tested source commit
`8cb3759b29ea8c7d843322a940a7ebcabfd9ff21`, tree
`6d27f2552d882b4418d16e4342e6854a436a952d` (= `main`), run from branch `chore/g2g3-reseal-8cb3759` before any
commit was made on it. Hardware `local-rtx-5090` (NVIDIA GeForce RTX 5090, sm_120, WSL Ubuntu-22.04,
CUDA 12.8, driver 595.97). Reason for the reseal: the shared CUDA library changed after the
`b6afb49` (RTX 5090) and `9e75b47` (H100) seals through `1dbcae0` (recovery-kernel cancel polling,
kernel pre-load, `inner_iteration_cap`) and `2bca11d` (HCW exact matrix-exponential step,
control-tracking term removal, relative KKT audit). G0/G1 were not re-run in this scope. The
procedure, tests, tolerances, timeouts and sanitizer targets are those of the earlier seals; the only
operational additions are a pre-step foreign-GPU wait (recorded per gate) and nice-10 `-j8` builds on
the shared host. No G4 campaign was launched from this evidence.

### G2

- Debug and RelWithDebInfo CUDA/native CTest: 70/70 and 70/70.
- Ten-update QP maximum CPU error `3.23909889e-07`, pinned one-shot error `3.90241894e-07`, natural residual `3.09112063e-07`.
- Managed SOCP cone distance `6.45576925e-11`, natural residual `6.45576925e-11`.
- CuPy/PyTorch/JAX producer maximum solution error `7.11332022e-08`; premature release PASS.
- 4 warm modes, checkpoint/restore True, default/non-default streams True, cancellation True, destruction True, post-create allocation delta 0, topology pointers stable True, error paths 5/5.
- 5 sanitizer logs (memcheck x2, racecheck, initcheck, synccheck) clean: True.
- Foreign GPU: 19 pre-step GPU checks, no foreign-process wait.
- Archive: `results/gpu/current-head-8cb3759-rtx5090/seals/g2-8cb3759b29ea.tar.gz`, SHA-256 `095f33dc83328290ea1533d0bc9531b17a316f004f0c7f8b5cd0057471fda45d`.

### G3

- Release and Debug CUDA/native CTest: 70/70 and 70/70.
- Tight canonical residuals: HCW `9.69295039e-07`, P1-C `1.42322019e-10`, P1-E `4.58086731e-07`, P1-D `2.82913893e-08`; maximum `9.69295039e-07`.
- Displaced HCW accepted 3 steps, retained change `1.18457409e-01`, terminal residual `2.92768846e-08`.
- Direct non-campaign pure-QOCO warmup regressions accepted 2/24/2 steps for P1-C/P1-D/P1-E; canonical residuals `8.51961803e-12`, `7.72710592e-12`, `4.27803326e-12`; maximum terminal residual `5.02714832e-11`.
- Fixed-tight PDHCG P1-C/P1-D/P1-E warmup representatives: P1-C-pd3 timeout (0 accepted), P1-D-pd6 timeout (0 accepted), P1-E-low-thrust timeout (0 accepted); all 3 remain honest negatives.
- Maximum CPU/GPU coefficient difference `2.75994505e-13`; production maximum canonical residual `9.56640559e-09`, nonlinear residual `2.92768846e-08`, CPU/GPU trajectory difference `0.00000000e+00`; topology allocation/copy deltas 0/0; hidden CPU fallback False.
- H1: supported from 20 intervals; 6 sizes, 7 measured repeats each, omega bootstrap interval [0.0, 0.0]; SCvx median seconds by N: 100: 0.2696, 10000: 31.01, 20: 0.05537, 2000: 5.972, 50: 0.1339, 500: 1.391.
- 16 memcheck/racecheck/initcheck/synccheck logs clean: True. No-device negative control: PASS.
- Nsight Systems kernel/memory records available: False/False (WSL); no timeline-residency claim is made.
- Foreign GPU: 29 pre-step GPU checks, no foreign-process wait.
- Archive: `results/gpu/current-head-8cb3759-rtx5090/seals/g3-8cb3759b29ea.tar.gz`, SHA-256 `609e0acbed65d7c4449148677cbd69b2703ba23ea277a53a7034da742c439de6`.

The ignored evidence root `results/gpu/current-head-8cb3759-rtx5090` contains 178 indexed artifacts; the root index is
`results/gpu/current-head-8cb3759-rtx5090/evidence-index.json`, SHA-256 `443a8caf16e09699c67f499d59078261cfb94b5408c59e07c0e03dd83cd4e4a2`. The compact summaries, indices, runner scripts and
archive hashes are committed; the archives and raw logs stay local-only (no immutable URI).
