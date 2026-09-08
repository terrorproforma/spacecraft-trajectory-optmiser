# Preserve objective magnitude during GPU Ruiz equilibration

An opt-in GPU scaling policy fixes the captured objective-normalization failure
while retaining all 205 independently certified legs in the 225-leg replay on
RTX 5090 and H100. It keeps the global objective multiplier `k` equal to one;
variable and constraint equilibration still run. The zero-Ruiz production default
and every physics, conic residual and objective acceptance limit remain unchanged.

The preceding [diagnosis](GPU_CONDITIONING_DIAGNOSIS.md) isolated multiplication
of the objective by 0.0001 as sufficient to trigger the captured failure. With
the new policy disabled, the rebuilt libraries fail all three QP repetitions on
each GPU. Enabled, they pass all three in **27 inner iterations** on each GPU.
These results qualify the original equations using independent long-double
sparse products; a vendor success status alone is insufficient.

## Implementation and configuration

`prepare_qoco_preserve_objective.py` adds the policy to the prepared CUDA numeric
update context and the CPU setup/reference equilibration. Device contexts capture
the flag at creation. Synchronous, queued and graph-owned numeric updates all
pass that immutable policy to the CUDA cost reduction, which uses a unit cost
multiplier when enabled. Numerical arrays and scale vectors stay on the device
through updates; this change introduces no new result download or CPU numerical
solve. Existing setup/reference CPU work is still present.

Patched `prepare_qoco_gpu.py` builds with device numeric updates include the
extension automatically, disabled by default. To test it with a newly prepared
library, set `SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE=1` **before creating any
solver workspaces**, and request two Ruiz passes with `--qoco-ruiz-iterations 2`.
Changing the environment after workspace creation is unsupported. Old libraries
do not implement this flag. The archived commands and library hashes identify
the exact local and Lambda builds used here.

## Complete trajectory replay

Both modes use the same new binary per GPU and the unchanged fixed 225-leg input,
original iteration/time budgets and independent certification. This is one full
pass per mode, not a repeated timing distribution. Solver time excludes the
independent certification step.

| Hardware | Zero-Ruiz solver time | Preserve-objective, two passes | Certified legs, both modes | Largest final-mass difference |
|---|---:|---:|---:|---:|
| RTX 5090 | 126.0303 s | 143.5143 s | 205 / 225 | 4.813e-8 kg |
| H100 | 160.7936 s | 158.8101 s | 205 / 225 | 2.032e-7 kg |

No previously certified leg is lost. The unqualified attempts remain rejected.
The local candidate changes three unsuccessful classifications from failure to
infeasible; that does not create three additional valid trajectories.

The builds measured here admit only zero-Ruiz workspaces: they create 72 workspaces in
this replay, whereas the candidate creates 225. Preserving objective magnitude
has not changed this eligibility rule. Local solve time rises 13.87%; the single
H100 observation is 1.23% lower. Neither establishes a general speedup. Reusing
scaled workspaces needs separate correctness and timing evidence before it can
be enabled.

The subsequent [scaled workspace experiment](GPU_SCALED_WORKSPACE_REUSE.md)
adds capability-checked reuse and records its separate measurements and
full-solver sanitizer limitations. The measurements above retain their original
configuration.

## Complete campaigns

Runs alternate baseline/candidate/candidate/baseline with two observations per
mode and the same new binary on each GPU. The one-ship fixture retains identical
initial plans and logical counts: 45,188,558 branches and 2,782,091 collection
options. All eight runs pass both final mission checkers at 548.254620 weighted
kg, with 46 of 47 attempted trajectory solves converging in every run.

| Hardware | Zero-Ruiz process median | Preserve-objective process median | Change |
|---|---:|---:|---:|
| RTX 5090 | 27.8505 s | 29.9529 s | 7.55% longer |
| H100 | 29.5550 s | 33.2093 s | 12.36% longer |

The candidate creates 47 workspaces per campaign versus 17 for zero Ruiz.
Numerical solve histories also vary, so setup counts alone do not explain the
whole difference. This is an opt-in accuracy fix for the scaling mode, not a
performance default. Existing zero-Ruiz runs remain the faster measured choice.

## Validation and evidence

Native numerical-update comparisons cover zero/one/four Ruiz passes, missing
quadratic diagonals, empty constraints, zero quadratic objectives, off-diagonal
terms, and 17/1031-variable systems. CPU/GPU matrices and scale vectors agree, and
independent transformation equations agree. Enabled-policy checks require
exactly `k = kinv = 1`. Memcheck, synccheck and racecheck pass on both GPUs.
All **107 broader tests pass on each GPU**, covering workspace reuse, retained
replay, SCvx, GPU selection, CLI qualification and independent verification.

The preparation tool rejects an unexpected vendor layout before writing either
file. Its final source reproduces the two compiled patched files byte for byte;
reapplication rejects without partial mutation. An initial preparation guard
omitted the third, graph-owned numeric-update path and rejected the source before
building. The clean v534/v535 builds include all three paths.

- [Local build, replay, tests and campaigns](../results/lambda/2026-09-08/gpu-preserve-objective-local-v537/summary.json)
- [H100 build and trajectory replay](../results/lambda/2026-09-08/gpu-preserve-objective-v535/analysis.json)
- [H100 tests and campaigns](../results/lambda/2026-09-08/gpu-preserve-campaign-v538/analysis.json)

The best retained fleet remains **12,805.194 weighted kg**. This work does not
establish a new fleet score, official leaderboard placement or complete GPU
control of route/fleet orchestration.
