# Classify initial bounds on CUDA

The native trajectory backend now classifies scalar and variable bounds on the
GPU during symbolic setup. Previously it downloaded four double-precision arrays
so the CPU could distinguish free, one-sided, two-sided and equality bounds.
CUDA now returns one byte per bound pair. Actual numerical bound values stay on
the GPU and enter QOCO through the existing device conversion and initialization.

This defaults on when device numerical initialization is active. Set
`SPACEPDHCG_TEST_QOCO_DEVICE_BOUND_TYPES=0` to restore host classification.
Generic adapter clients, explicit host initialization and diagnostic conversion
comparisons retain the host path. Symbolic sparse assembly and route/fleet
orchestration still run on the CPU; this is another step toward GPU residency.

## Preserved behavior

The classifier uses the same finite-value and exact-equality rules as the
existing conversion. Signed zeros compare equal; unequal adjacent floating-point
values remain two-sided bounds. Infinite bounds retain their original meaning.
NaNs are rejected before symbolic assembly. Subsequent numerical updates retain
the existing device checks for changed bound classes and invalid arithmetic.
No solver, objective, integration or physics acceptance tolerance changes.

Host-side symbolic assembly consumes only the compact classes and uses numerical
placeholders, as in the preceding device initialization change. Conversion maps
still select the actual device values before the first solve. No placeholder
problem is solved.

## Validation scope

The standalone CUDA probe checks all classes against independently specified
expected values, including NaNs, infinities, signed zero, neighbouring doubles,
empty inputs, missing pointers, nonzero pointer offsets and large grid-stride
tails. The native conversion probe also exercises device initialization with
mixed scalar/box/SOC/RSOC constraints, duplicate sparse entries, changed values,
invalid updates and state-origin modes.

Trajectory tests compare host classification, explicit GPU classification and
the new default under both thrust interpolations, both state origins and three
scaling policies. They retain independent propagation, the original 1e-5 kg
mass-comparison bound and numerical update checks. First-fresh-solve counters
must show fewer downloaded bytes and three fewer copy calls.

All **154 broad tests pass on each GPU**. The final default-enabled builds pass
**77 tests each**, including the explicit enable/disable and default comparisons.
The mixed native conversion probe passes all six combinations of zero/two/five
Ruiz passes and shifted/unshifted states on both GPUs. Memcheck, synccheck and
racecheck pass for the standalone classifier. Those sanitizer results cover that
helper; the previously documented full-solver cuDSS instrumentation failures
remain unresolved. See [the investigation](GPU_SCALED_WORKSPACE_REUSE.md).

## Measurements

For the first fresh solve in the full-catalogue campaign, adapter downloads fall
from **515,324 to 244,499 bytes**, a **52.55% reduction**, with **13 to 10 copies**
on both GPUs. The bound arrays themselves shrink from 288,880 downloaded bytes
to 18,055 classification bytes. These are first-fresh-solve counters, not sums
of cumulative counters across reused workspaces. No numerical update is omitted.

Complete campaigns alternate baseline/candidate/candidate/baseline using the
same binary and production zero-Ruiz pooling. Every run preserves identical
initial plans, 45,188,558 logical branches and 2,782,091 collection options.
Each attempts 47 legs, converges on 46, creates 17 workspaces and passes both
mission checkers at **548.254620 weighted kg**.

| Hardware | Host classification median | GPU classification median | Observed change |
|---|---:|---:|---:|
| RTX 5090 | 28.5799 s | 28.3968 s | 0.64% less time |
| H100 | 30.1844 s | 29.3083 s | 2.90% less time |

The original 225-leg replay retains the same **205 independently certified
trajectories** in both modes, on both GPUs. The other 20 attempts remain
unsuccessful. Both modes use 450 priming dispatches and 72 workspaces.

| Hardware | Host classification solver time | GPU classification solver time | Observed change |
|---|---:|---:|---:|
| RTX 5090 | 127.8692 s | 128.2501 s | 0.30% more time |
| H100 | 157.0263 s | 164.3139 s | 4.64% more time |

Solver time excludes independent certification and export. Maximum certified
mass differences are 4.0463e-8 kg locally and 1.0418e-7 kg on H100. Case 87
accounts for 7.1977 seconds of H100's 7.2876-second increase: it remains
unsuccessful and takes 25 versus 44 outer attempts. Both calls reuse a workspace,
so initial bound classification does not run in that call. This does not prove
the source of the numerical-path divergence. The legacy `infeasible` result
label is not a global infeasibility certificate.

Two complete campaigns per mode with overlapping timing ranges, and one full
replay per mode, do not establish a general speedup. The default advances GPU
residency while preserving verified results; the replay regressions are retained.
The best fleet remains **12,805.194102 weighted kg**, with no new submission.

## Reproduction

[The evidence bundle](../results/lambda/2026-09-08/gpu-bound-types-v577/summary.json)
contains the source snapshots, commands, runtime hashes, original replay inputs,
per-leg trajectories, mission results and timing reports. All 638 retrieved
Lambda files were checked against their archive manifest. The local archive has
its own member manifest, and published files have SHA-256 hashes.

Full replays and campaigns use explicit flag values on the experiment binary.
Final binaries change only the default flag policy; the regression test adds
the unset/default case. The 77 final tests validate those final source bytes.
Recipes retain the original machine paths and require the pinned QOCO/CUDA
runtime documented in the preceding initialization and scaled-workspace reports.
The existing visualizer dataset is unchanged; the new archives contain complete
campaign trajectories for inspection.
