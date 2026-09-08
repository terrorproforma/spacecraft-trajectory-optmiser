# Reuse GPU workspaces with objective-preserving Ruiz scaling

The optional scaled workspace pool reuses GPU QOCO allocations, sparse conversion
and vendor graphs between compatible trajectory legs. It requires the previously
validated objective-preserving scaling policy. Production still defaults to zero
Ruiz passes; scaled reuse remains an experiment.

The local fixed 225-leg replay reduces solver time from **138.2339 s to 124.8685 s**
(9.67%), retaining the same **205 independently certified legs**. Workspace
creation falls from **225 to 72**. The largest independently replayed final-mass
difference is **1.211e-7 kg**. This is one pass per mode, with two Ruiz passes in
both, identical binaries, unchanged budgets and unchanged acceptance tolerances.
Convergence histories differ, so the timing change cannot be attributed entirely
to setup savings.

The H100 replay likewise retains 205/225 certified legs and reduces workspace
creation from 225 to 72. Its largest final-mass difference is 6.877e-8 kg.

| Hardware | Fresh scaled workspaces | Reused scaled workspaces | Less solver time |
|---|---:|---:|---:|
| RTX 5090 | 138.2339 s | 124.8685 s | 9.67% |
| H100 | 166.9910 s | 158.9078 s | 4.84% |

Both modes reject the other 20 attempts. Their failure/infeasible classification
mix changes, without adding any accepted trajectory. Priming dispatch counts
remain 450 in both modes; this extension reuses allocations and setup, without
eliminating all host-dispatched priming.

## Complete campaign comparison

Four runs per GPU alternate baseline/candidate/candidate/baseline. Here the
baseline uses production zero Ruiz and existing pooling; the candidate uses two
objective-preserving Ruiz passes and scaled pooling. All eight missions retain
548.254620 weighted kg, the same initial plans and search counts, and pass both
mission checkers. Each attempts 47 legs, qualifies 46 and creates 17 workspaces.
The previous unpooled scaled experiment created 47 workspaces per campaign.

| Hardware | Production process median | Scaled reuse process median | Observed change |
|---|---:|---:|---:|
| RTX 5090 | 30.0638 s | 29.6743 s | 1.30% less time |
| H100 | 29.1546 s | 29.3794 s | 0.77% more time |

These two-observation timing ranges overlap. They do not establish a complete
campaign speedup or justify replacing the production zero-Ruiz default.

## Implementation

The adapter checks an optional library export,
`qoco_gpu_numeric_preserves_objective`, which reports the numerical context's
actual immutable policy. An environment flag alone is insufficient: libraries
without this capability continue to create fresh scaled workspaces. A context
whose objective multiplier is not fixed to one cannot enter the scaled pool.

The pool key now includes the number of Ruiz passes alongside device, interval
count, hold, free boundary settings, tolerance and environment configuration.
Restart drains the previous leg's numerical and replay metadata while its stream
is alive, clears solver history, and requires a complete numerical refresh for
the next leg. Only successful legs enter the pool. A failed or timed-out lease
is discarded. The next leg's geometry, dynamics, objective and scaling are
refreshed; previous primal solutions are not accepted as new solutions.

To enable the experiment with a rebuilt core and prepared QOCO library, set both
flags before creating any workspaces:

```bash
export SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE=1
export SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL=1
```

Use the CUDA graph outer loop and request `--qoco-ruiz-iterations 2`. The ordinary
workspace pool must also be enabled (its existing default). Numerical updates
remain on CUDA. First-workspace setup and route/fleet orchestration still contain
host work; this change does not make the entire application GPU controlled.

## Validation and limits

All **117 tests pass on RTX 5090 and H100**, including ten new cases for changed
geometry/mass/time, zero-order and cubic control, both state origins, retained
replay on/off, failed-lease discard, scaling-count isolation and old-library
fallback. Independent trajectory certification is unchanged. The final
preparation script reproduces the compiled vendor sources byte for byte and
rejects reapplication without partial writes.

Full-solver sanitizer validation is **not clean**. The H100 memory check passes
the changed-geometry/reuse test. H100 synccheck reports divergent warp barriers
inside `cudss::fwd_ker` during the initial fresh reference solve. Local full-solver
instrumentation fails with CUDA error 999 at the iterative-refinement launch on
both the preceding published build and the candidate, before reuse begins.
Excluding cuDSS kernels from local instrumentation does not cure that failure.
No failed or crashed sanitizer run is counted as a pass merely because its
printed error count is zero.

The preceding H100 core and QOCO library reproduce the same divergent-warp
failure with the same 98,016 reported errors before workspace reuse begins.
Racecheck aborts with host heap corruption/segmentation faults during fresh
reference solves on both the preceding and candidate H100 builds. The requested
cuDSS exclusion still reports cuDSS barriers on H100; it has not established
successful isolation or a clean synchronization check.

NVIDIA's [cuDSS release notes](https://docs.nvidia.com/cuda/cudss/release_notes.html)
describe a synccheck issue on Jetson Orin, which does not establish the cause of
the H100 failure here. The failed runs and baseline comparisons are retained.
The initial Lambda build also failed because its frozen source lacked Git
metadata required by CMake; the retry initializes a local source commit before
configuring and successfully builds the tested binaries.

The incumbent fleet remains **12,805.194 weighted kg**. These replay and one-ship
experiments do not constitute a new fleet score or an official submission.

## Retrieved evidence and visualiser

[Full summary, commands, raw archives and checksums](../results/lambda/2026-09-08/gpu-scaled-workspace-v553/summary.json)
include the failed build/instrumentation runs and both GPUs' successful tests,
225-leg replays and complete campaigns. The Lambda archive contains 877 verified
files. The displayed mission comes from H100 v548 candidate0: one ship, eight
mined asteroids, 509 exact replay samples and 3,010 cross-checked Kepler context
points. It is the performance fixture, not the incumbent fleet.

Open the [H100 benchmark mission](http://127.0.0.1:4173/?dataset=gtoc12-v548&epoch=69807&preset=oblique&z=1).
The complete local data directory is:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser\data\gtoc12-v548
```

If the viewer is stopped, launch it with PowerShell:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
node scripts/serve.mjs --port=4173
```

Then load the link above, or select **H100 scaled reuse v548** in Dataset.
