# Reuse GPU solver workspaces between trajectory legs

The zero-Ruiz CUDA Graph path now retains up to eight compatible QOCO workspaces
per host thread. In matched one-ship campaigns, complete-process median time
falls **5.40% on RTX 5090** and **5.03% on Lambda H100**, with unchanged verified
score. This is not a new fleet record or a fully GPU-controlled application.

## Implementation and the lifetime fix

The preceding [phase profile](GPU_SOLVER_PRIMING.md) found substantial repeated
setup and priming across 47 trajectory calls. The private cache retains sparse
conversion, device storage and vendor graphs. Its structural key includes CUDA
device, interval count, control hold, free endpoint modes and solver tolerance.
Changes to `SPACEPDHCG_` or `QOCO_` environment settings invalidate the cache;
configuration strings remain in memory and are never exported.

Rebinding updates the time grid, boundary states, fuel reference, gravitational
coefficient and mass-flow coefficient. The adapter forces numerical refresh
even though per-leg telemetry starts at zero. It clears accepted warm starts,
validation state and counters; the retained IPM graph initializes numerical and
iteration controls. Only successful SCvx workspaces enter the pool. Failed or
timed-out solves are destroyed, as are least-recently-used evictions.

The first prototype drained stale vendor metadata on the next acquisition.
That used a stream already destroyed by the preceding SCvx call and failed
with `context is destroyed`. The corrected release drains numerical metadata
while the producing stream still exists, after borrowed graphs have closed.
Failed v474/v475 logs and source snapshots are retained.

Reuse defaults on only when native QOCO replay, IPM graph and CUDA outer graph
are active, Ruiz iterations are zero, normal IPM initialization is enabled and
no QOCO snapshot directory is configured. `SPACEPDHCG_TEST_GTOC12_QOCO_POOL=0`
disables it. Scaled workspaces remain fresh: reusing their Ruiz packet is not
qualified. The earlier early-graph experiment remains disabled by default.

## Complete campaign measurements

Each same-binary comparison runs baseline/candidate/candidate/baseline with
pool off/on. The full-catalogue fixture uses one ship, beam width 16, 48
neighbours, three refinement candidates, four retiming attempts, two-day nodes
and at most 40 SCvx iterations. Two observations per mode establish measurements
for this workload, not a hardware-wide speed guarantee.

| Hardware / eight-entry pool | Fresh median | Reused median | Less time |
|---|---:|---:|---:|
| RTX 5090, complete process | 30.1515 s | 28.5219 s | 5.40% |
| H100, complete process | 32.2837 s | 30.6586 s | 5.03% |
| RTX 5090, CLI interval | 29.2856 s | 27.5889 s | 5.79% |
| H100, CLI interval | 31.3697 s | 29.8082 s | 4.98% |

Process time includes exit cleanup and diagnostic export. Initial search plans
and screening counts match exactly: 45,188,558 transfer branches and 2,782,091
collection options per run. Every run returns 548.254620 weighted kg and passes
both final mission checkers. Logical counts include cache hits; they do not
count unique fresh GPU computations.

Locally, workspace creations fall from 47 to 17 and priming solves from 141 to
111. Recorded structural keys predict 30 hits with eight entries, with no gain
from larger capacities in this fixture. Four entries gave only 12 hits and a
flat/slower local CLI median (29.0239 to 29.2092 s). Eight is a count bound, not
a measured GPU-memory ceiling.

H100 v478/v484 campaign labels were incorrect: the generator disabled early
graph entry for all runs and never enabled the pool. They are retained as
**baseline variability only**, excluded from speedup comparisons. Their explicit
pool test matrices remain valid. Corrected v487 toggles pooling and checks actual
workspace creation counts. Separate replay runners set the pool correctly.

## Fixed 225-leg replay

| Hardware / capacity | Fresh solver time | Reused solver time | Less time |
|---|---:|---:|---:|
| RTX 5090 / four | 135.3602 s | 123.0386 s | 9.10% |
| H100 / four | 165.6360 s | 150.9701 s | 8.85% |
| RTX 5090 / eight | 136.3887 s | 124.8857 s | 8.43% |
| H100 / eight | 170.4875 s | 155.4330 s | 8.83% |

These cumulative solve-call times exclude surrounding replay/certification
overhead. All comparisons preserve all **205 independently certified legs**.
The other 20 remain unsuccessful; their distribution between `failed` and
`infeasible` varies, and neither label proves global infeasibility. Maximum
certified returned-mass differences with eight entries are 6.996e-7 kg locally
and 1.030e-7 kg on H100. All physics and objective tolerances are unchanged.

## Final validation and remaining limits

Final default-enabled v490/v491 builds pass the 54-test refinement, CLI and final
verification suite on both GPUs. Pool tests alternate geometry, duration and
initial mass under zero-order/Lagrange holds, origin off/on and Ruiz 0/5, checking
physics certificates, mass, counter resets, timeout discard and disabling.
A separate test exercises default enablement.

The native probe compares 32 fresh/rebound assembly pairs bitwise while varying
holds, free endpoint modes, time grids, physics coefficients, boundary and fuel.
Invalid fuel must leave the preceding valid assembly unchanged. Memcheck,
synccheck and racecheck pass on both GPUs. This covers rebind/assembly, not the
entire vendor cuDSS/IPM graph.

A wider local confirmation takes 210.8269 seconds and evaluates 169,753,864
branches and 18,250,121 collection options. Four ships collect from 29 asteroids
and retain 2,088.668592 weighted kg, passing both mission checkers. Independent
maximum position error is 1.27505 km and velocity error 2.10933e-7 km/s. This
confirms an existing fleet, not a new search-quality record.

Outer SCvx graphs and other buffers still rebuild per leg. Vendor metadata
materialization still transfers a small packet to the host before retention;
adapter counters exclude opaque vendor-internal copies. Python still
orchestrates beam/fleet work. Exact-QP qualification outliers remain.

## Reproduce and inspect

Reports retain exact commands, stage logs and source/binary hashes. Recipes need
the pinned QOCO SOC-step library, CUDA/cuDSS and GTOC12 catalogue; replace original
machine paths for your environment. Final v490/v491 leave the pool flag unset;
comparison runners use explicit 0/1 with the same binary.

- [Local experiments, source snapshots and recipes](../results/lambda/2026-09-08/gpu-workspace-pool-local-v490/summary.json)
- [H100 corrected campaign](../results/lambda/2026-09-08/gpu-workspace-pool-v487/analysis.json)
- [H100 eight-entry replay](../results/lambda/2026-09-08/gpu-workspace-pool-replay-v486/analysis.json)
- [H100 final default build](../results/lambda/2026-09-08/gpu-workspace-pool-v491/report.json)

Published folders have SHA-256 manifests. Retrieved archives have an additional
manifest covering each member. The larger displayed fleet remains the
[v381 dataset](GPU_FLEET_RECOVERY.md#display-and-reproduce); the best fleet and
official-submission status are unchanged.
