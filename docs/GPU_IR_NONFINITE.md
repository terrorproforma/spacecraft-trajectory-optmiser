# Preserve finite corrections during GPU iterative refinement

The refinement controller previously treated a NaN residual as an improvement:
`new_norm >= best_norm` is false for NaN, so it accepted the correction and
overwrote the finite backup. Subsequent attempts could continue with invalid
directions until the iteration budget was exhausted.

Device refinement now rejects nonfinite correction norms and restores the saved
solution. A nonfinite initial norm or tolerance does not start refinement. This
does not mark the linear solve converged or bypass any outer qualification gate.
The preparation script applies matching guards to the diagnostic host-controlled
refinement path. The original QP, physics and objective tolerances are unchanged.

The original controller fails the new regression. Both RTX 5090 and H100 pass
14 controller cases, 9 norm cases, all four controller sanitizers, and 38 CLI and
trajectory checks. Eight additional cases run the actual kernels inside a CUDA
conditional graph on each GPU; H100 graph memcheck also passes. These sanitizer
claims cover the probes, not every vendor kernel.

Four balanced complete H100 campaigns compare only the guarded QOCO library:

| Run | Guard | Complete seconds | Verified kg |
|---|---|---:|---:|
| v339 | enabled | 53.293 | 548.255 |
| v340 | previous controller | 58.885 | 548.255 |
| v341 | previous controller | 59.477 | 548.255 |
| v342 | enabled | 52.865 | 548.255 |

Median complete time falls **59.181 -> 53.079 seconds (10.31% less time)**.
Both checkers accept all four missions. This is a small comparison of one fixed
campaign, not a universal speedup. The separate fleet score remains unchanged.

Exact difficult-QP graph replays remain mixed: RTX qualification changes from
27/32 to 30/32; H100 changes from 28/32 to 27/32. These samples do not establish
a reliability improvement. This fixes a demonstrated controller bug, but the
broader numerical repeatability issue remains open. The candidate is on the
development branch and has not been merged into main.

[Evidence and rejected arithmetic experiment](../results/lambda/2026-09-08/gpu-nonfinite-ir-v337/README.md).
