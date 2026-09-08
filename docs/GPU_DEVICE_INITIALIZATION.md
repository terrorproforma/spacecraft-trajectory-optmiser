# Device numerical initialization

The native trajectory backend now initializes QOCO numerical values directly
from GPU conversion buffers. Previously, a fresh workspace downloaded its first
matrices, objective and affine offsets, converted values on the CPU, and uploaded
them again. The new default retains these values on CUDA and removes one host
priming dispatch per fresh workspace.

This advances GPU residency without an established overall speedup. Symbolic
topology, bounds classification, vendor setup and route/fleet orchestration still
involve the CPU. The entire application is not yet GPU controlled.

## Implementation

The adapter constructs the symbolic pattern with zero-valued numerical
placeholders, retaining every sparse entry and conversion map. After vendor
allocation, a zero-Ruiz device update seeds the actual coefficients before any
configured Ruiz passes inspect the objective. The GPU audit also receives the
real device values. No placeholder problem is solved. Verbose and comparison
diagnostics can still request host values.

This defaults on only for the backend requiring device extensions. Set
`SPACEPDHCG_TEST_QOCO_DEVICE_INITIALIZATION=0` to restore the previous path.
Generic adapter clients retain their existing behavior. Production still uses
zero Ruiz; objective-preserving scaling and scaled pooling remain opt-in.

## Measurements

The first fresh solve in the fixed local campaign downloads **515,324 bytes in
13 copies**, down from **1,324,772 bytes in 19 copies**: 61.10% fewer bytes. These
are first-fresh-solve adapter counters, not sums of lifetime counters across
reused legs. One additional device numerical update initializes the workspace.

The original 225-leg fixture uses identical binaries per comparison, production
zero-Ruiz pooling, unchanged budgets and unchanged physics tolerances. Each mode
retains the same **205 independently certified legs**, creates 72 workspaces and
rejects the other 20 attempts. Priming dispatches fall from **522 to 450**.

| Hardware | Previous initialization | Device initialization | Observed solver time |
|---|---:|---:|---:|
| RTX 5090 | 120.5156 s | 124.1027 s | 2.98% more |
| H100 | 157.6918 s | 156.9875 s | 0.45% less |

Maximum independently replayed final-mass differences are 9.972e-8 kg locally
and 4.439e-8 kg on H100. Some convergence histories change; case 12 takes more
iterations in both candidate runs. These are single passes per mode and do not
isolate setup time from convergence variation.

Complete campaigns alternate baseline/candidate/candidate/baseline on each GPU.
All eight retain the same initial plans and logical search counts, attempt 47
legs, qualify 46, create 17 workspaces and pass both mission checkers.

| Hardware | Previous process median | Device process median | Observed change |
|---|---:|---:|---:|
| RTX 5090 | 27.8638 s | 27.6364 s | 0.82% less time |
| H100 | 29.0532 s | 29.2794 s | 0.78% more time |

Two observations per mode with overlapping timing ranges do not establish an
overall speedup. The default changes to remove numerical CPU round trips while
preserving verified results. The benchmark score remains **548.254620 weighted
kg** for one ship; the fleet incumbent remains **12,805.194102 weighted kg**.

## Validation and provenance

Final builds pass **65 tests on each GPU**, including explicit old initialization,
explicit device initialization and the new default. Tests cover both thrust
holds, shifted/unshifted states, zero Ruiz and two scaling policies. The selected
H100 initialization test passes Compute Sanitizer memcheck with zero errors.
Previously documented full-solver synchronization/race sanitizer failures remain
unresolved; this is not a claim of a clean full sanitizer suite.

Earlier broad runs stopped on obsolete numerical-update and priming-count
assertions. Those expectations now match the new operations; physics and
objective checks were not weakened. The first 111 passing tests plus the later
45-test run cover 142 unique tests, with 14 overlaps, on each GPU. Raw stopped
runs are retained.

Full replays and campaigns use explicit device initialization on experiment
builds. Final builds change only the flag default and verbose diagnostic guard;
the exact diff is archived. The 65-test final runs validate the final source
bytes. Binary/source hashes, original fixture, trajectories, commands and logs
are retained in the evidence bundle:

- [Summary](../results/lambda/2026-09-08/gpu-device-init-workspace-v566/summary.json)
- [Retrieved H100 archive](../results/lambda/2026-09-08/gpu-device-init-workspace-v566/lambda-raw.tar.gz)
- [Local archive](../results/lambda/2026-09-08/gpu-device-init-workspace-v566/local-raw.tar.gz)
- [Published file hashes](../results/lambda/2026-09-08/gpu-device-init-workspace-v566/files-sha256.json)
- [Earlier sanitizer investigation](GPU_SCALED_WORKSPACE_REUSE.md)

The existing `gtoc12-v548` visualizer dataset remains a separately verified
mission from the preceding tranche. It is not relabeled as this initialization
benchmark. The new evidence contains complete campaign trajectories for inspection.
