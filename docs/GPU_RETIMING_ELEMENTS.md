# GPU ephemerides for retiming tables

This describes the v215 endpoint-table checkpoint. The subsequent
[CUDA dynamic-programming port](GPU_RETIMING_DP.md) removes the CPU recurrence
described below; the measurements here retain their original scope.

Retiming previously propagated both endpoint bodies on the CPU and packed a
160-byte rendezvous request for every departure/duration combination. CUDA then
performed Lambert screening. The initial profile spent 0.824 of 0.869 seconds
in transfer-table construction; the dynamic-programming recurrence was small.

`spacepdhcg_orbitweaver_hop_elements_host` now uploads a pair of elliptic element
descriptors plus 16 bytes of epoch/duration per combination. CUDA solves Kepler's
equation, generates position and velocity, and assembles requests directly in the
existing retained workspace. The same Lambert kernel evaluates both directions,
applies Earth velocity allowances and chooses the lower cost. No extra device
allocation or CPU ephemeris fallback is introduced. The ordinary request buffer
provides temporary time storage under the workspace's existing exclusive lock.

`Retimer.leg_table` dispatches to this path inside an explicit CUDA screening
scope and caches the resulting table. The NumPy scope remains the reference.
Completed branch telemetry includes both directions exactly once; additional
`completed_element_hops` telemetry identifies the new path. Catalogue lookup,
small epoch-array packing, table downloads, dynamic programming, mass replay and
fleet orchestration remain on the CPU. This is a further GPU port, not a claim
that the whole application is GPU-native.

## Measurements

The fixed-order six-asteroid v209 route requires 412,116 Lambert branches during
retiming. Before/after paths use the same new native library; the baseline keeps
CPU endpoint propagation and ordinary CUDA hop screening. Six alternating-order
pairs are recorded. The table reports the median of the final five observations
per path, excluding initial runtime warm-up. Each observation constructs a fresh
Retimer and transfers its tables; this is not a cached-table lookup benchmark.

| GPU | CPU endpoints + CUDA screening | CUDA endpoints + screening | Speedup |
|---|---:|---:|---:|
| RTX 5090 | 0.658885 s | 0.256746 s | 2.57× |
| H100 80 GB | 0.364443 s | 0.145273 s | 2.51× |

Every observation chooses exactly the same body sequence, departure and arrival
epochs, and 524.0246406570842 kg planning yield. Ordinary SCvx refinement is a
separate cost. The first complete H100 certification takes 8.139 seconds for
13 arcs; a second certification with the new endpoint path takes 9.373 seconds.
Both pass; these single observations do not establish a full-pipeline speedup.

## Physics and reliability

46 local tests and 30 targeted H100 tests pass. The seven new tests include
Earth departures/returns, circular, inclined, high-eccentricity and retrograde
orbits, epochs before/after the element epoch, partial batches, invalid times,
workspace reuse, and a guard that rejects any CPU endpoint call during CUDA
retiming. Comparison uses CPU ephemerides with the existing CUDA Lambert operator;
existing Lambert parity tests cover the numerical screening solver separately.
H100 Compute Sanitizer memcheck and initcheck each run all seven new cases and
report zero errors. Solver tolerances, thrust limits and independent gates are
unchanged.

The emitted 524.025 kg mission passes the official checker and the independent
verifier before and after the port. The second run's maximum verifier discrepancy
is 0.565416 km in position and 9.8742e-8 km/s in velocity, accepted by the existing
mission verifier's rules. No discrepancy threshold was changed. This is one ship
visiting six asteroids, not an addition to the incumbent fleet or a new leaderboard
score.

The separate stalled hop 45738→25792, departing MJD 65428 with 2172.736594 kg,
reaches the same saturated-thrust plateau on CPU and GPU at 180 days, with about
69.79 m/s arrival velocity error. That is evidence of a poor schedule, not a proof
of global infeasibility. H100 repeats at 185, 190 and 200 days all pass independent
leg certification. The 185-day cases take 0.704 and 0.750 seconds. Route timing
must change at the planner level so downstream epochs and masses are rechecked.

## Evidence and reproduction

- [Local timings, H100 build/tests/timings, sanitizer logs and second mission](../results/lambda/2026-09-08/gpu-elements-v215/).
- [First certified improvement and the failed-hop timing diagnostics](../results/lambda/2026-09-08/gpu-retiming-v213/).
- `benchmark.py` in the measurement directory accepts an output directory and
  the v209 `ship_01/refinements.json`. Set `PYTHONPATH=src`, the pinned
  `SPACEPDHCG_GTOC12_DATA`, and `SPACEPDHCG_GTOC12_CUDA_LIBRARY` to a library built
  from this change. It compares both paths, asserts exact schedule agreement,
  and records runtime and input hashes.
- `SPACEPDHCG_GTOC12_GPU_TESTS=1 python -m pytest tests/test_gtoc12_gpu_elements.py`
  runs the new regression. All CUDA tests require exclusive use of the GPU.

The H100 source checksum file matches all six changed numerical/API/test files
in this checkout. Build snapshots have temporary Git identities; these are not
GitHub revisions. Raw artifacts and an evidence digest manifest are retained.
The initial configuration failure is retained and is excluded from timing
and validation claims.
