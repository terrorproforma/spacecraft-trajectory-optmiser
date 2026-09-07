# CUDA schedule dynamic programming

After moving endpoint propagation to CUDA, repeated price and mass-profile
iterations still allocated thousands of temporary NumPy arrays to choose camp
durations and flight times. `gtoc12_retime.cu` now performs those choices and all
per-cell authority, inflation and rocket-equation pricing on the GPU.

Each camp-layer thread handles one departure epoch; each leg-layer thread handles
one arrival epoch. Stages run in dependency order on a retained CUDA stream.
Only final scalar selection and the short path reconstruction use one thread.
The recurrence preserves strict `>` comparisons, ascending camp/TOF traversal,
earliest final-epoch ties, foreign-miner dates, pinned arrivals, bonus weights,
orphan credits, calibrated inflation and measured return-sweep overrides.
This compilation unit disables FMA contraction to retain the reference's
arithmetic order. Physical models, solver tolerances and acceptance checks are
unchanged.

The workspace retains an immutable table snapshot for the current visit order.
Price/mass changes upload only the small stage parameter array. Table replacement,
return-sweep replacement, visit-order changes and cache release cause a fresh
snapshot on the next call; no stale device pointer survives replacement. The
Python owner retains the source arrays while their identities serve as cache keys.
Changing contents of those internal cached arrays in place is not a supported
update: use the existing retimer cache invalidation/update methods.

`GpuLambert.retime_dp` selects this implementation within an explicit CUDA scope.
CPU scopes keep the NumPy reference. Unsupported/missing CUDA symbols fail
explicitly; there is no silent CPU fallback. Telemetry records completed DP calls,
table uploads and GPU use even when tables were already cached on the CPU.

## Measured performance

The fixture is the six-asteroid v209 route, requiring 412,116 Lambert branch
evaluations to build its retiming tables. Before/after use the same native
library and GPU endpoint-table path; only the DP dispatch changes.

| Workload | GPU | CPU DP | CUDA DP | Speedup |
|---|---|---:|---:|---:|
| Full retiming, including fresh tables | H100 | 144.939 ms | 72.335 ms | 2.00× |
| Full retiming, including fresh tables | RTX 5090 | 258.250 ms | 234.317 ms | 1.10× |
| One price evaluation, tables cached | H100 | 12.687 ms | 0.841 ms | 15.08× |
| One price evaluation, tables cached | RTX 5090 | 5.409 ms | 1.342 ms | 4.03× |

Full retiming uses six alternating-order pairs, reporting medians of the final
five observations per path after initial runtime warm-up. The cached benchmark
uses 30 alternating-order pairs across six prices, with one immutable table
upload and 31 total GPU calls including warm-up. It asserts identical arrival
and departure indices and objective agreement within 1e-9 kg. All full retiming
observations choose the same schedule and 524.024640657 kg planning yield.
The differing CPU/GPU balance on each host explains why an individual component
speedup does not translate into the same whole-retimer improvement.

## Validation

- 82 local tests and 75 H100 tests pass, covering the new DP and existing
  screening, retiming, return models and cooperative planning.
- Twelve DP parity cases cover synthetic random costs, bonus weights, orphan
  credit, foreign mining dates, zero-price ties, pinned/off-lattice arrivals,
  infeasible tables, cache replacement, changing mass and ratio limits, flat
  models and measured return overrides. H100 memcheck and initcheck each report
  zero errors. A final telemetry regression reruns all twelve cases on both GPUs.
- H100 re-flies all 13 legs of the selected mission in 7.726 seconds and passes
  both official and independent checks at 524.025 kg. The maximum independent
  position discrepancy is 0.565503 km and velocity discrepancy 9.8746e-8 km/s,
  accepted by the unchanged mission verifier. This is a repeat validation, not
  a new fleet score or a controlled SCvx speedup measurement.

## Fleet hill climbing

Follow-up: [the complete 23-ship scan and objective fix](GTOC12_CERTIFIED_OBJECTIVE.md)
recover the eight missing sources and pair the cluster timing settings with their
cluster search settings. The two passes below remain the original exploratory
measurements, not the latest fleet coverage.

Two initial retiming passes inspected all 23 incumbent ship labels. Fifteen had
exact-mass matching local source archives; eight were skipped pending source
reconciliation. The first pass used default timing settings and the second used
the existing cluster timing settings, including its established Earth-departure
authority envelope. The second pass finishes in 5.05 seconds and produces seven
proxy schedules, but none improves weighted yield while preserving raw haul.
No candidate is substituted into the fleet and no score increase is claimed.
These results are not a proof that the ships cannot improve: finer scheduling,
route-order changes and archived source reconciliation remain work to do.

The incumbent stays at **12,805.194 weighted kg**. The 524 kg independent ship
visits different asteroids, but is below the incumbent's lightest ship at
570.760 kg; its addition is also constrained by the fleet's ship-count rule.

## Remaining CPU work and reproduction

The [resident-table follow-up](GPU_RESIDENT_RETIMING_TABLES.md) now removes the
host table round trip for ordinary fixed-order retiming without return sweeps.
The paragraph below describes the original v220 implementation; custom tables
and return sweeps still use that host-table path.

Price bisection, forward mass-profile bookkeeping, route-order selection,
stage-descriptor preparation, table packing and global fleet optimisation remain
on CPU. Device-generated transfer tables currently cross the host boundary
before being retained by the DP workspace. Removing that round trip and batching
independent visit orders are further GPU-native optimisation opportunities.

All measurements, runtime/source hashes, test/sanitizer logs, the repeated
certified solution and negative fleet-search reports are in
[gpu-retime-dp-v220](../results/lambda/2026-09-08/gpu-retime-dp-v220/).
`benchmark_retime_dp_v219.py` and `benchmark_cached_dp_v219.py` take an output
directory followed by the v209 `ship_01/refinements.json`. Run from the repository
root with `PYTHONPATH=src`, the pinned `SPACEPDHCG_GTOC12_DATA`, and a newly built
`SPACEPDHCG_GTOC12_CUDA_LIBRARY`. The scripts serialize GPU use through the existing
lock and compare the CPU/GPU paths without changing physics settings.

The CUDA regression command is:

```sh
SPACEPDHCG_GTOC12_GPU_TESTS=1 python -m pytest tests/test_gtoc12_gpu_retime.py -q
```

Build snapshots carry temporary Git identities. `source-sha256.json` identifies
the final eight numerical/API/test files independently of those identities.
