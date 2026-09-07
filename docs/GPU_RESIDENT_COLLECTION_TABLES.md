# Resident collection costs through CUDA DP

**Publication held:** candidate `cc505a20` passes the focused tests below, but one
full campaign misses the 548.255 kg retimed mission after an Earth-departure
refinement failure. Another resident run succeeds. The [downloaded investigation](../results/lambda/2026-09-08/gpu-collect-resident-v298/README.md)
retains both outcomes. No overall speedup or complete-pipeline reliability is
claimed while this discrepancy remains unresolved; published main is unchanged.

Collection pair/return tables now remain as immutable float32 arrays in GPU
memory. The existing orbital-element grid operator builds the costs, and a
device kernel applies the original arrival-window gate and float32 rounding.
A single gather kernel copies the required epoch slices directly into the DP's
owned double buffers. No full cost table needs to pass through the CPU for a
collection-DP solve.

The selected hop and return costs travel with the final 632-byte result, so
Python result formatting does not fetch a full table. The legacy C solve entry
point still writes the original 496-byte result; the larger result uses the
new `spacepdhcg_collect_solve_v2` entry point. Existing clients retain their buffer
contract, while the new Python backend requires a rebuilt library.

## Ownership and remaining host work

`GpuLambert` owns a bounded LRU cache of device-table objects. A DP construction
retains its borrowed objects until native assembly has finished; cache eviction
therefore cannot invalidate an in-flight copy. Tests exercise capacities zero,
one and 20,000. Native creation checks device ownership, grid dimensions and
slice bounds, locks borrowed tables and finishes assembly before releasing them.
Cache release drops entries, and GPU-context shutdown closes remaining objects.

Host callers can explicitly download a table through `CollectPairTable.hop` or
`earth_return`; those consumers still exist in the wider search. Consequently,
the complete campaign is not claimed to have zero downloads. The dedicated
resident-DP tests forbid those methods and the native table-read wrapper and
require zero downloaded table bytes for the tested plans.

Cache lookup/control, small policy inputs, mining/subset mass preparation,
the controller between mass passes, calibrated geometry/phase inputs, certified
return-override preparation and broader beam/fleet orchestration still include
host work. Moving the cost arrays into device memory is a residency step, not
completion of the entire GPU-native application.

## Validation

RTX 5090 and H100 each pass 75 tests. Eight real captured tours match separately
constructed host-ephemeris tables and an uncached CPU DP, preserving the existing
route, epoch and objective tolerances. The zero-/one-entry cache tests compare
selected costs exactly and objectives within 1e-9 kg. A canary after the legacy
496-byte result confirms that the new native library does not overwrite it.

All four H100 Compute Sanitizer tools pass 42 cases with zero errors/hazards;
RTX 5090 memcheck also passes all 42. The initial H100 snapshot failed CMake
configuration because it had no Git metadata. A fresh snapshot with metadata
builds and passes; the failed configuration is retained separately from results.

## Warm complete collection tours

Six alternating pairs per fixture retain the pair tables and discard the first
pair. Both modes use CUDA DP and GPU ephemerides. `host` uses downloaded/packed
tables; `resident` gathers device tables directly. Times include both mass passes
and final result construction.

| GPU / asteroids | Host tables ms | Resident tables ms |
|---|---:|---:|
| RTX 5090 / 3 | 1.047 | 1.101 |
| RTX 5090 / 6 | 3.274 | 3.230 |
| RTX 5090 / 9 | 7.019 | 6.954 |
| H100 / 3 | 0.901 | 0.899 |
| H100 / 6 | 2.070 | 1.898 |
| H100 / 9 | 3.951 | 3.721 |

The nine-asteroid fixture is infeasible. These are proxy-tour timings, not new
certified missions per second. The three-asteroid RTX case is slightly slower;
the residency change is not presented as a uniform microbenchmark speedup.
An exploratory version launched one copy kernel per table and was slower on
warm local cases; the final version combines those copies into one gather.

## Reproduction

With the pinned catalogue, current CUDA library and GPU test environment set:

```sh
python -m pytest tests/test_gtoc12_gpu_resident_collect_tables.py \
  tests/test_gtoc12_gpu_collect_tables.py tests/test_gtoc12_gpu_collect_dp.py -q
python scripts/gpu/replay_gtoc12_collection_fixtures.py \
  results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json \
  build/resident-collection-replay.json
python scripts/gpu/benchmark_gtoc12_collection_dp.py \
  results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json \
  build/resident-collection-benchmark.json --resident-tables
```
