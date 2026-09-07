# Resident GPU retiming tables — 8 September 2026

Ordinary fixed-order retiming now constructs transfer tables inside the retained
CUDA DP workspace. Previously CUDA ephemeris and Lambert screening downloaded
every hop result, Python extracted costs and feasibility, and DP construction
uploaded those tables again. The new path transfers only orbital elements,
epochs, TOFs and stage descriptors, retaining numerical tables on the device.

## Implementation and scope

`spacepdhcg_orbitweaver_hop_grid_device` builds epoch/TOF pairs on CUDA, computes
endpoint states with the existing Kepler operator, evaluates the existing
short/long Lambert branches and stores scalar costs/flags directly into the DP
workspace. It reuses the bounded Lambert scratch allocation across chunks. A
stream completion barrier at the end of each table establishes readiness for
the DP workspace's stream; increasing the chunk count does not allocate more
scratch. These are blocking native bridges, not a graph-captured full pipeline.

`spacepdhcg_gtoc12_retime_create_elements` owns the immutable device tables.
`spacepdhcg_gtoc12_retime_path_host` returns one selected delta-v per leg alongside
the existing schedule and objective. Forward mass bookkeeping uses those costs
without requesting full host tables. Price/mass changes reuse the device tables;
cache release invalidates both the table generation and selected-path cache.
The selected costs are scoped to the retimer, visit order and exact schedule.

The path is automatic for an ordinary `Retimer` with no host tables or return
sweeps. Explicit/custom tables, CPU reference evaluation and return sweeps retain
the existing host-table implementation. This is not a CPU solver fallback: both
CUDA table paths use the CUDA ephemeris/Lambert and DP operators. Host fallback
here means table storage/packing, not acceptance of different physics.

Old C entry points remain available. Updated Python requires a rebuilt CUDA
library providing the new entries. No numerical tolerance or feasibility policy
changes. The small final path reconstruction also gathers the selected costs;
all interval/grid work remains distributed across blocks.

## Measurements

Both paths use the same rebuilt library, input mission, CUDA operators and
Python driver. Six alternating pairs construct fresh workspaces and transfer
tables; medians exclude the first pair. Timing covers complete retiming, including
native workspace preparation and Python price/mass iteration, not refinement.

| GPU | Previous host tables | Resident tables | Speedup |
|---|---:|---:|---:|
| H100 80 GB | 71.972 ms | 63.788 ms | 1.128× |
| RTX 5090 | 234.060 ms | 223.828 ms | 1.046× |

Every pair selects identical schedules and the same 524.024640657084 kg proxy
payload. The fixture evaluates **412,116 Lambert branches**, comprising 206,058
hop/table cells in 24 bounded batches. The resident path constructs those tables
once, performs six DP calls and reports **zero host table uploads**. These modest
whole-retiming gains do not imply a similar speedup for all mission solving, and
the cached-price speedups in the earlier DP report are separate measurements.

Telemetry counts completed branches after native construction succeeds. The
existing `retime_table_uploads` counts host snapshots only; new counters
`retime_resident_builds` and `retime_resident_cells` identify resident construction.
Table contents are not downloaded merely to count feasible branches.

## Accuracy and validation

- Four new tests compare real 15- and 30-day grids with CPU DP and independent
  host-table forward bookkeeping, including exact selected transfer costs.
  They check cache reuse/release, changed Earth TOF floors, explicit infeasible
  custom tables, and rejection of stale paths after an invalid pinned visit.
- Existing custom-table GPU tests retain coverage of weighting, cooperative
  visits, return overrides, ties, mass changes and infeasibility.
- The selected suite passes **84 tests on RTX 5090 in 25.06 s** and **84 on H100
  in 40.65 s**. H100 memcheck and initcheck each run 21 targeted cases with zero
  errors. A final Python cache-invalidation guard is then checked by all 21
  targeted cases on H100 (1.58 s); the final full local suite includes that guard.
- H100 v232 re-flies all 13 legs of the six-asteroid mission in **8.290 s**.
  The official checker accepts **524.025 kg** and the independent verifier
  accepts **524.0246406572 kg**, with no violations. Maximum replay discrepancies
  are 0.565716 km position, 9.87942e-8 km/s velocity and 4.75e-11 kg mass under the
  unchanged mission checks. This is a validation repeat, not a controlled
  refinement-speed improvement.

The incumbent remains **12,805.194 weighted kg**. No fleet replacement follows
from this repeat. The existing visualiser's 524 kg dataset remains the earlier
certified run with its original provenance; this repeat is archived separately.

## Reproduction and remaining work

[gpu-resident-retime-v231](../results/lambda/2026-09-08/gpu-resident-retime-v231/)
contains 25 hashed evidence artifacts, ten source hashes, native runtime hashes,
build/test/sanitizer logs, alternating measurements, the input, and the new full
mission solution. Configure `PYTHONPATH=src`, `SPACEPDHCG_GTOC12_DATA` to the pinned
catalogue and `SPACEPDHCG_GTOC12_CUDA_LIBRARY` to the rebuilt native library. From
the repository root, with the project's Python dependencies installed:

```sh
python results/lambda/2026-09-08/gpu-resident-retime-v231/benchmark_resident_v230.py \
  build/performance/resident-replay \
  results/lambda/2026-09-08/gpu-resident-retime-v231/input-refinements.json
```

The script takes the per-device lock and toggles `GpuLambert.resident_retime_tables`
to compare both table paths. Set `SPACEPDHCG_GTOC12_GPU_TESTS=1` for the GPU tests;
run them under the same lock. H100 build/configuration commands are recorded in
`h100/validation/report.json`; the complete mission runner is in `h100/mission-v232`.

Price bisection, forward mass bookkeeping, visit-order orchestration and global
fleet selection remain CPU work. Return sweeps and custom tables still cross the
host boundary. Eliminating those paths, reducing per-stage launch/barrier overhead
and batching independent visit orders are further GPU pipeline work.
