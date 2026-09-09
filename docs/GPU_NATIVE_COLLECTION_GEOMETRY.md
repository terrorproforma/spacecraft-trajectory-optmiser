# Native collection geometry and harvest-phase grids

The collection planner now derives orbital mean longitudes, pairwise semi-major
axis gaps, wrapped phase grids and the standard harvest-phase penalty on CUDA.
Paired search throughput improves **5.6–6.9% on H100** and **1.2–2.8% on RTX 5090**.
All candidate files remain byte-identical within each GPU, including comparison
with the preceding published native-mining runtime. The verified fleet score
remains **13,526.961241 weighted kg**; this checkpoint creates no new trajectory.

## Paired whole-search measurements

| GPU | Ship / candidates | Host geometry | Native geometry | Throughput gain |
|---|---|---:|---:|---:|
| H100 | 10 / 517 | 5.587234 s | 5.227055 s | 6.89% |
| H100 | 21 / 472 | 4.939793 s | 4.675890 s | 5.64% |
| RTX 5090 | 10 / 517 | 10.025945 s | 9.907333 s | 1.20% |
| RTX 5090 | 21 / 472 | 8.531645 s | 8.297360 s | 2.82% |

Each median uses three alternating measurements per mode, after one warm-up per
mode. Both modes use the same binary and input fleet, native mining/burn-pass
selection, resident tables, cached return rows and CUDA expansion/admission.
The control uses the preceding Python geometry construction. Timing covers
`RouteSearch.run`, excluding process startup and initial catalogue loading.

Native H100 throughput is approximately **99–101 surrogate candidates/s**; local
throughput is **52–57/s**. These are search candidates, not independently qualified
trajectories or PDHCG solves. The four measured ranges do not overlap, but there
are only two route searches and three measured repetitions per mode. RTX timings
include recorded desktop GPU activity and substantial memory use. The small local
gain is not a general isolated-device guarantee.

## Native implementation

Each plan supplies five orbital elements per selected asteroid: reference epoch,
semi-major axis, node, perihelion argument and mean anomaly. One CUDA kernel
computes each body's motion and longitude at the first requested epoch; another
distributes the pair/epoch grid across GPU blocks. The latter computes the same
wrapped phase, fitted-inflation geometry and weighted harvest-phase penalty.
This removes the host construction and upload of the three pairwise arrays.

The standard model uses the existing solar parameter, day length, astronomical
unit, phase threshold and penalty slope. The arithmetic keeps explicit operation
order and `--fmad=false`. Self/banned pairs remain zero. Flat, ratio and fitted
hop pricing, disabled phase priors, epoch slices and retained workspace updates
are covered. Existing C APIs retain their ABI; new geometry entry points accept
the small metadata block. Invalid metadata is rejected before updating the current
problem. Workspace growth uses the existing capacity/rebuild protocol.

The standard `HarvestPhasePrior` runs on the new path. Arbitrary custom Python
phase policies retain their existing callback semantics. The comparison switch
is `SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_GEOMETRY=0`. The default CUDA path requires
a rebuilt native library matching the Python package.

For the two measured searches, **2,184 geometry plans** upload **504,960 bytes of
orbital-element arrays**. This is only the new element-array traffic, excluding
other metadata, geometry-independent inputs, table traffic and kernel arguments.
Work counts remain **989 candidates, 3,279 DP passes, 51 workspace allocations and
2,133 rebinds**. Mining metadata and final result-readback byte counts are unchanged
from the preceding native-mining checkpoint.

## Verification and remaining CPU work

Both GPUs pass **248 tests**, **110 focused cases under each of memcheck,
racecheck and synccheck**, the native controller under every tool, and **43 full
leak-check cases with zero leaked bytes or allocations**. New tests compare full
geometry and penalty grids within `1e-12`, preserve selected tours, reject invalid
updates without changing the prior result, and prove the native path does not
construct a host phase grid. This grid-comparison tolerance does not change any
physics acceptance threshold. All 16 benchmark candidate files per GPU match their
respective previous hashes exactly.

Geometry for single selected-hop display diagnostics remains on the host, as do
catalogue-index lookup and element packing, table-handle selection, forward
scheduling and beam expansion input packing. The H100 raw profile is retained
to guide the next change. Its cumulative times include nested GPU work and cannot
be added or described as pure CPU time. The complete planner is not yet controlled
by native GPU code.

[Downloaded binaries, raw logs, candidates, profiles and portable audit](../results/lambda/2026-09-09/gpu-native-collection-geometry-v863/README.md)

The visualiser retains the v633 verified fleet. No dynamics tolerances, SCvx
defaults, trajectory controls, official checker results or fleet score changed.
