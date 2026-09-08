# GPU initial Earth beam

The initial Earth-to-asteroid beam now constructs endpoint states, solves Lambert
transfers, applies the thrust-authority screen, computes rocket-equation
propellant and mining scores, and ranks candidates in C++/CUDA. It returns only
the selected rows. The previous path built Cartesian endpoint states on the CPU,
downloaded the detailed screening grid, and used NumPy to score and sort it.

The CUDA path preserves the existing search policy: each asteroid block retains
its stable descending-score top-k, then their union is ordered by descending
score, asteroid ID, departure epoch and flight duration. Exact remaining ties
retain original row order. Cluster density and seeded-miner bonuses use the same
formula and operation order. Double-precision arithmetic, existing orbital
propagation and Lambert branch/root algorithms are retained. There is no relaxed
trajectory acceptance tolerance.

`SPACEPDHCG_TEST_GTOC12_EARTH_BEAM=0` selects the host baseline; CUDA screening
enables the new path by default. CPU screening and injected certified first-leg
search retain their existing paths. Python still prepares catalogue metadata,
filters the asteroid pool, identifies seeded neighbours, constructs route objects
and orchestrates later search. This does not complete GPU control of the planner.

## Measurements

An isolated measurement invokes the actual initial-beam method from the
full-catalogue one-ship campaign. After separate warmup calls, four observations
per mode preserve all 582 selected candidate identities and their ordering.
Proxy costs agree within the same numerical bounds as the ephemeris tests.
These timings cover initial-beam preparation and option rows, excluding later
trajectory refinement and mission certification.

| GPU | Host beam median | CUDA beam median | Stage speedup |
|---|---:|---:|---:|
| RTX 5090 | 3.521 s | 1.988 s | 1.77x |
| H100 | 2.656 s | 0.228 s | 11.67x |

The fixture screens 3,448,900 initial transfers after pool selection. Detailed
result downloads fall from **248,320,800 bytes** to **27,944 bytes** for 582
selected rows and two counters. That is 99.989% less output for this stage, not
the entire application's transfer volume. Those transfer requests now also use
GPU-generated ephemerides.

Complete CLI timing uses two baseline/candidate/candidate/baseline batches on
each GPU, four runs per mode combined:

| GPU | Baseline median | CUDA beam median | Less complete-run time |
|---|---:|---:|---:|
| RTX 5090 | 30.604 s | 29.993 s | 2.00% |
| H100 | 34.818 s | 32.250 s | 7.38% |

The ranges overlap. The first local batch was 4.81% slower overall despite faster
search; the second was 6.93% faster. Initial refinement time varies substantially
and explains the difference in those recorded stage totals. These observations
do not identify whether that variation comes from execution variability, tiny
changes to initial proxy values, or both. We retain both batches and do not claim
a universal end-to-end multiplier. Initial plan structure and nonnumeric fields
match exactly; the maximum numeric difference across their proxy fields is
9.10e-13. No failed or unqualified trajectory is accepted to improve timing.

Every full one-ship run retains 45,188,558 logical branch requests, 2,782,091
collection options and **548.254620 weighted kg**, accepted by both mission
checkers. Existing screening counters match except for the expected increase in
GPU-generated element hops; new counters report returned beam rows and bytes.
Logical requests include reused tables elsewhere and are not unique missions.

A wider local confirmation completes in **221.557 seconds**, retaining four
ships, 29 mined asteroids and **2,088.668592 weighted kg** with both checkers
passing. It preserves 169,753,864 logical branches and 18,250,121 collection
options. This is one wider validation, not a paired speed experiment. The best
retained fleet remains **12,805.194 weighted kg**.

## Validation and retained memory

Integration tests cover Earth allowances, weighted and unweighted candidates,
cluster and seed bonuses, duplicate epochs and durations, multiple asteroid block
sizes, partial screening batches, empty and invalid input, and changing workspace
dimensions. They forbid CPU screening and cost calculations in the CUDA path.
Native tests exercise 54 exact scoring and ranking cases, including top-k
truncation before the global sort, exact ties, all-invalid inputs and feasibility
counts. Memory, synchronization and race checks exercise the actual kernels.

The Lambert workspace retains target metadata, epoch/duration axes, per-block
rows, retained block winners and sort scratch. Its telemetry includes these
allocations. Buffers grow when required and are refreshed for changed inputs;
destruction releases them. The public API validates dimensions and finite input,
limits the total grid to UINT_MAX rows and each CUB sort to INT_MAX rows, and
requires output capacity for the requested limit. It is a blocking host-output
API, not a graph-capturable device search controller.

The final counter uses an explicit warp membership ballot before divergence.
This ensures partial-warps count feasible transfers without relying on a snapshot
of currently active threads. It changes telemetry handling, not transfer costs
or candidate ranking.

The zero-limit Python fast path also leaves evaluation counters unchanged when
no screening is performed. Final validation tests that guard explicitly.

The final native build passes 34 selected integration tests and all 54 native
scoring/ranking cases on RTX 5090 and H100, including memory, synchronization and
race checking. The final Python guard is rechecked with the 34-test suite and a
complete default-on campaign on each GPU. All mission acceptance gates retain
their original tolerances.

- [Local comparisons, isolated timing, wider fleet and final validation](../results/lambda/2026-09-08/gpu-earth-beam-local-v443/summary.json)
- [Combined H100 comparisons](../results/lambda/2026-09-08/gpu-earth-beam-v434/summary.json)
- [H100 isolated timing and native tests](../results/lambda/2026-09-08/gpu-earth-beam-v437/summary.json)
- [Second H100 comparison batch](../results/lambda/2026-09-08/gpu-earth-beam-v439/summary.json)
- [Final H100 native build and sanitizer validation](../results/lambda/2026-09-08/gpu-earth-beam-v442/summary.json)
- [Final H100 Python source and complete mission](../results/lambda/2026-09-08/gpu-earth-beam-v444/summary.json)

Archives retain raw reports, commands, source overlays and runtime hashes with
verified SHA-256 manifests. Performance comparisons use the opt-in v431/v434
native builds. Final v441/v442 builds add the explicit counter ballot; v443/v444
check the zero-limit Python guard on those unchanged final native binaries. The
saved local isolated-timing reproducer adds an optional output-path override
after measurement; its default path and all numerical work are unchanged. QOCO
and its numerical qualification thresholds are unchanged throughout.
