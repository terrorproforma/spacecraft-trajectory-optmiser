# Cooperative Lambert screening scan

Status: validated on the RTX 5090 and Lambda H100 with complete mission checks,
matching screening outputs, and native graph sanitizer coverage.

## Measured bottleneck

Direct wall timers around the complete one-ship campaign measured 24.302 seconds
inside 2,999 native hop-screening calls, including synchronization. Most calls
contained 451, 480 or 533 transfers. The existing one-thread-per-transfer kernel
launched only eight or nine blocks for these batches. Each thread scanned the
root-bracketing grid serially before bisecting the first root.

The initial cProfile capture contains an impossible 158.48-second self time in a
SciPy callback within an 81.59-second process. Its raw output is retained but is
not used for timing claims. Independent direct timers measured fleet-master
selection at 0.0044 seconds and confirmed the screening bottleneck.

## Change and accuracy boundary

A warp of 32 threads evaluates successive tiles of the existing scan grid for
one transfer. The kernel selects candidate brackets in original grid order and
uses the existing bisection routine. Invalid grid samples break adjacency as
before. A failed bisection allows the next candidate bracket to be considered.
Short/long-direction choice, cost ties, invalid outputs and iteration/tolerance
limits retain their previous definitions.

Four transfers occupy a 128-thread block. Batches through 16,384 transfers use
this cooperative scan; larger batches retain the original kernel. This applies
to the blocking host bridge, orbital-element/grid bridges and the device API,
including its graph-capture path. It changes GPU work distribution without
changing the physical model or substituting a different root solver.

## Local measurements

Paired microbenchmarks used baseline/candidate/candidate/baseline process order,
two excluded warmups and ten timed calls per batch per process. Across six scan
sizes (16–8,192 points) and twelve batch sizes (1–16,384 transfers), all output
fields matched exactly, treating NaNs as equal, in 216 comparisons. Inputs
include invalid flights, invalid body states and degenerate geometry.

| Batch size | Previous median | Cooperative median | Ratio |
|---|---:|---:|---:|
| 451 | 10.430 ms | 0.531 ms | 19.62× |
| 480 | 10.435 ms | 0.537 ms | 19.42× |
| 533 | 10.457 ms | 0.537 ms | 19.47× |
| 4,096 | 11.035 ms | 2.829 ms | 3.90× |
| 16,384 | 11.840 ms | 9.809 ms | 1.21× |

The complete one-ship CLI comparison retained identical search counts:
45,188,558 transfer branches and 2,782,091 collection options in every run.

| Run | Complete CLI seconds | Verified weighted kg |
|---|---:|---:|
| Baseline 0 | 76.038507 | 548.254620 |
| Candidate 0 | 42.288813 | 548.254620 |
| Candidate 1 | 43.058669 | 548.254620 |
| Baseline 1 | 74.739648 | 548.254620 |

Median runtime fell from **75.389077 to 42.673741 seconds**: **43.40% less time**,
or **1.77× faster**, for this fixture and GPU. Both mission checkers passed for
every run. Two samples per mode do not establish a universal speedup.

The subsequent four-ship comparison completed in **287.109863 seconds** with
the cooperative scan versus **469.526522 seconds** with the original kernel:
**1.64× faster**, or **38.85% less time**. Both searched 169,753,864 transfer
branches and 18,250,121 collection options, and returned the same four-ship,
29-asteroid fleet at 2,088.668592 weighted kg. Both final mission checkers passed.
This comparison has one run per mode, candidate first, so it provides a wider
fixture check rather than a statistical estimate across arbitrary missions.
[Four-ship reports, source snapshot and reproduction recipes](../results/lambda/2026-09-08/gpu-warp-fleet-v389/summary.json)
are archived with per-file checksums.

Validation also includes 120 passing tests, independent Kepler closure for
cooperative batches, eight hop tests under both memory and synchronization
checking, and the native device-graph probe under memory, synchronization and
race checking. The native probe compares cached host screening with uncached
device graph replay, including changed inputs and equal-cost direction ties.

[Raw evidence and checksums](../results/lambda/2026-09-08/gpu-warp-hops-v386/summary.json)
include both threshold experiments, complete campaign reports, source snapshots,
binary hashes, profiler captures and reproduction recipes. Each archive has a
manifest of the exact bytes of every member. CPU orchestration and independent
mission checking still remain outside the GPU; the fleet incumbent is unchanged.

## H100 validation

The H100 comparison uses the frozen v381 baseline source plus the archived warp
overlay, built for SM90. All 120 selected tests passed. Eight Python hop tests
passed both memory and synchronization checking; the separate uncached native
graph probe passed memory, synchronization and race checking. All fields match
exactly, treating NaNs as equal, in the same 216 microbenchmark comparisons.

| Run | Complete CLI seconds | Verified weighted kg |
|---|---:|---:|
| Baseline 0 | 53.594064 | 548.254620 |
| Candidate 0 | 44.063453 | 548.254620 |
| Candidate 1 | 43.943981 | 548.254620 |
| Baseline 1 | 53.400079 | 548.254620 |

Median runtime fell from **53.497072 to 44.003717 seconds**: **17.75% less time**,
or **1.22× faster**. All four campaigns have identical search counts and pass
both final mission checkers. This is two samples per mode on the named fixture,
not a universal cross-hardware estimate.
[H100 raw outputs, overlay, native probe and checksums](../results/lambda/2026-09-08/gpu-warp-hops-v387/summary.json)
are retained separately from the local measurements.

## Remaining costs after promotion

A fresh instrumented RTX 5090 campaign using the promoted source passed both
final mission checkers and the new final-status gate. Direct wall timers measured
3.561 seconds in 2,999 native hop calls, compared with 24.302 seconds in the
earlier diagnostic. The full profiled process took 43.275 seconds. The remaining
larger measured costs are 14.690 seconds across 47 native SCvx calls and 7.477
seconds across 50 retiming calls. Independent mission verification took 3.324
seconds across four calls; fleet-master selection took 0.0045 seconds.

These are inclusive, potentially nested diagnostic timers and must not be added
together or substituted for the paired benchmark medians. They shift the next
performance investigation toward SCvx and retiming.
[Profile, complete mission output, runtime hashes and reproduction commands](../results/lambda/2026-09-08/gpu-pipeline-profile-v392/timers.json)
are archived separately.
