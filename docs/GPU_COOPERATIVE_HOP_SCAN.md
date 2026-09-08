# Cooperative Lambert screening scan

Status: validated on the RTX 5090; H100 validation is queued behind the ongoing
v381 fleet campaign. This candidate is not yet promoted to main.

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
every run. Two samples per mode do not establish a universal speedup. A wider
four-ship paired comparison is running separately.

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
