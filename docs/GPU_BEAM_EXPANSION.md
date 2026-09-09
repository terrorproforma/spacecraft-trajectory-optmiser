# CUDA beam expansion and ranking

Deploy-phase child pricing and heuristic ranking now run in a retained CUDA
workspace. A search uploads the parent/deployment metadata and screened transfer
grids for one depth. GPU threads apply the existing visited/banned-pair masks,
authority limit, ratio/flat inflation, rocket-equation cost, arrival deadline,
weighted mining estimate, asteroid prices and heuristic penalties. CUB sorts
the resulting children on the device.

Python downloads ranked results in small slices and constructs only the prefix
needed by the existing reserve, diversity and Earth-return admission checks.
The beam no longer needs hundreds of thousands of Python child/leg objects
before it can select a few dozen survivors. Parent histories stay shared until
a selected child is materialized. The CUDA workspace retains capacity across
depths and rejects reads after a failed rank, preventing stale result reuse.

Ranking retains the original descending score, ascending epoch, lexicographic
deployment sequence and stable input-order tie break. Arithmetic is FP64 with
FMA contraction disabled for this operator. For exact Python float weights,
the score uses the compensated sum behavior of Python 3.12 and later; older
Python versions use sequential sums. This follows the
[CPython summation implementation](https://github.com/python/cpython/blob/v3.12.13/Python/bltinmodule.c#L2464).
Custom numeric weights and overridden heuristic methods retain the existing
Python search behavior. They are not silently interpreted as the standard
native heuristic.

Within an explicit CUDA screening scope, native expansion is enabled by default.
`SPACEPDHCG_TEST_GTOC12_GPU_EXPANSION=0` selects the previous expansion/ranking
path for paired measurements. The ordinary CPU scope remains available for
independent comparison. No trajectory dynamics or physical tolerance changes
are part of this optimization.

## Evidence and remaining work

A fresh profile of the preceding runtime measured about 51.8 million Python
function calls for the ship-10 search. Expansion consumed 11.5 profiled seconds
and selection another 4.3 cumulative seconds. Those are profiling observations,
not clean benchmark timings. Collection scheduling and its repeated native
calls remain a separate substantial cost.

The first native full-route run prices/ranks 621,850 valid children out of
1,969,920 input slots over nine depths, while Python materializes only 10,123
children for admission. The route produces the same 517 surrogate candidates;
the second fixed route produces 472. Candidate records are compared recursively:
discrete fields, topology, order and list lengths must match exactly. FP64 fields
are checked at absolute difference at most 2e-9. The largest observed difference
is 9.095e-13. This comparison does not certify surrogate candidates as trajectories.

Three measured samples per mode, after one warm-up per mode, give these
whole-search medians. Both modes use the same native library and source; only
the expansion flag changes. Timings exclude process startup and catalogue loading.

| GPU | Route | Python expansion (s) | CUDA expansion (s) | Throughput gain |
| --- | --- | ---: | ---: | ---: |
| RTX 5090 | Ship 10 | 20.851 | 12.961 | 1.61x |
| RTX 5090 | Ship 21 | 18.969 | 10.818 | 1.75x |
| H100 | Ship 10 | 18.113 | 7.013 | 2.58x |
| H100 | Ship 21 | 17.279 | 6.343 | 2.72x |

Combined throughput is about 41.6 surrogate candidates/s on RTX 5090 and 74.1
on H100. These are search rates, not certified solutions/s or inner-solver gains.
Ship 21 still materializes 95,632 of 630,066 valid children for host admission.
Across both routes, Python constructs 105,755 of 1,251,916 children, a 91.55%
reduction. Each route still uploads 141.95 MB of expansion inputs; eliminating
that transfer and moving admission to CUDA are further opportunities.

A separate beam-width-128 run generates 1,960 surrogate candidates, versus 989
at width 64, screening 64,828,728 Lambert branches. This takes 26.95 seconds on
H100 and 48.52 seconds on RTX 5090. It includes 509 distinct asteroid orders
absent from the narrower search. None was trajectory-refined in this checkpoint,
so it does not promote the checkpoint fleet's 13,023.704900978253 weighted kg /
14,291.0061601653 raw kg score. Independent fleet work may advance separately.

[Downloaded reports, exact source/binaries and reproducible audit](../results/lambda/2026-09-09/gpu-beam-expansion-v835/README.md).

Final default-on validation passes 183 tests on each GPU. Forty-five tests pass
under each of CUDA memcheck, racecheck and synccheck, with separate endpoint
controller checks under all three tools. Tests cover exact authority and arrival
boundaries, stable ties, flat/ratio models, weighted prices, cluster bonuses,
unreachable lookahead, empty inputs, retained capacity and invalidation after
failed calls. The initial v830 fixture failure is preserved: it assigned to a
read-only property before any of the six new tests could exercise CUDA; the
corrected fixture uses the property's backing field.
Seven focused GPU tests also pass full leak checking on each device, with zero
bytes leaked, zero remaining allocations and zero reported errors.

The frozen base is `3404e926355beb3383f391c3a4bb80ce3d618afa` with seven owned
source/build/test files overlaid. Native compilation is v830, paired search
measurement uses v833, and v834 enables the already-measured path by default
and tests that default. Their C++ source and native binaries are identical.
Concurrent SCvx mass-merit changes after that base are outside these timings.

The complete search is still not GPU controlled. Hop screening is dispatched
from the host and its grids are copied into the new pool. Beam admission,
collection scheduling, some priors and Python route construction remain.
The next architectural steps are a direct resident screening-to-expansion
interface and device-controlled collection scheduling/admission.
