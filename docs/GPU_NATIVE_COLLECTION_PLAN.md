# Native collection mining and burn scheduling

The collection planner now computes mining rewards, subset masses and its
two-pass burn policy on CUDA. On the same two route searches, whole-search
throughput improves **5.3–7.5% on H100** and **2.6–5.9% on the local RTX 5090**.
Candidate files remain byte-identical within each GPU, including comparison with
the previously published shared-return runtime. The verified fleet score remains
**13,526.961241 weighted kg**; this change supplies no new trajectory or score.

## Measured complete search time

Each entry is the median of three alternating measurements per mode after one
warm-up per mode. Both modes use the same newly built binary, resident tables,
cached return rows, CUDA expansion/admission and CUDA DP operators. The control
retains the preceding host mining/subset/pass preparation.

| GPU | Route | Host preparation | Native preparation and pass selection | Throughput gain |
|---|---|---:|---:|---:|
| H100 | Ship 10, 517 candidates | 5.992309 s | 5.576231 s | 7.46% |
| H100 | Ship 21, 472 candidates | 5.200408 s | 4.940643 s | 5.26% |
| RTX 5090 | Ship 10, 517 candidates | 10.469986 s | 9.891108 s | 5.85% |
| RTX 5090 | Ship 21, 472 candidates | 8.602863 s | 8.381183 s | 2.64% |

The native rates are **92.71 and 95.53 surrogate candidates/s on H100**, and
**52.27 and 56.32/s locally**. These timers cover `RouteSearch.run`, excluding
process startup and initial catalogue loading. They are not qualified trajectory
or PDHCG solution rates. Measured ranges do not overlap between modes in these
four comparisons, but this is a small benchmark set. The RTX run records desktop
GPU activity and substantial memory use despite an empty reported compute-process
list before each sample. Treat its modest gains as measurements in that desktop
environment, not an isolated-device performance guarantee.

## What moved to CUDA

The old path constructs weighted mining values for every asteroid and epoch,
enumerates all `2^k` subsets in Python, and uploads each pass's mass vector. It
downloads the first DP result, calculates mean hop propellant on the host, then
decides whether to request another pass.

The new native API uploads only the epoch, deployment and weight vectors for that
work. Parallel kernels compute the mining matrix and subset masses. Device logic
selects the heavy, estimated-burn or nominal-burn fallback policy, retaining the
first result when the existing policy calls for it. Only the selected result and
small diagnostics are downloaded. Explicit burn requests still run one pass.

The new path preserves increasing-index subset summation, the existing weighted
mining arithmetic, the minimum-stay boundary, the dry-mass floor, and NumPy's
short-array mean reduction order. The ordered DP comparisons, epsilon ties,
backtracking, return-cell rules and physical models are unchanged. CUDA
compilation retains `--fmad=false` for this translation unit.

Across both benchmark routes on either GPU:

- **3,279 DP passes** remain 3,279 passes, grouped into **2,184 native plan calls**.
- Result readbacks decrease from **2,072,328 to 1,432,704 bytes**, a **30.86%** reduction.
- Schedule metadata uploads total **4,477,152 bytes**. This excludes the existing
  geometry, policy and table traffic; it is not a total-transfer claim.
- Retained workspace counts stay at **51 allocations and 2,133 rebinds**.
  The speed gain is not attributed to fewer workspace creations.

New plan entry points preserve the old C ABI, including the 496-byte legacy result
and 632-byte extended result. The new output is 656 bytes. Plan metadata is copied
before create/update returns. Rejected updates preserve the previous problem;
legacy updates disable plan solving until valid plan metadata is supplied again.
CUDA failures invalidate the new solve path. Old callers remain supported.

CUDA route planning selects this path by default. The explicit comparison switch
is `SPACEPDHCG_TEST_GTOC12_NATIVE_COLLECT_PLAN=0`. A rebuilt native library is
required for the default path; the Python package must match its native runtime.

## Verification and remaining work

Both GPUs pass **236 tests**, including the CPU reference, previous CUDA passes,
exact complete-tour/diagnostic comparisons, tie cases, fitted pricing, phase and
return overrides, invalid metadata, update ownership and legacy ABI checks.
Each also passes **98 focused cases under memcheck, racecheck and synccheck**, the
native controller under each tool, and **31 full leak checks with zero leaked
bytes or allocations**. The legacy ABI test explicitly invokes the old path so
the new default cannot bypass its assertion.

All 16 benchmark candidate files per GPU have the same respective route hashes;
all candidate values and orderings match. The downloaded source/binary manifests,
raw test logs, eight benchmark runs per GPU and H100 profiles are checked by a
portable saved-evidence audit covering **1,995 archive members**. That audit
reconstructs medians and byte counts and checks actual frozen source files; it
does not repeat GPU execution.

The H100 profile contains no call to the old host `_solve_collect_dp` path. For
ship 10 it records 1,152 native plan calls and 5.47 million total profiled calls.
Collection scheduling, forward scheduling, expansion input packing and
geometry/policy preparation remain. Profile cumulative times include nested CUDA
calls and overlap; they must not be added or labelled as pure CPU time. Moving
those inputs and scheduling decisions into retained native structures is the next
architecture task. The entire planner is not yet GPU-controlled.

[Downloaded evidence and audit instructions](../results/lambda/2026-09-09/gpu-native-collection-plan-v859/README.md)

The v633 verified fleet remains the visualiser dataset. No dynamics tolerances,
SCvx defaults, trajectory controls, leaderboard score or physics checkers changed.
