# GPU joint candidate evaluation

The joint itinerary search can evaluate a complete timing neighborhood with CUDA
instead of calling the Python evaluator once per candidate. On the local RTX 5090,
the final compatibility snapshot evaluates the tested neighborhoods **8.04–8.88x
faster with an empty Lambert cache** and **7.58–7.89x faster with a populated cache**.
These measurements cover search-surrogate evaluation and its controller. They do
not include full low-thrust refinement or independent trajectory certification.

The feature remains opt-in: `SPACEPDHCG_TEST_GTOC12_JOINT_BATCH` defaults to `0`.
Enabling it requires an active CUDA Lambert backend. This evidence set contains
no new H100 measurement: its Lambda upload was blocked by automatic approval
review, and all measurements below were made locally.

## What runs on the GPU

[gtoc12_joint.cu](../cpp/cuda/src/gtoc12_joint.cu) evaluates one candidate per CUDA
thread in blocks of 128. Each thread follows that itinerary's mining, thrust,
transfer-time, mass and payload-sizing rules in visit order. Independent candidates
occupy separate threads and can span multiple blocks. The implementation uses
FP64 and is compiled with `--fmad=false` to preserve the reference arithmetic.
It retains device buffers and a stream across calls, with explicit capacity and
device-ownership checks.

The wrapper first uses GPU evaluation to identify candidates that need geometry,
then batches uncached Lambert work through the CUDA backend. Exact-epoch measured
legs and previously computed Lambert costs remain usable. Full evaluation returns
the same feasibility/failure fields and bookkeeping as the Python reference.
Unsupported model overrides are rejected instead of being approximated silently.

The optional `spacepdhcg_gtoc12_joint_best_host` entry point reduces eligible
candidate scores on CUDA, keeps the first highest-scoring row on ties, and copies
back only that row's detailed state. A nonwinning invalid mining stay remains an
error. This reduction uses one 128-thread block; the preceding candidate
evaluation is distributed across blocks.

Python still constructs candidate epoch arrays, prepares policy/visit metadata,
looks up and deduplicates cached costs, materializes the selected plan, and controls
the wider search. The implementation therefore advances GPU execution of the
candidate evaluator; it does not make the whole campaign a C++-only GPU program.

With `SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION` unset, the wrapper uses the
native winner entry point when the loaded core exports it. Older cores continue
to perform candidate arithmetic on CUDA and select the winning returned row on
the host. Explicit `0` selects that compatible path. Explicit `1` requires the
native winner entry point and fails before metadata, preflight or geometry work
if the symbol is missing. The joint batch opt-in itself remains unchanged.

## Measured neighborhoods

Both implementations use the same incumbent, candidate order, geometry policy
and stopping/selection criteria. Each case uses three A/B/B/A blocks, providing
six timings per implementation. The mesh step is three days. Every full-candidate
and ordered-winner comparison passed.

| Incumbent | Lambert cache | Candidates | Scalar median | Batched median | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Ship 01 | Empty | 186 | 84.988 ms | 9.572 ms | 8.88x |
| Ship 01 | Populated | 186 | 33.817 ms | 4.285 ms | 7.89x |
| Ship 02 | Empty | 156 | 66.888 ms | 8.322 ms | 8.04x |
| Ship 02 | Populated | 156 | 24.444 ms | 3.223 ms | 7.58x |

The batched medians correspond to approximately **18,700–19,400 candidate
evaluations/s with an empty cache** and **43,400–48,400/s with a populated cache**.
These candidates are timing alternatives for two existing route orders, not
independently certified trajectories or complete fleet solutions.

The timer includes move generation, epoch copies, Python bookkeeping, cache
lookups, uncached geometry, C API packing/transfers, joint arithmetic, plan
materialization and winner selection. It excludes file loading and checksums,
baseline evaluation, CUDA/workspace warmup, fixture/cache construction, the full
parity audit and report writing. An empty cache here means an empty per-run
Lambert cache with CUDA already initialized. SCvx and the independent checkers
are outside both timed paths. The table does not establish an equivalent speedup
for a complete optimization campaign.

## Validation and reproducibility

The v596 core was built for `sm120` from published base
`bd944af935e5b71e50ef7ff4d34ea7674602832a` plus the recorded joint overlays. Its
SHA-256 is `86d7fdc952e82f7e5557c1651a1722900d83cdd98d8a709a4b6274fd83af4671`.

- Both standalone native tests passed normally and under memcheck, synccheck and
  racecheck. They were compiled with `-UNDEBUG`, so their assertions and CUDA calls
  remained active. The sanitizer logs report zero errors and zero race warnings.
- The 42 original Python cases and eight winner-selection cases passed normally
  and under each of the three sanitizers, with zero skips. Coverage includes
  incumbent bookkeeping, policy/failure precedence, measured legs, payload rounds,
  retained workspaces, compact output buffers and selection boundaries through
  4,097 candidates.
- After the final compatibility-only wrapper change, **62 tests passed on the new
  core** and **54 passed on the older v590 core**, with winner selection unset to
  exercise automatic API compatibility. Ruff check and format passed for that
  wrapper and its compatibility tests. Native source and the v596 core were
  unchanged; the final benchmark above uses this final Python snapshot.

The test-only CMake addendum explicitly registers `gtoc12_joint_smoke` and
`gtoc12_joint_selection_test`. A fresh Release, test-enabled build discovered and
passed both CTest targets with zero skips in 0.49 seconds. Their compile commands
place `-UNDEBUG` after `-DNDEBUG`, and both executables retain assertion checks.
That addendum and its build/CTest evidence are recorded separately from the
unchanged v596 core's original build provenance.

The first v596 harness attempt correctly stopped when seven historical-fixture
tests skipped. The exact Git-tracked fixture was then pinned and copied into the
isolated source, and the complete Python checks were rerun. Both attempts remain
in the evidence so a partial test run cannot be mistaken for the final result.

[Published evidence](../results/local/2026-09-09/joint-candidates-v596/) contains
the source manifest and archive, exact commands, separate native/sanitizer/final
compatibility logs, the benchmark's full provenance and per-file SHA-256 records.
Its `original-v590/` directory separately preserves the earlier native snapshot
based on `8a5dffee2826e37fe024f1c0be9c9b6d302e186b`, the original 42-test evidence
and the earlier v593 benchmark. All original native overlay hashes were checked
against their build record before archiving.

The separately verified fleet improvement is documented in
[GPU orphan recovery](GPU_ORPHAN_RECOVERY.md). That experiment used the older
v590 core with host winner selection explicitly enabled via selection `0`;
these candidate-speed measurements do not change or re-certify its result.
