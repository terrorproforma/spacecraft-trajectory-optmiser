# GPU-resident itinerary geometry

The optional resident geometry path keeps itinerary preflight, exact-epoch cost
lookup, orbit propagation, Lambert screening, mass/mining arithmetic and winner
selection on CUDA. It removes the Python loop over candidate legs and the
intermediate preflight and computed-cost downloads. No physics tolerances change.

The caller supplies trial epochs, stage metadata, one orbit pair per stage and
sorted sparse cache/measured-leg overrides. GPU threads handle independent
candidate legs, then the existing joint evaluator propagates each candidate's
ordered mass history. The GPU winner selector retains first-in-order ties and
the existing strict improvement threshold. The native call uses one retained
workspace and stream, with one final synchronization.

```bash
export SPACEPDHCG_TEST_GTOC12_JOINT_BATCH=1
export SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION=1
export SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY=1
```

Use a current CUDA library through `SPACEPDHCG_GTOC12_CUDA_LIBRARY` and the CUDA
Lambert backend. The feature remains opt-in; requesting it from an older library
without `spacepdhcg_gtoc12_joint_geometry_host` fails explicitly. Setting only the
resident geometry flag to `0` gives a comparison using the same batched CUDA
arithmetic and device winner selector.

Python still constructs mesh moves, packs stage metadata and sparse authoritative
overrides, runs the search controller, and materializes the winner. Host validation
and transfers remain. This is a step towards a GPU-controlled search, not a claim
that the whole application is GPU-native C++. Independent CPU physics checks still
gate acceptance of refined trajectories.

## Component measurements

Six A/B/B/A blocks compare the previous staged CUDA wrapper with the resident
path. The first block is warmup; ten samples per mode remain for each case.
The same real archived ships contain 186 candidates / 20 visits and 156 candidates
/ 17 visits. Timing includes metadata, geometry, transfers, arithmetic and winner
materialization. Fixture loading, cache cloning, CUDA context warmup and correctness
checks are outside the timer. Cold cache means absent Lambert keys, not a cold
CUDA context. Ratios are **additional speedup over the previous batched CUDA
wrapper**, not over a CPU-only mission solver.

| GPU | Ship 1, cold | Ship 1, warm | Ship 2, cold | Ship 2, warm |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 4.38× | 2.35× | 4.77× | 2.13× |
| H100 | 6.40× | 3.67× | 5.86× | 3.20× |

Every measured winner matches exactly, including all evaluation fields. The
two-level real-ship controller also matches exactly, but accepts zero moves on
this fixture; this component check alone establishes no new trajectory or score.

## Correctness and current limits

Both GPUs pass all 71 tests without skips. The nine resident-geometry tests also
pass CUDA memcheck, synccheck and racecheck on each GPU. Tests cover real cold/warm
neighborhoods, full and compact results, workspace reuse, authoritative NaN/inf
cached costs, ABI layout, sorted unique record keys, invalid flags and preservation
of output buffers on invalid input. Warm-cache arithmetic matches exactly;
uncached all-candidate comparisons retain 5e-5 absolute / 2e-8 relative allowance
for independently propagated geometry. These are comparison tolerances, not
relaxed trajectory acceptance criteria. The sanitizer result applies to this
geometry/joint path, not the separately documented QOCO/cuDSS issues.

Each call downloads 24 bytes of geometry counters plus its requested final
results. The new path makes no Python `paired_hops` calls or preflight downloads.
Sparse known costs remain authoritative, including nonfinite values. Calculated
costs are not inserted into Python's Lambert cache: repeated uncached queries
currently recompute on CUDA. GPU cache retention and GPU mesh generation remain
follow-up work. Lazy geometry buffers are retained at candidate capacity; sparse
record storage grows only when a larger override table is supplied.

## Full-fleet recovery and evidence

Four A/B/B/A processes on each GPU replay the same abandoned-miner search.
Every run passes both complete-fleet physics checkers and reproduces
**12,810.135953 weighted kg / 14,051.854894 raw kg**, with 23 ships, 195 collected
asteroids and 196 deployed miners. All four retained proxy plans have exactly
matching event epochs and payloads between modes. Tiny differences below 1e-6 kg
in refined fleet totals are numerical replay differences, not new score records.

| GPU | Staged median process | Resident median process | Resident time change |
| --- | ---: | ---: | ---: |
| RTX 5090 | 78.99 s | 79.08 s | +0.11% |
| H100 | 137.57 s | 138.27 s | +0.51% |

There are only two processes per mode. These observations establish no whole-run
speedup. The processes include CPU verification and export, and reuse a historical
fleet rather than constructing 23 ships from scratch. Native conic convergence
also varies between replays. Component speedups must not be presented as an
equivalent whole-campaign gain.

Each run searches 18 visit orders through 36 retiming driver calls and evaluates
23,142 joint candidates in 132 batches, followed by 36 native leg solves. The
resident path replaces 1,481,088 preflight download bytes with 3,168 counter bytes;
88,576 bytes of final selected results remain. It computes 221,023 joint geometry
hops and rejects 206,881 before geometry. Its total logical Lambert request count
is higher because computed cache entries are not retained; throughput denominators
must use the mode's actual counters. These are surrogate candidates, not 23,142
independently certified trajectories.

[Downloaded results and reproducible evidence](../results/lambda/2026-09-09/gpu-joint-geometry-v659/)
include both frozen source/binary archives, all test and sanitizer logs, all eight
campaigns, timing samples and SHA-256 manifests. The public header's buffer-ownership
comment was clarified after measurement; `published-source.json` verifies that
runtime source and the final test are otherwise byte-identical to the tested files.
The original local 69-test run and subsequent 71-test run are both retained.

## Display the H100 replay

Select **H100 resident geometry v659** in the existing web visualiser. The complete
downloaded solution is:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-joint-geometry-v659\h100-best\Result.txt
```

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath node -ArgumentList @('scripts/serve.mjs', '--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-geometry-v659&epoch=69807&preset=oblique&z=1'
```

## Next GPU boundary

`JointItinerary.optimise_epochs` still builds every trial matrix in Python.
For N visits, its fixed ordered move set contains `10*N - 14` candidates.
A GPU generator can form those rows from one incumbent epoch vector, preserving
the positive/negative sign order, launch/return moves, individual arrival and
departure moves, prefix/suffix shifts and whole-itinerary shift. Feeding those
device arrays directly into resident geometry would remove the next matrix
construction and upload boundary. GPU acceptance and mesh progression can then
reuse the same buffers. This follow-up must preserve winner ties, strict objective
improvement, stopping behavior and measured-leg overrides; broader deployment
and asteroid choices remain necessary to raise the fleet score.
