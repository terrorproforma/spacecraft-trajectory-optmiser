# Batched CUDA itinerary search and device winner selection

The optional continuous-itinerary evaluator now processes independent candidate
epoch vectors on CUDA. Each candidate preserves the ordered mass propagation,
mining bookkeeping, calibrated Lambert costs, measured-leg reuse and rejection
gates of the scalar reference. Candidate evaluation spans multiple GPU blocks;
the small final winner reduction uses one 128-thread block.

The native `spacepdhcg_gtoc12_joint_best_host` entry point selects the highest
feasible objective strictly above `minimum + 1e-9`, preserves the earliest row
on ties, and returns only that row's details. Invalid mining stays remain errors
even when their candidate cannot win. Retained workspace storage owns the
results, selection record and temporary arrays. No physics tolerances change.

Distinct binary64 epochs now have distinct Lambert-cache and measured-leg keys.
The previous five-decimal rounding could alias nearby continuous candidates;
batching must not allow an earlier speculative evaluation to substitute the
wrong cost for a later candidate.

## Use and remaining CPU work

Build the current CUDA library and select it through
`SPACEPDHCG_GTOC12_CUDA_LIBRARY`. Before launching an itinerary or cluster search:

```bash
export SPACEPDHCG_TEST_GTOC12_JOINT_BATCH=1
export SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION=1
```

Use `--screening-backend cuda --workers 1` with the relevant CLI command. Native
SCvx refinement additionally uses `--seed-backend cuda --discretisation-backend
cuda --assembly-backend cuda --convex-solver qoco --outer-loop-backend cuda
--gpu-execution graph` and a validated `SPACEPDHCG_QOCO_LIBRARY`.

The joint batch feature remains opt-in. Device selection is selected automatically
when the loaded joint library exposes its optional entry point. An older joint
library retains its full-result API when no selection override is set; explicitly
requesting the missing entry point fails before metadata, geometry or evaluation
work. Setting the selection flag to `0` provides a matched full-download
comparison using the same CUDA arithmetic.

Python still constructs mesh moves, compiles metadata, examines preflight failure
rows, packs a retained Lambert cache, and materializes the selected route. The
native host API also validates and uploads inputs. This is not yet a completely
GPU-controlled search. Independent CPU physics verification remains the final
acceptance check, not part of the GPU throughput count.

## Component measurements

Frozen builds on RTX 5090 and H100 evaluate the same two archived 23-ship-fleet
members. Their three-day mesh neighbourhoods contain 186 candidates / 20 visits
and 156 candidates / 17 visits. Three A/B/B/A blocks compare scalar CUDA Lambert
with the batched CUDA controller, including epoch construction, geometry,
packing, transfers, arithmetic and winner materialization. Context warmup,
fixture loading, state cloning and output auditing are outside the timer.

| GPU | Ship 1, empty cache | Ship 1, complete cache | Ship 2, empty cache | Ship 2, complete cache |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 9.08× | 7.74× | 7.79× | 7.58× |
| H100 | 12.56× | 10.92× | 12.81× | 11.07× |

These ratios describe the search surrogate, not a full mission or SCvx solver.
An empty cache means per-run Lambert keys are absent; the CUDA context is warm.
Every candidate is compared with the scalar reference outside timing. Complete
cache tolerances are 5e-9 absolute / 5e-12 relative; independently propagated
empty-cache comparisons allow 5e-5 / 2e-8. Winning indices must match.

The separate exact comparison isolates device selection from full-result CUDA
evaluation. Final-result downloads fall from **126,480 to 688 bytes** for ship 1
and **91,104 to 592 bytes** for ship 2 (99.46% and 99.35% less). Preflight still
downloads 64 bytes per candidate, and input uploads are unchanged. Observed
selection-only timing ratios are approximately 0.98–1.02× locally and
1.00–1.01× on H100: lower traffic has not established a substantial additional
speedup for these small neighbourhoods.

## Validation and evidence

The frozen native builds each pass 50 Python tests, including real archived
ships, measured-cost calibration, failure precedence, foreign miners, exact
epoch keys, first ties, thresholds, and winner batches of 1, 127, 128, 129, 257
and 4,097. The original local run skipped seven missing archive fixtures; those
fixtures were supplied and all 50 then passed without skips. Both the full-API
and selection-API native probes pass memcheck, synccheck and racecheck on both
GPUs. This does not resolve the separately documented QOCO/cuDSS sanitizer
limitations.

The benchmark controller check also agrees exactly between full-download and
device-selection modes across a two-level real-ship search. That particular
archived ship accepts zero moves; synthetic tests separately exercise two
accepted moves. No improved physical trajectory is inferred from that check.

The measurements pin the Python implementation before the later optional-ABI
compatibility guard. That guard changes routing for older libraries and moves
explicit missing-capability errors ahead of work; the measured current-library
paths retain the same native arithmetic and selection policy.

The final compatible wrapper passes **62 tests on each GPU**, including twelve
CPU-only tests of old/new API routing and early explicit-capability errors. The
later CMake changes register both native probes and keep their assertions active
in Release builds; they do not change the measured library source.

## Complete fleet replays

The abandoned-miner recovery case from [v595](GPU_ORPHAN_RECOVERY.md) was replayed
with the new core, comparing full-result downloads (A) and device selection (B).
Each process includes initial fleet verification, GPU search and native SCvx,
candidate fleet verification, viewer export and IO. It starts from the historical
23-ship incumbent, so the resulting gain is a reproduction of v595, not another
increment to that score.

| GPU | A, complete process seconds | B, complete process seconds | Median change |
| --- | --- | --- | --- |
| RTX 5090 | 79.186, 79.026 | 77.589, 78.090 | 79.106 → 77.839 s; 1.60% less |
| H100 | 137.434, 137.686 | 139.145, 122.268 | 137.560 → 130.706 s; 4.98% less |

The intended order was A/B/B/A. The last local A stopped before GPU work because
another process held the GPU lock, then passed in a separate queued retry.
The failed attempt is preserved. Only two successful samples per mode were taken,
and the H100 B range spans both A samples. These timings do not establish a
reliable end-to-end speedup from winner selection.

All eight successful processes reproduce **14,051.854894 physical kg /
12,810.135953 weighted kg**, within 1e-6 kg, with 23 ships and 195 collected
asteroids from a footprint of 196 deployed miners. Both complete-fleet physics
checkers pass at unchanged tolerances in every run. Each evaluates 18 new orders
at two grids, 106,024,898 logical Lambert branches, 23,142 joint candidates in
132 batches, and 36 native leg solves. Those are distinct stages and their
counts must not be combined into a "solutions per second" figure.

Across each run, final joint-result downloads fall from **15,359,152 to 88,576
bytes** (99.42% less). Preflight downloads remain **1,481,088 bytes**. The next
CPU work to remove is epoch construction and cache/metadata packing, followed by
the preflight round trip; these measurements do not justify claiming that work
has already moved to CUDA.

## Downloaded evidence and visualiser

[Published results](../results/lambda/2026-09-09/gpu-joint-itinerary-v642/) include
the summary, complete timing reports, final compatibility checks, the selected
H100 solution/export and regenerated viewer dataset. The two raw archives retain
frozen sources, both native libraries, original validation logs, all fleet
attempts, and the local retry. Their per-file manifests and archive hashes were
verified after retrieval: 1,629 local members and 1,621 H100 members.

The existing visualiser dataset is **H100 GPU itinerary v642**. Its import checks
11,679 exact replay samples and 73,562 Kepler context points; the maximum asteroid
context difference is 3.59e-6 km. The displayed weighted score and H100 hardware
were checked in the running WebGL2 page.

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath 'node' -ArgumentList @('scripts/serve.mjs', '--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-joint-v642&epoch=69807&preset=oblique&z=1'
```

The exact H100 solution is:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-joint-itinerary-v642\h100-best\Result.txt
```

On a fresh checkout, the archived `viewer-dataset` directory can be copied to
`results/lambda/2026-09-06/visualiser/data/gtoc12-joint-v642` before starting the
server. The larger verifier export is also retained for importing again from
the checksum-verified asteroid catalogue.


## Audited existing H100 v632 evidence

The [v632 evidence bundle](../results/lambda/2026-09-09/joint-candidates-v632/)
preserves all **80 files retrieved from the existing Lambda run**, with their
original [retrieval hashes](../results/lambda/2026-09-09/joint-candidates-v632/evidence/download-manifest.json)
and [viewer-file hashes](../results/lambda/2026-09-09/joint-candidates-v632/evidence/viewer-download-manifest.json).
The redundant transport archives are omitted; their hashes remain recorded.
Retrieval and publication started no new GPU run and performed no new trajectory
verification. The [source manifest](../results/lambda/2026-09-09/joint-candidates-v632/evidence/source-manifest.json)
and [recorded source/configuration identity](../results/lambda/2026-09-09/joint-candidates-v632/evidence/remote-provenance.json)
distinguish the measured implementation from the later compatibility wrapper.

The separate selection-only transfer measurement forces an eligible winner with
`minimum_objective=-math.inf`, as shown by its
[driver](../results/lambda/2026-09-09/joint-candidates-v632/supporting-source/benchmark_joint_selection_v634.py).
That file is preserved as a labeled local audit reference, separately from the
remote downloads; the result JSON itself does not record the threshold. The
[selection report](../results/lambda/2026-09-09/joint-candidates-v632/evidence/benchmark-v634/selection.json)
therefore measures returning the best candidate even when it would not improve
the incumbent. In the
[main scalar-versus-batched report](../results/lambda/2026-09-09/joint-candidates-v632/evidence/benchmark-v634/scalar-vs-batched.json),
ship 2 has **no improving winner** in either cache condition. Its matching null
winner is a correct search outcome, not evidence of a new physical solution.

All four v636 retained best fleets in the
[campaign reports](../results/lambda/2026-09-09/joint-candidates-v632/evidence/campaign-v636/)
pass both recorded official and independent checkers at **12,810.135953 weighted
kg / 14,051.854894 physical kg**, to the displayed precision. They reproduce the
v595 improvement rather than adding another score increment. Candidate1's
[second refinement](../results/lambda/2026-09-09/joint-candidates-v632/evidence/campaign-v636/candidate1/candidates/ship_15_attempt_02/refinement.json)
failed on return leg 17, asteroid 13077 to Earth, with
`virtual control remains 1.380e-01`. The earlier verified best fleet remained
retained. Its **120.727-second campaign timer** consequently covers different
downstream work from the runs that also certified and checked their second
candidate; it is not evidence of an end-to-end selection speedup. The reports do
not establish why that refinement outcome differs.

The later [compatibility validation](../results/lambda/2026-09-09/joint-candidates-v632/evidence/compatibility-validation-v640/report.json)
and [test output](../results/lambda/2026-09-09/joint-candidates-v632/evidence/compatibility-validation-v640/pytest.log)
record **62 passing tests** on the H100 core with the exact final wrapper SHA-256
`5f70cf173929b8324e9e40885200e71272732f3e8210b5372dc238a07ae25689`.
This is separate from the earlier performance/campaign wrapper `b8a423…`.
The [local v596 evidence](GPU_JOINT_CANDIDATES.md) remains a separate measurement
and validation set; its local speedup table is unchanged by this retrieval.
