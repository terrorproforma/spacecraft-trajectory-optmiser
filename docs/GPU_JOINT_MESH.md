# GPU-generated itinerary neighbourhoods

The optional CUDA mesh path generates all `10*N - 14` ordered timing moves from
one incumbent arrival/departure vector. Trial epochs feed directly into resident
preflight, sparse cost lookup, orbit propagation, Lambert screening, mass/mining
evaluation and winner selection. The host receives only the selected epoch vector
and evaluation. The GPU retains incumbent and trial buffers across calls.

```bash
export SPACEPDHCG_TEST_GTOC12_JOINT_BATCH=1
export SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY=1
export SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION=1
export SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_MESH=1
```

Use the CUDA Lambert backend and a current `SPACEPDHCG_GTOC12_CUDA_LIBRARY`.
Setting only the mesh flag to zero compares against Python-generated epochs with
the same resident CUDA geometry and arithmetic. An explicitly requested mesh
requires the new native entry point and device winner selection. Custom instance
move generators are rejected. The new API accepts finite positive mesh steps and
rejects inputs whose shifted epochs overflow, preserving output buffers on errors.

Move order, first-winner ties, strict improvement thresholds, measured-leg reuse
and physics tolerances are unchanged. CUDA reproduces the generator's binary64
additions, including zero components for touched visits and copies for untouched
visits. `spacepdhcg_gtoc12_joint_mesh_host` also supports downloading all generated
epochs for auditing; normal search uses the compact selected-vector output.

## Measurements and validation

Six A/B/B/A blocks compare CPU and GPU mesh generation with resident geometry.
The first block is warmup, leaving ten timing samples per mode per case. Timers
include epoch generation, metadata packing, geometry, transfers, arithmetic and
winner materialization. Fixture loading, cache cloning and correctness checks
are outside timing; the CUDA context is warm in both cache conditions.

| GPU | Ship 1, cold cache | Ship 1, warm cache | Ship 2, cold cache | Ship 2, warm cache |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 1.104× | 1.108× | 1.069× | 1.149× |
| H100 | 1.420× | 1.358× | 1.399× | 1.336× |

These are additional component ratios against the preceding resident-geometry
path. They are neither whole-mission speedups nor certified trajectories per
second. The two ships contain 186 candidates / 20 visits and 156 candidates /
17 visits. Epoch uploads fall from 59,520 to 320 bytes and 42,432 to 272 bytes
respectively; returning the selected epochs adds 320 or 272 download bytes.

Both GPUs pass **87 tests** without skips. The combined 25 resident-geometry and
mesh tests pass CUDA memcheck, synccheck and racecheck on both GPUs. Generated
matrices match every CPU epoch bit-for-bit for multiple visit counts and steps,
including signed-zero inputs. Full/compact API behavior and actual search outcomes
retain their independent oracle. The broader QOCO/cuDSS sanitizer limitations
remain separate from these successes.

The real controller test starts from a worse feasible neighbouring itinerary.
Ship 1 accepts six moves through two mesh levels; ship 2 accepts one. Both modes
return exactly the same epochs and evaluation fields, with 1,117 and 469 candidate
evaluations respectively. A test disables the Python move generator during the GPU
run, confirming that the integrated controller uses native generation.

The first new test run accidentally included the incumbent as an extra expected
row because its shared fixture helper prepends that row. The production move set
does not include it. The corrected fixture excludes that extra row; native code
was unchanged. The original failed test/log and corrected test are both retained
with hashes, along with the subsequent passing suite.

## Remaining control work

Python still drives accepted-step counting, mesh-level progression and the
deadline check, compiles metadata and sparse overrides, and materializes plans.
This tranche removes candidate-matrix construction and transfer, not the entire
host controller. Computed geometry costs still recompute on CUDA rather than
populating Python's Lambert cache. Independent CPU physics verification remains
the final fleet acceptance gate.

The next boundary is to retain metadata and the incumbent across complete mesh
levels, with GPU acceptance and stopping state. The existing native SCvx uses
CUDA conditional graphs; the same mechanism can drive the generate → geometry →
evaluate → select → accept loop without a Python round trip for each move. Such
a loop needs explicit tests for ties, no improvement, accepted-move limits,
mesh transitions, deadlines and final incumbent preservation before promotion.

## Fleet replays and reproduced convergence failure

All eight retained best fleets pass both complete-fleet physics checkers and
reproduce **12,810.135953 weighted kg / 14,051.854894 raw kg** with 23 ships,
195 collected asteroids and 196 deployed miners. The four saved proxy plans
match exactly in event epochs and payloads across CPU/GPU mesh modes.

Each process attempts 18 orders, 36 retiming driver calls, 23,142 joint
evaluations in 132 batches and 36 native leg solves. GPU mesh generation handles
128 of those batches; four single-incumbent evaluations remain. It uploads
39,888 bytes of incumbent epochs and downloads 39,888 bytes of selected epochs,
in addition to the existing 88,576 result and 3,168 geometry-counter bytes.

| GPU | CPU mesh median process | GPU mesh median process |
| --- | ---: | ---: |
| RTX 5090 | 78.42 s | 79.86 s |
| H100 | 132.10 s | 138.28 s |

These raw medians are not evidence of an end-to-end speedup. Only two processes
per mode were measured, and completed verification work differs: one local GPU
mesh run and one H100 CPU mesh run fail to refine the second, lower-scoring
alternative return. They retain the verified first improvement and skip the
second complete-fleet check. Conic convergence variation therefore confounds
whole-process comparisons. All failed alternatives and logs are retained.

The local failed return is now reproduced in a separate diagnostic without mesh
generation. Three repeated calls with identical captured numerical input hashes
produce two converged/certified results and one failed result. Disabling QOCO
workspace reuse still produces a failure, with fresh workspace creation checked
in every call. Five Ruiz equilibration iterations also fail in one of three calls.
Neither setting is promoted as a fix. Across these nine calls, all six converged
legs pass the existing independent CPU certificate and all three failures are
rejected. GPU-generated seeds and assembled conic matrices remain to be captured
to locate the divergence below the Python input boundary.
[Exact inputs, repeated failure and rejected workarounds](../results/local/2026-09-09/return-replay-v672/).

The subsequent capture finds byte-identical first conic problems across six
calls. Ordinary native dispatch also reproduces the trajectory failure. Eight
standalone replays of that exact conic input all fail the independent accuracy
audit; increasing regularization to 1e-8 or disabling both IPM and factor graphs
also fails eight of eight. This narrows the investigation to inner-solver
numerical accuracy for a concrete input, without yet establishing a validated
fix. [Captured problems, failed comparisons and independent oracle](../results/local/2026-09-09/return-qp-v680/).

[Downloaded H100 result and both complete evidence archives](../results/lambda/2026-09-09/gpu-joint-mesh-v670/)
preserve frozen source/binaries, all validation and sanitizer logs, all eight
campaign attempts, benchmark samples and hashes. The source was frozen from
`2cbdd425` plus this mesh implementation; concurrent persistent-solver snapshot
tooling was excluded from the measured builds.

## Display the downloaded H100 result

Select **H100 GPU mesh v670** in the existing web visualiser. Complete solution:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-joint-mesh-v670\h100-best\Result.txt
```

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process node -ArgumentList @('scripts/serve.mjs', '--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-mesh-v670&epoch=69807&preset=oblique&z=1'
```
