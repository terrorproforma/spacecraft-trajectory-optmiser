# GPU-controlled whole-itinerary epoch search

The fixed-order timing search now has an optional native CUDA entry point that
keeps the complete mesh schedule on the device. It evaluates the initial route,
generates neighbourhoods, propagates geometry, evaluates candidates, selects and
accepts improvements, advances mesh levels and checks its deadline inside one
conditional graph. Python uploads the immutable metadata and starting epochs
once, then receives the retained final epochs, evaluation and work counts.

This removes the Python loop around `evaluate_mesh`, repeated metadata uploads
and per-neighbourhood winner downloads. It retains the existing multi-block
candidate and Lambert kernels. The controller only updates a small device state;
it does not serialize the candidate workload into a persistent single block.

## Exact search contract

Candidate order is unchanged (`10*N - 14` moves). The highest feasible objective
wins, equal scores retain the earliest candidate, and acceptance still requires
more than `incumbent.objective + 1e-9`. A failed neighbourhood advances the mesh;
the configured accepted-move limit also advances it. Every accepted route keeps
its epochs, forward masses, inflations, proxy costs and collected quantities.
The final unsuccessful neighbourhood cannot overwrite that retained result.

All geometry uses the existing exact-epoch measured/cached overrides and the
same pinned ephemerides and Lambert arithmetic. Those inputs are immutable for
the duration of the search. No new approximations, relaxed physical thresholds,
extra search moves or CPU numerical fallback are introduced.

The deadline is soft, checked between complete neighbourhoods. Native setup time
is deducted from the remaining host budget before launch; a device timer governs
the loop. An already expired deadline still evaluates and returns the starting
route, matching the previous controller. Empty meshes, zero move budgets and
infeasible starting points preserve the same return contract. A nonfinite mining
stay remains an error, including when its candidate could not win.

CUDA conditional WHILE nodes keep the decision on the device, using
`cudaGraphSetConditional` to finish the loop. See the
[NVIDIA conditional graph documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).
The deadline uses the `%globaltimer` nanosecond register on the tested SM90 and
SM120 targets. NVIDIA describes that register as target-specific; this is not a
portability claim for future GPUs or non-NVIDIA targets.
[PTX timer specification](https://docs.nvidia.com/cuda/archive/12.8.0/pdf/ptx_isa_8.7.pdf).

## Enable and compare

Use a core exposing `spacepdhcg_gtoc12_joint_search_host`:

```bash
export SPACEPDHCG_TEST_GTOC12_JOINT_BATCH=1
export SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SEARCH=1
```

The flag selects this path inside `JointItinerary.optimise_epochs` when a CUDA
Lambert backend is active. An explicitly requested search rejects a core without
the new capability and unsupported custom move/evaluation implementations.
The default remains unchanged during this experimental rollout.

The measured baseline uses the same new binary with device search disabled and
the previous resident geometry, device mesh and device selection enabled:

```bash
export SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION=1
export SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY=1
export SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_MESH=1
export SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SEARCH=0
```

Final telemetry counts the initial evaluation and every attempted neighbourhood,
including failed neighbourhoods. Accepted moves and actual geometry work are
reported separately. There is one final epoch upload/download pair per whole
search, rather than one pair per neighbourhood. Instrumentation/report counts
are materialized after execution and do not control the CUDA loop.

## Paired stage measurements

Each case runs seven alternating-order pairs; the first pair is warmup, leaving
six measured samples per mode. Timings include metadata preparation, native
allocation/reuse, graph creation, kernel execution, transfers and final Python
plan construction. GPU context creation, catalogue parsing and independent
low-thrust certification are outside this search-stage timing.

| Case | RTX baseline / device search | RTX ratio | H100 baseline / device search | H100 ratio |
| --- | ---: | ---: | ---: | ---: |
| Ship 1 incumbent | 12.409 / 9.082 ms | 1.37x | 7.061 / 2.871 ms | 2.46x |
| Ship 1 perturbed | 17.151 / 12.107 ms | 1.42x | 9.417 / 3.427 ms | 2.75x |
| Ship 2 incumbent | 13.108 / 8.191 ms | 1.60x | 7.963 / 2.865 ms | 2.78x |
| Ship 2 perturbed | 8.109 / 5.205 ms | 1.56x | 4.960 / 2.166 ms | 2.29x |

Every sample returns exactly the same epochs, objective, plan quantities and
accepted-move count as its baseline. These are search-proxy throughput results;
they do not measure certified trajectories per second or a whole-fleet speedup.

## Validation and failures retained

Both GPUs pass 111 tests without skips. The 24 new cases cover complete search
parity and repeated workspace reuse, invalid settings, missing native capability,
initial infeasibility, empty mesh, zero move budget, sub-ULP tied candidates,
expired deadlines and active device deadlines. The other 87 cases cover existing
GPU evaluator, geometry, selection, mesh and controller behaviour. Initial local
validation skipped seven archived-route tests; the final run includes their
actual archived fixture.

The first controller used a block reduction inside the conditional loop.
Synchronization checking reported divergent-barrier errors on both GPUs. The
final controller uses a full-warp shuffle reduction with stable ordering and no
block barrier. The complete moving-search regression passes racecheck and
synccheck on both GPUs and memcheck on H100. Local active-loop memcheck still
returns CUDA error 999; local empty-loop memcheck passes. A local complete-memory-
clean claim is therefore not supported. Original failing source, binaries and
logs are retained separately from the final candidate.

The first H100 configure attempt lacked a Git repository needed for build
provenance; initialization of the isolated repository fixed it. The first H100
benchmark failed before GPU work because an editable import hook hid the frozen
package. The rerun removes that hook, as the existing test harness does, and uses
the same pinned source and binary. Neither failed attempt is a performance sample.

## Whole mission and remaining GPU boundaries

The local complete comparison runs 18 orders and 36 native low-thrust legs in
each mode; all 36 converge and both retained fleets pass official and independent
physics checks. Both modes evaluate 23,142 joint candidates and 106,462,390
Lambert branch requests across the complete campaign, including DP screening.
The four whole epoch searches accept 112 moves across 128 neighbourhoods.

| Local complete-campaign traffic | Previous mesh controller | Device search |
| --- | ---: | ---: |
| Joint host calls | 132 | 4 |
| Epoch bytes uploaded | 39,888 | 1,248 |
| Epoch bytes downloaded | 39,888 | 1,248 |
| Result/detail bytes downloaded | 88,576 | 2,688 |
| Geometry-stat bytes downloaded | 3,168 | 96 |
| Additional final search-report bytes | 0 | 128 |

The local process took 76.936 s baseline and 75.867 s candidate. This is one pair;
it does not establish a repeatable whole-campaign speedup. The retained score is
unchanged: **12,810.136 weighted kg / 14,051.855 raw kg**, with 23 ships. Both H100
runs also converge on all 36 legs and pass both complete-fleet physics checkers,
retaining that score. H100 process time is 135.002 s baseline and 134.643 s
candidate, again one pair with no established whole-campaign speedup.

This change moves the entire fixed-order epoch-search loop onto CUDA. Python
still controls route/order enumeration, learning between certifications,
insertion/seed exploration, fleet assembly and some input packing. Low-thrust
refinement uses the existing GPU QOCO/SCvx path; the persistent PDHCG backend is
not introduced into that path here. File output and independent CPU physics
certification remain explicit boundaries whose cost belongs in full-run timing.

[Downloaded H100 result, raw archives and source/binary hashes](../results/lambda/2026-09-09/gpu-device-search-v707/)
retain all four complete campaigns and the initial failing implementations.

Select **H100 GPU epoch search v707** in the web visualiser. Full solution:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-device-search-v707\h100-best\Result.txt
```

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process node -ArgumentList @('scripts/serve.mjs', '--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-search-v707&epoch=69807&preset=oblique&z=1'
```
