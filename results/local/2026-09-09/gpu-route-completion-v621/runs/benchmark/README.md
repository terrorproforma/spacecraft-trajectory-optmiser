# Prepared completion throughput comparison (v621)

This benchmark compares the **unmodified production** `RouteSearch._finish_cpu` and `RouteSearch._finish_many` on the same historical completion requests. It has not been launched during preparation. There are no new Lambert calls, trajectory solves, search candidates, refinements, certificates or score changes.

The frozen Python source is the final-b adapter snapshot (`completion-adapter-cpu-v621b`, report SHA256 `2e7175e8f818d11987fc240b5511c54b498f5485798501703c03e0906359bc8a`). The normal full CUDA core is `cb977ff09b206de807996a8a21d7ccec41bb6a2a4c35a37c10ebe4883b83d2fb`, bound through build manifest `43c3c2974af7a9d16ba060724855a56b5122acf9c53ff29d68243df612d82969`. Prior direct-C-ABI parity and the four production adapter GPU tests passed; the latter evidence is frozen as `adapter-gpu-prerequisite.json`. The benchmark uses none of that test's observation wrappers inside its method timers.

## Inputs and finite work

Only the 20 historical v619 controls are used: ships 23, 1, 4, 7 and 10, with flat-proxy versus measured deployment prefixes and the frozen flat versus existing-fit collection models. They belong to the historical v595/v616 fleet. No synthetic control is timed. Both methods use identical `CollectPairTable` provenance, including cached saved features and generic table returns; even the small batch is this archival DP workload, not an observed live heuristic scheduling invocation.

Each model has ten controls: one CPU-admitted flat-prefix ship-4 case and nine final-mass rejections. The admitted control is first in the fixed cycle so every group includes both outcomes. `plan.json` records every repeated index and the exact acceptance count. Larger batches repeat these controls; there are still only 20 unique historical cases.

| Batch size | Why it is included |
|---:|---|
| 4 | Size of the four schedule modes; this is an archival DP request benchmark at that size |
| 24 | Actual saved v616 beam and chain-shortlist setting |
| 48 | Frozen `SearchSettings.chain_tour_candidates` default; pruning can produce fewer requests |
| 64, 256, 1024 | Prospective batching/scaling probes, not observed live shortlist sizes |

For each of 12 model/size groups, each backend runs once for first-call cost and four times with reuse. The warm order is CPU/GPU/GPU/CPU followed by GPU/CPU/CPU/GPU. First-call order alternates between groups. This is **60 GPU completion calls and 14,200 proxy candidate evaluations per backend**. The worker has a 180-second deadline; there is no automatic retry or extra warmup.

## Timings and qualification

The outer GPU method timer includes normal production packing, workspace checks/allocation, CUDA transfer/kernel/readback, and plan reconstruction. The CPU timer includes the scalar loop and returned plan/reason list. Production telemetry separately records packing, native-call, kernel and total adapter time. No benchmark instrumentation is inserted into either production method.

Request construction and source/input checks occur outside these method timers. Backend setup, close, post-call numerical comparison and raw capture are measured separately. The first GPU call in each group creates a fresh completion workspace; subsequent calls must retain its workspace and host-buffer identities. Only the first group can include initial process CUDA context setup, so first-call values are not claimed as repeated cold-device measurements. Normal Python garbage collection remains enabled.

Every invocation checks the prescribed accepted/rejected decisions, unchanged deployment/collection epochs, cargo and collection order, retained prefix, forward leg metadata and inflation, final mass and propellant. Every GPU call also checks all native per-leg/results against the frozen oracle, including failed cases, with the previously fixed FP64 tolerance. Numeric tolerance never changes classification. Each GPU readback is retained as compact compressed NPZ; timings, comparison results and outcome hashes are retained per call. Whole repeated production plans are not duplicated. First CPU/GPU outcome summaries per group support reconstruction and audit.

Throughput is **proxy candidate evaluations per second**. It is not independently certified solutions per second or a measured whole-search speedup. Large repeated batches show possible completion scaling; they do not establish that the live mission search can supply those batches cheaply. Four warm samples give a bounded comparison, not a broad statistical performance claim. Default-off policy should remain until practical host-inclusive results warrant a change.

## Preparation checks and launch boundary

Fourteen construction-only tests passed in 6.78 seconds. They forbid GPU loading and CPU propellant calculation, verify all source/input hashes, and compare packed role/model/epoch/authority/feature fields against the already-frozen oracle. `validation-01` retains initial lint errors; `validation-02` retains the detected canonical-signature error on an unused `_Partial` NaN score sentinel. Tagged nonfinite encoding fixes the signature without altering physical input fields. `validation-03` records the passing checks.

`launch.py` checks the final ready manifest, frozen Python and native source/binary hashes, prior adapter success, Python binary identity, the RTX 5090 UUID, and an empty compute-process list. It acquires `/home/angus/.spacepdhcg-gpu.lock` nonblockingly. A single-use marker prevents duplication. The child inherits the lock descriptor; the supervisor reaps only its owned process group on timeout (10-second TERM then 30-second KILL allowance). Busy/preflight/partial failures are preserved, with no automatic rerun.

`inputs/gpu-profile.json` is the unchanged prior parity-run pin document: its embedded three-call validation budget describes that earlier test. This benchmark's authoritative budget is in **`plan.json`**, as enforced by `run.py` and `launch.py`.

Root must review the frozen runner before executing once:

```powershell
wsl -d Ubuntu-22.04 -- env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /home/angus/worktrees/spacepdhcg-literature-venv/bin/python -B /mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser/build/performance/completion-benchmark-v621/launch.py
```
