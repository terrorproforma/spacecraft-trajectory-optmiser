# GTOC12 CUDA interval refinement

GTOC12 `solve_leg` can now run interval propagation and variational RK4 on the
GPU through `ScvxSettings(discretisation_backend="cuda")`. The same path is
available on the `run`, `cluster-fleet`, `fleet-master`, `retime-returns` and
`joint-itinerary` commands with `--discretisation-backend cuda`. The default
remains the existing NumPy reference. Missing CUDA configuration fails explicitly.

This is a migration step, not a complete GPU-native refiner. The initial Lambert
seed, Clarabel solve, merit/trust decisions and independent verification still run
on the CPU. Sparse assembly also uses the CPU unless the explicit
[`--assembly-backend cuda` path](GTOC12_GPU_ASSEMBLY.md) is selected. The CLI currently requires `--workers 1` for
CUDA: the existing forked CPU worker arrangement is not a tested GPU batch driver.
Aggregate reports distinguish requested backend from measured GPU use; completed
leg summaries record their actual discretisation backend. A selected CUDA option
alone does not set `gpu_used=true`.

## Physics and device execution

The GTOC12 model cannot be replaced directly with the older native low-thrust
model: nonlinear mass flow is proportional to the norm of the interpolated thrust
vector, whereas its convex mass-control sensitivity uses the Gamma surrogate.
The new FP64 CUDA code preserves both, the nondimensional units, fixed-step RK4,
ZOH hold, and the clamped four-node Lagrange stencil. No fast-math or fused multiply
add is enabled for this source. Neither physical constraints nor tolerances change.

Each interval has its own GPU block (128 threads for linearisation, 32 for state
propagation). The time steps within an interval remain ordered. This propagates
from each interval's reference state; it is not a sequential whole-trajectory
rollout. The independent verifier remains the authority for the final trajectory.

The native C API retains topology, inputs and A/B/c/propagated output buffers.
`launch_device` accepts device state/control pointers and a CUDA stream, performs
no allocation or host synchronization, and leaves outputs on the GPU for a future
native assembler. A device invalid-dynamics flag accompanies the outputs. The
caller must order external-stream consumers and complete them before workspace
reuse/destruction. Workspaces are bound to the creating device and not thread-safe.

The transitional Python bridge uploads states/controls, launches the native
operation, downloads coefficients and synchronizes once for the CPU assembler.
Its workspace persists across outer iterations and polishing. It owns immutable
node times/stencils and closes on normal return, timeout and Python exceptions.

## Local use

The checked CUDA build includes the new API in `libspacepdhcg_cuda.so` and builds
`gtoc12_discretisation_test` through the normal CMake CUDA test targets. Configure
for the actual GPU architecture: 120 for the local RTX 5090, 90 for H100. The
existing pinned PDHCG checkout remains a read-only build dependency.

The frozen local runtime from this tranche is:

```text
/home/angus/build-spacepdhcg-gtoc12-v85/final/libspacepdhcg_cuda.so
```

From WSL in this repository:

```bash
export PYTHONPATH=src
export SPACEPDHCG_GTOC12_CUDA_LIBRARY=/home/angus/build-spacepdhcg-gtoc12-v85/final/libspacepdhcg_cuda.so
python -m spacepdhcg gtoc12 run --run-id cuda-refinement \
  --output results/gtoc12/runs/cuda-refinement --discretisation-backend cuda
```

This command also requires the pinned GTOC12 catalogue and Python dependencies,
just like the existing CPU workflow. For commands with a `--workers` option, pass
`--workers 1`. To rerun the GPU regression in the prepared local environment, set
`SPACEPDHCG_GTOC12_GPU_TESTS=1` and use the literature virtualenv's pytest. Serialize
GPU tests and benchmarks with `/home/angus/.spacepdhcg-gpu.lock`.

## Matched measurements on RTX 5090

Two warmups and nine measured repetitions per setting, alternating CPU and GPU
calls in one process, with one BLAS thread. Interval times include host output
allocation and transfers, but exclude initial workspace construction. Full-leg
wall time includes workspace creation/destruction and CPU solver work; the
independent certificate is computed outside the timed solve. GPU utilization was
0% before this serialized experiment. The static viewer stays open and does not
animate while paused. These are local WSL wall timings, not isolated CUDA-event
kernel latency or H100 measurements.

| Intervals | Hold | Linearisation CPU ms | CUDA bridge ms | Speedup |
|---:|---|---:|---:|---:|
| 75 | ZOH | 3.340 | 0.397 | 8.41x |
| 75 | Lagrange | 6.716 | 0.627 | 10.72x |
| 257 | ZOH | 8.700 | 0.515 | 16.89x |
| 257 | Lagrange | 19.694 | 0.774 | 25.44x |
| 2,001 | ZOH | 62.301 | 1.286 | 48.46x |
| 2,001 | Lagrange | 158.484 | 2.061 | 76.90x |

The representative 150-day, 75-interval ZOH leg takes **137.914 ms CPU versus
118.281 ms with CUDA interval dynamics**, a **1.166x** improvement (~14.2% less
wall time). All 22 samples, including warmups, converge and pass the independent
certificate, with final mass within 1e-5 kg of the fixed 2445.3111007852112 kg
reference. This is a measured representative-leg improvement, not a claim about
fleet-search throughput or every trajectory.

Additional matched checks cover node spacing 0.5/2/8 days, both holds, and free
endpoint v-infinity. Five extra pairs converge and retain the same qualification
and final-mass agreement. Both solvers report infeasible and fail qualification
on the free-v-infinity pair; this does not establish physical infeasibility. The
failure is retained and is not counted as a success.
A sampled profile places most remaining time in CPU Clarabel and sparse assembly.

## Verification and evidence

- The combined GTOC12, telemetry, planner-schema and native SCvx regression run
  passes all 247 tests with no skips. The additional lifecycle memcheck passes
  with zero errors and zero leaked allocations.
- Twenty-four interval combinations span 3/17/257/2,001 intervals, both holds and
  1/8/16 RK4 substeps. A/B/c/states match the CPU reference within 2e-12 absolute
  and relative tolerance. Analytic constant-thrust mass flow and affine closure
  are checked independently of that reference.
- The native device API passes 24 changing-input CUDA Graph replays across both
  holds, with analytic mass and affine closure, on an external nonblocking stream.
- Native device API passes memcheck/leak-check, initcheck, racecheck and synccheck.
  The host bridge/reused workspace/complete leg pass memory, initialization and
  synchronization checks; full-leg racecheck was not run in this tranche.
- Ownership, repeated updates, polishing substeps, invalid state rejection,
  recovery after invalid input, closed handles, missing libraries, exception
  cleanup, and the existing outer-loop time limit are covered. This is not a
  new mid-kernel cancellation implementation.
- CLI checks cover all five commands, explicit selection, honest unknown aggregate
  GPU-use fields, missing library errors and rejection of unsupported worker counts
  before catalogue loading/search.

[Checkpoint and reproduction sources](../artifacts/performance/gtoc12-discretisation-v85-checkpoint.json),
[paired timings](../artifacts/performance/gtoc12-discretisation-v85-benchmark.json),
[additional qualification including the failed case](../artifacts/performance/gtoc12-discretisation-v85-additional-legs.json),
[CUDA checks](../artifacts/performance/gtoc12-discretisation-v85-checks.json),
[final combined regression](../artifacts/performance/gtoc12-discretisation-v85-regression.json).

Runtime v87 adds a fixed native conic layout and GPU coefficient assembly using
these device outputs. It retains all structurally possible entries, but the
transitional bridge still downloads matrices for Clarabel. Replacing Clarabel
requires equivalent objective, cone and residual qualification at the existing
1e-9 conic tolerance. Host decisions and true GPU batching remain open.
