# Native GTOC12 conic assembly

The explicit CUDA assembly path now runs interval linearisation and conic numerical
assembly on the GPU. It removes Python sparse-matrix construction from the outer
loop. The fixed topology is compiled in C++ once; values, right-hand sides, the
linear objective and quadratic smoothness objective are filled in parallel.

The device API leaves those arrays on the GPU for the next solver integration.
The current Python bridge downloads them for CPU Clarabel. This path is available
for development and qualification; it is **not the default or a general full-solve
speed improvement**. The representative complete solve currently regresses,
despite the faster coefficient path. CPU seed generation, Clarabel, outer-loop
decisions and independent verification remain; the full GPU-native goal is open.

## Representation and physics

The layout exactly represents the existing GTOC12 convex subproblem:

```text
minimize 0.5 x' P x + q' x
subject to A x + s = b
s in Zero(equalities) x Nonnegative(inequalities) x SOC(4) x ...
```

Variables keep the existing order: nodal states, nodal four-component controls,
interval virtual controls, absolute-value epigraphs, optional departure v-infinity
and optional arrival v-infinity. Both thrust holds and all four endpoint choices
are supported. ZOH fixes the unused final control to zero.

The native layout includes dynamics, endpoint conditions, mass and solar-radius
bounds, state/control trust regions, virtual-control penalties, thrust SOCs,
optional v-infinity SOCs, fuel weights and the upper-triangular control-smoothness
Hessian. Units, RK4 substeps, norm-of-interpolated-thrust mass flow and the Gamma
variational surrogate follow the existing model. No fast math or fused multiply
add is enabled. Conic tolerance remains 1e-9; independent physics gates are unchanged.

The matrix pattern includes all potentially nonzero entries, including those that
are zero at the current iterate. The mass variational row has only Phi[6,6] and
Psi[6,3]: its other derivatives are identically zero and omitted at construction.
The topology never changes as thrust switches between coast and burn.

That distinction matters numerically. Initial runtime v86 stored the identically
zero mass-row entries too. Matrix parity passed, but the representative ZOH leg
took 28 iterations and violated the thrust gate (0.6000002424951428 N). A diagnostic
probe removing zeros restored qualification. Omitting only the proven structural
zeros then restored qualification while retaining a fixed pattern. Runtime v87
implements that rule; the rejected v86 runtime and evidence remain recorded.
This is evidence of sensitivity to sparse representation, not a general convergence
reliability fix. No output clipping or tolerance relaxation was used.

## API and use

`spacepdhcg_gtoc12_conic_launch_device` accepts device state, control and parameter
pointers and an external CUDA stream. It runs interval dynamics, then multi-block
assembly with 256 threads per block (up to 1,024 blocks). It allocates nothing and
does not synchronize with the host. Device parameters can change before a graph
replay. Output CSC topology and numerical buffers persist until reuse/destruction;
a device flag reports invalid dynamics, parameters or arithmetic.

The workspace is device-bound and serialized. Callers must finish external-stream
work before reusing or destroying it. The host bridge owns a nonblocking stream,
uploads states/controls/parameters, and downloads one packed numerical buffer plus
the validity flag before synchronizing. Topology is copied from retained host
metadata only at Python construction. Bridge outputs have independent values and
share immutable topology. No new mid-kernel cancellation is implemented.

Frozen runtime:

```text
/home/angus/build-spacepdhcg-gtoc12-v87/final/libspacepdhcg_cuda.so
```

With the existing WSL environment and GTOC12 catalogue:

```bash
export PYTHONPATH=src
export SPACEPDHCG_GTOC12_CUDA_LIBRARY=/home/angus/build-spacepdhcg-gtoc12-v87/final/libspacepdhcg_cuda.so
python -m spacepdhcg gtoc12 run --run-id cuda-assembly \
  --output results/gtoc12/runs/cuda-assembly \
  --discretisation-backend cuda --assembly-backend cuda
```

All five refinement commands expose the assembly option. Commands with a workers
option require `--workers 1` for CUDA; missing native configuration or selecting
CUDA assembly with NumPy dynamics fails explicitly. Completed leg reports identify
actual assembly and dynamics backends; selecting a backend alone does not claim
that GPU work occurred. CPU assembly remains the default.

## Matched RTX 5090 measurements

One process, one BLAS thread, rotating order across three backends, two warmups and
nine measured repetitions. Coefficient times include linearisation, assembly and
host output/transfers; initial workspace construction is excluded. Full-leg times
include construction/destruction and CPU work, with independent certification
outside the timer. GPU utilization was 0% before the serialized run. These are WSL
wall timings, not isolated CUDA-event kernel times or Lambda measurements.

| Intervals | Hold | CPU dynamics + assembly, ms | CUDA dynamics + CPU assembly, ms | CUDA dynamics + assembly bridge, ms |
|---:|---|---:|---:|---:|
| 75 | ZOH | 6.229 | 3.278 | 0.431 |
| 75 | Lagrange | 11.785 | 4.646 | 0.626 |
| 257 | ZOH | 19.238 | 9.532 | 0.542 |
| 257 | Lagrange | 37.436 | 15.716 | 0.796 |
| 2,001 | ZOH | 177.027 | 108.062 | 1.615 |
| 2,001 | Lagrange | 320.229 | 164.920 | 2.572 |

The coefficient path at 2,001 intervals is 66.92x faster for ZOH and 64.12x for
Lagrange than CUDA dynamics plus CPU assembly. Against NumPy dynamics plus CPU
assembly, the corresponding ratios are 109.62x and 124.51x. These are phase gains.

The representative 150-day ZOH leg medians are **140.045 ms NumPy, 123.214 ms CUDA
dynamics, 151.595 ms CUDA dynamics plus assembly**. The new path is 23.0% slower
than CUDA dynamics with CPU assembly, and is not promoted as the faster full-leg
backend. All 33 runs converge and pass independent certification, with final mass
within 1e-5 kg of the fixed 2445.3111007852112 kg reference. The new pattern takes
seven outer iterations versus five on the earlier path. A separate instrumented
sample attributes about 130 of 146 ms to Clarabel construction and solve.

Five additional CPU/CUDA pairs qualify across 0.5/2/8-day grids and both holds.
The free-endpoint-velocity case still produces solver-reported infeasible results
and fails qualification on both backends. Physical infeasibility is not proven;
this failed case is retained and excluded from successful qualification claims.
The additional cases are single runs, not timing distributions.

## Checks and evidence

- Matrix, RHS, objective and cone parity: 32 combinations across 3/17/257/2,001
  intervals, both holds and all endpoint options; each updates inputs three times
  with 1/8/16 substeps, changing trust/penalty/smoothness values and coast-to-burn
  transitions. Absolute matrix error <=2e-12; RHS absolute/relative <=2e-12;
  objective exact. Topology remains sorted, unique and unchanged.
- Three complete-leg tests retain convergence, independent physics qualification
  and final-mass agreement <=1e-5 kg. Invalid inputs/parameters, flag reset, closed
  handles, ownership of returned values and exception cleanup are tested.
- Native API: 48 changing-input graph replays, independent layout counts and
  objective/Hessian checks, both holds and small/large grids. All four CUDA
  sanitizer tools pass, with zero reported errors or race warnings and no leaks.
- Host bridge: five tests covering complete legs, invalid input recovery and
  solver-exception cleanup pass memory, initialization and synchronization checks.
  Full-leg racecheck was not run.
- The focused combined selection passes 78 tests. The broader GTOC12, telemetry,
  planner-schema and existing native SCvx regression passes all 284 tests with no
  skips (377.33 s). The additional dynamics lifecycle memcheck reports no errors
  or leaks. Ruff and Git whitespace checks pass.

[Checkpoint and reproduction sources](../artifacts/performance/gtoc12-conic-v87-checkpoint.json),
[matched timing samples](../artifacts/performance/gtoc12-conic-v87-benchmark.json),
[focused tests](../artifacts/performance/gtoc12-conic-v87-checks.json),
[sanitizers](../artifacts/performance/gtoc12-conic-v87-sanitizers.json),
[additional cases](../artifacts/performance/gtoc12-conic-v87-additional-legs.json),
[combined regression](../artifacts/performance/gtoc12-conic-v87-regression.json),
[rejected v86](../artifacts/performance/gtoc12-conic-v86-rejected.json).

## Next native connection

The retained device outputs are in Clarabel's combined CSC convention. The existing
native QOCO adapter consumes canonical Q/A/F arrays. A fixed conversion can split
scalar equalities/inequalities from SOC rows once, mirror upper-P topology once,
and update numerical values on the GPU. Scalar bounds are b/b for equality rows
and -infinity/b for inequality rows. SOC data are F=-A_soc and offset=b_soc.
Variables are free because all bounds already appear as rows.

The QOCO adapter currently configures its own tolerance to 1e-8, whereas this
GTOC12 reference uses 1e-9. The new connection must expose the correct requested
tolerance and independently qualify objective, residuals and nonlinear physics;
the adapter currently rejects nonfinite audited residuals, but its success return
alone does not enforce a caller-selected audit threshold. That connection, CPU outer
decisions and true GPU trajectory batching remain to be implemented and measured.
