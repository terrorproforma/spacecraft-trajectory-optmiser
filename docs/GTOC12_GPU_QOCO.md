# Experimental native GPU GTOC12 conic solver

GTOC12 can now pass retained CUDA conic matrices directly to the GPU QOCO
adapter. This connects native interval integration, native conic assembly,
parallel numerical conversion, the GPU conic solve and a retained device primal.
It is an **experimental opt-in**, not a faster or generally qualified replacement
for the default Clarabel backend. Difficult transfer cases still fail. No overall
speedup or fully GPU-native trajectory pipeline is claimed.

## Selection and qualification

Set `SPACEPDHCG_GTOC12_CUDA_LIBRARY` to the new native core and
`SPACEPDHCG_QOCO_LIBRARY` to the patched GPU QOCO library. The tested Linux paths
are recorded in the checkpoint. Refinement CLI commands accept:

```text
--discretisation-backend cuda --assembly-backend cuda --convex-solver qoco --qoco-ruiz-iterations 0
```

The equivalent `ScvxSettings` fields are `discretisation_backend="cuda"`,
`assembly_backend="cuda"`, `convex_solver_backend="qoco"` and
`qoco_ruiz_iterations=0`. Existing defaults and the legacy QOCO adapter's tolerance
remain unchanged. Missing device extensions cause an explicit error; this path
does not fall back to a CPU conic solver.

The requested conic tolerance remains the reference setting, 1e-9 by default.
The new path drives QOCO internally to 1/100 of that tolerance, then independently
requires finite audited primal and dual residuals and a global relative objective
gap at or below the requested tolerance. Raw QOCO status 1 or 2 is necessary but
insufficient. The global objectives are reduced from the original unscaled
device matrices and mapped primal/dual vectors:

```text
primal = 0.5 x'Qx + q'x
dual   = -0.5 x'Qx - b'y - h'z
gap    = abs(primal - dual) / max(1, abs(primal), abs(dual))
```

The existing residual audit includes dual-cone feasibility and per-cone
complementarity. The added global gap catches inaccurate objectives that those
local checks can miss. Rejected solves leave the host primal output untouched;
the Python bridge returns NaNs and records the rejection. Unavailable audit
values are JSON null. `solver_reports` records every attempted subproblem, with
actual inner iterations, status, residuals, objectives and adapter counters.
Adapter timing and transfer counters are cumulative for the retained workspace;
take successive differences for per-attempt costs rather than summing snapshots.

Conic qualification is not a nonlinear physics certificate. Complete trajectories
must still pass independent propagation, endpoint/thrust/mass checks and the
fixed objective comparison. A solver-reported infeasible result does not prove
physical infeasibility.

## Conversion and device ownership

The wrapper compiles CSC maps once, mirrors upper-triangular P into full symmetric
Q, and splits equality/inequality A from SOC F. Equalities have b/b scalar bounds;
inequalities have -infinity/b bounds. Variables are free because their bounds
already occur as explicit rows. SOC data are F=-A_soc and offset=b_soc, with the
required permutation from Clarabel's `[t,vector]` into canonical `[vector,t]`.

For ZOH, all four final control components are already fixed to zero by equality
rows. The wrapper omits exactly their redundant Gamma bounds and final thrust
cone. It retains reference-dependent control trust rows. The original assembly
API is unchanged; tests audit the returned primal against its uneliminated
constraints, including the omitted rows.

The new C API is in `gtoc12_qoco_c_api.h`. Instances own buffers, maps and solver
state, are device-bound and serialized, and support repeated numerical updates.
`solve_device` takes caller device inputs and returns a retained device primal.
`solve_host` uploads states, controls and parameters, then downloads only a
qualified primal. The native orchestration is synchronous and not graph-capturable;
there is no mid-solve cancellation. External stream work must finish before
workspace reuse or destruction.

Initial topology/conversion setup still downloads matrices and performs CPU
metadata work. Successful retained numerical updates use device buffers; solver
recreation can repeat setup transfers. QOCO control flow and scalar diagnostics,
the Python SCvx outer loop, seed generation and independent verification remain
on the CPU. Adapter transfer counters exclude the bridge, four objective scalars
and opaque QOCO-internal transfers. They are not total process traffic.

## Hill-climb record

- v88 established the device connection. Its residual-only gate accepted some
  subproblems with objective error above 1e-8. Those results are not promoted.
- v89 added the global GPU objective-gap gate. Representative transfer reliability
  remained poor, including a physics-certified leg that missed the fixed mass.
- v90 tightened both internal tolerance and iterative refinement. All six transfer
  attempts failed physics checks. Its iterative-refinement changes were reverted.
- v91 removed only the redundant final ZOH constraints. One of six transfer
  attempts passed physics and the fixed mass target. A coast test failed once and
  passed on rerun; both records are retained. This is unresolved numerical
  sensitivity, not evidence of repeatable convergence.
- v92 combined the reduced formulation with a tighter internal solve tolerance,
  leaving iterative refinement unchanged. One of six transfer attempts qualified:
  ZOH/Ruiz 0 reached 2445.3111007852112 kg, matching the fixed reference, in 1.297 s.
  The other five failed physics. This single success is not a timing distribution.
- v93 retains v92 numerics at the normal requested tolerance. It clears all
  unavailable residual fields on invalid input and prevents internal tolerance
  underflow for very small positive requests.

## Final repeatability comparison

On the local RTX 5090, all 18 attempts for the fixed 150-day ZOH leg converged and
passed independent physics certification and the unchanged 1e-5 kg mass gate.
One warm-up per backend is retained in the artifact and excluded from the timing
median. The other five runs per backend rotate execution order:

| Dynamics / assembly / solver | Median full-leg time | Outer iterations |
|---|---:|---|
| NumPy / NumPy / Clarabel | 147.461 ms | 5 each |
| CUDA / CUDA / Clarabel | 145.656 ms | 7 each |
| CUDA / CUDA / QOCO, Ruiz 0 | 818.922 ms | 8, 9, 9, 13, 6 |

GPU QOCO is **5.62x slower** than CUDA assembly with Clarabel on this fixture.
Its five measured runs pass the fixed mass comparison with maximum error
2.12e-9 kg, but 11 of their 45 subproblem attempts are rejected by qualification.
Convergence remains sensitive, and this successful fixture does not override the
failed hold/equilibration cases above. The backend is not promoted as an
optimization win.

Using only the final cumulative snapshot from each measured QOCO workspace, the
adapter solve region totals 3.764 s out of 4.047 s full-leg wall time. Setup totals
0.171 s, numerical update 0.016 s and residual audit 0.007 s. These are host wall
regions, not a GPU kernel profile, but they put the next investigation inside
the conic solver rather than in the Python bridge.

See [all timing attempts](../artifacts/performance/gtoc12-qoco-v93-benchmark.json)
and the source/runtime checkpoint for reproduction. Rejected cases remain in the
artifacts; they must not be filtered out when comparing backends. The next work is
convergence and conditioning on the same fixed problems before moving outer-loop
computation or adding trajectory batching.

## Validation and reproduction

The final regression passes all 297 tests without skips. This includes all holds
and endpoint options on the coast fixtures, repeated numerical updates, audits
against original CPU constraints/objectives, invalid-input recovery, rejection
without output, missing-extension errors and exception cleanup. The legacy native
conversion test also passes its CPU oracle and mutation contracts.

The native device/host API test passes all four CUDA sanitizer tools: no memory
errors or leaks, no race hazards or warnings, and no initialization or
synchronization errors. Four Python bridge lifecycle tests pass memory,
initialization and synchronization checking.
These checks cover the stated fixtures, not full-leg race checking or cancellation
during QOCO iterations.

- [Source, frozen runtimes and reproduction helpers](../artifacts/performance/gtoc12-qoco-v93-checkpoint.json)
- [Full regression and lifecycle memory check](../artifacts/performance/gtoc12-qoco-v93-regression.json)
- [Native sanitizer and legacy conversion checks](../artifacts/performance/gtoc12-qoco-v93-sanitizers.json)
- [Python bridge initialization and synchronization checks](../artifacts/performance/gtoc12-qoco-v93-host-sanitizers.json)
- [Earlier failing hold/equilibration attempts](../artifacts/performance/gtoc12-qoco-v92-legs.json)

The final core is `/home/angus/build-spacepdhcg-gtoc12-v93/final/libspacepdhcg_cuda.so`
inside WSL. It uses the frozen QOCO78 library and cuDSS/CUDA paths in the checkpoint.
Restore its embedded `helper_sources` to their named paths to reproduce the build,
benchmark and checks. Use a fresh runtime directory (v94 or later); frozen
experimental runtimes and their recorded results must not be overwritten.
