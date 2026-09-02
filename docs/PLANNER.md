# SpacePDHCG planner (`spacepdhcg plan`)

The planner is the user-facing layer on top of the validated single-GPU device SCvx stack:
native persistent PDHCG workspace (G2), device-resident SCvx drivers with analytic
coefficients for HCW, 3-DoF/6-DoF powered descent and low thrust (G3), the native pure-QOCO
GPU-IPM outer loop with production nonlinear handback, device KKT/CGLS recovery, and the
CPU reference solvers. It turns one JSON problem document into a certified (or honestly
uncertified) plan with node/dense histories, telemetry, and a certificate.

## Components

| layer | location | role |
|---|---|---|
| Problem schema | `src/spacepdhcg/planner/schema/problem.schema.json` | JSON Schema 2020-12, versioned `1.0.0` |
| Python problem layer | `src/spacepdhcg/planner/problem.py` | schema + semantic validation, unit normalisation, family metadata |
| Native problem model | `cpp/include/spacepdhcg/planner/problem.hpp` | canonical-document parser with the native default table |
| Family adapters | `cpp/include/spacepdhcg/planner/families.hpp` | frozen transcriptions from user values, device layout metadata, initial references, RK4/ZOH replay, device-equivalent metrics |
| C ABI | `cpp/include/spacepdhcg/c_api.h` (`spacepdhcg_planner_*`) | topology/values/reference/rollout/evaluate for the CPU reference |
| Native executable | `cpp/cuda/tools/spacepdhcg_plan.cu` | persistent SCvx loop on the device driver, strict JSON result, deterministic exit codes |
| Python API/CLI | `src/spacepdhcg/planner/{api,cli,native_runner,cpu_reference,result,viewer_export}.py` | `plan(problem, options) -> PlanResult`, `spacepdhcg plan` |
| Examples | `examples/planner/` | one problem per family + commands |

## Backends

| backend | device policy | notes |
|---|---|---|
| `pure_qoco` (default) | `SPACEPDHCG_CUDA_SCVX_PURE_QOCO`, fixed inner tolerance 1e-8, 200 IPM iterations, primal warm start | the qualified P1-C/P1-D/P1-E path; needs `SPACEPDHCG_QOCO_LIBRARY` |
| `pdhcg` | `SPACEPDHCG_CUDA_SCVX_ADAPTIVE` with the frozen G4 forcing ceilings/iteration limits | fast persistent path |
| `pdhcg_recovery` | `SPACEPDHCG_CUDA_SCVX_FIXED_TIGHT`, inner tolerance min(tol, 1e-6), 1,000,000 iterations | enables the device projected-KKT/CGLS recovery (iteration limit ≥ 350 000) |
| `cpu_reference` | Python SCvx loop + Clarabel over the native transcription ABI | CPU only; labelled `execution: cpu_reference`; never substituted silently |

Presets: `frozen_adaptive_pure_qoco` (default for `pure_qoco`/`cpu_reference`),
`frozen_adaptive_pdhcg`, `fixed_tight_pdhcg`. Trust-region (initial 1, min 1e-4, max 8,
shrink 0.5, expand 1.8, acceptance 0.05, strong 0.75, boundary 0.8), penalty (feasibility
100, virtual = transcription L1 weight), and forcing values come from the frozen G4 policy
(`cpp/include/spacepdhcg/scvx/g4_policy.generated.hpp`) and can be overridden per document.

## Certificate

`certificate.certified` is `true` only when all gates pass at `certificate_tolerance`
(default = solver tolerance, 1e-6): solver API success, outer convergence, canonical
residual, device dynamics/path/terminal residuals, virtual control, independent host replay
parity (≤ 1e-9), independent dynamics/path/terminal residuals of the replayed controls, no
hidden CPU fallback, steady-state residency (zero post-create topology allocations/copies),
host/device coefficient parity (≤ 5e-12), and a successfully evaluated replay. The dense
continuous-time violation (RK4/ZOH replay with `dense_replay_substeps`) is reported and
flagged but not gated.

## Result document

`plan-result.json` (`result_kind: spacepdhcg_plan_result`, schema 1.0.0) contains `status`
(code/message/exit code/solver status), `problem` (resolved canonical document with units,
orders, solver policy), `summary`, `solver_residuals`, `independent_replay`,
`model_evaluation`, `initial_reference_evaluation`, `trajectory` (node times/states/controls),
`dense_replay`, `iterations` (per-iteration phase, tolerances, trust radii/action,
predicted/actual reductions, ratio, residuals, recovery telemetry, fingerprints), `timings`
(CUDA startup separate from topology/coefficients/workspace/update/scaling/solve/recovery/
replay/acceptance/D2H), `backend` (execution, policy, QOCO disposition, residency counters,
device), and `certificate`.

## Exit codes

`0` certified · `2` not certified (max iterations, trust region exhausted, time limit, gate
failure) · `3` inner solver failure (QOCO unavailable/numerical, PDHCG failure) · `64` invalid
problem/usage · `65` I/O · `66` CUDA unavailable/device error · `70` internal.

## Limitations

See `examples/planner/README.md` (fixed final time only, frozen terminal patterns, altitude
bound fixed at 0, node-level certificate, pure-QOCO speed). The planner never modifies the
frozen transcriptions or tolerances; requests outside their expressiveness fail closed with
an explicit message.
