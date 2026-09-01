# Gate G3 report — device-resident deterministic SCvx

## Decision

**G3 FAIL. G4 is not authorised.**

The analytic device coefficient path and values-only persistent CQP path are working, but the
authoritative final-forcing qualification failed before a production outer-loop or matched-quality
H1 campaign could be accepted. On the correctness-sized 3-DoF qualification, a requested
canonical residual of `1.0e-6` reached the backend iteration limit at 1,000,000 iterations. The
independently reported natural residual was `5.76465191e-4`, with relative primal residual
`1.38560229e-8` and relative dual residual `7.55280608e-5`. This is not reclassified as a pass.

Evidence was sealed from source commit `832aaf4` under
`results/gpu/g3/g3-20260901T000027Z-832aaf4`. The archive SHA-256 is
`ed37f0e34d6f10df387bbd2ef8b35d27f2ab39ab85b3ac4b204107731bf127e1`.

## Acceptance criteria

1. **All deterministic families meet final quality gates — FAIL.**
   HCW, 3-DoF powered descent, low thrust, and 6-DoF all execute the device coefficient,
   direct-CSC, values-only update, retained warm-state, solve, and independent-residual path.
   Their deliberately loose repair-phase run has maximum natural residual `2.42428348e-3`,
   which is below its `1.0e-2` repair forcing request and is therefore not itself a final-gate
   failure. The separate final qualification requested `1.0e-6` and achieved
   `5.76465191e-4` with iteration-limit termination, so no final accepted SCvx result is claimed.

2. **Production SCvx trust/forcing outer loops and CPU/GPU parity — FAIL / not qualified.**
   The repository has a resident all-family CQP integration test, not a complete production
   nonlinear outer driver. Consequently accepted/rejected histories, trust histories,
   predicted/actual reductions, exact-penalty and virtual-control convergence, nonlinear dense
   replay, and final CPU/GPU trajectory parity were not generated. Implementing or benchmarking
   those paths after the final inner-forcing failure would not satisfy the locked inexact-SCvx
   contract.

3. **Analytic device coefficient correctness — PASS.**
   Maximum CPU/device differences were:
   HCW `2.842e-14`, 3-DoF `8.327e-17`, low thrust `4.139e-13`, and 6-DoF `1.110e-16`.
   Maximum quaternion radial sensitivity was `1.891e-16`. Production finite differences are
   disabled. Exact affine reconstruction and direct fixed-CSC writes pass, with stable pointers.

4. **Steady-state residency and lifetime invariants — PARTIAL PASS.**
   The persistent integration records zero post-create topology allocation delta, zero topology
   index-copy delta, zero update-allocation delta, stable value/primal/dual pointers, full-retained
   warm starts, and `hidden_cpu_fallback=false`. These counters establish the tested values-only
   CQP path. They do not establish the missing production nonlinear outer loop.

5. **Nsight Systems trace — NEGATIVE RESULT RETAINED.**
   Nsight Systems 2024.6.2 recorded CUDA API activity (73 kernel-launch API calls in the
   representative integration run), but its generated database contained neither CUDA kernel
   records nor GPU memory records under this WSL environment. The kernel and memory summary
   reports were explicitly skipped. Therefore no kernel-timeline residency claim is made from
   this trace; the raw `.nsys-rep`, SQLite database, profile log, and stats are retained in the
   ignored evidence archive.

6. **H1 decision — UNRESOLVED / NOT QUALIFIED.**
   H1 is neither supported nor rejected. A preregistered matched-quality size sweep cannot be
   evaluated without a production SCvx outer loop that passes the final forcing and nonlinear
   result gates. There is no defensible scale boundary or confidence interval. Kernel-only and
   repair-phase timings are deliberately not substituted for `T_CQP` or `T_SCvx`.

7. **Build, test, and sanitizer gates — PASS for implemented scope.**
   Ruff passed; Python passed 88 tests. Debug and Release CUDA/native CTest each passed 50 tests
   under warnings-as-errors builds. The coefficient kernels passed memcheck, racecheck,
   initcheck with unused-memory tracking, and synccheck for all four models. The bounded
   persistent integration path passed all four tools; the unbounded convergence run under
   racecheck was stopped after 484 seconds and is not represented as completed sanitizer
   coverage.

## Implemented commits

- `9c42bf5` — analytic HCW, 3-DoF, low-thrust, and 6-DoF device coefficients.
- `eef02bd` — fully initialised sanitizer-clean outputs.
- `bf50194` — all-family values-only persistent integration and retained-state checks.
- `832aaf4` — tight final-residual qualification and reproducible G3 evidence runner.

## Preserved blockers

- Final requested canonical residual: `1.0e-6`.
- Achieved natural residual: `5.76465191e-4`.
- Backend termination: iteration limit after 1,000,000 iterations.
- Production nonlinear outer-loop parity: unavailable because the final forcing gate failed.
- H1: unresolved and unqualified; no scale boundary claimed.
- Nsight GPU kernel/memory timeline: unavailable in the captured WSL trace.

G4 must remain stopped until the final canonical residual is achieved, the production all-family
outer loops pass matched CPU/GPU nonlinear quality checks, and H1 is evaluated with the locked
end-to-end schema.
