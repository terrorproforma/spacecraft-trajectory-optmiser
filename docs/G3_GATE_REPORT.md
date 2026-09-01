# Gate G3 report — device-resident deterministic SCvx

## Decision

**G3 FAIL. G4 is not authorised.**

The exact failed CQP has now been reproduced, dumped at full precision, and compared with the CPU
reference and pinned upstream PDHCG. The data, topology, stream, warm-state, scaling-refresh, dual
sign/order, and result-copy paths are not the cause. The current persistent first-order iteration
has a stationarity floor near `5.7e-4`; pinned upstream PDHCG also fails the same absolute
physical-unit natural-residual gate. The authoritative final request remains `1.0e-6` and is not
reclassified as a pass.

Evidence was sealed from source commit `832aaf4` under
`results/gpu/g3/g3-20260901T000027Z-832aaf4`. The archive SHA-256 is
`ed37f0e34d6f10df387bbd2ef8b35d27f2ab39ab85b3ac4b204107731bf127e1`.
That negative archive is immutable and remains part of the record.

## Exact reproduction and independent comparison

The reproduced CQP has 57 variables, 57 scalar rows, 49 affine-cone rows, eight SOC blocks, and
full-data SHA-256
`e6547c15187fbac09dd0c1cbb7a4eff1b4c8561eb07b2f71aaa30dff6c031817`. The hash covers Q/A/F
topology and values, `c/l/u`, affine offsets, variable bounds, and cone inventory.

- Persistent CUDA, cold, 1,000,000 iterations: iteration limit, natural residual
  `5.76465191e-4`, relative primal `1.38560229e-8`, relative dual `7.55280608e-5`.
- Pinned upstream one-shot PDHCG, cold, 1,000,000 iterations: iteration limit, independently
  recomputed absolute natural residual `4.16380274e-3`, relative natural residual
  `9.01459953e-7`, relative primal `2.08086094e-6`, relative dual `8.19524041e-7`, and relative
  objective gap `1.69020000e-2`.
- CPU Clarabel, unchanged CQP: solved in 18 iterations; independently recomputed natural residual
  `1.90266292e-9`, objective `1.049277319925314`, terminal residual `9.15863522e-14`, linearised
  dynamics residual `2.74302883e-13`, and nonlinear RK4 dynamics residual `2.27373675e-13`.

Supplying the CPU primal and correctly sign-converted `[dual_A, dual_F]` pair to upstream PDHCG
terminates in 410 iterations with independently recomputed natural residual `2.75861566e-7`.
Primal-only warm start does not. This isolates the ordering/sign convention and proves that the
same upstream model can recognise a qualified KKT pair; it is diagnostic evidence, not an allowed
production handoff.

## Root cause

The definitive blocker is an algorithmic convergence/capability mismatch on this ill-conditioned
CQP, not a formulation or lifecycle corruption. Q spans `1e-8` to `1e-2`, affine coefficients span
approximately `6.67e-5` to `1.732`, and physical bounds reach `15,000`. The persistent iteration
makes primal feasibility tight but stalls in physical-unit stationarity. Upstream's scaling and
adaptive restarts move the primal/dual balance, but still do not produce a point satisfying the
locked absolute natural residual and objective quality.

Diagnosis also found a separate residual-contract defect: the persistent reporter omitted scalar
and affine-cone dual natural terms and stopped on backend-relative primal/dual quantities. It now
computes the complete projection natural residual and uses that canonical quantity for requested
tolerances. A focused SOC regression proves that a primal-feasible, stationary point with a
non-normal cone dual is rejected. Correcting this defect does not hide or resolve the tight
stationarity floor.

## Controlled ablations

| Identical-CQP run | Iterations | Absolute natural residual | Result |
|---|---:|---:|---|
| Persistent, cold, non-default stream | 1,000,000 | `5.76465191e-4` | FAIL |
| Persistent, cold, default stream | 1,000,000 | `5.76465191e-4` | FAIL; bit-identical |
| Persistent, loose then tight, scaling reused | 1,000,000 tight | `5.76198300e-4` | FAIL |
| Persistent, loose then tight, scaling refreshed | 1,000,000 tight | `5.76198300e-4` | FAIL |
| Persistent, tight then identical tight re-solve | 2 × 1,000,000 | `5.74567542e-4` | FAIL |
| Experimental cone-compatible diagonal steps | 1,000,000 | `5.92542419e-4` | FAIL; not retained |
| Experimental diagonal steps, primal weight 10 | 1,000,000 | `5.67979534e-4` | FAIL; not retained |
| Upstream default, cold | 1,000,000 | `4.16380274e-3` | FAIL |
| Upstream own primal-dual re-solve | 2 × 1,000,000 | native residuals worsened | FAIL |
| Upstream Ruiz 20 | 300,000 | `6.25859475e-4` | FAIL |
| Upstream Curtis-Reid 5 | 300,000 | `6.89690486e18` | numerical divergence |
| Upstream PC alpha 0.5 | 300,000 | `6.12314437e-1` | FAIL |
| Upstream PC disabled | 300,000 | `9.85548172e-2` | FAIL |
| Upstream faster restart thresholds | 300,000 | `1.38199590e-3` | FAIL |
| Upstream restart Kp 0.5 / 0.0 | 300,000 | `1.08278805e-1` / `2.88757434e-2` | FAIL |
| Upstream bound/objective rescaling disabled | 300,000 | `1.33778925e-3` | FAIL |
| Upstream reflection 0.5 | 300,000 | `4.77791026e-3` | FAIL |

The persistent checkpoints were `1.13954252e2` at 1,000 iterations, `1.64544282e-1` at 10,000,
`5.90049919e-4` at 100,000, `5.87181378e-4` at 300,000, and `5.76465191e-4` at 1,000,000. The
near-flat tail and negligible second-million improvement rule out a merely short iteration cap.
All primary buffers and both solver paths are float64; float precision is not the source of the
floor.

## Acceptance criteria

1. **All deterministic families meet final quality gates — FAIL.**
   HCW, 3-DoF powered descent, low thrust, and 6-DoF all execute the device coefficient,
   direct-CSC, values-only update, retained warm-state, solve, and independent-residual path.
   Their deliberately loose repair-phase run now stops on the canonical residual and has maximum
   natural residual `9.81253727e-3`, below its `1.0e-2` repair forcing request. The separate final
   qualification requested `1.0e-6` and achieved
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
   persistent integration and focused canonical-residual regression paths passed all four tools.

## Implemented commits

- `9c42bf5` — analytic HCW, 3-DoF, low-thrust, and 6-DoF device coefficients.
- `eef02bd` — fully initialised sanitizer-clean outputs.
- `bf50194` — all-family values-only persistent integration and retained-state checks.
- `832aaf4` — tight final-residual qualification and reproducible G3 evidence runner.
- Follow-up diagnostic/residual-contract commit and archive: recorded below after sealing.

## Preserved blockers

- Final requested canonical residual: `1.0e-6`.
- Achieved natural residual: `5.76465191e-4`.
- Backend termination: iteration limit after 1,000,000 iterations.
- Production nonlinear outer-loop parity: unavailable because the final forcing gate failed.
- H1: unresolved and unqualified; no scale boundary claimed.
- Nsight GPU kernel/memory timeline: unavailable in the captured WSL trace.

The narrow next technical requirement is either a persistent cone-preserving scaling/restart or
other first-order strategy demonstrated to reduce the independently recomputed absolute natural
residual below `1.0e-6`, or a separately authorised and labelled final GPU interior-point polish
with all handoff/setup cost. CPU Clarabel is reference evidence and is not relabelled as PDHCG
success. The frozen G3 work does not authorise silently substituting it for the resident backend.

G4 must remain stopped until the final canonical residual is achieved, the production all-family
outer loops pass matched CPU/GPU nonlinear quality checks, and H1 is evaluated with the locked
end-to-end schema.
