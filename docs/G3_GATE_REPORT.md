# Gate G3 report — device-resident deterministic SCvx

## Current-head regression closure — 2026-09-03

G3 **PASS** on tested source commit
`b6afb49d7fc7da5ed1ac9003c3bcae5d35506026` under `single-gpu-v1`.
The maximum tight canonical residual is `9.69295039e-7`; displaced HCW accepts three steps,
and direct non-campaign pure-QOCO P1-C/P1-D/P1-E warmup regressions accept 2/24/2 steps.
The matched fixed-tight PDHCG representatives remain retained 150-second negatives with zero
accepted steps. All 62 Debug and 62 Release CTests, H1, and sixteen sanitizer runs pass with no
fallback or post-create topology churn. The current local-only archive is
`results/gpu/current-head-b0cd570/seals/g3-b6afb49d7fc7.tar.gz`, SHA-256
`08ccc71bec848cea24c64b1efbf97e50962a2db14a4d6a2099b7750451756810`.

G4 is scientifically authorised by G0-G3, but launch remains operationally blocked until an
official capability is regenerated for the final clean executable. No G4 campaign was launched.
See `CURRENT_HEAD_G0_G3_REPORT.md` for complete current-head evidence. Historical wording and
negative archives are retained below.

## Decision

**G3 historical PASS amended: displaced-reference outer-loop parity was not established.**

The original G3 campaign established nominal-reference residency, lifecycle, coefficient-generation,
and solver diagnostics. Gate G4 later exposed that this wording overstated the production outer-loop
coverage: the all-family parity fixture started from references whose nonlinear rollout already
matched the reported final trajectory, so rejected candidates still produced zero trajectory
difference. It did not verify coefficient evolution after an accepted displaced step.

The corrective implementation now updates reference tracking objectives, trust-cone centres and
radii, exact-penalty epigraph costs, low-thrust radial halfspaces, 6-DoF quaternion linearisations,
and nonlinear dynamics in place. Direct CPU/device coefficient parity has maximum absolute
difference `1.8118839761882555e-13` (contract `5e-12`), but the
frozen displaced-start outer-loop qualification remains above matched-quality tolerances because
the current PDHCG production solve eventually exhausts the trust region. Therefore this amendment
does not reseal G3 as a displaced-start PASS, and it does not authorise G5.

The final clean campaign ran from implementation commit
`9dcc070938594c12b0e54cad0b57d553600f4522`. Transactional projected-KKT/CGLS recovery now removes
the previously diagnosed stationarity floor without a host solve. Production CUDA outer drivers
execute HCW, 3-DoF powered descent, low thrust, and 6-DoF through device coefficient generation,
values-only update, retained warm state, forcing, identical-CQP re-solve, nonlinear replay,
trust/merit acceptance, checkpoint rollback, and compact host diagnostics.

The maximum tight canonical residual is `7.39885877e-7`, the maximum final nonlinear residual is
zero for the qualified fixed-point parity fixtures, and the maximum CPU/GPU trajectory difference
is zero. All repeated topology allocation and topology-copy deltas are zero, and
`hidden_cpu_fallback=false`.

The full evidence directory is
`results/gpu/g3/g3-20260901T050705Z-9dcc070`. The final resealed archive, including explicit
1,000,000-iteration solve plus 50,000-iteration recovery costs, is
`results/gpu/g3/g3-20260901T050705Z-9dcc070-final-9f766c1.tar.gz`, SHA-256
`0a404b546a5fd83f2a466967ac76b5ed2bf61958c725e73a24f1e66a42096ff8`.

The earlier negative archives and controlled-ablation archive below remain immutable. They are the
pre-recovery baseline and were not deleted or reclassified.

Evidence was sealed from source commit `832aaf4` under
`results/gpu/g3/g3-20260901T000027Z-832aaf4`. The archive SHA-256 is
`ed37f0e34d6f10df387bbd2ef8b35d27f2ab39ab85b3ac4b204107731bf127e1`.
That negative archive is immutable and remains part of the record.

The definitive follow-up was sealed from source commit `5fb59fb` under
`results/gpu/g3/g3-20260901T004121Z-5fb59fb`. Its archive SHA-256 is
`49bb2418cd5b3ae10c785ff3f93137ac13f5101f3cc7c53cadd3a41a5643482f`. It
contains the exact full-precision CQP dump, independent CPU and upstream qualifications, warm-dual
convention check, expected tight failure, builds/tests, sanitizers, and the retained WSL Nsight
limitation.

The focused controlled-ablation archive
`results/gpu/g3/diagnostic-20260901T100544Z.tar.gz` has SHA-256
`ffbec34391bc35ef0582747d81633cd21a7fe5edf02902d01f36954267f35269`.

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

## Historical root cause and closure

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
non-normal cone dual is rejected. Correcting this defect did not hide the floor. The final
device-only recovery closes it with bound/cone image projections, canonical dual reconstruction,
restarted CGLS, projected KKT correction, and transactional rollback for every rejected candidate.

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

1. **All deterministic families meet final quality gates — PASS.**
   The unchanged `1.0e-6` request produced HCW `0`, 3-DoF `1.42322019e-10`, low thrust
   `7.39885877e-7`, and 6-DoF `9.31049726e-9`.

2. **Production SCvx trust/forcing outer loops and CPU/GPU parity — PASS.**
   All four production paths run coefficient, update, scaling, solve/recovery, residual, nonlinear
   replay, exact-penalty/virtual-control, reduction-ratio, trust decision, checkpoint, and compact
   diagnostic stages. HCW accepted its zero step at radius `1`; the nonlinear families rejected
   non-improving proposals and retained their already-qualified CPU truth references at radius
   `0.5`. Maximum CPU/GPU trajectory, virtual-control, terminal, path, and dynamics differences
   are zero. This sequence difference is explained by the identical starts being nonlinear fixed
   points. Objectives are HCW `0`, 3-DoF `1.10164746`, low thrust `0`, and 6-DoF `1.10169199`.

3. **Analytic device coefficient correctness — PASS.**
   Maximum CPU/device differences were:
   HCW `2.842e-14`, 3-DoF `8.327e-17`, low thrust `4.139e-13`, and 6-DoF `1.110e-16`.
   Maximum quaternion radial sensitivity was `1.891e-16`. Production finite differences are
   disabled. Exact affine reconstruction and direct fixed-CSC writes pass, with stable pointers.

4. **Steady-state residency and lifetime invariants — PASS.**
   Production allocates all trajectory, replay, metric, checkpoint, projected-KKT, and CGLS
   scratch at create time. The loop records zero post-create topology allocation, zero topology
   index copy, no H2D trajectory/CQP transfer, stable pointers, device-to-device accepted-reference
   updates, compact D2H diagnostics only, and `hidden_cpu_fallback=false`.

5. **Nsight Systems trace — NEGATIVE RESULT RETAINED.**
   Nsight Systems 2024.6.2 recorded CUDA API activity (76 kernel-launch API calls in the
   representative integration run), but its generated database contained neither CUDA kernel
   records nor GPU memory records under this WSL environment. The kernel and memory summary
   reports were explicitly skipped. Therefore no kernel-timeline residency claim is made from
   this trace; the raw `.nsys-rep`, SQLite database, profile log, and stats are retained in the
   ignored evidence archive.

6. **H1 decision — SUPPORTED from 20 intervals.**
   The preregistered HCW sweep covered `20`, `50`, `100`, `500`, `2,000`, and `10,000` intervals
   with seven measured repeats per coordinate, matched canonical/nonlinear quality, and no
   censoring. Repeated topology allocation/copy overhead is zero, so median
   `omega_persist=0` and its seeded 10,000-resample bootstrap 95% interval is `[0, 0]` at every
   coordinate. Median `T_SCvx` rises from `0.053785286 s` at 20 intervals to `27.8976435 s` at
   10,000 intervals. The sustained supported boundary is 20 intervals.

7. **Build, test, sanitizer, and negative-control gates — PASS.**
   Ruff passed; Python passed 91 tests. Debug and Release native/CUDA warnings-as-errors CTest each
   passed 52 tests. Sixteen logs cover memcheck, racecheck, initcheck with unused-memory tracking,
   and synccheck across variational kernels, persistent integration, recovery failure/lifetime
   paths, and all-family production drivers. Every sanitizer summary is clean. The no-device
   negative control failed as required.

## Implemented commits

- `9c42bf5` — analytic HCW, 3-DoF, low-thrust, and 6-DoF device coefficients.
- `eef02bd` — fully initialised sanitizer-clean outputs.
- `bf50194` — all-family values-only persistent integration and retained-state checks.
- `832aaf4` — tight final-residual qualification and reproducible G3 evidence runner.
- `e27c2e4` — complete natural-residual contract, exact CQP diagnostics, controlled root cause,
  focused regression, and expanded evidence runner.
- `5fb59fb` — clean full-precision CQP serialization for independent replay.
- `15c8f10` — transactional projected-KKT/CGLS recovery and adversarial lifecycle/property tests.
- `7324b9c` — production resident SCvx drivers, parity coverage, transfer ledger, and H1 harness.
- `a0c84f5` — complete automated G3 qualification and machine decision.
- `36579b1`, `9dcc070` — create-time scratch initialization and deterministic sanitizer lifecycle.
- `84a2076` — correct per-tool sanitizer/test summary parsing.
- `9f766c1` — explicit 1M-plus-50k recovery cost telemetry.

## Resolved blockers and retained limitations

- Final requested canonical residual remains `1.0e-6`; maximum achieved is `7.39885877e-7`.
- The historical 3-DoF floor was `5.76465191e-4`; final recovery reaches `1.42322019e-10`.
- 3-DoF recovery used 50,000 device iterations and `13.5649619 s` of
  `18.9728335 s` CQP time after the 1,000,000-iteration request. 6-DoF recovery used 50,000
  iterations and `16.7068145 s` of `25.5232444 s`.
- Production nonlinear outer-loop parity is qualified with maximum trajectory difference zero.
- H1 is supported from 20 intervals under the preregistered confidence rule.
- Nsight GPU kernel/memory timeline: unavailable in the captured WSL trace.

The WSL Nsight limitation prevents a kernel-timeline residency claim; counters, transfer ledgers,
stable pointers, and sanitizer evidence establish the gate instead. CPU Clarabel remains only the
truth model and was never used as a production fallback. With every frozen G3 criterion passing,
G4 is authorised.

## Unified-roadmap regression

After native QOCO, G5, G6, and G7 integration, the Release persistent CW, persistent SOC,
production SCvx, and recovery GPU regressions all passed serially. The G7 one-GPU test also
exercised the concrete G3 backend callback/route seam. Native-QOCO P1-C produced clean memcheck,
racecheck, initcheck, and synccheck runs; a separate initcheck unused-pool diagnostic is attributed
to third-party cuDSS/cuBLAS allocations. These integration checks preserve the G3 implementation
status but do not replace the original frozen G3 evidence archive.
