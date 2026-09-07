# GPU reference-centred solver coordinates

**Unmerged candidate.** v138's broad suite recorded 331 passes and one strict
coast equality failure, 1.3945688767003704e-9 against 1e-9. Option-disabled
integration also recorded one coast failure and 88 passes. Reference centring
has not eliminated the issue; these failures prevent promotion to main.

The GTOC12 QOCO bridge can solve for corrections to the current reference
states. Write the physical primal as `x = delta + o`, where `o` contains the
reference states and zeros for controls, virtual controls, slacks and endpoint
velocity variables. This is a change of coordinates, not a change to dynamics,
feasibility, objective or tolerances.

For the conic formulation `A x = b`, `G x + s = h` and objective
`0.5 xᵀ P x + cᵀ x`, the translated solver data are:

```text
P' = P
c' = c + P o
b' = b - A o
h' = h - G o
```

The constant objective offset `0.5 oᵀ P o + cᵀ o` does not affect the minimizer.
The bridge reconstructs physical `x` on GPU before auditing the original
matrices, objective and duals, so reported objectives include that offset and
the original qualification thresholds still apply. The native implementation
supports a nonzero objective/Hessian over translated coordinates, even though
the current GTOC12 state prefix has neither.

## Why this change

The v136 coast failure was a real dynamics error, not an audit discrepancy.
A fresh rejected iterate had original equality error 7.34418854222032e-7;
the GPU audit reported exactly the same absolute residual. Its independently
recomputed objective agreed to about 4e-31. The worst rows were interval
position dynamics. This is an inaccurate candidate with near-zero objective;
the qualification gate correctly rejected it.

The balanced diagnostic ran 48 coast solves on published v135 and 48 on v136.
The stricter original-equality check failed once on v135 and three times on
v136. One v136 solve also failed the normalized conic gate. The problem
therefore predates deferred reporting, but these results do not establish the
cause of every v136 failure or its failure rate. Full raw iterates and original
constraint diagnostics are retained for independent inspection.

Reference-centred coordinates avoid asking the linear solver to recover tiny
corrections on top of absolute positions around 2.7 scaled distance units. The
translated RHS is computed directly from the original matrices and reference;
it is never replaced with zero or an assumed feasible trajectory.

## GPU implementation and contracts

The original audit retains its own coefficients. A separate retained packed
buffer contains the translated solver values. Row gathers reuse the audit's
compiled sparse row structure and use no floating-point atomics. The symmetric
quadratic translation includes both halves of the upper-triangular matrix.
The reference prefix is copied into owned GPU memory; subsequent SCvx updates
cannot overwrite it while the solver or audit consumes it.

Translation, reconstruction and audit can be captured together. Allocation
happens when the origin workspace is first enabled. Each subsequent attempt
uses retained buffers and device copies/kernels. The first solve and each
solver rebuild need a numerical update after symbolic setup; this is reported
as an additional device numeric update. No new host trajectory transfer is
introduced. Warm-start requests and accepted-primal caching are rejected for
the native translated mode; GTOC12 already uses independent cold solves.

The full outer SCvx dispatch, report finish, setup and recovery are still
host-controlled. This conditioning change supports the GPU-native pipeline;
it does not complete that pipeline by itself. The full conditional-IPM
sanitizer failure remains separate from the translation kernels' checks.

## Reproduction

Use native v138 with the unchanged prepared QOCO v128 runtime. The option is
captured at GTOC12 workspace creation and remains disabled by default:

```sh
export SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN=1
export SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1
export SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY=1
export SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY=1
export SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION=1
export SPACEPDHCG_TEST_GTOC12_DEVICE_ASSEMBLY_VALIDATION=1
export SPACEPDHCG_TEST_GTOC12_DEFERRED_REPORTS=1
```

The native API rejects nonfinite translated values. v138 also fixes recovery
when the first origin is invalid before any solve completes: it classifies the
failure as numerical, reports zero iterations, and rebuilds before retrying
with valid inputs. Changing a reference while a deferred solve is pending is
rejected. A direct mixed-cone quadratic test covers changing full/prefix
origins, a nonzero Hessian and objective, the unique physical solution, and
both CPU-oracle and device-validation execution with zero/three Ruiz passes.

## Retained measurements

All measurements below use the local RTX 5090. These are synthetic solver
fixtures, not new fleet solutions or leaderboard scores.

| Check | Result |
| --- | --- |
| v137 broad suite / GPU integration | 332 / 89 passed |
| v137 centred coast, strict original constraints | 36 of 36 passed |
| v136 and uncentred v137 coast comparison | Each had one original-equality failure in 36 solves |
| v137 translation/audit graph replay | 24 cases; memcheck, initcheck and synccheck reported zero errors |
| v138 mixed-cone native tests | Eight runs passed, including four translated recovery cases |
| v138 GTOC12 guard / GPU integration | Passed / 89 passed |
| v138 broad suite | 331 passed, one strict coast equality failure |
| v138 option-disabled integration | 88 passed, one strict coast equality failure |
| v137 three-way full-transfer comparison | All 36 transfers passed independent physics and mass gates |
| v138 versus v137 full-transfer comparison | All 24 transfers passed the same gates |

The first complete-transfer comparison measured medians of 274 ms for v136,
328 ms for uncentred v137 and 299 ms for centred v137. The later paired
comparison measured 343 ms for centred v137 and 291 ms for centred v138.
Both batches retain every warmup and measured run. Iteration counts and
timings vary substantially; no additional reliable speedup is established.
The final-mass reference remains 2445.3111007852112 kg with a 1e-5 kg gate.

The raw rejected coast iterates remain in the evidence. The finite sample of
successful centred solves does not prove that all convergence variability is
eliminated. Full conditional-IPM sanitizer runs remain unqualified; the
standalone translation/audit checks do not establish whole-solver sanitizer
coverage. The previous v136 checkpoint is historical failure evidence and is
not rewritten by the later results.

## Lambda campaign snapshot

At 2026-09-07 06:14 UTC the separate G4 campaign had completed 210 of 396
groups and was running the next group on the H100. Its 1,890 recorded attempts
comprised 1,296 timeouts and 594 numerical failures; none recorded success.
This is source commit `1dbcae098fd8871d4e0ac087e2306534704eff59`, not native
v138 and not a GTOC12 fleet search. The campaign was left running.

The completed groups' results, raw output, error logs and input descriptions
were downloaded to `results/lambda/2026-09-07/g4-progress-v138/`. The compressed
snapshot `completed-results.tar.gz` has SHA-256
`a72cb735ae2584dcdf156805f945f2a7b98da5d8caa2cf0dae2ed358810f163d`.
Its `snapshot-summary.json` contains the counts. Running-group files were
excluded. The existing visualiser's GPU solver progress panel displays this
dated summary separately from its unchanged fleet data.

Open `http://127.0.0.1:4173/` and scroll to **GPU solver progress**. If the
local server is stopped, start it from PowerShell:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
& 'C:\Program Files\nodejs\node.exe' scripts/serve.mjs --port=4173
```
