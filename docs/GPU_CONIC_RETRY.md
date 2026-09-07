# Bounded recovery from inaccurate GPU subproblems

This is a development candidate. The complete comparison below does not establish
a lower failure rate, and cuDSS numerical repeatability remains open. It has not
been merged into main.

## Failure and diagnosis

Resident-table campaign v299 returned a verified 475.975 kg mission after missing
the 548.255 kg retimed candidate. Its Earth departure to asteroid 57530, from MJD
64343 to 64808 at 3000 kg initial mass, exhausted refinement with virtual control
0.07504. Other complete runs found and verified the higher-value mission.

The captured first convex subproblem has 5,839 variables, 1,648 equalities and
10,280 cone rows. Replaying identical coefficients with the pinned H100 QOCO
binary reproduces numerical qualification failures. In v308, only 7/16 replays
at the current 1e-8 factor regularisation pass an independent original-equation
audit. This establishes an inner-solver repeatability problem independently of
collection-table memory residency; it does not prove that every campaign miss
has the same cause.

Increasing regularisation, tightening iterative-refinement parameters, enabling
Ruiz scaling and rescaling the objective failed to produce a reliable setting
in the subsequent sweeps. v309's row named `baseline` is an exploratory **1e-9
internal-tolerance** comparison, not the production 1e-11 internal tolerance.
Every external diagnostic audit retained its 1e-9 residual/objective-gap gate.
A separate compensated-residual prototype failed all 32 local candidate replays
and was rejected; it is not part of the production change.

## Controller change

An unqualified `QOCO_SOLVED_INACCURATE` result previously shrank the SCvx trust
region immediately. Such a result is not an admissible nonlinear step and does
not establish that the trust region is too large.

The CUDA controller now permits two cold retries at the unchanged reference and
trust region. It never copies an unqualified candidate into the trajectory. A
third consecutive inaccurate failure takes the existing trust-shrink path.
Qualified candidates reset the retry allowance. Other solver failures and
invalid qualified candidates retain immediate rejection and shrink behavior.

Retries consume the existing SCvx iteration and wall-time budgets. Their count
and control reside in device state, including the outer CUDA graph. There are
no new host decisions or host trajectory transfers. Internal IPM tolerances,
external 1e-9 conic gates and nonlinear physics certificates are unchanged.

The C record layout is unchanged. `conic_rejected=2` records rejection followed
by an unchanged-trust retry; `1` records rejection followed by shrink. Python
history exports the trust values and `retry_unchanged`. The final formatter also
distinguishes an unqualified convex solve from a qualified convex result rejected
by the nonlinear/physical candidate checks. One H100 replay exercised the latter.

## Verified scope so far

On both RTX 5090 and H100:

- The actual CUDA controller passes 22 branch cases and 18 retry transitions.
  These check unchanged trajectories, bounded retry count, subsequent shrink,
  reset after acceptance, original qualification and device-packet precedence.
- Outer-graph, deadline and input-guard tests pass, as do all 20 GPU SCvx tests.
- The controller passes memcheck, initcheck, synccheck and racecheck with zero
  errors/hazards. This sanitizer scope is the controller, not all vendor kernels.
- The exact 465-day departure converges and passes independent physics
  certification in 24/24 replays per GPU. Every accepted step passes the original
  conic residual and objective-gap gates. An additional audit checks that each
  reported retry leaves the next attempt's trust values unchanged.

The local arc median is 0.616 seconds and its maximum is 2.178 seconds. H100's
median is 0.928 seconds and maximum is 2.802 seconds. These are isolated candidate
replays, not matched speedup measurements. The 48 replays contain 62 unchanged
retries; rejected attempts remain visible in the histories and solver reports.

Eight balanced H100 campaigns compare the previous and retry controllers using
the same resident-table implementation and fixed search configuration. All eight
return 548.255 kg and pass both official and independent mission checkers.

| Run | Controller | Complete seconds |
|---|---|---:|
| v316 | Retry | 61.107 |
| v317 | Previous | 60.554 |
| v318 | Previous | 60.169 |
| v319 | Retry | 60.515 |
| v320 | Retry | 59.460 |
| v321 | Previous | 60.585 |
| v322 | Previous | 59.733 |
| v323 | Retry | 60.797 |

Median time changes from 60.362 to 60.656 seconds, about 0.5% slower. Neither
controller failed in this small comparison, so it establishes neither a failure-rate
improvement nor a speedup. The incumbent fleet score remains 12,805.194 weighted kg.

[Diagnostic evidence and local/H100 validation](../results/lambda/2026-09-08/gpu-conic-retry-v314/)
includes source/binary hashes, the exact QP, complete audit distributions, raw
arc histories and rejected experiments. Compact QP replay samples retain the
first and last full vectors per setting; hashes identify full vector logs kept
on Lambda. The frozen v313/v314 Python formatter predates the final rejection-label
correction; its `qualified` solver-report field is authoritative for that distinction.
