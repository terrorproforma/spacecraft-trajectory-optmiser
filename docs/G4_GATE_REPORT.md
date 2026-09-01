# Gate G4 report — adaptive and hybrid matched-quality study

Status date: **2026-09-01**
Decision: **FAIL**
H5: **unresolved**
H6: **unresolved**
G5 authorised: **no**

## Scope and frozen policy

The G4 policy was frozen before evaluation in commit `8f318b9`. The machine-readable contract is
`benchmarks/g4_policy.json`; it fixes all six required modes, forcing and re-solve rules, trust
policy, warm-start and scaling modes, recovery inclusion, quality tiers, tuning/evaluation split,
the full Paper 1 family matrix, H5/H6 thresholds, bootstrap seed/method, solver-order seed, timeout
handling, and 50 ms GPU power sampling.

Implementation and qualification telemetry were committed in `509d9dd`; the G3 compatibility
tolerance correction is `b70bf15`. Evaluation was run from the clean full commit
`b70bf15f3fc84c6ed9138df4cf466a3907c51e06`.

## GPU interior-point baseline

The pinned baseline is QOCO-GPU commit `09f049597deef2a7ead15b3da19a9456ff7d4e53`
(reported version 0.3.2), tree `c85fe82f71a67921868fc761c242de11ac46f4a2`, with
`nvidia-cudss-cu12==0.7.1.6`, CUDA 12.8.93, `sm_120`, float64, and sparse direct cuDSS
factorisation. Its native form is

\[
 \min_x \tfrac12 x^T P x+c^Tx,\qquad Ax=b,\qquad h-Gx\in
 \mathbb R_+^l\times\prod_i\mathcal Q_i.
\]

This supports the required quadratic objective, equalities, nonnegative rows, and SOCs. Exact box
splitting and an orthonormal rotated-SOC-to-SOC conversion are declared in the lock. QOCO exposes
`qoco_set_x0` for an unequilibrated primal start, but exposes no dual warm-start API.

The pinned source built and its CUDA/cuDSS demo executed on the RTX 5090 in six IPM iterations with
objective 4.042, native primal residual `1.413e-8`, and native dual residual `8.258e-9`. This is
availability evidence only. No production trajectory CQP was executed through QOCO, so neither a
pure-IPM trajectory comparison nor a hybrid timing claim is made. The missing production adapter
and unavailable dual handoff make H6 unresolved.

## Qualification result

The first nontrivial tight-quality coordinate in each required nonlinear family failed the
matched-quality prerequisite. The campaign therefore stopped before timing the full matrix:
unqualified trajectories may not enter H5/H6 speed comparisons.

| Family | Coordinate | Canonical residual | Terminal residual | SCvx time (s) | GPU energy (J) |
|---|---:|---:|---:|---:|---:|
| P1-C 3-DoF | N=20, dispersion=0.01 | `5.674e-4` | `1.000` | `35.771` | `4002.01` |
| P1-D 6-DoF | N=20, dispersion=0.05 | `1.480e-1` | `4.990` | `70.501` | `7981.64` |
| P1-E low-thrust | N=100, dispersion=0.01 | `4.731e-1` | `70.800` | `164.240` | `18665.82` |

All three samples:

- returned no accepted step and ended at the outer-iteration limit;
- violated the requested forcing threshold after the identical-CQP re-solve;
- retained matching 64-bit numeric CQP fingerprints across the re-solve;
- reported zero post-create topology allocations and zero hidden CPU fallback;
- were excluded from all performance claims.

The initial diagnosis identified a real architectural omission but overstated its causal role in
this particular archive. The one-outer-iteration qualification materialised its first CQP from the
displaced CPU reference, so it never reached a second reference update. The missing production
updates were nevertheless in scope and are now implemented for the reference-tracking objective,
trust-cone centres and radii, exact-penalty epigraph cost, low-thrust radial halfspaces, 6-DoF
quaternion linearisation, and nonlinear dynamics.

The corrective rerun found two additional production defects: nonlinear merit used unscaled
magnitudes inconsistent with the CPU SCvx policy, and an identical-CQP refined re-solve updated
telemetry but acceptance still evaluated the stale pre-refinement primal. Both are corrected, and
all four fixtures compare every CPU/device numerical vector with maximum absolute difference
`1.8118839761882555e-13` (contract `5e-12`) while changing trust radii and exact-penalty weights
in place. Even after those fixes, displaced P1-C remains
unqualified: a 15-outer fixed-loose diagnostic accepted one step after eight trust reductions but
ended at the minimum radius with scaled dynamics `1.705e-3`, terminal `8.092e-4`, and virtual
control `2.000e-3`. This is retained as new negative evidence, not as a replacement for the frozen
archive. G4 therefore remains FAIL pending an accurate production IPM/hybrid path or a further
solver correction under unchanged policy.

The compact coverage ledger retains 1,080 family/interval/policy/warm-start/quality coordinates.
Unexecuted first-order coordinates are explicitly censored after qualification failure. Pure-IPM
and hybrid coordinates are explicitly unsupported because no production adapter execution exists;
the QOCO smoke run is not relabelled as a trajectory result.

## H5 and H6 decisions

### H5 — unresolved

There are zero matched-quality nontrivial family samples and therefore no valid five-repeat paired
bootstrap, no two-family support region, and no sustained three-coordinate boundary. Adaptive
forcing cannot be supported or rejected from unqualified trajectories.

### H6 — unresolved

There is no archived production QOCO trajectory solve or qualified PDHCG-to-QOCO handoff. The
pinned baseline accepts primal starts but not dual starts. No Pareto-frontier or time-advantage
claim is permitted.

## Timing and energy limitations

Power was sampled with `nvidia-smi` over each isolated benchmark process boundary. The traces are
GPU-only and no idle subtraction was applied. WSL delivered maximum sampling gaps of 1.938 s,
1.930 s, and 1.862 s despite the requested 50 ms cadence, so the integrated energies above are
diagnostic only. Display/shared-machine isolation was not established.

An earlier qualification directory is preserved as negative evidence because other GPU tests ran
concurrently with its power traces. It is excluded from energy conclusions.

## Build, test, and sanitizer status

- Ruff: passed.
- Python: 98/98 tests passed.
- CUDA/native Release: 52/52 tests passed.
- CUDA/native Debug: 52/52 tests passed.
- G3 production regression: passed at the original `1e-6` contract.
- Compute Sanitizer on the modified telemetry/outer lifecycle:
  - memcheck: 0 errors;
  - racecheck: 0 hazards, 0 errors, 0 warnings;
  - synccheck: 0 errors;
  - initcheck: 0 errors.

The sanitizer runs validate telemetry allocation, hashing, stream use, and lifetime behavior. They
do not qualify the failed nonlinear trajectories.

## Criterion-by-criterion decision

1. Policies frozen before evaluation: **PASS**.
2. Mathematically valid GPU IPM pinned and executable: **PASS** for baseline availability.
3. Production pure-IPM trajectory execution: **PASS** for exact HCW trajectory QP/SOCP adapter
   correctness on RTX 5090; **FAIL** for the required nonlinear P1-C/P1-D/P1-E executions.
4. Audited hybrid conversion: **PASS** for qualified primal handoff with explicit dual discard;
   **FAIL** for a matched-quality nonlinear G4 handback.
5. Per-outer policy telemetry and identical-CQP re-solve audit: **PASS**.
6. Matched final nonlinear quality: **FAIL**.
7. H5 resolved under preregistered rules: **FAIL / unresolved**.
8. H6 resolved under preregistered rules: **FAIL / unresolved**.
9. Full primary matrix and repeat/instance count: **FAIL**; censored after qualification failure.
10. Affected tests and four sanitizer tools: **PASS**.
11. Negative, unsupported, and censored records retained: **PASS**.

Gate G4 therefore **does not pass**, and Gate G5 is **not authorised**.

## Integrated harness and adapter follow-up

The audited harness commits and pinned-QOCO adapter were integrated after the corrective campaign.
The CUDA outer executable now rejects a mismatched generated policy SHA, consumes quality-tier
fixed-tight tolerances, scaling and warm modes, and the frozen adaptive phase, re-solve, polish,
and trust parameters instead of duplicating those constants. Requested and actual runtime values
are emitted separately.

Serialized RTX 5090 adapter validation passed 18/18 tests, including exact trajectory QP and SOCP
agreement with Clarabel, persistent same-pattern update, accepted primal warm start, independent
canonical residuals, and explicit discard of the unsupported dual start. This closes the adapter
ABI/runtime gap but does not convert the failed nonlinear qualification into a pass. An integrated
one-outer P1-C diagnostic under policy SHA
`9ab3b444e3dd21fdd2a75c3cebfe8fd8374f9e5ff672cd757afaeb6036530024`
reproduced canonical residual `5.653990342580073e-4`, terminal residual `1e-3`, identical-CQP
re-solve fingerprint, no accepted step, and forcing failure. A 15-outer diagnostic exceeded
30 minutes before completion and was terminated and retained as timeout evidence; it is not
reported as an executed matrix point. H5 and H6 therefore remain unresolved.

## Evidence

Clean corrective qualification after the fixed-pattern update implementation:

- `results/gpu/g4/qualification/g4-20260901-2cbebb3-clean-corrective/`
- execution commit recorded by the manifest:
  `2cbebb3b1b66666e6dc95e879260174b32abf22e`
- manifest SHA-256: `fe218070ec0332a3747e9a1f3f6c5705f79c313810d3bb070d3314d48adcd49c`
- decision SHA-256: `d7143c2cc338423a21c0cf606aa71b84812bc784f26efd7e0c30fd8821a09744`
- coverage SHA-256: `ed06f436c410660eb9dc0abc76847c5ab7dae8a6a4603697bad5dbe9608815bc`
- corrected canonical residuals: P1-C `5.654e-4`, P1-D `1.283e-2`,
  P1-E `1.369e-2`;
- corrected scaled terminal residuals: P1-C `1.000e-3`, P1-D `5.243e-2`,
  P1-E `7.080e-2`; qualified samples: 0/3; forcing failures: 3/3.

This corrective directory supplements rather than replaces the frozen original failure archive.
The earlier `g4-20260901-f21f93d-corrective2` directory was produced while the final
initial-boundary fix was still uncommitted and is retained only as contaminated negative evidence,
not clean qualification evidence. That fix was committed as `2cbebb3` before this rerun. The
aborted first corrective directory is likewise retained as additional negative evidence.

Primary isolated qualification archive:

- `results/gpu/g4/qualification/g4-20260901T072000Z-b70bf15-isolated.tar.gz`
- SHA-256: `2bda322a9f85d59c870aeb244dd5b2c64a23ef5f38a706eb2b9a4d051baae5c8`
- evidence-index SHA-256:
  `84db411403dbde4871011e8d3fbf4d85e1534d44e2f53698ab724e007678f7d7`

Preserved contaminated preliminary archive:

- `results/gpu/g4/qualification/g4-20260901T070000Z-509d9dd-contaminated.tar.gz`
- SHA-256: `cac80c4fa5bb6797a36f94cfeac91d5b8d79469e2c739f911f8503a98d9e876d`

The isolated archive contains the exact execution commit, policy and solver-lock hashes, commands,
environment, raw outer-iteration diagnostics, power traces, compact 1,080-coordinate coverage
ledger, QOCO build/smoke logs, test logs, all sanitizer logs, decision record, and artifact index.
