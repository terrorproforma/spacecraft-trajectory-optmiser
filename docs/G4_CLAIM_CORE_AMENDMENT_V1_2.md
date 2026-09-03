# G4 claim-core amendment `single-gpu-v1.2`

Versioned, preregistered amendment to the H5/H6 claim core
(`benchmarks/g4_h5_h6_claim_core.json`, SHA-256 `40dc2174…0c8d0b`) under campaign scope
`single-gpu-v1`. It supersedes `single-gpu-v1.1`
(`benchmarks/g4_claim_core_amendment_v1_1.json`, SHA-256 `c691467e…8a1f5`) and inherits its
`contamination`, `deterministic_replay`, `censoring` and `schedule` sections verbatim, so the
amended 396-group schedule identity is unchanged. The machine-readable contract is
`benchmarks/g4_claim_core_amendment_v1_2.json`, locked by
`benchmarks/g4_claim_core_amendment_v1_2.sha256`, validated by the reused
`experiments/schema/g4_claim_core_amendment.schema.json` (conditional v1.2 sections) and by
`spacepdhcg.experiments.g4_execution_contract.validate_amendment_v1_2_sections`. The scope
registry (`benchmarks/campaign_scopes/single-gpu-v1.json`, `spacepdhcg.campaign_scope`) lists it
under `amendments` with `supersedes: single-gpu-v1.1`.

Frozen at `2026-09-03T06:45:00Z`, before any group ran under it. At that moment checkpoint
`build-integration-report/g4-claim-core-ccd5596` (v1.1) held only `pure-gpu-ipm` groups. Those
records were opened solely to establish the rule-A probe evidence below; they are retained as the
labelled diagnostic stratum `ipm_no_equilibration_v1_1` and are never H6 evidence.

Every record produced under the amendment carries `policy_amendment: "single-gpu-v1.2"` in the
checkpoint metadata, in every raw attempt, every `execution_group_result` and the decision notes.

## Rule A — IPM baseline equilibration (recorded per attempt)

`docs/COMPARATIVE_SOLVER_CAMPAIGN.md` requires each system to run its native algorithm at
documented default/best-feasible settings with matched final quality, not matched internals.
The `pure-gpu-ipm` baseline and the IPM stage of `hybrid-pdhcg-ipm` therefore ignore the
PDHCG-oriented campaign `scaling_mode` axis and use QOCO's own equilibration setting.

**What "native default" turned out to mean.** The request assumed QOCO's default is "Ruiz on".
At the pinned QOCO commit (`09f0495…`, library SHA-256 `3db21490…e131ac`) `set_default_settings`
sets `ruiz_iters = 0`: the shipped default is *no* equilibration. The amendment records
`mode: qoco_native_default`, `ruiz_iterations: 0`, and the executor passes exactly that to QOCO
for every IPM solve regardless of the coordinate `scaling_mode`. Each IPM attempt echoes
`amendment.ipm_equilibration = {mode, ruiz_iterations, requested_ruiz_iterations, scaling_mode,
qoco_status_code}`; non-IPM attempts carry `null`.

**Recorded `scaling_mode`.** `pure-gpu-ipm` attempts record `scaling_mode:
not_applicable_ipm_native` in the Paper 1 identity block and the runtime block (the enum was
widened in `experiments/schema/paper1_result.schema.json`). `hybrid-pdhcg-ipm` keeps the
coordinate's `scaling_mode` because its PDHCG stage still consumes it; its IPM stage is governed
by the same rule and echo.

**Probe evidence (foreground, contaminated, 120 s deadline).** P1-E N=100 group `a8bdf61e`,
conditioning 4.0 (the failing coordinate) and conditioning 0.0:

| probe | library | Ruiz | policy | cond. | outcome |
| --- | --- | --- | --- | --- | --- |
| campaign v1.1 | pinned | 0 | pure-gpu-ipm | 4.0 | numerical 8/8, QOCO status 5 after 27–200 iterations, 28–213 s |
| Ruiz on | pinned | 5 | pure-gpu-ipm | 4.0 | numerical 6/6 at iteration 8 with **NaN iterates**, 43–49 s |
| Ruiz on | scratch-fixed | 5 | pure-gpu-ipm | 4.0 | numerical after 54 iterations, no NaN, 183 s |
| Ruiz on | scratch-fixed | 5 | pure-gpu-ipm | 0.0 | numerical 6/6 after 21–24 iterations, no NaN, 36–88 s |
| Ruiz off | scratch-fixed | 0 | pure-gpu-ipm | 0.0 | **qualified 3/3**, 11–13 QOCO iterations per subproblem, 128–156 s |
| either | scratch-fixed | 5 / 0 | hybrid-pdhcg-ipm | 4.0 | timeout 6/6 in the PDHCG stage; IPM stage never reached |

Conclusion, stated plainly: native equilibration does **not** rescue the failing conditioning-4.0
coordinate, and on the converging conditioning-0.0 coordinate it turns qualified solves into
numerical failures. The conditioning-4.0 IPM result stands as a genuine IPM negative on this
problem class; the amendment is recorded anyway so the selection is explicit in every record.

**The NaNs are a QOCO CUDA-backend defect, not data scaling.** Two defects were traced and fixed
in a scratch build only (the pinned library is unchanged for the campaign):

1. `src/equilibration.c`: `safe_div(1.0, 0.0)` returns `QOCOFloat_MAX`, so structurally empty
   rows of `G`/`A` in the SCvx formulation receive a `DBL_MAX` scale and the scaled problem
   overflows to NaN (QOCO status 5 at iteration 8).
2. `algebra/cuda/cuda_linalg.cu scale_arrayf`: no host fallback, so Ruiz (host data) never scales
   `P` and `c` while the bookkeeping scale `k` is applied, collapsing the objective scale.

The scratch fixes separate "Ruiz is broken in this build" from "Ruiz does not help this
problem class"; the second is what the probes show once the first is removed.

## Rule B — deadline classification by measured wall time

An IPM solve (or any backend solve) may be uninterruptible. If a launched attempt's measured wall
time exceeds the attempt deadline it is classified `timeout` with reason
`wall_exceeded_deadline`, never `numerical`. The completed solve's residuals, iteration counts and
the solver's own disposition are attached as diagnostics
(`amendment.deadline_classification.{rule, wall_exceeded_deadline, attempt_deadline_seconds,
measured_wall_seconds, solver_disposition}`). An `executor_defect` keeps its disposition.
Wall-clock timeouts are censoring, exactly like cooperative timeouts.

## Rule C — N=2000 hard bound (unchanged, recorded)

An executor that exceeds the group deadline is killed, restarted exactly once
(`restart_generation` 1) and a second breach yields an error record for the group. Nothing
changes; the amendment records it.

## Diagnostic stratum `ipm_no_equilibration_v1_1`

`pure-gpu-ipm` groups completed under v1.1 in `g4-claim-core-ccd5596` are moved to the ledger
state `diagnostic` (disposition `ipm_no_equilibration_v1_1`) by
`run_g4_campaign.py label-stratum`; every record and run directory is retained verbatim with a
`diagnostic_stratum.json` sidecar. The successor checkpoint cites the stratum in its
`diagnostic_strata` metadata (`init --cite-diagnostic-stratum`), the decision refuses a v1.2
checkpoint without that citation, and `migrate` refuses to import records taken under a
different amendment. The stratum is excluded from H6 and cited in `docs/G4_GATE_REPORT.md`.

## Executor and tooling

- Executor: `SPACEPDHCG_G4_POLICY_AMENDMENT=single-gpu-v1.2` selects rules A/B; the QOCO adapter
  takes `ruiz_iterations` explicitly (`spacepdhcg_native_qoco_create`) and reports the value used
  plus the QOCO status code; `--g4-amendment-v1-2-selftest` checks the selection and
  classification logic without a GPU.
- Capability: declares `single-gpu-v1.2` in `policy_amendments_supported` with
  `ipm_equilibration` and `deadline_classification` blocks; the `pure_gpu_ipm` probe runs with
  the recorded equilibration and pins the QOCO library hash.
- Scheduler/decision: `run_g4_campaign.py` validates the echoes on every raw attempt;
  `decide_g4_claim_core.py` re-derives rule B from the recorded wall time and deadline, checks the
  recorded `scaling_mode`, and reports `diagnostic_strata`.
