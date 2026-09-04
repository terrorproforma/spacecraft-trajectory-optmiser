# Active campaign scope: `single-gpu-v1`

Scope record: `benchmarks/campaign_scopes/single-gpu-v1.json`
Scope schema: `experiments/schema/campaign_scope.schema.json`
Status: **active completion goal**

This versioned scope changes completion planning, not historical evidence or preregistration.
Existing G5 code, runbooks, schemas, logical-rank tests, one-rank evidence, and the original
`full-multi-gpu-v1` campaign remain intact. Physical 2/4/8-GPU work is tracked in
[`DEFERRED_MULTI_GPU_BACKLOG.md`](DEFERRED_MULTI_GPU_BACKLOG.md) and does not block this scope.

## Exact active requirements

1. G0 host/native truth and environment freeze.
2. G1 pinned upstream one-shot CUDA correctness on one physical GPU.
3. G2 persistent one-GPU PDHCG workspace correctness and lifecycle.
4. G3 device-resident deterministic SCvx correctness on one physical GPU.
5. G4 one-GPU adaptive, inexact, pure-IPM, and hybrid experiments.
6. Paper 1 one-GPU evidence, H1/H2/H3/H5/H6 decisions, and in-scope products.
7. OrbitWeaver one-GPU coarse-to-refined-to-scenario simulations.
8. OrbitWeaver one-GPU pricing/restricted-master integration and independent certification.
9. OrbitWeaver one-GPU trajectory and mission visualisation.

The Paper 1 products in scope are F01-F06, F08-F11, T01-T05, T07, and T08. F03 may describe the
implemented decomposition architecture but may not imply physical scaling. F07, F12, and T06 are
excluded from this freeze and listed explicitly as deferred; they are never emitted empty.

## Hypothesis disposition

- H1, H2, H3, H5, and H6 remain active under their original thresholds and decision rules. Their
  evidence must use one physical GPU and matched end-to-end quality.
- H4 is `deferred-not-in-scope`. It is not supported, rejected, mixed, or unresolved because the
  physical multi-GPU experiment is intentionally outside this campaign.
- H4 has no manuscript claim ID in a `single-gpu-v1` configuration. T08 records its deferred
  disposition. Scoped tooling rejects an H4 claim or any 2/4/8-GPU/P1-F record.

No preregistered threshold or historical decision is changed. Legacy `1.0.0` campaign
configurations remain readable and resolve to `full-multi-gpu-v1`.

## G6 freeze semantics

A `single-gpu-v1` freeze may succeed only when:

- every configured in-scope G4 coordinate is complete at the frozen instance/repeat requirements;
- all evidence is immutable, hash-verified, portable, independently replayed, and from the
  configured clean repository commit;
- H1/H2/H3/H5/H6 have ordinary preregistered decisions and H4 has the scoped deferred disposition;
- every included product is traceable to in-scope runs;
- the build manifest and every source product carry `campaign_scope_id=single-gpu-v1`;
- F07/F12/T06 are listed as deferred products rather than generated from absent data.

The same tool refuses `full-multi-gpu-v1` unless physical P1-F records exist at 2, 4, and 8 GPUs.
A freeze seal remains a completeness seal, never a scientific PASS statement.

## G7 one-GPU acceptance

G7 is complete in this scope when one-GPU records cover:

1. coarse convex arc evaluation;
2. refined SCvx promotion;
3. scenario expansion and risk evaluation;
4. route pricing and restricted-master integration;
5. independent high-fidelity certification;
6. trajectory/mission visualisation.

Every accepted result must be converged or iteration-limited with an independently accepted
certificate. A schema-v2 manifest carries `campaign_scope_id=single-gpu-v1`, uses
`ownership=single_gpu`, names exactly one device, and cannot claim
`physical_multi_gpu_tested`. Completion explicitly excludes scaling, throughput, energy, memory
crossover, and tractability-frontier claims.

## Integration order

1. Complete and seal current-head G4 one-GPU evidence.
2. Build and verify the scoped Paper 1 products and decisions.
3. Freeze `single-gpu-v1` from a clean commit.
4. Run the complete one-GPU OrbitWeaver simulation/certification/visualisation flow.
5. Record G7 `complete-in-scope`.
6. Later, on suitable hardware, execute the preserved G5 runbook and distributed OrbitWeaver
   backlog as a separate `full-multi-gpu-v1` (or successor) campaign.
