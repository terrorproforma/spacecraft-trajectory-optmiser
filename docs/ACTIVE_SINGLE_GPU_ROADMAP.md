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

## Track L — literature reference reproduction (campaign Phase 0-1)

A separate track, not a gate: it freezes literature inputs and reproduces published references
before any comparative performance claim. Owner document: `docs/LITERATURE_TARGETS.md`; report:
`docs/REFERENCE_REPRODUCTION_REPORT.md`; registry: `benchmarks/literature/targets.json`.

Scope and status (2026-09-03):

1. Provenance store with evidence labels for every literature value — done, frozen by test.
2. P1-C Acikmese-Ploen 2007 / Blackmore 2010 Mars 3-DoF — reproduced: lossless SOCP 400.63 /
   400.09 kg and the repository SCvx with the accurate option (variational RK4, multiple-shooting
   merit, stall stop) 399.36 / 398.84 kg against 399.5 / 399.4 kg published (the frozen
   forward-Euler default stays at 405.65 / 413.43 kg and is recorded as the before value);
   pure-QOCO GPU leg deferred behind `spacepdhcg literature gpu-run` while the G4 session owns
   the device.
3. P1-D Szmuk-Acikmese 2018 free-final-time 6-DoF — reproduced on the independent CPU core
   (t_f = 3.39008 UT) and on the NEW native `pd6_fft` topology (t_f = 3.39254 UT, envelope
   0.01 UT); `pd3_fft`/`pd6_fft` CUDA coefficient kernels build for `sm_120`, their device
   parity test is deferred until the G4 campaign releases the device. Frozen fixed-time
   topologies and G4 policy hashes untouched.
4. P1-D-MC Chari 2024 — seeded samples committed; CPU independent batch run; pure-QOCO
   `pd6_fft` GPU batch deferred (preflight-gated); persistent device SCvx batch blocked.
5. P1-E Earth-Mars — reproduced; Earth-Dionysus (five revolutions) — reproduced through the MEE
   formulation with revolution bookkeeping and trust-weight continuation.
6. TOPS — pinned revision ingested, metadata selection frozen; two-body cases run (P4 Cartesian,
   P3 multirev and P1 highly-elliptic free-time through MEE); CR3BP/solar-sail unsupported.
7. GTOPX — evaluator built from pinned source; best-known vectors verified.
8. GTOC12 — official verifier reproduces published solution files; GTOC9 — official examples
   validate under re-implemented rules; GTOC5 — data pinned, scoring blocked.

Track L does not modify sealed evidence, the G4 policy, or the frozen fixed-time native
transcription topologies; its free-final-time variants are additive (`pd3_fft`, `pd6_fft`).

## Integration order

1. Complete and seal current-head G4 one-GPU evidence.
2. Build and verify the scoped Paper 1 products and decisions.
3. Freeze `single-gpu-v1` from a clean commit.
4. Run the complete one-GPU OrbitWeaver simulation/certification/visualisation flow.
5. Record G7 `complete-in-scope`.
6. Later, on suitable hardware, execute the preserved G5 runbook and distributed OrbitWeaver
   backlog as a separate `full-multi-gpu-v1` (or successor) campaign.
7. In parallel (Track L), keep `benchmarks/literature/` frozen; rerun `spacepdhcg literature run
   all` only from a clean commit and regenerate the reproduction report.
