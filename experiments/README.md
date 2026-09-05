# Experiment evidence records

SpacePDHCG and OrbitWeaver benchmark design is frozen in `benchmarks/paper1_matrix.json` and
`benchmarks/paper2_matrix.json`. The scientific rules governing those runs are in
`docs/BENCHMARK_PROTOCOL.md`. Complete-system comparisons, literature-result admissibility, and the
secondary GTOPX and historical GTOC tracks are defined in
`docs/COMPARATIVE_SOLVER_CAMPAIGN.md`; machine-readable external references are registered in
`benchmarks/literature_baselines.json`.

Every measured run should emit one machine-readable record conforming to `experiments/schema/run_manifest.schema.json`. The standard-library implementation is `spacepdhcg.experiments.RunManifest`; it records repository identity, upstream revisions, host/GPU metadata, problem and solver identity, requested/achieved quality, timing, status, notes, and artifact references.

Publication evidence must never use a moving upstream revision. `third_party/pdhcg.lock.json` pins the PDHCG revision used by the GPU validation workflow. The manually dispatched `.github/workflows/gpu.yml` records the CUDA/GPU environment and verifies that exact revision before running one-shot correctness tests.

Large raw results belong in workflow artifacts or an external immutable artifact store. Repository commits should contain only compact manifests, hashes, summaries, and scripts needed to reconstruct tables and figures.

Paper 1 G6 adds a separate archived-evidence envelope, preregistered decision record, and campaign
freeze configuration. Their schemas are `paper1_evidence.schema.json`,
`paper1_decision.schema.json`, and `paper1_campaign.schema.json`. This separation lets G4/G5
producers retain their compact result interface while G6 adds immutable archive/replay/residual
links and campaign-level completeness requirements. See `docs/PAPER1_FREEZE_TOOLING.md`.
