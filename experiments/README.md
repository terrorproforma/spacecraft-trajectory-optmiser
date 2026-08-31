# Experiment evidence records

SpacePDHCG and OrbitWeaver benchmark design is frozen in `benchmarks/paper1_matrix.json` and `benchmarks/paper2_matrix.json`. The scientific rules governing those runs are in `docs/BENCHMARK_PROTOCOL.md`.

Every measured run should emit one machine-readable record conforming to `experiments/schema/run_manifest.schema.json`. The standard-library implementation is `spacepdhcg.experiments.RunManifest`; it records repository identity, upstream revisions, host/GPU metadata, problem and solver identity, requested/achieved quality, timing, status, notes, and artifact references.

Publication evidence must never use a moving upstream revision. `third_party/pdhcg.lock.json` pins the PDHCG revision used by the GPU validation workflow. The manually dispatched `.github/workflows/gpu.yml` records the CUDA/GPU environment and verifies that exact revision before running one-shot correctness tests.

Large raw results belong in workflow artifacts or an external immutable artifact store. Repository commits should contain only compact manifests, hashes, summaries, and scripts needed to reconstruct tables and figures.
