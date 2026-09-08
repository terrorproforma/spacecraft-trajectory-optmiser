# Verified GPU fleet-exchange improvement

New verified score: **12,842.970672 weighted kg**, +32.834719 kg over the previous
incumbent. The H100 solution is `h100-best/Result.txt`; fresh independently
propagated visualiser samples are in `h100-best/viewer/`.

`summary.json` records warm fleet-selection timings, actual work and complete
qualification run times. Packing proposals are not trajectory solves. All four
full mission checks (two runtimes on two GPUs) pass both verifiers; every run
converges all 33 native solves for the two replacement routes.

The raw archives contain:

* `v727`: frozen source, build logs, native library and initial correctness tests.
* `v728`: alternating exchange-enabled/disabled comparisons on the exact pool.
* `v729`: fresh replacement refinement using the previously qualified runtime.
* `v730`: final 90-test regression set and full-pool memory/race/sync sanitizers.
* `v732`: GPU selection and fresh refinement with the new v727 native library,
  official/independent checks and trajectory export.

`FILES.json` inside each archive records all members, lengths and SHA-256 hashes.
The archive receipts and adjacent member lists were checked before and after
retrieval. Official catalogue/verifier binaries are external pinned dependencies,
not redistributed here. `sha256.json` covers the published files except itself.

From a Linux/WSL Python environment with the project's dependencies and CUDA:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  results/lambda/2026-09-09/gpu-fleet-exchange-v733/reproduce/replay.py \
  results/lambda/2026-09-09/gpu-fleet-exchange-v733/local.tar.gz \
  --node-cap 0 --exchange-rounds 16 --repeats 5 --output exchange-replay.json
```

Use the H100 archive on H100, and `--exchange-rounds 0` to disable exchanges.
The replay verifies/extracts the archived source/library, acquires the GPU lock
and refuses to overwrite output. It replays packing only. The v729/v732 workers
retain the original paths, configuration, inputs and full-qualification commands.

The visualiser dataset is `gtoc12-exchange-v733`. Full interpretation and loading
instructions are in `docs/GPU_FLEET_EXCHANGES.md` at the repository root. Wider
Python orchestration and independent CPU mission audits remain; the entire
application is not yet fully GPU native.
