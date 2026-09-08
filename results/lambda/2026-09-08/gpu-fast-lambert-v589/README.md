# CUDA Lambert root measurements, 8 September 2026

`summary.json` compares the complete A/B/B/A campaign and isolated screening on
RTX 5090 and Lambda H100, with final default-build validation. See
`docs/GPU_FAST_LAMBERT_ROOTS.md` in the repository for method and limitations.

`lambda-raw.tar.gz` contains 161 verified source/runtime/result files, including
the final H100 trajectory at `campaign-final/default/output/fleet/Result.txt`.
`local-raw.tar.gz` contains local runs, both profiling attempts, the original
empty-fixture test failure, frozen binaries and reproduction workers.
The archive member manifests verify every file. `retrieval.json` records the
remote archive checksum; `summary.json` records the local archive checksum.
`files-sha256.json` covers published files. Original experiment binaries have the
new method opt-in; final binaries enable it by default. Exact source is archived.

All ten campaigns pass official and independent mission verification. These
one-ship runs retain 548.254620 weighted kg; no new fleet record is claimed.
The best fleet remains 12,805.194 weighted kg. Lambert sanitizer passes do not
resolve the separately documented vendor full-solver sanitizer failures.

The imported final H100 trajectory is displayed in the existing local viewer:
http://127.0.0.1:4173/?dataset=gtoc12-v588&epoch=69807&preset=oblique&z=1
Dataset files live in `results/lambda/2026-09-06/visualiser/data/gtoc12-v588`.
