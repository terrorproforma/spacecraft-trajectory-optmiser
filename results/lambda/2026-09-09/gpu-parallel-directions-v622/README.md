# Parallel Lambert direction experiments

The final strategy is opt-in: `SPACEPDHCG_TEST_GTOC12_PARALLEL_DIRECTIONS=1`
uses a full warp per direction through 1,024 transfers, then the existing large
dispatcher. The default remains unchanged. See `docs/GPU_PARALLEL_DIRECTIONS.md`.

`summary.json` contains all final-policy campaign times and raw final-policy
microbenchmark samples. Ranges overlap; no reliable overall speedup is established.
All eight final-policy campaigns pass both mission verifiers and retain
548.254620 weighted kg. The best fleet remains 12,805.194 weighted kg.

`lambda-raw.tar.gz` contains 155 hash-verified files, including both rejected
layouts, final-policy tests, source snapshots, runtime libraries and all four H100
campaigns. H100 trajectories are under
`parallel-directions-campaign-v617/<baseline0|candidate0|candidate1|baseline1>/output`.
`local-raw.tar.gz` includes 260 verified files covering the corresponding local
experiments, the full half-warp campaign, final-policy campaign and initial setup
failures. Sources are based on 8a5dffee with recorded overlays.

`retrieval.json` authenticates the remote archive. `summary.json` records the
local archive checksum. Member manifests check every archived file, and
`files-sha256.json` authenticates the published files. The three native sanitizer
passes apply to Lambert, not the separately documented full-QOCO/cuDSS failures.
