# Collection-DP profile and exact-mass cache fix

See [findings, validation and GPU port requirements](../../../../docs/GPU_COLLECTION_DP_PORT.md).
Source fix: `a02de775`; baseline: `49fb0d47`.

`summary.json` contains direct timing, cache audits and the corrected mission's
verification. `v272/timing.json` holds eight real inputs for the GPU port.
`v275/output/fleet/Result.txt` is the corrected full-campaign solution; its exact
viewer export is preserved, while the live viewer retains v269 provenance.

The recorded source paths identify frozen Lambda snapshots. The formatted test
file in v276 supersedes the earlier formatting fingerprint recorded at v275
startup; solver source is identical. `validated-source/` is byte-exact to the
final validation manifest. Native CUDA/QOCO binaries are unchanged by this fix.

The original v270 wrapper lost cProfile output at SystemExit. v271 saves its raw
profile, but its aggregate times are inconsistent and must not be used as a
performance claim. Direct v272 timers are authoritative for the measured DP
share; complete v275 runtime is a separate, single observation.

All archives are verified in `retrieval.json`; `sha256.json` pins every evidence
file except itself. Local baseline/current replays use the same final replay
script, with `--reference archived --mass-key rounded` for the old solver and
the default uncached reference for the fixed solver. The older exploratory
`local-replay` log/JSON is retained separately.
