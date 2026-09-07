# CUDA collection-table ephemerides

See [implementation and reproduction](../../../../docs/GPU_COLLECTION_TABLES.md).
Code commit: `f491eda6`. Native libraries are unchanged from the preceding CUDA DP.

`summary.json` records matched complete campaigns, CPU v292/v294 and CUDA v291/v293.
Every run passes both checkers at 548.255 kg. `v*/output` includes each source
solution, exact verifier-derived viewer export and run report. These replays do
not replace the incumbent fleet or the v269 live viewer's provenance.

`h100` contains 71 tests, six clean memcheck cases, eight independent table/tour
replays and the cold-table benchmark. `rtx5090` contains the corresponding local
evidence. The final test formatting is separately validated in `h100-final-tests`
and local memcheck; its final hash supersedes the initial H100 source snapshot.
`source` preserves that original snapshot. `final-source-sha256.json` identifies
final repository sources using normalized LF text. All mission runs use identical
CUDA/QOCO binary hashes; only the collection-table preparation hook changes.

The downloaded archive hash is recorded in `summary.json`; `sha256.json` pins
every evidence file except itself. CUDA branch counts include the extra rectangular
grid cells subsequently masked by the original arrival-window gate.
