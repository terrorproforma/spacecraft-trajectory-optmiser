# Multi-block CUDA collection-tour DP

See [implementation, timing scope and reproduction](../../../../docs/GPU_COLLECTION_DP_CUDA.md).
Code: `aff67ecd`, plus the final epsilon-order regression cases in this publication.

`summary.json` records the matched full-campaign comparison: 90.590 s CPU collection
DP versus 62.526 s CUDA collection DP, two runs per mode. All four campaigns pass
both mission checkers at 548.255 kg. Other numerical backends are identical.
This is a single-ship performance replay, not a new fleet leaderboard score.

- `v280`, `v284`, `v285`, `v286`: full run reports, exact commands, logs, Result.txt
  solutions and verifier-derived viewer exports. CPU comparison runs are v284/v286.
- `h100`: build, 69 integrated tests, four clean 28-case sanitizer runs, eight real
  input replays, paired warm benchmark, source and binary fingerprints.
- `rtx5090`: corresponding 69 integrated tests, replay, benchmark, memcheck and
  four additional epsilon-order tests.
- `h100-final-tests`: final 32 native cases, including epsilon ordering, run on
  the same binary. This test source supersedes the earlier 28-case test hash.
- `source`: original validated source snapshot; `final-source-sha256.json` pins
  the final repository sources using normalized LF text.
- `comparison`: sequential matched-run controller and terminal report.
- `rejected-wrapper-v281`: failed Python bootstrap, before any campaign started;
  excluded from timing statistics.

`collect-dp-final-v287.tar.gz` is the byte-verified downloaded Lambda archive;
its SHA-256 is in `summary.json`. `sha256.json` pins every evidence file except
itself. Binary paths identify retained builds on their respective machines.

The current live visualiser retains the earlier v269 verified 548 kg mission:
http://127.0.0.1:4173/?dataset=gtoc12-v269&epoch=69807&preset=oblique&z=1
Its original samples and compute provenance are unchanged. The latest replay's
own export is `v285/output/fleet/viewer/trajectories.json`, with its manifest and
source solution archived alongside it; new timings have not been attributed to
the old viewer run.
