# Verified fleet regeneration v799

Both GPUs produce a full-fleet-checker-qualified **12,992.035662 weighted kg /
14,271.047228 raw kg** solution, with 23 ships and 200 asteroids. The weighted
gain over v780 is 148.479966 kg. Nine routes are replaced. The H100 replay is in
`h100-best/Result.txt` and the existing viewer's `gtoc12-regeneration-v799` dataset.

See [the full report and copy-paste viewer instructions](../../../../docs/GPU_FLEET_REGENERATION.md).

## Contents

`local.tar.gz` and `h100.tar.gz` each contain 1,127 payload files and `FILES.json`.
Every payload's byte count and SHA-256 are checked after transfer. They retain:

- `runtime`: frozen v788 source, source manifest, CUDA/QOCO libraries and bonus table.
- `v792`: all-ship weighted pilot, cProfile data and every surrogate route.
- `v793`: four native refinements and the rejected 23-ship fleet. This fleet fails
  the raw-mass ship-count rule and must not be promoted.
- `v794`: wider raw-mass searches, exact settings, telemetry and every candidate.
- `v795`: 26 refinements, all 373 native calls, 18 certified routes, selected fleet
  and both complete-fleet checker results.
- `v796`: combined 44-column CUDA selection and complete recheck.
- `v798`: retained-workspace budget sweep through an exhaustive 17,607,866-node
  pass, complete recheck and exact-sample viewer export.

Concise reports are extracted under `local` and `h100`. `summary.json` separates
stage timings and counters. Native call outcomes differ slightly between GPUs;
both have 365 converged calls and eight rejected calls in the main refinement.
The exact candidate schedules and cargo match. Four approximate proxy fields
have differences no larger than 2.30e-11; raw archives retain both versions.

## Audit without a GPU

Run Python 3 from this directory:

```text
python audit_saved.py
```

The standard-library auditor verifies all archive members, source/library hashes,
solution fingerprints, native call counts, selected-column certification, exact
candidate schedules/cargo and emitted score/raw-mass bookkeeping. It preserves
the rejected v793 fleet as rejected. `saved-audit.json` records the successful
audit; `candidate-comparison.json` records the cross-GPU proxy differences.
This is a saved-data audit, not a rerun of physical propagation.

## Rerun the numerical experiments

Each raw archive retains the actual `run.py` and `launch.py` for every phase.
They use the frozen source in `~/spacepdhcg-grid-v788/repo/src` and the supplied
SM120 (local) or SM90 (H100) native library. Recreate the archived directory names
under the target user's home, install the source's Python dependencies, supply
the pinned competition data and the matching CUDA/cuDSS runtime, then run the
phase launchers in order. The launchers record all environment settings and hold
the shared GPU lock. Use fresh output directories; do not overwrite completed
evidence. The `reproduce` folder retains the generators and viewer importer
preparation script; machine-specific launchers should be reviewed before reuse.

The final pool is optimal only among its 44 certified columns. Full GPU fleet
control and global mission optimality remain ongoing work.
