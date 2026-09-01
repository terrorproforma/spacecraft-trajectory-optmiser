# G4 checkpointed campaign scheduler

`scripts/gpu/run_g4_campaign.py` schedules the frozen 24,883,200-row Cartesian ledger without
materialising it in memory. Each coordinate has a SHA-256 identity over canonical JSON. Execution
order applies the frozen solver-order rotation while preserving the canonical ledger ordinal.

## Durability

- `checkpoint.sqlite3` uses WAL journaling and `synchronous=FULL`.
- `journal.jsonl` is append-only and fsynced after every claim and terminal transition.
- Every attempt uses `runs/<coordinate-id>/<random-attempt-id>/`; files use exclusive atomic
  creation and can never replace earlier evidence.
- Restart converts an interrupted attempt into an immutable `interrupted` record and creates a new
  attempt directory for the same coordinate.
- Valid launched-process timeout, OOM, unsupported, numerical, qualified and unqualified results
  count as completed. Invalid records are quarantined and remain incomplete.
- A non-blocking process lock enforces one measured GPU worker.

## Fail-closed executor capability

The current `device_scvx_integration_test --g4-sample` interface does not consume the frozen
evaluation seed or conditioning span. Consequently it cannot truthfully execute the complete
matrix yet. The scheduler refuses to claim a row unless an independently produced capability JSON
pins:

- the exact executable SHA-256 and policy SHA-256;
- application of seed, conditioning, dispersion, transfer and solver-order parameters;
- the common timing boundary;
- independent nonlinear replay.

This prevents nominal fixture repetitions from being mislabeled as distinct matrix rows. A
capability must only be emitted after the production executable implements and tests those
parameters.

## Commands

```bash
python scripts/gpu/run_g4_campaign.py init \
  --campaign build/g4-campaign

python scripts/gpu/run_g4_campaign.py status \
  --campaign build/g4-campaign

python scripts/gpu/run_g4_campaign.py run \
  --campaign build/g4-campaign \
  --executable build-integration-cuda-release/cuda-tests/device_scvx_integration_test \
  --capabilities build/g4-executor-capabilities.json
```

The final command is also the restart command. It resumes an interrupted coordinate before
claiming new work. `--max-runs N` creates bounded checkpoint chunks. The frozen per-process timeout
is read from `benchmarks/g4_policy.json`; rows are never classified from a prediction.

Energy is sampled around each launched process using 50 ms `nvidia-smi` requests. Records include
sample count, maximum observed gap, validity, integrated joules and the shared-display-GPU caveat.
CUDA startup remains separately reported by the executable and excluded from the accepted common
timing boundary.
