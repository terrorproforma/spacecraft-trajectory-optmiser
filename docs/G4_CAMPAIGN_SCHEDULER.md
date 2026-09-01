# G4 checkpointed campaign scheduler

`scripts/gpu/run_g4_campaign.py` schedules the frozen 24,883,200-row Cartesian ledger without
materialising it in memory. Each coordinate has a SHA-256 identity over canonical JSON. Execution
order applies the frozen solver-order rotation while preserving the canonical ledger ordinal.

## Durability

- `checkpoint.sqlite3` uses WAL journaling and `synchronous=FULL`.
- Checkpoint metadata pins the exact clean source commit, policy hash and ledger cardinality.
- `journal.jsonl` is append-only and fsynced after every claim and terminal transition.
- Every attempt uses `runs/<coordinate-id>/<random-attempt-id>/`; files use exclusive atomic
  creation and can never replace earlier evidence.
- Restart converts an interrupted attempt into an immutable `interrupted` record and creates a new
  attempt directory for the same coordinate.
- Valid launched-process timeout, OOM, unsupported, numerical, qualified and unqualified results
  count as completed. Invalid records are quarantined and remain incomplete.
- A non-blocking process lock enforces one measured GPU worker.

## Hash-pinned executor capability

`device_scvx_integration_test --g4-sample` consumes and reports the complete coordinate contract.
Evaluation seeds drive deterministic family-specific physical inputs. Conditioning bins apply an
equivalent positive scaling across dynamics equality rows; both coefficients and equality bounds
are transformed, so the physical feasible set is unchanged. The independent CPU expectation is
transformed identically and checked against the GPU buffers before solve. Workspace scaling remains
a separate runtime policy.

`scripts/gpu/generate_g4_executor_capability.py` queries the compiled executor and writes a
content-addressed capability record that pins:

- the exact source commit, executable, policy and matrix SHA-256 values;
- numerical application of family, intervals, policy, quality, conditioning, scaling, warm-start,
  family classes and evaluation seed;
- execution-only treatment of warmup/repeat identity and frozen solver order;
- the common timing boundary;
- independent nonlinear replay.

Every successful row must repeat all requested/applied axis values and the coordinate, capability,
policy and matrix hashes. It also emits deterministic instance/problem/coefficient hashes,
conditioning-factor extrema, measured coefficient dynamic range and CPU/GPU coefficient parity.
The scheduler quarantines any disagreement.

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

Generate the capability only from a clean committed source and its final executable:

```bash
python scripts/gpu/generate_g4_executor_capability.py \
  --executable build-integration-cuda-release/cuda-tests/device_scvx_integration_test \
  --output build/g4-executor-capabilities.json
```

The final command is also the restart command. It resumes an interrupted coordinate before
claiming new work. `--max-runs N` creates bounded checkpoint chunks. The frozen per-process timeout
is read from `benchmarks/g4_policy.json`; rows are never classified from a prediction.

Energy is sampled around each launched process using 50 ms `nvidia-smi` requests. Records include
sample count, maximum observed gap, validity, integrated joules and the shared-display-GPU caveat.
CUDA startup remains separately reported by the executable and excluded from the accepted common
timing boundary.
