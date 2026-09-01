# G4 checkpointed campaign scheduler

`scripts/gpu/run_g4_campaign.py` preserves the frozen 24,883,200-row Cartesian logical ledger while
scheduling 2,764,800 persistent execution groups. Each group contains two warm-ups followed by seven
measurements in one process, session, and workspace. Raw attempts remain distinct; warm-ups are
excluded from statistics. See `docs/G4_EXECUTION_CONTRACT.md`.

## Durability

- `checkpoint.sqlite3` uses WAL journaling and `synchronous=FULL`.
- Checkpoint metadata pins the exact clean source commit, policy hash, schedule kind, and group
  cardinality.
- `journal.jsonl` is append-only and fsynced after every claim and terminal transition.
- Every attempt uses `runs/<coordinate-id>/<random-attempt-id>/`; files use exclusive atomic
  creation and can never replace earlier evidence.
- Restart converts an interrupted attempt into an immutable `interrupted` record and creates a new
  attempt directory for the same coordinate.
- A group completes only after all nine raw attempt records validate and every measured attempt
  carries a strict Paper 1 result. Invalid evidence is quarantined and remains incomplete.
- A non-blocking process lock enforces one measured GPU worker.

## Hash-pinned executor capability

`device_scvx_integration_test --g4-session` consumes and reports the complete execution-group
contract.
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
- one process, persistent session/workspace, separate attempt records, and independent policy reset
  under execution-contract version `g4-persistent-group-v1`.

Every raw attempt must repeat its group, physical instance, and repeat identities with an exact
disposition, reason, timing, and launch state. A timeout or OOM is legal only after actual launch.
The scheduler quarantines any disagreement or incomplete measured result.

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

The final command is also the restart command. It resumes an interrupted group before claiming new
work. `--max-runs N` creates bounded group chunks. The frozen per-attempt timeout is enforced by the
persistent executor; the outer process has only a nine-attempt safety boundary. Rows are never
classified from a prediction, and larger rows stay pending until launch.

Energy is sampled around each launched process using 50 ms `nvidia-smi` requests. Records include
sample count, maximum observed gap, validity, integrated joules and the shared-display-GPU caveat.
CUDA startup remains separately reported by the executable and excluded from the accepted common
timing boundary.
