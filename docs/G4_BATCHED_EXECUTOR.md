# G4 persistent campaign executor

## Baseline decomposition

The first qualified row (`ordinal=0`, fixed-tight) took 356.864485 s wall time.
The native record attributes 354.418484 s (99.315%) to CQP work and
354.280054 s to the SCvx driver. It executed 100 outer iterations and
3,230,000 PDHCG iterations, with no recovery. The residual 2.446001 s is the
strict upper bound for process startup, CUDA context creation, problem and
coefficient construction, workspace setup, replay/validation, JSON emission,
and pipe handling because the old schema did not expose those fields
individually. The scheduler journal adds 0.7456 s between the measured process
boundary and the durable terminal transition, including validation, result
serialization, exclusive file creation, SQLite FULL synchronization, and
journal fsync.

The next policy row was actually executed for 600.259016 s and retained as a
timeout. Energy was invalid in both rows: the first maximum gap was 0.317344 s
and the timeout maximum gap was 0.527754 s.

Removing every non-CQP cost could improve the first row by only 1.0069x. If
all 24,883,200 rows cost the first row's 354.418484 s of CQP work, the compute
lower bound is 279.3 years; actual timeout rows make that optimistic. A
one-year campaign requires 2,838.6 rows/hour (1.268 s/row), 281.3x the
observed 10.09 rows/hour. A 30-day campaign requires 34,560 rows/hour and
3,425x observed throughput. Process persistence is necessary, but cannot
provide the required speedup by itself.

## Implemented execution path

- `device_scvx_integration_test --g4-server` initializes CUDA once and accepts
  tab-delimited frozen row requests over stdin. Every response carries the
  coordinate content address so the scheduler rejects protocol cross-talk.
- `PersistentExecutor` keeps one server alive, enforces the real 600 s row
  boundary, restarts after a timeout/crash, and records the process generation
  and one-time CUDA startup separately from the accepted timing boundary.
- Every request remains sequential and independent. No concurrent stream or
  batched-kernel timing is admitted yet.
- Full stdout, including every progress record, is retained as a deterministic
  gzip object under `objects/sha256`; per-attempt results retain the object
  hash, byte count, exact terminal records, and progress-record count.
- Direct NVML sampling replaces per-sample `nvidia-smi` subprocesses. The
  sampler obtains synchronized start/end boundary samples, can be pinned to a
  CPU core, records cadence gaps, and preserves the shared-display warning.
- Migration requires an exclusive nonblocking lock on the source campaign.
  Terminal evidence is copied immutably, coordinate hashes are recomputed
  against the frozen ledger, duplicate imports are no-ops, and future claims
  skip imported ordinals.
- New G4 sample records expose topology, coefficient generation, workspace
  creation, update, scaling, solve, residual, replay, acceptance, transfer,
  recovery, and QOCO timing fields for the representative pilot.

## Launch gate

The old worker must remain active until all of the following pass: Release and
Debug CUDA builds, sanitizers, old/new equivalence over all frozen axes and
failure modes, leakage tests, crash/restart tests, direct-NVML cadence tests,
and a representative throughput pilot. Cross-row workspace reuse, bounded
cache/backpressure, in-process cancellation with final progress, and grouped
physical-instance scheduling are not implemented by process persistence
alone. Therefore migration and launch are prohibited until those items are
complete; changing a timeout into a predicted or synthetic classification is
also prohibited.
