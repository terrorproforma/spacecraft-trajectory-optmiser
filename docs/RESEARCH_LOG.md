# Research log

## 2026-08-30 — Programme lock and repository bootstrap

### Decisions

- The first paper is **B + D + C**, implemented in that order.
- The second paper is **E**, built on the stable continuous trajectory oracle.
- The repository is the system of record; code, documents, experiments and decisions are committed here.
- The first executable spine is a fixed-pattern CW rendezvous QP with repeated numerical updates and warm starts.
- CPU OSQP is the initial correctness/persistence baseline, not a claim about final GPU performance.
- The hot-loop design prohibits symbolic sparse reconstruction.
- Fixed grids are used until persistent-buffer performance is established.
- Accuracy and feasibility comparisons are end to end; inner-solver kernel timing alone is insufficient.

## 2026-08-30 — M0 executable spine passes CI

### Evidence

- Python 3.11 and 3.12 CI passed lint, seven unit tests, coverage collection and the repeated-solve benchmark.
- The 12-interval CW smoke problem used 114 variables and 120 scalar constraints.
- Three repeated workspace updates produced a maximum dynamics defect of approximately `2.84e-14` and a maximum terminal error of approximately `4.52e-23`.
- The measured CPU reference values are smoke-test observations, not performance claims and not cross-hardware benchmark results.

### Interpretation

The fixed symbolic structure, numerical update path, warm-start lifecycle and independent trajectory checks are operational. M0 is therefore complete at the CPU reference level.

## 2026-08-30 — Upstream PDHCG interface audit

### Findings

- PDHCG natively separates scalar rows `l <= A x <= u`, affine cone rows `F x + g in K`, variable bounds and variable cone blocks.
- Its SOC coordinates are `[v, w, z]`, with total length `v_dim + 2`.
- The Python model exposes coefficient setters and primal-dual starts, but `optimize()` calls a one-shot `solve_once` core binding.
- The public C API has create, start-value, solve and free functions but no persistent updateable solver workspace.
- Python multi-GPU support is not currently exposed; distributed solving is through the C++/MPI/NCCL executable.

### Decision

ADR 0001 locks a native-compatible canonical form. The repository now adds a persistent Clarabel CPU reference that converts PDHCG cone coordinates explicitly. The actual B contribution remains a lower-level PDHCG extension with persistent device buffers and in-place coefficient updates.

### Immediate next work

1. Close the native SOC correctness gate in CI.
2. Add a one-shot PDHCG adapter for CUDA-enabled integration tests.
3. Define the C++ `PersistentCQP` ownership and stream semantics against upstream internal types.
4. Add random trajectory-banded QP/SOCP cross-solver fixtures.
5. Begin the nonlinear 3-DoF powered-descent transcription only after the conic bridge is stable.
