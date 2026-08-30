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

### Immediate next work

1. Implement the canonical CQP data model.
2. Implement exact-discrete CW dynamics and fixed-pattern QP construction.
3. Implement the persistent OSQP lifecycle.
4. Add tests and a repeated-solve timing command.
5. Use CI results to close M0 and then introduce SOC constraints and the first PDHCG bridge.
