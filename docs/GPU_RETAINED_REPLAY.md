# Replay the first solve of a retained trajectory workspace

An opt-in path removes a redundant synchronous startup solve when a compatible
GPU QOCO workspace is reused. It retains the existing numerical update, cold IPM
initialization, GPU residual/objective qualification and independent trajectory
certification. It does not reuse the preceding trajectory's solution.

Per-leg counters reset on reuse, but the vendor graph remains built. Previously,
the zero solve count forced synchronous priming even for that retained graph.
The adapter now tracks that distinction separately. After rebinding numerical
data, it can use the existing updated-data GPU replay. A genuinely cold/rebuilt
workspace still primes normally; a vendor readiness rejection also falls back
to the existing priming path. The retained marker is consumed once and cleared
on solver rebuild.

Enable the experiment with:

```bash
export SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY=1
```

Unset it or set it to `0` for the published default behavior. Workspace pooling
must also be enabled. Its existing eligibility rules still exclude nonzero Ruiz
scaling. This flag is independent of the earlier experimental early-graph flag,
which remains off in these comparisons.

## Complete campaigns

Each comparison alternates baseline/candidate/candidate/baseline on the same
binary. Both modes retain GPU option tables and the eight-entry workspace pool.
Native phase/replay traces are enabled in both modes; these are diagnostic
measurements, not isolated kernel timings.

| Hardware | Baseline process median | Candidate process median | Observation |
|---|---:|---:|---|
| RTX 5090 | 28.0839 s | 27.8027 s | 1.00% less time; ranges overlap |
| Lambda H100 | 29.5337 s | 29.3840 s | 0.51% less time; ranges overlap |

All eight campaigns preserve identical initial plans and logical search counts:
45,188,558 branches and 2,782,091 collection options. Both final mission checkers
pass, retaining 548.254620 weighted kg. Each candidate records exactly 30 first
solves successfully submitted through retained updated-data replay. Local median
startup-phase time falls from 5.4096 to 4.3684 seconds, but the overall timing
benefit is not firmly established.

## Fixed trajectory replay

The fixed 225-leg input is unchanged, with SHA-256
`09da720e7eca6481e9504ffa02840b5f35b9c2398ed229098c574c885510c846`.
Each mode runs in its own process with pooling enabled. Timings below sum solve
calls, excluding independent certification and artifact export.

| Hardware | Baseline solver time | Candidate solver time | Certified legs |
|---|---:|---:|---:|
| RTX 5090 | 127.9127 s | 128.2862 s | 205 / 225 in both |
| Lambda H100 | 158.4798 s | 146.1420 s | 205 / 225 in both |

Locally, host-dispatched startup attempts fall from 522 to 369 while workspace
creations remain 72. All baseline certified trajectories remain certified; the
maximum final-mass difference is 6.2378e-8 kg. Total solver time is 0.29% greater,
so this is not a demonstrated fixed-replay speedup. Individual difficult legs
vary substantially: case 87 takes 3.7904 versus 7.5412 seconds, while case 175
takes 4.4185 versus 2.0439 seconds. Neither `failed` nor the legacy `infeasible`
label proves global infeasibility.

H100 also reduces startup attempts from 522 to 369, retaining all 205 certified
trajectories with a maximum final-mass difference of 2.6328e-8 kg. Its total is
7.79% faster, but that is not evidence that retained replay caused the gain:
the 72 fresh-workspace cases, where the changed branch does not run, take
88.4127 versus 74.8762 seconds. The 153 reused-workspace cases take 70.0671
versus 71.2658 seconds. The corresponding local reused subset takes 54.0299
versus 57.0666 seconds. These are descriptive subsets, not separately controlled
benchmarks; neither overturns the full-run totals. Large numerical-path
variation prevents a portable speedup claim, so the new path remains opt-in.

## Validation and measurement limits

All 107 tests pass on each GPU. The new test rotates geometry and changes
duration and mass under zero-order/Lagrange thrust and state-origin off/on. It
checks independent physics, the unchanged 1e-5 kg reference-mass comparison,
first-solve counters and explicit successful updated-data replay. Existing tests
cover timeout discard, scaled-workspace exclusion, solver physics, CLI final
gates, the official example and three reference fleets.

The first local test attempt, v511, failed an incorrect transfer-byte assertion.
Adapter reports include cumulative counters from retained audit/conversion
objects, so those fields are not a per-leg byte ledger. The corrected test uses
the native replay trace to prove which path executed; no physics assertion was
weakened. These results make no transfer-bandwidth claim. A local replay runner,
v516, also failed before launching any solves after reading a report while its
producer was writing it; v518 began after that producer had terminated.

The best fleet score remains unchanged. Host route/fleet orchestration and
independent CPU verification remain, and this experiment does not establish a
fully GPU-controlled application or a portable throughput improvement.

## Reproducible evidence

- [Local checks, campaigns, full replay and retained failures](../results/lambda/2026-09-08/gpu-retained-replay-local-v518/summary.json)
- [H100 complete campaign comparison](../results/lambda/2026-09-08/gpu-retained-replay-v514/analysis.json)
- [H100 225-leg replay](../results/lambda/2026-09-08/gpu-retained-replay-legs-v517/analysis.json)

Archives include input/source snapshots, commands, logs and library hashes.
SHA-256 manifests cover every published file and each archive member. No QOCO
vendor binary or physics acceptance tolerance changed in this experiment.
