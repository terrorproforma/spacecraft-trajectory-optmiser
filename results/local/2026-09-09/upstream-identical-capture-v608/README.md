# Pinned upstream identical-capture comparison — v608

The original pinned PDHCG implementation accepts both real known-qualified
GTOC12 conic points at iteration zero. Neither real cold-start problem qualifies
within 100,000 iterations. This separates the known-point stopping defect in
our persistent adapter from a broader cold-convergence problem; it does not
establish a performance win, trajectory certificate or fleet improvement.

Eight native calls ran once on the local RTX 5090 under the shared GPU lock.
Four exact known-point calls (two analytic and two real) pass the unchanged
independent KKT gate. Two analytic cold calls report native OPTIMAL but narrowly
fail the stricter common gap gate. Both real cold calls report ITERATION_LIMIT.
All outcomes, including failures, are retained.

| Case | Iterations | Native API wall seconds | Common primal | Common dual | Common gap | Qualified |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mixed-seeded | 0 | 0.288106 | 0 | 8.88178e-17 | 0 | yes |
| shifted-seeded | 0 | 0.206753 | 0 | 0 | 0 | yes |
| mixed-cold | 610 | 0.584366 | 5.94552e-10 | 3.86208e-10 | 1.28852e-09 | no |
| shifted-cold | 660 | 0.556870 | 9.11962e-10 | 4.144e-10 | 1.90649e-09 | no |
| conditioning-cold | 100,000 | 37.332046 | 4.85763e-06 | 8.9182e-09 | 0.00132932 | no |
| difficult-cold | 100,000 | 34.817582 | 3.55732e-06 | 2.85111e-05 | 0.0170812 | no |
| conditioning-seeded | 0 | 4.335978 | 9.16887e-16 | 2.72821e-16 | 1.26052e-11 | yes |
| difficult-seeded | 0 | 2.724576 | 3.04311e-12 | 4.73696e-16 | 3.9994e-11 | yes |

## What was held fixed

- Upstream commit `167c8b72b4b96d2f94d405b8763e485514192b81`, tree
  `62b05e6c1bedd385f6c267af3645ae4aae0421b4`, clean source verified locally.
  The existing `0001-free-quadratic-state.patch` is included and hashed.
- The exact original-coordinate captures and QOCO known points from v606.
  Full symmetric Q is constructed once, all scalar inequalities retain their
  dual variables, standard SOC permutations are unchanged, and every upstream
  Pi is negated before the established original-coordinate conversion.
- Native tolerances 1e-9, presolve disabled, upstream default scaling, restart,
  infinity norm and inner iterations retained. Real cold limits are 100,000
  iterations and 60 seconds; real seeded limits are one iteration and 10 seconds.
  A process timeout adds at most 20 seconds to each native time limit.
- Independent original-equation primal/dual/gap and per-block complementarity
  tolerance 1e-9, cone violation 1e-8. The unchanged Python auditor receives only
  a documented termination-code adapter (upstream OPTIMAL is code 1); its
  mathematical formulas and tolerances are not modified. Every C++ qualification
  verdict agrees with the independent Python verdict.

Four hidden-device CPU input checks and one duplicate-option rejection passed.
Analytic cold points verify the dual sign/permutation within 1e-6; that mapping
check does not relax their strict qualification result in the table.

## Interpretation and limits

The older persistent v606 implementation imported the same two real known points
correctly, rejected them under its absolute natural-residual rule and moved away
from them. Upstream's iteration-zero acceptance supports the planned opt-in
common-KKT stopping intervention. Acceptance of a supplied optimum is a
correctness regression test, not evidence of fast optimisation from a cold start.

At 100,000 iterations, upstream's conditioning capture has a much smaller gap
than the old persistent replay; the difficult capture does not improve its gap.
Neither qualifies. Comparing their unqualified wall times as a speedup would be
misleading. These captures have zero Hessians, so their mathematical primal
proximal maps reduce to explicit linear steps. Their literal sparse matrices
retain off-diagonal zero entries, however. Upstream's structural classification
selects its sparse-Q BB inner-solve path rather than its non-Q fast path. The
recorded 200,000 inner iterations are consistent with that dispatch. An exact
zero-removal ablation would be a separate experiment; none was run here.
Missing quadratic CG alone cannot explain these failures.

Timing is the synchronous one-shot native C API including its setup, GPU work
and host transfers. It is not pure GPU event time, a warmed repeated-workspace
benchmark, solutions/second, or whole-SCvx time. The recorded 200,000 inner
iterations in each real cold case are upstream telemetry, not 200,000 certified
trajectory solutions. The test does not connect upstream or persistent PDHCG to
the native GTOC12 outer loop. Current GTOC12 refinement still uses GPU QOCO.

The verified 23-ship incumbent remains 14,051.854893908598 raw kg and
12,810.135953048577 fixed-bonus weighted kg. No viewer dataset is promoted.

## Evidence and reproduction

`manifest.json` records the static reference archive, exact compiled source
hashes, compiler/link command and executable hash. `diagnostic-sources.zip`
contains the exact replay/importer, headers, patch, lock and independent auditor;
the upstream tree is identified by the repository lock. Public-header additions
in this source snapshot do not invoke the new persistent policy in the reference
executable. Binaries are retained locally in the build directory, not included
here. `run/report.json`, each raw stdout log and `run/run.py` retain parameters,
initial-point hashes, original vectors and independent results.

The new repository target is `upstream_snapshot_replay` with BUILD_TESTING on.
Run the analytic fixtures before either real capture. Each call requires an
explicit `--iterations` and `--deadline-seconds`; `--initial-point` selects a
separate known-point diagnostic, while `--validate-only` makes no CUDA API call.
Use the shared GPU lock, preserve complete stdout/stderr, and apply the common
external gate to every result. The recorded comparison is not rerun by packaging.


The separate CUDA-toolchain compile and hidden-device input check also pass;
their logs are retained in `nvcc-cmake-check`. The initial `-Wpedantic` experiment
rejected nvcc-generated GCC line directives, not diagnostic C++ source; those
logs remain in `nvcc-check`. The source itself passed strict g++ warnings.
This compile validation did not add any GPU solver calls.

Iteration-zero upstream acceptance preserves accuracy but does not preserve
every primal bit: scaling/unscaling changes 1,756 and 2,354 entries by at most
4.440892098500626e-16. `seed-roundtrip.json` records that distinction. No
optimisation step was taken, and both common certificates remain qualified.
