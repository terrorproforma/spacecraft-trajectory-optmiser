# Independent CPU review of upstream replay v608

The eight saved calls contain no identified sign, Hessian symmetry, cone order,
coordinate translation, status conversion or original-equation KKT defect.
An independent 65-digit Decimal calculation from the exact represented FP64
coefficients and exported vectors agrees with every qualification verdict:
four supplied known points qualify at zero iterations, and all four cold
starts fail the common accuracy gate. This establishes a known-point acceptance
result, not a cold-solve speedup or a mission score improvement.

`review_saved_evidence.py` reproduces that check with Python's standard library.
It verifies all 37 indexed artifact files, the eight archived source hashes,
source aggregate, manifest, runner and executable identities, raw/report
agreement and all eight original-coordinate vector mappings. `findings.json`
contains the input and log hashes, independent metrics and stored-Q counts.
No solver was invoked and no raw output or acceptance gate was changed.

The native mapping is consistent with upstream's opposite dual-sign convention:
seed Pi is the negative of the established native normal, and exported Pi is
negated before recovering original equality/nonnegative/SOC multipliers.
The original upper P is mirrored once into full symmetric Q; standard SOCs
place the radius last with native length `vector_dimension + 2`. The shifted
objective constant is supplied once and the origin is added once when exporting
x. Original-objective and KKT evaluation use the original capture. Upstream
copies the supplied matrices and initial vectors, so temporary host seed
lifetimes are valid. Exact numeric vector mapping was checked; it does not
assert signed-zero bit identity.

Two review findings have been resolved:

- Both real Hessians are numerically zero but contain stored off-diagonal
  zeros: 1,893/2,100 full entries, including 1,260/1,398 off-diagonal entries.
  Upstream classifies these as sparse Q and dispatches to its BB inner solve.
  The final artifact README now distinguishes that implementation path from
  the mathematically explicit linear proximal map. The recorded 200,000 inner
  iterations at 100,000 outer iterations are consistent with the BB counter
  plus the unconditional outer-step counter. No exact-zero-removal ablation
  has been run. Missing quadratic CG alone does not explain these failures.
- The archived CMake hash input for `accelerator_c_api.h` lacked a corresponding
  configure dependency. Root added that header and the upstream patch to the
  current source's configure dependencies. This prevents stale provenance on
  incremental changes; it does not invalidate the separately hashed saved
  source and executable. The original archived CMake remains preserved.

Iteration-zero upstream scaling/unscaling changes 1,756 conditioning and 2,354
difficult primal entries by at most 4.440892098500626e-16. Both independent
certificates still pass. This is not a bit-exact seed roundtrip or an optimizer
step. Native OPTIMAL in the two analytic cold cases does not satisfy the common
gap gate: their gaps are approximately 1.29e-9 and 1.91e-9. Upstream's native
gap denominator is `1 + abs(primal) + abs(dual)`; the common audit uses
`max(1, abs(primal), abs(dual))`.

Timing scope is explicit: `solve_wall_seconds` includes the one-shot solve API's
internal setup, work, host transfers and final synchronization. Earlier host
`create_qp_problem` work is recorded separately as `setup_seconds`. These are
not pure GPU event times, qualified solutions/second, warmed workspace or full
SCvx measurements. The archived Python runner/build/packaging copies preserve
their original execution context; they are not portable launchers from their
copied archive locations. The supplementary nvcc validation is a CUDA-toolchain
compile and hidden-device input check, not a fresh complete CMake build.

Source references verified in this review:

- `cpp/cuda/tests/upstream_snapshot_replay.cu`: canonical import, seed Pi
  negation, one-shot solve, output normal negation and separate common audit.
- `_upstream/pdhcg/src/pdhcg.c:232`: literal matrix conversion;
  `:452`: objective constant; `:569`: copied start values.
- `_upstream/pdhcg/src/cone_utils.c:24`: SOC length;
  `src/kernels/pdhcg_soc_cone_kernels.cu:21`: radius-last convention.
- `_upstream/pdhcg/src/kernels/pdhcg_kernels.cu:680`: opposite dual sign.
- `_upstream/pdhcg/src/utils.cu:834`: structural Q classification;
  `src/pdhg_core_op.cu:749`: sparse-Q BB dispatch;
  `:668` and `:762`: inner-count updates;
  `:1292`: native gap normalization.
- `_upstream/pdhcg/src/solver.cu:121`: pre-update iteration-zero termination.
- `cpp/cuda/CMakeLists.txt`: linked pinned reference target, source hash inputs
  and corrected configure dependencies.

Reproduce from the repository root, CPU only:

```powershell
& 'C:/Users/Angus/.local/bin/python3.12.exe' -B build/performance/upstream-review-v608/review_saved_evidence.py --output build/performance/upstream-review-v608/findings.json
```

The reviewed artifact index is
`ca8db521f65608fa311d69cc419bcd14643c74207e572ca9a34333c56d35390a`.
The replay executable identity is
`d3e4673f6974e59500cae6f9b072f7a1305a68fd0862f31ea95c6e8b7e9d52e7`.
