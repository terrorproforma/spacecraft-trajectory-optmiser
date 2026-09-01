# QOCO-GPU production adapter

`spacepdhcg.backends.QOCOGPU` targets the QOCO C ABI pinned by
`third_party/qoco_gpu.lock.json` at commit
`09f049597deef2a7ead15b3da19a9456ff7d4e53`. It works with either that
revision's CUDA/cuDSS build or its builtin CPU build; the latter exists only
for static and ABI correctness tests.

## Exact mapping

The adapter maps the canonical objective to QOCO's upper-triangular CSC
Hessian after verifying that the original Hessian is symmetric and positive
semidefinite. Scalar and variable equalities become `A*x = b`. Upper and lower
bounds become, respectively, `u - C*x >= 0` and `C*x - l >= 0`, with box rows
split exactly.

QOCO requires the nonnegative orthant before all SOC blocks. The converter
therefore records row provenance while reordering scalar bounds, variable
bounds, affine cones, and variable cones. Native SOC coordinates
`[vector..., radius]` are permuted to QOCO's `[radius, vector...]`. Rotated
SOCs use the orthonormal map

`(w, u, v) -> ((u+v)/sqrt(2), w, (u-v)/sqrt(2))`.

Exponential, power, and PSD cones are classified as unsupported rather than
approximated. A numeric update that changes bound type, cone dimensions, or
CSC sparsity is also rejected explicitly.

## Lifecycle and ownership

QOCO copies setup arrays and retains one symbolic workspace. Same-pattern
matrix and vector updates use `qoco_update_matrix_data` and
`qoco_update_vector_data`. The pinned cleanup function frees the
`QOCOSolver` allocation itself, so the ctypes bridge allocates that object
with the C allocator and releases it exactly once. Context-manager use is
recommended:

```python
with QOCOGPU(problem, library_path="build/qoco-g4/libqoco.so") as solver:
    solution = solver.solve()
```

Setup, conversion, numeric update, symbolic analysis, and solve/polish times
are retained in `last_report`, outside the frozen `CQPSolution` schema.
Native residuals are diagnostic only. Published `CQPSolution` residuals are
recomputed independently in original, unequilibrated canonical coordinates.

## Hybrid handoff

`PDHCGQOCOHybrid` runs a PDHCG backend first and offers a solved, finite
primal to QOCO. Qualification records the canonical primal residual and
whether the start was accepted. QOCO exposes no dual-start API, so a supplied
PDHCG dual is always discarded and reported as such. The QOCO result is
converted back to canonical primal/dual ordering before optional nonlinear
handback. Failed QOCO results are returned with a failure classification, and
`solve_and_handback` refuses to pass a failed result to the nonlinear owner.

## Serialized GPU validation still required

CPU/static tests validate conversion, ownership, the C ABI, same-pattern
updates, warm-start routing, residuals, and agreement with Clarabel. They do
not qualify the CUDA backend or any G4 performance claim. Once the single RTX
5090 is available, run the existing pinned build script and then serialize:

1. one production trajectory QP and one SOCP through `QOCOGPU`;
2. one cold and one accepted-primal warm solve;
3. one PDHCG-to-QOCO hybrid polish with dual-discard evidence;
4. failure-path checks for CUDA/cuDSS loading and device OOM;
5. only after matched nonlinear quality, the unchanged G4 policy matrix.
