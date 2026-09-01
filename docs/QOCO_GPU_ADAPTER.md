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

`PDHCGQOCOHybrid` runs a PDHCG backend first and applies the frozen `1e-6`
handoff gate before calling QOCO. Both reported PDHCG residuals, an
independently recomputed canonical primal residual, finite canonical ordering,
and the exact final CQP numeric fingerprint must pass. An ineligible predictor
raises `QOCOHybridIneligibleError` with a complete `HybridRunReport`; QOCO is
not run and the result is not labelled hybrid. QOCO exposes no dual-start API,
so a supplied PDHCG dual is always discarded and reported as
`discarded-unsupported-by-pinned-qoco`. The QOCO result is
converted back to canonical primal/dual ordering before optional nonlinear
handback. Failed QOCO results are returned with a failure classification, and
`solve_and_handback` refuses to pass a failed result to the nonlinear owner.

## Nonlinear candidate handback

`handback_qoco_candidate` keeps pure `pure-gpu-ipm` and qualified
`hybrid-pdhcg-ipm` records distinct. It rechecks independent canonical QOCO
quality and the exact CQP/topology fingerprints before invoking a
`DeviceNonlinearOwner`. The owner must prove canonical identity ordering,
device replay, complete family path inventory, and zero hidden CPU fallback.

The native `spacepdhcg_cuda_scvx_driver_handback_qoco` C ABI copies canonical
primal and dual vectors to the resident workspace, gathers state/control
trajectories, and evaluates RK4 dynamics, path, terminal, virtual-control and
quaternion metrics on the declared CUDA stream. It applies the same frozen
predicted/actual reduction, restoration, trust shrink/retain/expand, and
transactional commit rules as the production outer driver. Rejected
candidates leave the resident nonlinear reference unchanged. Conversion,
setup, polish, host-to-device transfer, replay, and acceptance costs are
reported separately.

## Serialized GPU validation

The pinned CUDA/cuDSS library was loaded on the RTX 5090 and the adapter's 18-test
qualification passed. It includes exact trajectory QP and SOCP agreement with Clarabel,
same-pattern updates, cold and accepted-primal warm solves, independent residuals, and explicit
dual discard. The CUDA runtime requires both `build/qoco-cudss-lib` and the pinned
`nvidia/cu12/lib` directory on `LD_LIBRARY_PATH`; omitting them is correctly classified as setup
failure rather than solver evidence.

This validates the original adapter and CUDA ABI, not the new nonlinear
handback. The dedicated short `device_scvx_qoco_handback_test` remains deferred
until the single RTX 5090 is free. No performance, energy, or full frozen
matrix result is implied by the compile/static coverage.
