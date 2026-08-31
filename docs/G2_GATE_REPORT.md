# Gate G2 report

## Decision

**G2 FAIL. G3 is not authorised.**

Implementation commit `e17b6d2307868f01658489e04fb93d7596c17fa4`
passes the persistent box-QP/SOC numerical, lifecycle, pointer, stream,
allocation, checkpoint, cancellation, and standard sanitizer cases. G2 remains
failed because the frozen DLPack producer matrix is not implemented/executable,
the persistent kernel does not yet support exponential/power/PSD cones, and
the required track-unused initcheck mode reports library/descriptor
allocations.

## Implemented design

- lock-verified linked adapter against PDHCG commit
  `167c8b72b4b96d2f94d405b8763e485514192b81`, tree
  `62b05e6c1bedd385f6c267af3645ae4aae0421b4`;
- deterministic build-copy cleanup/initialisation patch, SHA-256
  `7f212ac5ef6afa96b7084092bfcff80602c2c976e1a7a9f3305d1817cb7554f4`;
- persistent owned Q/A/F topology and descriptors, cone metadata, numerical
  buffers, primal/dual/internal iterates, scaling/step state, stream/events,
  mapped cancellation state, diagnostics, and allocation ledger;
- no-exception-style status C ABI for create, values update, warm starts,
  solve/query/wait, independent residuals, reset/refresh, checkpoint/restore,
  cancel, diagnostics, and destroy;
- CUDA/device/managed pointer, dtype, shape, stride, access, alias, stream,
  lifetime callback, and fingerprint validation;
- one persistent block kernel with no inner-loop host staging and a single
  compact post-completion diagnostic transfer.

## Passing evidence

- CUDA Debug and Release builds: PASS with `CMAKE_CUDA_ARCHITECTURES=120`;
- CUDA CTest: 5/5 in both configurations;
- Python: 86/86; host native Debug: 41/41; Ruff: PASS;
- ten-update box-QP stress:
  - worst CPU primal error `5.74777776e-7`;
  - worst pinned one-shot primal error `5.09637298e-7`;
  - natural residual `1.26827116e-6`;
  - post-create allocation/topology/index-copy deltas all zero;
- managed-memory SOC:
  - persistent `x=[1.00000006795, 0]`;
  - pinned one-shot `x=[1.00000007468, 0]`;
  - cone distance `3.39757522e-8`;
  - natural residual `1.48236289e-7`;
- Compute Sanitizer standard runs:
  - memcheck: zero errors, zero leaks;
  - racecheck: zero hazards;
  - initcheck: zero errors;
  - synccheck: zero errors.

## Blocking negative evidence

1. `--track-unused-memory yes` is rejected by CUDA 12.8 because the option is
   flag-only in this tool version. The compatible flag form exits 99 and
   reports 11 unused cuBLAS/cuSPARSE/reference descriptor allocations.
2. The versioned C++ DLPack adapter is not connected to the persistent C ABI.
   CuPy, PyTorch, and JAX are also absent from the sealed environment, so no
   producer result is claimed.
3. Exponential, power, and PSD cone descriptors are explicitly rejected. SOC
   and rotated SOC are supported.
4. A host-only Release warnings-as-errors build exposes a pre-existing GCC
   `maybe-uninitialized` warning in `lambert_family.hpp`; the ordinary Debug
   warnings-as-errors gate passes 41/41 and the CUDA Release gate passes.

## Sealed archive

- directory:
  `results/gpu/g2/g2-20260831T225156Z-e17b6d2`
- reproducible archive:
  `results/gpu/g2/g2-20260831T225156Z-e17b6d2.tar.gz`
- archive SHA-256:
  `7efb893b913f83aa2e8c0444a161c9189db5a3cc347f6e25c15e5927eae7dec6`
- evidence-index SHA-256:
  `60239662da3683ab5d328d99a9806f05de03bf9ee59f6f590c8c727f27e30b6c`

The archive contains exact commands, complete logs, environment/lock data,
acceptance JSONL, negative results, and per-file SHA-256 records.
