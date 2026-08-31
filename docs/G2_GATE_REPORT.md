# Gate G2 report

## Decision

**G2 PASS. G3 is authorised.**

The blocker-closing implementation is commit
`66e714b466989159b3fe29d756d9ce7d7cf9e7fe`; the reproducible evidence runner
and its WSL/CUDA fixes are commits `f3c4969`, `fcf8d5e`, and `c804a2d`.
All G2 acceptance requirements in `GPU_EXECUTION_ROADMAP.md` sections
8.14-8.16 now pass.

## Implemented design

- lock-verified linked adapter against PDHCG commit
  `167c8b72b4b96d2f94d405b8763e485514192b81`, tree
  `62b05e6c1bedd385f6c267af3645ae4aae0421b4`;
- deterministic build-copy patch SHA-256
  `dd31b99869bda77400d8cf33c2048ad424d345a17b17898420f5dab96766ecc6`;
- persistent owned topology, cone metadata, numerical buffers, retained
  iterates, scaling, streams/events, diagnostics, cancellation, and allocation
  ledger;
- no-exception C ABI for the complete lifecycle plus legacy/versioned DLPack
  create, asynchronous update, and asynchronous warm-start ingestion;
- one-shot Python capsule consumption, producer stream handoff, and retained
  managed-tensor deleters without host copies, conversion, or global sync.

## Passing evidence

- fresh CUDA Debug and Release Werror builds: PASS for SM 120;
- CUDA CTest: 7/7 in both configurations;
- Python: 88/88; host Debug and Release Werror: 41/41 each; Ruff: PASS;
- actual versioned DLPack producers:
  - CuPy 14.2.0: PASS;
  - PyTorch 2.11.0+cu128: PASS;
  - JAX/JAXlib 0.11.1 with CUDA 12 plugin: PASS;
  - every producer passes non-default stream handoff, premature update
    producer release, invalid-dtype rejection, pointer stability, and the
    `[2/3, 1/3]` box-QP reference;
- independent legacy/versioned fixture passes exact-once deletion, CUDA device
  and managed storage, rank/shape/stride/alignment/dtype/access/device/
  fingerprint rejection, and host-span rejection;
- ten-update box-QP worst CPU error `5.74777776e-7`, worst pinned one-shot
  error `5.09637298e-7`, natural residual `1.26827116e-6`, and zero
  post-create allocation/topology/index-copy deltas;
- managed SOC matches CPU and pinned one-shot with cone distance
  `3.39757522e-8` and natural residual `1.48236289e-7`;
- Compute Sanitizer memcheck (including DLPack lifetime), racecheck, flag-only
  tracked initcheck, and synccheck all report zero errors/hazards/leaks.

## Resolved findings and cone scope

1. CUDA 12.8 uses flag-only `--track-unused-memory`; the roadmap now has the
   valid command. Five application/cuSPARSE scratch reports were eliminated by
   deterministic initialization, and the unused persistent cuBLAS handle was
   removed. The pinned one-shot comparator remains exercised by CTest; the
   prescribed persistent-only initcheck target does not instantiate comparator
   cuBLAS internals. No suppression or threshold is used. Historical raw
   findings remain in the prior sealed archive.
2. Pinned PDHCG supports exponential, power, and PSD cones. The persistent G2
   boundary explicitly returns `SPACEPDHCG_CUDA_UNSUPPORTED` for them and tests
   this separately from numerical failure. This is not a G2 blocker:
   authoritative section 8.14 scopes correctness references to QP/SOCP and
   section 8.16 does not require the wider inventory. No relaxation or epigraph
   substitution is performed.
3. The Release Lambert warning was fixed with an initialized prior sample plus
   presence flag. Fresh Release Werror compilation and 41/41 tests pass.

## Acceptance enumeration

1. CPU and pinned one-shot numerical equality: PASS.
2. Zero post-create topology allocations/index copies: PASS.
3. Declared stream use, including real DLPack producers: PASS.
4. Warm starts and checkpoint/restore retention: PASS.
5. Independent residuals: PASS.
6. Required sanitizer matrix: PASS.
7. No hidden CPU fallback: PASS.
8. Validation, ownership, cancellation, destruction, and lifecycle paths:
   PASS.
9. Issue #2 implementation criteria: PASS; H1 measurement remains G3 work.

## Sealed archive

- directory: `results/gpu/g2/g2-20260831T232513Z-c804a2d`
- archive: `results/gpu/g2/g2-20260831T232513Z-c804a2d.tar.gz`
- archive SHA-256:
  `9e3bdb1075bbbe5c075b182e7925fc5bbfbe4a21845f5456b7b93ef97f6e66ff`

The archive contains exact commands, complete logs, environment/lock data,
producer versions/results, acceptance JSONL, sanitizer diagnostics, and
per-file hashes. The previous FAIL archive is retained unchanged.
