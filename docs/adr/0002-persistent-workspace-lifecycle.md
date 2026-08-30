# ADR 0002 — Persistent workspace lifecycle, streams and rescaling

- **Status:** accepted
- **Date:** 30 August 2026
- **Decision scope:** contribution B C++/CUDA implementation

## Context

Upstream PDHCG's public API follows a create/solve/free lifecycle. Its internal solver state,
however, already owns the objects needed for persistence: GPU primal-dual iterates, reflected
points, SpMV descriptors, preconditioner data, cone projection workspaces, residual buffers and
inner-solver warm starts.

Successive convexification keeps matrix dimensions, sparse index arrays and cone layout fixed
while updating numerical values. Re-entering the public one-shot path would repeat host copies,
CSR conversion, presolve, scaling, CUDA allocation and descriptor creation at every outer
iteration. Avoiding that work is contribution B.

Persistence is not only an allocation question. Updating Q, A, F and the bound/objective vectors
also raises three correctness questions:

1. when are borrowed device pointers safe to reuse;
2. when may scaling and preconditioners be retained; and
3. which solver iterates remain valid after numerical coefficients change.

## Decision

The public contribution-B contract is the C++20 interface in
`cpp/include/spacepdhcg/persistent_cqp.hpp`.

### 1. Immutable structure

Workspace creation copies exactly once:

- matrix dimensions;
- CSC or CSR offset/index arrays for Q, A and F;
- affine and variable cone descriptors;
- solver/device configuration.

Sparse index arrays, cone order, row dimensions and variable count cannot change through an
update. Any change requires a new workspace.

### 2. Device numerical values

`NumericValuesView` points to device-resident arrays whose ordering matches the immutable sparse
patterns. It contains Q/A/F values, linear objective, scalar bounds, affine offsets and variable
bounds.

`update_values` is asynchronous. Input pointers are borrowed until all work enqueued by that
call has completed on the supplied stream. The implementation copies or transforms them into
workspace-owned buffers before returning control of the corresponding completion event.

Python/JAX interoperation will pass these pointers through DLPack or the CUDA array interface.
No NumPy/SciPy canonicalisation is allowed in the hot loop.

### 3. Stream semantics

The caller supplies an opaque stream interpreted by the CUDA implementation as `cudaStream_t`.
The same stream must order:

```text
rollout / differentiation
        -> coefficient update
        -> optional rescaling
        -> warm-start update
        -> PDHCG solve
        -> residual calculation
        -> SCvx acceptance calculation
```

Different streams are permitted only when the caller supplies the required event dependencies.
The first implementation will reject overlapping updates and solves on one workspace rather than
silently introducing races.

### 4. Workspace state machine

```text
ready
  -> update_pending
  -> ready
  -> solving
  -> solved
  -> update_pending ...
```

Any CUDA, validation or solver failure enters `failed`. Cooperative cancellation returns either
`solved` with an interrupted report or `ready`, depending on whether a consistent iterate was
materialised. Cancellation never frees the workspace.

### 5. Warm starts

The workspace owns copies of:

- primal and dual iterates;
- current/reflected PDHG points;
- cone-projection warm starts;
- quadratic proximal/inner-solver state where mathematically compatible.

The canonical dual ordering is `[dual_A, dual_F]`. A caller may reset iterates while retaining
allocations, descriptors and scaling.

### 6. Rescaling policy

Numerical updates select one of three policies:

- **`reuse`:** retain existing Ruiz/cone scaling and preconditioner unconditionally;
- **`refresh_if_needed`:** compute device-side relative-change diagnostics and refresh when a
  threshold or reuse-count limit is exceeded;
- **`force_refresh`:** recompute numerical scaling before the next solve.

The default is `refresh_if_needed`. Initial thresholds in the interface are hypotheses, not
settled algorithmic constants. Experiments must record when scaling was refreshed and how that
affected convergence.

A first implementation may conservatively force refresh whenever Q/A/F values change, then
relax this rule after correctness is established. It may not silently reuse invalid scaling to
obtain attractive timing numbers.

### 7. Timing and residual accounting

Every solve report separates:

- update time;
- rescaling/preconditioner time;
- PDHCG iteration time;
- final residual time;
- total elapsed time.

Residuals are computed from the updated, unscaled mathematical problem. Kernel-only timings are
not accepted as end-to-end evidence.

### 8. Upstream integration boundary

The first implementation should be a thin integration layer against a pinned upstream PDHCG
commit, not an uncontrolled fork. Changes that generalise cleanly—persistent state creation,
numerical update hooks and stream-aware solves—should be structured so they can be proposed
upstream.

SpacePDHCG remains responsible for:

- the stable C++ contract;
- spacecraft/JAX bindings;
- scenario-aware partitioning;
- SCvx tolerance and acceptance logic;
- benchmark instrumentation.

## Consequences

### Positive

- “Persistent” has a precise, testable meaning.
- The API accommodates JAX/CUDA buffers without host reconstruction.
- Rescaling reuse becomes a measured algorithmic choice rather than hidden state.
- The same lifecycle can support one GPU, scenario groups and future time partitions.
- A CPU compiler can validate the interface before CUDA implementation begins.

### Costs and risks

- Upstream internal types are not a stable public ABI.
- Asynchronous pointer lifetime and stream ordering require careful tests.
- Reusing scaled data may improve speed but damage convergence if thresholds are poorly chosen.
- Explicit device ownership makes error handling and cancellation more complex than the one-shot
  Python path.

## Verification gates

1. The interface header compiles under strict C++20 warnings in ordinary CI.
2. CUDA tests prove fixed index arrays are allocated/copied once.
3. Address and memory-check tooling reports no leaks or use-after-free across repeated updates.
4. One-shot and persistent PDHCG agree on exact-optimum QP/SOCP fixtures.
5. Independent unscaled residuals agree with reported residuals.
6. Reuse, adaptive refresh and forced refresh are compared on coefficient-change sweeps.
