# Pinned PDHCG persistence code map

This map is the pre-implementation audit required by
`docs/GPU_EXECUTION_ROADMAP.md` section 8.2. It describes the exact checkout
locked by `third_party/pdhcg.lock.json`:

- repository: `https://github.com/Lhongpei/PDHCG`
- commit: `167c8b72b4b96d2f94d405b8763e485514192b81`
- tree: `62b05e6c1bedd385f6c267af3645ae4aae0421b4`
- licence: Apache-2.0; the upstream copyright and licence notices remain in
  every compiled upstream source and deterministic patch.

The public API at this revision is one-shot. `solve_qp_problem()` calls
`optimize()`, which allocates, preprocesses, solves, extracts, and destroys a
complete CUDA solver state on every call. Reusing only `qp_problem_t` would not
be a persistent GPU implementation.

## Chosen G2 integration seam

G2 uses the allowed **linked internal adapter** strategy:

1. CMake verifies the locked commit and tree before importing any source.
2. The adapter compiles the selected pinned single-GPU upstream sources
   directly from `_upstream/pdhcg`; the ignored checkout is never edited and no
   upstream snapshot is copied into the repository.
3. The upstream target remains the one-shot GPU reference. SpacePDHCG owns the
   explicit persistent lifecycle, stream binding, topology/value buffers,
   diagnostics, and allocation ledger under `cpp/cuda`.
4. Every imported source path, the exact commit/tree, and patch digest are
   recorded by CMake and in the G2 evidence manifest.
5. The deterministic patch
   `third_party/patches/pdhcg/0001-free-quadratic-state.patch` is applied only
   to an ignored build-tree copy after lock verification. It frees
   quadratic-state allocations omitted by the pinned destructor and
   initializes spectral, CSC-conversion, and SpMV scratch allocations so
   application-owned memory is fully initialized under initcheck. It does not
   change numerical kernels.

The patch SHA-256 is
`dd31b99869bda77400d8cf33c2048ad424d345a17b17898420f5dab96766ecc6`.

The adapter boundary is deliberate: pinned upstream launches many kernels on
the implicit default stream and destructively rescales/extracts state. Calling
that one-shot orchestration from a repeated solve would violate G2. The
persistent target therefore retains its own CUDA execution state while linking
the pinned implementation for matched-quality one-shot comparisons.

## Public entry and one-shot orchestration

| Stage | Pinned location | Work and lifetime |
|---|---|---|
| Public solve | `include/pdhcg.h`, `src/pdhcg.c:894` | `solve_qp_problem()` validates fixed cone sections, copies/defaults parameters, and calls `optimize()`. |
| Host problem construction | `src/pdhcg.c` | `create_qp_problem()` converts dense/CSC/COO descriptors to owned host CSR and clones vectors/cone metadata. `qp_problem_free()` releases those host allocations. |
| Top-level orchestration | `src/solver.cu:35` | `optimize()` performs optional QCQP conversion and presolve, scaling/preprocessing, CUDA-state creation, iteration, result extraction/postsolve, and unconditional CUDA-state destruction. |
| QCQP conversion | `src/qcqp_transform.c` | May create an extended SOCP problem and topology; this is creation-only and cannot occur in a values-only update. |
| Presolve | `src/presolve_wrapper.c` | Allocates PreFOS adapters/reduced host topology and may solve early. Persistent G2 creation freezes the resulting topology; presolve is never rerun by a numerical update. |
| Dummy row | `src/utils.cu` | A zero-row problem may be cloned with a dummy constraint. This is also creation-only topology mutation. |

## Scaling and preprocessing

`optimize()` calls `rescale_problem()` in `src/preconditioner.c:653`.

- `deepcopy_problem()` allocates a full host copy of vectors, CSR matrices,
  starts, cone metadata, and QCQP metadata.
- Curtis-Reid, Ruiz, Pock-Chambolle, and bound/objective scaling allocate host
  scratch and mutate the copied numerical values.
- `preprocess_qp_problem()` classifies Q as none, diagonal, sparse, low-rank,
  or combined and may allocate a diagonal Q or dense/diagonal low-rank middle
  representation.
- `rescale_info_t` owns the scaled host problem, processed representation, row
  and column scaling vectors, scalar scaling factors, and timing.
- `rescale_info_free()` destroys the processed/scaled host representations
  immediately after `initialize_solver_state()` in the one-shot path.

Persistent ownership must therefore retain scaling vectors, modes, coefficient
change metrics, reuse count, refresh reason, and the processed Q
representation. A refresh may recompute numerical scaling, but must preserve
the fixed sparse pattern and consistently rescale retained iterates.

## CUDA state creation

`initialize_solver_state()` in `src/solver_state.cu:688` is the principal
persistence boundary.

### Host allocations

- `pdhg_solver_state_t`, matrix wrappers, quadratic-objective and inner-solver
  records;
- cone bucket/layout arrays and temporary permutations;
- temporary warm-start/scaling/finite-bound/ones vectors;
- `pdhcg_spmv_ctx_t` records;
- optional PSD and distributed cone runtime records.

### Device allocations and initial transfers

- A CSR and explicit transposed CSR: row offsets, column indices, and values;
- Q sparse/diagonal/low-rank matrices and low-rank middle state;
- objective, scalar/variable bounds, affine offsets, finite-bound copies;
- row and variable scaling vectors;
- initial/current/PDHG/reflected primal and dual iterates;
- primal/dual products, slacks, residuals, deltas, and ones vectors;
- inner projected-BB/proximal buffers and diagonal preconditioner state;
- cone starts, dimensions, power parameters, fixed mask, projection/residual
  warm starts, complementarity and power-violation scratch;
- PSD eigensolver matrices, eigenvalues, info, packed-index data, and solver
  workspace;
- distributed split-cone metadata and residual scratch when enabled.

The `ALLOC_AND_COPY*` macros perform `cudaMalloc` plus H2D copies.
`cusparseCsr2cscEx2` constructs the transpose using a temporary device buffer.
Cone metadata and warm starts are uploaded during
`initialize_cone_runtime()`. Those index/metadata allocations and copies are
strictly creation-only in G2.

### Descriptors and handles

- `cusparseCreate()` and `cublasCreate()` create handles.
- `cusparseCreateDnVec()` creates dense-vector descriptors for iterates,
  products, and Q/low-rank work.
- `pdhcg_spmv_ctx_create()` in `src/spmv_backend.cu` creates a cuSPARSE CSR
  descriptor, persistent SpMV preprocessing/plan, and device scratch for A,
  A-transpose, Q, R, and R-transpose.
- PSD cone projection creates cuSOLVER-backed bucket workspaces.

The one-shot path binds none of these resources to a caller stream. G2 must
bind every handle and launch to the workspace consumer stream and keep the
descriptors/scratch alive until destroy.

### Creation-time spectral work

`initialize_quadratic_term_information()` and
`initialize_step_size_and_primal_weight()` call the eigenvalue/singular-value
estimators in `src/pdhg_core_op.cu`. At this revision those estimators allocate
temporary device vectors and descriptors, perform host-returning BLAS
reductions, then free everything. G2 treats this as scaling/preconditioner
creation or explicit refresh work; it is forbidden in an ordinary retained
solve.

## Iteration call graph

`src/solver.cu:120` owns the outer loop:

```text
compute_residual / compute_infeasibility_information
  -> A*x and A^T*y through retained SpMV contexts
  -> Q*x through diagonal kernels or retained Q/R/R^T contexts
  -> scalar, box, variable-cone, and affine-cone residual kernels
  -> cuBLAS reductions and termination quantities
check_termination_criteria
should_do_adaptive_restart -> perform_restart
pdhg_update
  -> A^T*y
  -> LP, diagonal-Q, or projected-BB/proximal primal update
  -> variable-cone projection through cone_dispatch.cu
  -> A*x
  -> scalar-row or affine-cone dual update/projection
compute_fixed_point_error
halpern_update
```

`src/cone_dispatch.cu` dispatches SOC, rotated-SOC, exponential, power, and
PSD projection/residual kernels using the retained cone runtime. No allocation
is expected in this dispatch.

`src/pdhg_core_op.cu` has no per-outer-iteration device allocation in its main
update path, but it does contain synchronous host-result BLAS operations and a
device-to-host scalar copy in the projected-BB inner convergence check. The
one-shot outer loop also reads residuals and stopping decisions on the host.
These synchronization points must be explicit in diagnostics/timing; they
cannot be represented as a fully host-free asynchronous hot loop.

## Residuals and result extraction

`compute_residual()` in `src/pdhg_core_op.cu:1038` computes upstream:

- scalar-row and affine-cone primal residuals;
- variable-cone feasibility and projected-gradient/natural residuals;
- stationarity, cone dual membership, and complementarity;
- primal/dual objectives and absolute/relative gap.

G2 additionally computes canonical independent residuals in SpacePDHCG-owned
kernels/scratch and compares them with CPU truth and the upstream diagnostics.
The independent result is not inferred from upstream termination status.

`create_result_from_state()`:

1. recomputes `A^T y`, Q products, and reduced cost;
2. rescales the retained iterate **in place**;
3. host-allocates result vectors;
4. copies primal, dual, and reduced cost D2H;
5. copies compact scalar diagnostics into `pdhcg_result_t`.

The persistent path must not use this destructive extraction. It retains
scaled internal iterates, copies only requested compact diagnostics
asynchronously to pinned host storage, and exports primal/dual through
caller-provided device views or checkpoint buffers.

## Cleanup map

`pdhg_solver_state_free()` in `src/solver_state.cu:1060` destroys:

- all numerical/topology/iterate/residual/cone/inner-solver CUDA allocations;
- A/A-transpose/Q/R/R-transpose SpMV contexts and their descriptors/scratch;
- dense-vector descriptors;
- cuBLAS and cuSPARSE handles;
- PSD and distributed split-cone runtimes;
- all host wrappers.

`rescale_info_free()`, `pdhcg_presolve_info_free()`, and `qp_problem_free()`
release preprocessing, presolve, transformed, and dummy-row host objects.
`pdhcg_result_free()` releases extracted host results.

Persistent destroy must order completion/cancellation, event synchronization,
external-borrow release, descriptor/handle destruction, device frees, and
host frees. A failed create uses the same partial-cleanup path and ledger.

## Permutation and distributed inspection

- `src/permute.cu` deep-copies and structurally permutes CSR rows/columns and
  vectors on the host, validates that cone blocks remain contiguous, and
  repermutes extracted solutions. It is creation-only.
- `src/partition_utils.c` computes legal cone-aware partition cuts using host
  dynamic programming scratch. It is creation-only.
- `distributed/distributed_solver.cu` repeats the one-shot
  scale/initialize/iterate/extract/free structure around a 2-D MPI grid.
- `distributed/distributed_utils.cu` creates grid/partition metadata, extracts
  local CSR blocks, serializes/broadcasts problems and scaling state, and
  includes D2H gather copies.
- `distributed/distributed_conic.cu` allocates/uploads split-cone indices,
  types, masks, statistics, and residual buffers and frees them with solver
  state.

G2 is intentionally single-GPU. The inspected distributed paths establish
which partition/cone metadata a later multi-GPU workspace must retain; they
are not linked into the G2 target.

## Persistence invariants

After successful create:

- dimensions, Q/A/F index pointers, cone metadata, permutations, descriptors,
  and their ledger entries are immutable;
- topology allocation count and index-copy count remain unchanged;
- values-only update changes only Q/A/F values, objective, bounds, affine
  offsets, finite-bound representations, scaling metadata, and epochs;
- solve reuses all topology, descriptors, scratch, handles, scaling state, and
  eligible iterate state;
- no call reaches `create_qp_problem()`, `optimize()`,
  `initialize_solver_state()`, `create_result_from_state()`, or
  `pdhg_solver_state_free()` in the repeated solve hot path;
- no CPU numerical solver fallback exists.

Any unavoidable default-stream launch, device-wide synchronization,
per-solve topology/index operation, or one-shot reconstruction is a G2
failure and must be reported rather than hidden.
