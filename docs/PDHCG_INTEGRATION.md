# PDHCG integration contract

## Upstream lock

The adapter was designed against upstream `Lhongpei/PDHCG` commit:

```text
167c8b72b4b96d2f94d405b8763e485514192b81
```

That revision includes the conic Python API and initial PSD-cone support. Any integration run
must record the actual upstream commit or installed package version because the API is new and
may change.

## Two distinct backends

SpacePDHCG deliberately distinguishes:

1. **`PDHCGOneShot`** — maps the canonical problem exactly into public `pdhcg.Model`, applies
   parameters and primal-dual starts, then calls `optimize()`. It exists for correctness,
   upstream compatibility and GPU integration tests.
2. **`PersistentCQP`** — the contribution-B C++/CUDA extension that will retain preprocessing,
   scaling, sparse descriptors, device buffers and iterates between SCvx subproblems.

Calling the one-shot adapter “persistent” would be incorrect. Upstream's public Python
`Model.optimize()` invokes `_core.solve_once`; the public C lifecycle is create/solve/free.
There is currently no public numerical-update call for a live device solver state.

## Canonical mapping

| SpacePDHCG | Upstream `pdhcg.Model` |
|---|---|
| `Q` structure and values | `objective_matrix` |
| linear objective `c` | `objective_vector` |
| scalar matrix `A` | `constraint_matrix` |
| scalar lower/upper bounds | `constraint_lower_bound`, `constraint_upper_bound` |
| affine cone matrix `F` | `affine_cone_matrix` |
| affine cone offset `g` | `affine_cone_offset` |
| affine cone blocks | `affine_cones=ConeSpec(...)` |
| variable lower/upper bounds | `variable_lower_bound`, `variable_upper_bound` |
| variable cone blocks | `variable_cones=ConeSpec(...)` |
| canonical primal start | `setWarmStart(primal=...)` |
| dual ordered `[dual_A, dual_F]` | `setWarmStart(dual=...)` |

Cone names are mapped as:

```text
SECOND_ORDER          -> soc
ROTATED_SECOND_ORDER  -> rsoc
EXPONENTIAL           -> exp
POWER                 -> power
POSITIVE_SEMIDEFINITE -> psd
```

The `v_dim` and start-index conventions are kept native. No SOC coordinate permutation is
performed for PDHCG; the permutation exists only inside the Clarabel reference adapter.

## Current verification

CPU CI injects a strict fake upstream module and checks that:

- Q, A and F dimensions and values reach `Model`;
- every SOC block receives the correct start and `v_dim`;
- variable and scalar bounds are preserved;
- tolerances and iteration limits map to upstream parameter names;
- primal and dual warm starts use the correct native dimensions;
- returned objective, residual, iteration and runtime fields map into `CQPSolution`.

This verifies API semantics without claiming that CUDA kernels have executed. A real integration
run requires a CUDA-capable host, a compatible upstream build and the actual `pdhcg` package.

## Persistent extension boundary

The production extension will have to retain or reconstruct stable ownership around upstream
internal objects such as:

- `processed_qp_problem_t` and rescaling metadata;
- `pdhg_solver_state_t`;
- cuSPARSE SpMV contexts for A, A-transpose and Q;
- cone projection workspaces and warm starts;
- current, reflected and PDHG primal-dual iterates;
- residual and preconditioner buffers.

The implementation must distinguish updates that permit reuse of scaling/preconditioners from
updates that require numerical rescaling. That policy is part of the experiment, not an
assumption.

## GPU integration gate

A CUDA integration run is accepted only when it records:

- SpacePDHCG commit;
- upstream PDHCG commit/package version;
- CUDA toolkit, driver and GPU model;
- solver parameters;
- objective and primal/dual residuals;
- independent trajectory feasibility;
- one-shot model-build, preprocessing and solve time separately.

The one-shot results establish correctness and the baseline cost that contribution B must remove.
