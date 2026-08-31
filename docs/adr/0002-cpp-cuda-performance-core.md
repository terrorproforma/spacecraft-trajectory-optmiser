# ADR 0002 — C++/CUDA numerical core with a thin Python control plane

- **Status:** accepted
- **Date:** 31 August 2026
- **Decision owners:** SpacePDHCG programme

## Context

The reference implementation began in Python because NumPy, SciPy, Clarabel, and OSQP make
mathematical structure easy to inspect and test. That was appropriate for establishing the CQP
contract, spacecraft models, and solver-independent interfaces. It is not the intended production
architecture.

SpacePDHCG's claimed contribution depends on eliminating repeated symbolic construction,
host-device transfers, allocations, and Python control flow from the SCvx/PDHCG hot loop. Merely
calling CUDA kernels from a Python outer loop would leave several target bottlenecks intact.

At the same time, rewriting every experiment, plot, and research utility in C++ would slow
scientific iteration without improving solver latency. NumPy and SciPy already execute most dense
and sparse arithmetic in compiled native libraries; Python is costly chiefly when it controls
fine-grained repeated operations or rebuilds data crossing the CPU/GPU boundary.

## Decision

SpacePDHCG uses a two-layer architecture.

### C++20/CUDA numerical core

The compiled core owns:

- fixed CQP topology and numerical buffers;
- sparse matrix-vector and transpose products;
- cone projections and residual evaluation;
- spacecraft rollout and variational dynamics;
- fixed-grid continuous-time path-constraint integration;
- coefficient canonicalisation into preallocated arrays;
- persistent primal/dual/scaling state;
- adaptive forcing, trust-region acceptance, and hybrid handoff;
- scenario partitioning and NCCL collectives;
- high-volume OrbitWeaver arc-cost kernels.

CUDA extends these C++ types. It does not introduce a second incompatible model or CQP contract.
CPU kernels remain the executable truth model for their GPU counterparts.

### Thin Python control plane

Python owns:

- transparent reference formulations;
- unit and cross-backend correctness tests;
- experiment configuration and launch;
- machine-readable result aggregation;
- plots, tables, and manuscript generation;
- rapid prototyping of new mission models;
- user-facing notebooks and examples.

Python may submit a complete episode or batch to the compiled core, but it may not orchestrate
individual PDHCG iterations, trajectory nodes, cone blocks, or GPU collectives in the production
path.

## Binding boundary

The preferred binding order is:

1. a stable C++ API;
2. a narrow C ABI where cross-toolchain compatibility is useful;
3. pybind11 bindings for ergonomic Python use;
4. DLPack or the CUDA array interface for zero-copy device arrays.

The binding layer must expose opaque workspace handles and borrowed device buffers. It must not
convert persistent CUDA arrays into NumPy/SciPy objects on every SCvx iteration.

## Migration map

| Current reference responsibility | Production owner |
|---|---|
| Python `CQPStructure`/`CQPValues` | C++ `core/cqp.hpp` |
| SciPy sparse products | C++ CPU oracle, then cuSPARSE kernels |
| Python HCW dynamics/transcription | C++ `core/hcw.hpp` and `core/cw_cqp.hpp` |
| Python 3-DoF model | C++ `core/powered_descent.hpp` |
| Future robust 6-DoF model | C++ `core/powered_descent_6dof.hpp` |
| Python forcing/trust policies | C++ `core/scvx_policy.hpp` |
| Python scenario layout | C++ `core/scenario.hpp` |
| Python repeated first-order solves | C++ persistent host PDHG, then CUDA PDHCG |
| Host-side path checks | C++/CUDA continuous-time certificates |
| Python route-arc loops | Batched C++ OrbitWeaver kernels |

## Performance rule

A rewrite is justified when at least one of the following applies:

- the operation is executed per node, scenario, inner iteration, or route arc;
- the operation owns or updates persistent GPU memory;
- crossing the Python/native boundary would require data copies or synchronisation;
- deterministic latency or thread-level parallelism matters;
- the code is shared by CPU and CUDA implementations.

A rewrite is not justified merely because C++ can execute scalar code faster. Large reference
operations already dispatched to BLAS, sparse libraries, or solver binaries may remain in Python
until profiling identifies them as material end-to-end costs.

## Consequences

### Positive

- The production path can be fully device-resident.
- CPU and GPU implementations share data structures and semantics.
- Python remains useful for scientific transparency and reproducibility.
- OrbitWeaver can issue large batched calls rather than millions of Python callbacks.
- Future onboard or embedded builds need not carry a Python runtime.

### Costs

- C++/Python parity tests are mandatory.
- Binding and packaging complexity increases.
- Templates and device code must be kept disciplined to avoid long build times.
- Automatic differentiation requires either explicit variational equations, a C++ AD library, or
  generated derivatives rather than relying only on JAX.

## Rejected alternatives

### Python/JAX everywhere

Rejected as the production baseline because the target solver is an upstream C++/CUDA codebase,
its distributed implementation uses MPI/NCCL, and persistent sparse/cone ownership must survive
across outer iterations without host recanonicalisation.

### C++ everywhere

Rejected because experiment management, plotting, reference modelling, and manuscript generation
do not materially benefit and would become harder to inspect and modify.

### Separate Python and C++ mathematical models

Rejected because silent formulation drift would invalidate crossover comparisons. The C++ core
must be cross-checked against transparent Python references until the compiled implementation is
the frozen publication model.
