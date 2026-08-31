# C++ core architecture

## Decision

SpacePDHCG is a **C++20/CUDA numerical engine with a thin Python research interface**.
Python remains first-class, but it is not permitted in the production hot loop.

The rule is:

> No Python object creation, SciPy sparse reconstruction, Python callback, or host/device
> round trip may occur per knot point, per scenario, per PDHCG iteration, or per accepted
> SCvx iteration in the production path.

Python is retained for experiment composition, reference solvers, independent verification,
plotting, paper tables, mission configuration, and rapid formulation work. C++/CUDA owns
repeated numerical execution.

## Why not rewrite everything

Python calling compiled NumPy, SciPy, Clarabel, OSQP, or CUDA kernels is not intrinsically
slow. The current cost problem is more specific:

1. sparse structures and coefficient arrays are assembled through Python/NumPy/SciPy;
2. the reference SCvx loop creates a backend workspace for each outer iteration;
3. scenario matrices are materialised on the host for correctness checks;
4. no persistent device ownership exists yet;
5. the Python upstream PDHCG interface is single-GPU and one-shot.

Rewriting plotting, experiment control, configuration parsing, and correctness tests in C++
would add risk without improving the solver. Rewriting the coefficient fill, propagation,
linearisation, persistent workspace, scenario operators, and route-arc evaluation does improve
latency, memory traffic, and multi-GPU scalability.

## Language boundary

### C++20 host core

The host core owns:

- fixed CSC topology and cone metadata;
- validated numerical CQP buffers;
- topology fingerprints and update epochs;
- spacecraft dynamics, Jacobians, integration, and nonlinear checks;
- SCvx forcing, trust-region, inexact-error, and hybrid-solver policies;
- scenario trees, block-arrow indexing, non-anticipativity maps, and partition plans;
- Lambert, Hohmann, rocket-equation, and other analytical OrbitWeaver screening methods;
- benchmark manifests and deterministic result records;
- the native backend/workspace interface.

The first C++ tranche is in:

- `cpp/include/spacepdhcg/core/fixed_cqp.hpp`
- `cpp/include/spacepdhcg/dynamics/powered_descent_3dof.hpp`
- `cpp/include/spacepdhcg/scvx/policies.hpp`
- `cpp/include/spacepdhcg/distributed/scenario_layout.hpp`
- `cpp/include/spacepdhcg/orbitweaver/lambert.hpp`

These components are dependency-light and compile in ordinary CPU CI. They are the actual
production types that CUDA code will consume, not disposable mocks.

### CUDA and distributed core

CUDA/NCCL owns:

- persistent device buffers for `Q`, `A`, `F`, bounds, offsets, iterates, and residuals;
- sparse matrix-vector products and cone projections;
- PDHCG preprocessing and scaling state;
- in-place numerical coefficient updates;
- device-side propagation, linearisation, and CQP coefficient fill;
- stream/event ordering, CUDA graph capture where useful, and asynchronous copies;
- scenario-local kernels and shared-control reductions;
- multi-GPU partition ownership and NCCL collectives;
- device-resident solution and diagnostic buffers.

### Python interface

Python owns:

- mission/problem declarations;
- experiment grids and reproducibility manifests;
- high-accuracy Clarabel/OSQP reference solves;
- independent nonlinear propagation and certification checks;
- result analysis, plotting, tables, and paper generation;
- a thin native binding that passes configuration and borrows arrays without rebuilding the
  problem.

The binding should use pybind11 or nanobind for control objects and DLPack or the CUDA array
interface for zero-copy array exchange. It must expose explicit ownership and stream semantics.

## Runtime architecture

```text
Python mission/experiment configuration
                 |
                 v
C++ immutable problem topology + model configuration
                 |
                 v
C++/CUDA persistent workspace creation (once)
                 |
                 v
GPU propagation and linearisation
                 |
                 v
in-place CQP coefficient fill
                 |
                 v
PDHCG solve with adaptive tolerance and warm start
                 |
                 v
GPU nonlinear acceptance checks and trust update
                 |
                 +---- repeat without rebuilding topology ----+
                 |
                 v
optional interior-point polish / certification
                 |
                 v
compact result record returned to Python
```

## Migration gates

### Gate CXX-0 — host-core compilation

Complete when all core headers compile with C++20, `-Wall -Wextra -Wpedantic -Werror`, and
address/undefined-behaviour sanitizers. This is exercised by `.github/workflows/cpp-core.yml`.

### Gate CXX-1 — cross-language numerical parity

For identical fixtures, Python and C++ must agree on:

- sparse dimensions and nonzero counts;
- topology fingerprint;
- dynamics and analytic Jacobians;
- linearised dynamics coefficients;
- scenario/control ownership maps;
- cone row ordering;
- objective and constraint activity;
- Lambert endpoint velocities.

The parity tolerance is problem-specific and must be declared in the fixture manifest.

### Gate CXX-2 — native fixed-pattern transcription

C++ fills all changing powered-descent and rendezvous CQP values into preallocated arrays.
Python may construct a reference copy, but production solving cannot depend on SciPy assembly.

### Gate CXX-3 — persistent native backend

A single C++ object owns upstream PDHCG preprocessing, scaling, device descriptors, iterates,
and solution buffers across outer SCvx iterations. Updating values may not allocate or alter
symbolic topology.

### Gate CXX-4 — device-resident SCvx

Propagation, differentiation, coefficient fill, solve, residual checks, acceptance, and trust
updates remain on-device. Only compact iteration diagnostics leave the GPU.

### Gate CXX-5 — scenario-distributed execution

Whole-scenario partitions run on multiple GPUs. Shared-control and risk aggregates are reduced
through NCCL. Results must match the monolithic CPU oracle before performance claims are made.

## Performance policy

A C++ implementation is not automatically faster. Every benchmark must separate:

- topology construction;
- numerical coefficient update;
- scaling/preconditioning refresh;
- host-to-device transfer;
- solver iteration time;
- residual and nonlinear-check time;
- collective communication;
- end-to-end accepted-trajectory time.

The project will not claim speedup from a language change alone. The intended gains come from
persistent ownership, fewer allocations, less data movement, regular memory layout, kernel
fusion, and parallel execution.

## Repository policy

Python reference code is not deleted when a C++ equivalent appears. It remains an independent
oracle until the corresponding paper result is frozen. Production benchmarks must identify
which implementation path was used and may not label the Python reference path as the native
SpacePDHCG engine.
