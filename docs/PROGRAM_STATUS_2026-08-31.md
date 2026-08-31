# SpacePDHCG and OrbitWeaver programme status

Status date: **2026-08-31**.

This document distinguishes completed engineering from GPU-dependent experiments. “Implemented”
does not mean “demonstrated faster”; performance claims require the benchmark protocol and real
recorded hardware.

# Executive state

The project is now deliberately split as:

```text
Python reference / experiments / plotting / route orchestration
                         │
                    stable C ABI
                         │
C++20 fixed CQP + dynamics + transcription + SCvx + robust routing core
                         │
          future persistent PDHCG CUDA implementation
                         │
            future NCCL/MPI multi-GPU implementation
```

The hot numerical path is no longer intended to remain Python. Python remains valuable as a
transparent reference, test oracle, and scientific interface.

# Status legend

- **COMPLETE-REFERENCE** — implemented and covered by CPU/native correctness tests.
- **PREPARED** — interface and host logic exist; a CUDA/NCCL implementation is still required.
- **GPU-BLOCKED** — cannot be honestly completed without compatible accelerator hardware.
- **OPEN-CPU** — useful work remains and does not require a GPU.
- **EXPERIMENT-BLOCKED** — implementation may exist, but the paper result needs hardware runs.

# A — common modelling and benchmark foundation

| Item | Status | Evidence |
|---|---|---|
| Canonical QP/SOCP/CQP representation | COMPLETE-REFERENCE | Python and C++ fixed-pattern structures |
| Stable topology fingerprint | COMPLETE-REFERENCE | matching Python/C++ FNV fingerprint and parity probe |
| HCW rendezvous QP/SOCP | COMPLETE-REFERENCE | repeated numerical updates and independent diagnostics |
| Known-optimum trajectory-banded fixtures | COMPLETE-REFERENCE | deterministic QP/SOCP families |
| Nonlinear 3-DoF powered-descent model | COMPLETE-REFERENCE | analytic Jacobians, Euler/RK4 propagation, path checks |
| 14-state 6-DoF powered-descent model | COMPLETE-REFERENCE | quaternion rigid-body dynamics and analytic Jacobians |
| Long-horizon two-body low-thrust model | COMPLETE-REFERENCE | analytic gravity/thrust Jacobians and mass depletion |
| 3-DoF fixed-pattern CQP transcription | COMPLETE-REFERENCE | virtual control, thrust/glide/trust SOCs |
| 6-DoF fixed-pattern CQP transcription | COMPLETE-REFERENCE | thrust/torque/rate/glide/trust SOCs and quaternion linearisation |
| Low-thrust fixed-pattern CQP transcription | COMPLETE-REFERENCE | thrust/trust SOCs and radial tangent constraints |
| Higher-order production transcription | OPEN-CPU | current fixed-grid C++ transcriptions use Euler linearisation |
| Continuous-time inter-node certification | OPEN-CPU | independent checks exist; formal dense-output enforcement remains |

# B — persistent device-resident SCvx

| Item | Status | Notes |
|---|---|---|
| Immutable C++20 CQP ownership | COMPLETE-REFERENCE | owns CSC topology and cone metadata |
| Mutable numerical update buffers | COMPLETE-REFERENCE | topology-preserving updates with validation |
| Persistent workspace state machine | COMPLETE-REFERENCE | update/solve/reset/cancel lifecycle |
| Host/device/unified pointer descriptors | COMPLETE-REFERENCE | API boundary is frozen |
| Scaling reuse/refresh controller | COMPLETE-REFERENCE | matrix/vector change thresholds and reuse budget |
| Primal-dual warm-start lifecycle | COMPLETE-REFERENCE | backend and session contracts |
| Checkpoint/restart | COMPLETE-REFERENCE | deterministic topology-locked binary state |
| Native persistent outer SCvx driver | COMPLETE-REFERENCE | create once, update thereafter, adaptive solves and trust region |
| Dense C++ ADMM debug backend | COMPLETE-REFERENCE | small QP/SOCP correctness only; not a performance solver |
| Stable C ABI | COMPLETE-REFERENCE | C++ dynamics/Jacobians/Lambert callable from Python `ctypes` |
| Pinned upstream one-shot PDHCG adapter | PREPARED | exact data/cone compatibility exists |
| Real one-shot PDHCG correctness run | GPU-BLOCKED | requires NVIDIA CUDA 12.4+ runtime |
| Concrete persistent PDHCG CUDA workspace | GPU-BLOCKED | must own upstream preprocessing, scaling, iterates and device buffers |
| In-place device coefficient updates | GPU-BLOCKED | requires CUDA implementation and measurement |
| Device-side dynamics/Jacobian fill | GPU-BLOCKED | C++ formulas exist; CUDA kernels do not |
| CUDA graph capture and stream overlap | GPU-BLOCKED | meaningful only on real hardware |
| Zero-copy Python accelerator exchange | PREPARED | C ABI exists; DLPack/CUDA array path remains |

# D — adaptive inexact and hybrid solving

| Item | Status | Notes |
|---|---|---|
| Repair/progress/refinement/polish forcing policy | COMPLETE-REFERENCE | Python and C++ policy implementations |
| Fixed-tolerance comparator | COMPLETE-REFERENCE | experiment baseline |
| Re-solve-before-shrink rule | COMPLETE-REFERENCE | implemented in outer drivers |
| Inexact error ledger | COMPLETE-REFERENCE | accumulated and relative error diagnostics |
| Hybrid first-order/IPM solve plan | COMPLETE-REFERENCE | explicit handoff and final-polish contract |
| Conditional convergence theorem skeleton | COMPLETE-REFERENCE | assumptions and proof obligations documented |
| Native outer-driver policy lifecycle | COMPLETE-REFERENCE | backend created once and updated between SCvx iterations |
| PDHCG adaptive-tolerance sweep | EXPERIMENT-BLOCKED | needs real PDHCG CUDA runs |
| QOCO-GPU/CuClarabel comparison | EXPERIMENT-BLOCKED | needs compatible builds and GPU hardware |
| Empirical crossover map | EXPERIMENT-BLOCKED | requires matched-quality results |
| Final theorem with verified numerical conditions | OPEN-CPU | mathematical refinement can continue without a GPU |

# C — robust scenario and multi-GPU optimisation

| Item | Status | Notes |
|---|---|---|
| Scenario information histories | COMPLETE-REFERENCE | deterministic scenario tree |
| Shared-prefix non-anticipativity | COMPLETE-REFERENCE | exact sparse rows |
| Block-arrow variable layout | COMPLETE-REFERENCE | scenario-local blocks and shared arrowhead |
| Deterministic scenario partition | COMPLETE-REFERENCE | whole-scenario load balancing |
| Communication-volume model | COMPLETE-REFERENCE | ring-allreduce accounting |
| Monolithic robust CQP oracle | COMPLETE-REFERENCE | probability-weighted local blocks and exact shared controls |
| 3-DoF robust assembly | COMPLETE-REFERENCE | native and Python paths |
| 6-DoF robust assembly | COMPLETE-REFERENCE | generic C++ scenario bundle exercised on 14-state CQP |
| Expected/worst/VaR/CVaR risk evaluation | COMPLETE-REFERENCE | native risk aggregation and Python robust references |
| Condensed shared-control oracle | COMPLETE-REFERENCE | CPU interior-point correctness formulation |
| Native scenario-local CUDA shards | GPU-BLOCKED | requires device ownership and kernels |
| NCCL non-anticipativity reductions | GPU-BLOCKED | requires multi-GPU node |
| Communication/computation overlap | GPU-BLOCKED | requires profiling on real topology |
| Strong and weak multi-GPU scaling | EXPERIMENT-BLOCKED | requires 2/4/8 GPU runs |
| Device-side worst/CVaR optimisation epigraphs | OPEN-CPU | native risk evaluation exists; full CQP augmentation remains |

# E — OrbitWeaver multi-destination optimisation

| Item | Status | Notes |
|---|---|---|
| Stable arc-oracle contract and fidelity ladder | COMPLETE-REFERENCE | analytical through certified levels |
| Hohmann/phasing screening | COMPLETE-REFERENCE | Python/C++ analytical screen |
| Zero-revolution Lambert solver | COMPLETE-REFERENCE | universal-variable C++ solver and parity checks |
| Edelbaum low-thrust screening | COMPLETE-REFERENCE | native circular radius/inclination estimate |
| Deterministic beam search | COMPLETE-REFERENCE | time/mass-dependent native search |
| Time-expanded moving-target graph | COMPLETE-REFERENCE | exhaustive scheduled-arc truth graph |
| Exact small-instance elementary labels | COMPLETE-REFERENCE | route truth model up to 64 targets |
| Optimistic route lower bounds | COMPLETE-REFERENCE | non-elementary dynamic-programming bound |
| Request cache and warm-start tokens | COMPLETE-REFERENCE | Python oracle layer |
| Coarse convex arc adapter | OPEN-CPU | can use committed CQP transcriptions and CPU references |
| Refined deterministic SCvx arc adapter | PREPARED | native outer driver exists; production backend still missing |
| Robust SCvx arc adapter | PREPARED | scenario oracle exists; production multi-GPU backend missing |
| Multi-revolution Lambert | OPEN-CPU | not hardware dependent |
| Dynamic discretisation discovery | OPEN-CPU | not hardware dependent |
| Column generation/pricing | OPEN-CPU | exact labels and lower bounds are available foundations |
| Multi-spacecraft assignment/master problem | OPEN-CPU | not hardware dependent |
| Final high-fidelity certification | OPEN-CPU | requires a selected dynamics/fidelity stack, not a GPU in principle |
| Massive route × scenario throughput | EXPERIMENT-BLOCKED | central Paper 2 scaling result needs GPUs |

# Paper and experiment infrastructure

| Item | Status |
|---|---|
| Paper 1 benchmark protocol | COMPLETE-REFERENCE |
| Paper 1 full experiment manifest | COMPLETE-REFERENCE |
| Paper 2 full experiment manifest | COMPLETE-REFERENCE |
| Hardware/software provenance requirements | COMPLETE-REFERENCE |
| Failure/OOM/timeout reporting rules | COMPLETE-REFERENCE |
| Paper 1 real result tables and plots | EXPERIMENT-BLOCKED |
| Paper 2 large-scale result tables and plots | EXPERIMENT-BLOCKED |

# Exactly what the absence of GPU runs blocks

No-GPU status blocks only claims or implementations that require a CUDA execution environment:

1. importing and running the pinned upstream PDHCG package;
2. validating upstream one-shot PDHCG against committed CPU references;
3. implementing and debugging the concrete persistent CUDA workspace against upstream internals;
4. measuring allocation-free numerical updates and device residency;
5. benchmarking PDHCG against GPU interior-point solvers;
6. measuring GPU memory, utilisation, energy, and crossover points;
7. implementing and validating NCCL/MPI scenario reductions;
8. measuring strong/weak scaling and communication overlap;
9. validating GPU-specific precision, determinism, stream, and graph-capture behaviour;
10. making any speedup, throughput, memory-scaling, or energy claim.

The absence of a GPU does **not** block dynamics, transcriptions, sparse topology, outer-loop logic,
robust formulations, route algorithms, correctness oracles, theory, manifests, or paper structure.

# Remaining pre-GPU priorities

In descending leverage:

1. higher-order discrete linearisation and continuous-time constraint checking;
2. full C++ worst-case/CVaR optimisation augmentation;
3. multi-revolution Lambert and broader low-thrust arc screening;
4. dynamic discretisation discovery for time-expanded routes;
5. column generation and multi-spacecraft master problem;
6. packaged native wheels and DLPack/CUDA-array interface;
7. paper outlines, notation lock, and figure-generation scripts;
8. additional adversarial and property-based tests.

# First GPU-day run order

When compatible hardware is available:

1. record hardware/software manifest;
2. build the exact pinned upstream PDHCG commit;
3. run known-optimum QP and SOCP fixtures through the one-shot adapter;
4. compare objective and residuals against Clarabel/OSQP;
5. run HCW and 3-DoF CQP fixtures;
6. implement/activate persistent workspace ownership;
7. compare one-shot versus persistent cold and warm solves;
8. connect the native outer SCvx driver;
9. run 6-DoF and low-thrust scaling;
10. move to scenario sharding and NCCL only after single-GPU correctness is closed.
