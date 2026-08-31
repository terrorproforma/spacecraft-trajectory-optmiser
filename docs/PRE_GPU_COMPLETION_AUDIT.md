# Pre-GPU completion audit

**Status date:** 31 August 2026  
**Scope:** SpacePDHCG contributions B, C and D plus the OrbitWeaver E programme.

This ledger separates three categories that had become conflated in earlier status notes:

1. work that can be completed and verified on ordinary CPU CI;
2. software whose interfaces and mathematical truth models can be completed pre-GPU but whose production implementation requires CUDA or NCCL;
3. empirical claims that cannot exist until controlled GPU experiments are run.

## Architecture decision: C++ runtime, Python reference

The project is not intended to become a Python numerical runtime.

### C++ owns the production path

- spacecraft dynamics and variational propagation;
- fixed-grid trajectory transcription and canonical CQP buffers;
- successive-convexification acceptance, forcing and trust-region logic;
- persistent solver ownership and the stable C ABI;
- scenario layouts, block-arrow operators and risk augmentations;
- OrbitWeaver arc, graph, route and master-problem kernels;
- future CUDA kernels, stream scheduling and NCCL collectives.

### Python remains deliberately thin

- high-transparency Clarabel and OSQP correctness oracles;
- regression and C++/Python parity tests;
- experiment manifests, plots, tables and paper artefacts;
- fast research prototyping before a formulation enters the native hot path.

Rewriting the reference and analysis layer in C++ would reduce scientific transparency without accelerating the device-resident solve loop. The performance requirement is therefore **no Python in the repeated production hot loop**, not **no Python in the repository**.

## Completed before GPU access

### Canonical optimisation and persistence contracts

- immutable fixed-pattern CQP representation;
- mutable numerical-value buffers;
- stable fingerprints and topology-change rejection;
- CPU persistent-session truth model;
- C++ persistent-workspace lifecycle, epochs, states and stream-order contracts;
- stable C ABI and C++/Python parity tests;
- one-shot upstream PDHCG adapter and exact upstream revision lock.

### Deterministic trajectory stack

- HCW rendezvous QP and SOCP families;
- nonlinear 3-DoF powered-descent dynamics and SCvx;
- 6-DoF powered-descent dynamics and fixed-pattern transcription;
- long-horizon low-thrust two-body dynamics and transcription;
- continuous inter-node path checking;
- automatic fixed-grid Gauss–Lobatto, Simpson, trapezoidal and midpoint path sampling;
- nonlinear path linearisation into immutable affine sample patterns;
- integral violation-state CT enforcement inside the CQP;
- selectable Euler or RK4 discrete-flow linearisation with invariant sparse topology;
- exact dense-ADMM and high-accuracy CPU references for supported problem classes.

### Inexact and hybrid control logic

- adaptive repair, exploration, refinement and polish phases;
- fixed-tolerance comparator;
- re-solve-before-trust-region-shrink policy;
- accumulated-error and empirical relative-forcing diagnostics;
- backend portfolio and warm-start handoff contracts;
- a written conditional inexact-SCvx stationarity argument and proof obligations.

### Robust scenario stack

- scenario trees and information-history non-anticipativity;
- deterministic block-arrow and condensed shared-column formulations;
- scenario partition and communication accounting;
- partition-invariant CPU forward/transpose operator truth models;
- expected-value, worst-case and CVaR aggregation;
- fixed-pattern affine-loss risk-CQP augmentations;
- robust nonlinear post-evaluation.

### OrbitWeaver

- zero- and multi-revolution universal-variable Lambert families;
- explicit short/long direction and lower/higher parameter branch metadata;
- family-aware Lambert screening with matching impulses and mass closure;
- low-thrust feasibility screening;
- concrete native coarse-convex low-thrust arc adapter;
- concrete persistent deterministic low-thrust SCvx arc adapter;
- typed, request-compatible warm-reference transfer between fidelity stages;
- deterministic beam search;
- time-expanded moving-target graph and elementary-route solver;
- route-driven dynamic discretisation discovery;
- exact small-instance multi-spacecraft route-column master;
- route-column dominance and reduced-cost pricing primitives;
- dependency-free two-phase restricted-master simplex with dual-price recovery;
- full iterative column generation with incumbent, lower bound and iteration records;
- native multi-fidelity trajectory-oracle contract, exact cache and warm-start pipeline.

### Native delivery

- root CMake builds the full host-native smoke suite and C ABI;
- warnings-as-errors and ASan/UBSan gates;
- native-core and native-parity merge-surface checks;
- installable CMake package exports;
- external package-consumer test.

## Pre-GPU work still available

The following items remain legitimate CPU-side work and should be pursued before expensive hardware campaigns:

- complete adversarial, randomized and property-based tests for the risk, route-master, Lambert-family, CT-sampling and oracle layers;
- finish higher-order variational integration alternatives to finite-difference RK4 where analytic or automatic derivatives provide a clear benefit;
- connect robust scenario SCvx to the OrbitWeaver robust-fidelity stage on the host truth path;
- define and implement a final high-fidelity certification adapter with a selected force model and independent acceptance policy;
- add native wheel packaging and a documented accelerator-pointer exchange design;
- construct reproducible scale-sweep manifests and synthetic memory/work estimates;
- expand the formal inexact-SCvx argument into theorem, assumptions, lemmas and counterexample tests;
- lock Paper 1 and Paper 2 notation, table schemas and figure-generation interfaces;
- prepare paper tables and plotting schemas without inventing hardware results.

## Partially hardware-blocked

### B — persistent device-resident SCvx

Pre-GPU contracts, topology, ownership, update semantics, warm starts and CPU truth models can be complete. The following cannot be verified without CUDA:

- upstream PDHCG internal-state ownership across solves;
- device allocation exactly once;
- in-place coefficient updates from device pointers;
- stream-safe asynchronous solve and residual retrieval;
- CUDA graph capture and overlap;
- proof that host transfer and recanonicalisation overhead is below the H1 threshold.

### C — scenario-aware multi-GPU optimisation

Scenario semantics and algebra can be complete on CPU. Hardware is required for:

- real device sharding;
- NCCL reductions and non-anticipativity coordination;
- communication/computation overlap;
- multi-GPU warm-start ownership;
- strong and weak scaling;
- comparison against generic PDHCG partitioning.

### D — adaptive inexact and hybrid solving

Policy, theory and replay harnesses can be complete pre-GPU. Hardware is required for:

- PDHCG matrix-vector and cone-projection counts;
- QOCO-GPU and CuClarabel crossover measurements;
- memory-failure boundary of factorisation-based solvers;
- end-to-end validation of H5 and H6;
- energy measurements.

### E — OrbitWeaver

The route architecture can advance substantially on CPU. Hardware is required for the intended scale of:

- batched arc refinement;
- robust scenario expansion for every candidate arc;
- simultaneous route-candidate and scenario parallelism;
- high-throughput production cost-oracle benchmarking.

## Strictly blocked empirical claims

No responsible result can currently be reported for:

- GPU speedup, latency or throughput;
- persistence overhead percentage;
- peak GPU memory;
- multi-GPU efficiency;
- NCCL byte/time measurements;
- energy per accepted trajectory;
- H1 through H6 performance conclusions;
- the node/scenario crossover against GPU interior-point solvers.

These are experimental outputs, not missing documentation or implementation details.

## Gate to the first hardware campaign

The first CUDA campaign should begin only after the host-native branch is green and frozen. Its minimum sequence is:

1. reproduce random banded QP/SOCP and HCW reference solutions with one-shot PDHCG;
2. verify the persistent native update lifecycle on one GPU;
3. run deterministic 3-DoF SCvx with device-resident coefficient updates;
4. execute adaptive-versus-fixed inner-accuracy sweeps;
5. add scenario bundles on one GPU;
6. expand to NCCL multi-GPU runs;
7. compare against GPU interior-point baselines at matched nonlinear quality.

Until then, the repository may claim completed architecture and CPU-verified mathematics, but not accelerator performance.
