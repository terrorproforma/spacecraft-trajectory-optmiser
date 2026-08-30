# Milestones and gates

## M0 — Repository and numerical contract

**Deliverables**

- Packaged Python project and CI.
- Immutable CSC structure plus mutable CQP values.
- Backend lifecycle protocol.
- Exact-discrete CW dynamics.
- Fixed-pattern rendezvous QP.
- Persistent OSQP baseline and repeated-update benchmark.

**Gate**

The same solver workspace must solve at least two initial/target pairs through numerical updates and warm starts, while independent checks confirm initial state, terminal state, dynamics and control limits.

## M1 — B0/B1 conic bridge

**Deliverables**

- SOC thrust-magnitude constraints.
- Conic residual and projection checks.
- PDHCG-CQP adapter in reference mode.
- QOCO-GPU and CuClarabel adapters where available.
- Random trajectory-banded QP/SOCP generator.

**Gate**

All backends agree within declared tolerances on objective and feasibility across the correctness suite.

## M2 — B: persistent device-resident CT-SCvx

**Deliverables**

- Nonlinear 3-DoF powered-descent model.
- Fixed-grid multiple-shooting transcription.
- Virtual control, trust regions and exact-penalty merit function.
- Persistent GPU coefficient updates.
- Device-side rollout and acceptance logic.

**Gate**

After initialisation, full host-device/canonicalisation overhead is measured and tested against H1. Accepted trajectories pass independent nonlinear checks.

## M3 — D: adaptive inexact and hybrid solving

**Deliverables**

- Outer residual definition.
- Forcing-rule tolerance controller.
- Re-solve-before-shrink logic.
- Interior-point polish policy.
- Fixed-accuracy, adaptive and hybrid benchmark modes.

**Gate**

H5 and H6 are either supported or rejected with a reproducible crossover explanation.

## M4 — C: scenario-aware multi-GPU robustness

**Deliverables**

- Scenario tree and common open-loop prefix.
- Block-arrow canonicalisation.
- Generic and scenario-aware partitions.
- NCCL communication instrumentation.
- Robust 6-DoF powered-descent and rendezvous cases.

**Gate**

Weak/strong scaling, memory crossover and communication results are reproducible. H2–H4 are resolved rather than assumed.

## M5 — Paper 1 release

**Deliverables**

- Frozen benchmark configurations.
- Machine-readable result manifests.
- Reproduction commands and hardware metadata.
- Draft manuscript and figures generated from committed results.
- Versioned software release.

## M6 — E prototype: continuous oracle for moving-target routing

**Deliverables**

- Stable batched `evaluate_arcs` API.
- Multi-fidelity arc evaluation.
- Analytical screening and convex lower bounds.
- Beam-search or column-generation prototype.
- Fixed-sequence multi-rendezvous refinement.

**Gate**

The route layer can replace an analytical arc estimate with a refined trajectory result without depending on backend-specific data structures.

## M7 — E robust multi-destination engine

**Deliverables**

- Multiple spacecraft and moving targets.
- Uncertainty-aware route scoring.
- Candidate-route × scenario parallelism.
- Mission-level resource, timing and propellant constraints.
- High-fidelity final-route certification.

This is the second-paper programme. It starts after the trajectory oracle is credible, but its API requirements constrain the design from M0 onward.
