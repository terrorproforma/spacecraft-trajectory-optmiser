# OrbitWeaver Paper 2 outline

## Working title

**OrbitWeaver: Multi-Fidelity Bilevel Optimisation for Robust Multi-Destination Spacecraft
Missions**

Paper 2 treats SpacePDHCG as a stable continuous trajectory oracle. It does not depend on a
particular solver implementation above the oracle contract.

# Core problem

Optimise:

- target selection;
- spacecraft assignment;
- visit order;
- departure and arrival epochs;
- waiting and service times;
- propulsion and resource state;
- continuous trajectory variables;
- uncertainty and recourse.

The cost of moving from target `i` to `j` is time-, mass-, fidelity-, and scenario-dependent rather
than a static TSP edge weight.

# 1. Introduction

- Why moving-target mission design is mixed discrete–continuous.
- Why monolithic full-fidelity MINLP is usually the wrong computational architecture.
- Why the continuous oracle must support batches, lower bounds, warm starts, and requested
  fidelity.
- Contributions: multi-fidelity arc refinement, exact small-instance truth models, route-search
  integration, and robust route × scenario parallelism.

# 2. Related work

- GTOC-style asteroid tours;
- Keplerian/time-dependent TSP;
- servicing and debris-removal routing;
- column generation and labelling;
- beam/tree/evolutionary search;
- Lambert and low-thrust arc approximation;
- bilevel and surrogate-assisted mission design.

# 3. Oracle contract

Define a request by:

- initial state and epoch;
- target state/ephemeris and arrival window;
- spacecraft/resource model;
- scenario set;
- requested fidelity and accuracy;
- optional warm-start token.

Define a response by:

- feasibility;
- lower bound;
- nominal/risk objective;
- delta-v, propellant, and flight time;
- terminal and path diagnostics;
- gradients or sensitivity data when available;
- trajectory and reusable warm-start token.

# 4. Multi-fidelity arc ladder

1. Hohmann/phasing analytical screen.
2. Zero- and multi-revolution Lambert screen.
3. Edelbaum/low-thrust analytical screen.
4. Coarse fixed-grid convex arc.
5. Refined deterministic SCvx arc.
6. Robust SCvx arc.
7. Final certified high-fidelity trajectory.

Define promotion and rejection criteria between levels.

# 5. Route representations

## 5.1 Time-expanded graph

Nodes are target–epoch pairs; arcs carry time-dependent estimates and lower bounds.

## 5.2 Exact elementary labels

Use the committed small-instance solver as a truth model for route heuristics.

## 5.3 Beam search

Explain deterministic ordering, resource state, admissible lower bounds, and batched oracle calls.

## 5.4 Column generation

Master problem over complete routes; pricing by resource-constrained labels using continuous arc
bounds.

## 5.5 Multi-spacecraft master problem

Target coverage/selection, one route per spacecraft, depots, resource transfer, and service
compatibility.

# 6. Robust route optimisation

- scenario sets attached to arcs or complete routes;
- non-anticipativity until information arrival;
- expected, worst, and CVaR objectives;
- adaptive allocation of scenarios and solve accuracy to promising branches;
- route × scenario parallelism.

# 7. Search policy

Define branch priority from:

- current feasible objective;
- optimistic route lower bound;
- uncertainty/risk bound;
- arc fidelity confidence;
- remaining resource slack;
- cached warm starts and nearby solved arcs.

Computational effort should increase only as a branch becomes competitive.

# 8. Experimental protocol

Refer to:

- `benchmarks/paper2_matrix.json`;
- `docs/BENCHMARK_PROTOCOL.md`.

## 8.1 Arc calibration

Compare analytical estimates to refined/certified legs across geometry, duration, revolution
family, and propulsion regime.

## 8.2 Exact route benchmarks

Compare beam and future column-generation methods against exact elementary labels.

## 8.3 Large deterministic missions

Target count, epoch resolution, visits, beam width, cache effectiveness, and fidelity allocation.

## 8.4 Multi-spacecraft missions

Assignment, depots, service times, and resource coupling.

## 8.5 Robust missions

Candidate routes × scenarios × GPU count, final risk, and mission feasibility.

# 9. Results structure

1. Arc estimator error and rejection safety.
2. Exact small-instance optimality gaps.
3. Route quality versus runtime.
4. Candidate screening/refinement funnel.
5. Cache and warm-start effectiveness.
6. Multi-spacecraft scaling.
7. Robust route × scenario scaling.
8. Final mission examples and certified trajectories.

# 10. Mission demonstrations

Primary demonstration:

- multi-spacecraft servicing or debris removal among moving Earth-orbit targets.

Extensions:

- asteroid tours;
- cislunar logistics;
- propellant depots;
- distributed observation;
- autonomous constellation maintenance;
- on-orbit assembly.

# 11. Limitations

- finite ephemeris/epoch discretisation;
- local optimality of continuous refinement;
- fidelity-model error;
- scenario coverage;
- route lower-bound strength;
- computational proof limits at large scale.

# Planned figures

1. Bilevel route/oracle architecture.
2. Fidelity promotion funnel.
3. Time-expanded target–epoch graph.
4. Exact-label versus beam-search quality.
5. Arc calibration plots.
6. Candidate count by fidelity level.
7. Route quality/runtime Pareto front.
8. Multi-spacecraft route visualisation.
9. Robust route × scenario scaling.
10. Final certified mission timeline and trajectories.
