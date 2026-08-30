# Architecture

## 1. Non-negotiable design invariants

1. **Fixed symbolic structure in the hot loop.** A solve episode may update matrix values, vectors, tolerances and warm starts, but not sparse index arrays or cone layout.
2. **Reference and performance paths solve the same canonical problem.** The transparent CPU path is the numerical oracle for the CUDA path.
3. **Persistence is measured end to end.** Setup, canonicalisation, coefficient updates, transfers, solver time and nonlinear verification are reported separately.
4. **No hidden feasibility claims.** Every accepted trajectory is independently propagated and checked against dynamics, terminal conditions and path constraints.
5. **Multi-GPU partitioning follows scenario structure before generic nonzero balancing.** Shared controls are explicit, not concealed in a monolithic sparse matrix.
6. **E consumes a stable oracle.** Multi-destination routing is not allowed to reach through the trajectory API into solver internals.

## 2. Layered system

```text
Mission / route optimiser (E)
            │ batched arc requests
            ▼
Trajectory oracle API
            │ models + scenario sets + requested fidelity
            ▼
SCvx outer loop
            │ fixed-pattern CQP values
            ▼
Persistent solver interface
       ┌────┴────────────────────────┐
       ▼                             ▼
reference CPU backend        PDHCG/QOCO/CuClarabel GPU backends
       │                             │
       └──────── correctness ────────┘
```

## 3. Canonical CQP contract

The core representation separates immutable symbolic structure from mutable numerical values.

```python
structure = CQPStructure(
    quadratic=CSCStructure(...),
    constraint=CSCStructure(...),
    cones=(...),
)

values = CQPValues(
    quadratic_values=...,
    constraint_values=...,
    linear_objective=...,
    lower_bounds=...,
    upper_bounds=...,
)
```

A backend may retain device copies of both the structure and values. Calling `update()` may not change dimensions, CSC index arrays or cone blocks. This invariant is enforced by tests before GPU code exists.

## 4. Initial CW variable layout

For `N` control intervals, state dimension `n_x = 6`, and control dimension `n_u = 3`,

```text
z = [x_0, x_1, ..., x_N, u_0, u_1, ..., u_(N-1)].
```

The QP constraint rows are:

```text
initial state equality       6
linear dynamics              6N
terminal state equality      6
component control bounds     3N
```

The initial QP intentionally uses component bounds so it can be solved by OSQP. The next B1 increment replaces or augments these with second-order-cone thrust constraints and a conic reference backend.

## 5. Backend lifecycle

```text
construct(structure, initial_values)
update(new_values)
warm_start(primal, dual)
solve(tolerance, iteration_limit)
read solution and residuals
```

The first implementation uses OSQP to prove that numerical updates and warm starts can be performed without rebuilding the workspace. The same lifecycle will then be implemented by `PersistentCQP` on CUDA.

## 6. Device-resident target

After one-time allocation:

- sparse index arrays remain resident;
- matrix/vector coefficients are updated in place;
- primal and dual iterates remain resident;
- rollout, linearisation, acceptance and trust-region logic execute on device;
- host communication is limited to control, compact diagnostics and requested result extraction.

## 7. Scenario decomposition target

Scenario-local variables and operators are assigned to a scenario GPU group. Non-anticipative controls and risk aggregates form the narrow coupling interface. The implementation will report both generic PDHCG partitioning and scenario-aware partitioning.

## 8. Numerical evidence hierarchy

1. Algebraic unit tests.
2. Agreement with a high-accuracy CPU reference.
3. Repeated-update persistence benchmark.
4. Nonlinear trajectory feasibility checks.
5. Single-GPU crossover experiments.
6. Multi-GPU weak and strong scaling.
7. Robust mission-level results.

No layer is skipped merely because a later benchmark appears visually plausible.
