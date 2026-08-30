# ADR 0001 — Mirror PDHCG's native canonical form

- **Status:** accepted
- **Date:** 30 August 2026
- **Decision scope:** B0/B1 solver bridge and all later persistent backends

## Context

The upstream PDHCG implementation solves

\[
\begin{aligned}
\min_x\quad &
\frac12x^\top(Q+R^\top D R)x+c^\top x+c_0\\
\text{s.t.}\quad&
\ell_c\leq Ax\leq u_c,\\
&Fx+g\in\mathcal K_a,\\
&\ell_v\leq x\leq u_v,\\
&x_J\in\mathcal K_v.
\end{aligned}
\]

Its public Python `Model` has coefficient setters and accepts primal-dual starts. However,
`Model.optimize()` calls the one-shot `solve_once` binding. The public C API similarly exposes
`create_qp_problem`, `set_start_values`, `solve_qp_problem`, and `qp_problem_free`, but no
persistent device workspace or in-place coefficient update function.

Consequently, wrapping the current Python class would repeatedly canonicalise, preprocess,
allocate, scale and transfer a problem. That would not satisfy contribution B.

Upstream SOC coordinates also differ from the conventional solver ordering:

```text
PDHCG SOC:  [v (v_dim slots), w, z],  ||v||² + w² <= z²
Clarabel:   [z, v, w]
```

A rotated SOC is stored by PDHCG as `[v, s, t]` with `||v||² <= 2 s t`.

## Decision

SpacePDHCG's canonical data model mirrors the upstream native split exactly:

1. sparse quadratic matrix `Q` and linear objective `c`;
2. scalar affine rows `l <= A x <= u`;
3. native affine cone rows `F x + g in K`;
4. variable bounds;
5. variable cone blocks.

Immutable CSC index arrays are separated from mutable numerical values. Every persistent
backend must reject updates that change matrix dimensions, sparse index arrays, cone layout,
or the finite/equality pattern of scalar bounds.

The CPU conic reference uses Clarabel and performs an explicit coordinate conversion:

- SOC: `[v,w,z] -> [z,v,w]`;
- rotated SOC: `[v,s,t] -> [s+t, sqrt(2)v, s-t]`;
- exponential and power cones: identity coordinates;
- PSD cones: disabled until both projects' `svec` ordering is verified.

The initial QP baseline remains on OSQP. OSQP rejects native cones and finite variable bounds
unless those bounds have already been represented as scalar affine rows.

The later CUDA backend will extend PDHCG below its one-shot public API with a lifecycle of the
form:

```text
initialize_structure(...)
update_values(device pointers, stream)
set_warm_start(device pointers)
solve_async(tolerance, iteration limit, stream)
get_device_solution()
```

## Consequences

### Positive

- The reference and performance paths solve the same mathematical object.
- Native quadratic objectives and affine cones survive without CVXPY epigraph lifting.
- The trajectory transcription can be developed before CUDA hardware is introduced.
- Any upstream contribution can target a precise missing lifecycle rather than a speculative
  wrapper.

### Costs

- Scalar lower/upper rows require conversion for solvers using `Ax+s=b` form.
- Dual variables must be mapped back from solver-specific row duplication and cone transforms.
- Low-rank `R^T D R` objectives are deferred until the sparse-Q bridge is stable.
- Explicit warm starts are unavailable through Clarabel's current public Python interface, so
  Clarabel remains a correctness/polish reference rather than the persistence performance
  target.

## Verification gate

The decision is validated when the same fixed CW rendezvous structure can be solved repeatedly
as:

- a scalar-bound QP through persistent OSQP; and
- a native affine-SOC QP through persistent Clarabel updates,

with independently checked dynamics, endpoint and thrust feasibility.
