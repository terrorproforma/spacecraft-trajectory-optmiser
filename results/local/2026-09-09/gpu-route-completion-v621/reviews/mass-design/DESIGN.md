# Exact mass-coordinate reduction: feasible, with a conditioning constraint

Both pinned captures permit exact affine elimination of physical mass variables
and their initial/continuity equalities while retaining **every virtual-control
variable**, all original costs and cones. A direct prefix implementation is not
ready to promote: it creates a strongly coupled cumulative operator that the
current infinity-norm Ruiz scaling does not sufficiently address.

Run the coefficient-only CPU audit from the repository root:

```powershell
& 'C:/Users/Angus/.local/bin/python3.12.exe' -B build/performance/mass-structure-review-v621/audit_mass.py
```

The script reads only the original coefficient captures and frozen source, not
known/computed solutions. It verifies source/input hashes, exact row structure,
and rational-arithmetic forward/reverse/dual identities. `findings.json` contains
the full measurements. No GPU, solver, compiler or weight search was run.

## Observed coefficients

Frozen `gtoc12_conic.cu:153-174` orders seven state components with physical mass
last and emits the initial-mass equality. The actual captures verify exactly

```
m_0 = 1
m_(k+1) - m_k + g_k Gamma_k - nu_(m,k) = a_k.
```

Here `nu_(m,k)` is the original retained mass virtual control, not zeroed or
discarded. All `g_k` are positive in these captures, but `a_k` is **nonzero** and
must be retained. Do not replace the captured affine equation with a guessed
physical mass-flow formula or set Gamma equal to the thrust-vector norm.

| Capture | Intervals | Eliminated masses/equalities | Retained virtual variables | Variables after mass + L1 reduction |
|---|---:|---:|---:|---:|
| conditioning | 210 | 211 | 1470 | 3580 |
| difficult | 233 | 234 | 1631 | 3974 |

All physical-mass objective coefficients and the entire Hessian are exactly
zero. Remaining equalities have no numerical mass entries for these captured
linearizations; this is not a general dynamics assumption. There are exactly
three scalar mass-bound rows per node and no mass entries in SOC rows.

## Primal, transpose and dual maps

Partition original variables into eliminated masses `m` and retained variables
`q`, and reorder only the eliminated equality block as initial mass followed by
mass dynamics: `L m + H q = b_E`. Then

```
d = L^-1 b_E,       S = -L^-1 H,       m = d + S q.
m_j = 1 + sum_(k<j) (a_k + nu_(m,k) - g_k Gamma_k).
```

Apply `S` by a prefix sum; apply `S^T` by a suffix sum. For a mass covector `w`,
each virtual mass receives `sum_(j>k) w_j`, and Gamma receives that same suffix
multiplied by `-g_k`. The constant `d` contributes to affine offsets, never to
the transpose. Forward-only reconstruction is insufficient.

Write retained equality rows as `[C_m C_q]`, original cone rows as `[G_m G_q]`.
The reduced data are

```
C' = C_q + C_m S,       b' = b_C - C_m d
G' = G_q + G_m S,       h' = h - G_m d
c' = c_q + S^T c_m,     constant' = constant + c_m^T d.
```

For these captures `c_m=0`, so the original objective and L1 coefficients remain
unchanged. Scalar/SOC cone geometry remains original; only affine arguments
change. The exact L1 proximal step remains separable in retained coordinates
when its metric is diagonal.

Given reduced equality duals `y_C` and original cone duals `z`, recover eliminated
equality duals by

```
w = c_m + C_m^T y_C + G_m^T z
y_E = -L^-T w.
```

Thus `y_initial=-sum_j w_j` and `y_mass,k=-sum_(j>k) w_j`. Scatter these into
their original equality-row positions. Original mass stationarity is zero in
exact arithmetic, and original retained stationarity equals reduced
stationarity. Original dual objective becomes
`constant' - b'^T y_C - h'^T z`, including the correct constant sign.

Recover these mass equality duals before completing original virtual-epigraph
pair duals, or use the equivalent reduced gradient for that completion. Otherwise
mass virtual-control subgradients omit the eliminated equality normal. Original
primal/dual feasibility, stationarity, global gap and per-block complementarity
must still be recomputed on all original rows. A supplied already-qualified
original seed must be checked untouched before canonical mass reconstruction.

## Why this requires new operator-aware scaling

Three mass bounds per node become cumulative constraints on earlier controls
and mass virtuals. Explicit numerical operator entries after both mass and L1
reductions grow from **21463 to 152919**, and **23815 to 185746**. Use structured
forward/reverse scans rather than materializing this fill.

There is also a provable conditioning warning. Give the K mass-virtual columns
the unit vector `1/sqrt(K)`. Three copies of each cumulative prefix produce

```
||K_reduced||^2 >= (3/K) sum_(j=1..K) j^2 = (K+1)(2K+1)/2.
```

The norm lower bounds are **210.74985** and **233.74987**. These cumulative
mass-virtual entries are all unit magnitude, so infinity-norm Ruiz alone leaves
that subblock unchanged. The old spectral step near 0.327 cannot be reused for
it; reciprocal global weighting does not change this product condition. A
smaller global step can protect stability while slowing unrelated coordinates.
This rules out treating elimination as a drop-in mask plus the old scaling.

Before any GPU intervention, the next prerequisite is a CPU-verified,
operator-aware diagonal metric or another explicitly derived structured metric,
with matched forward/adjoint and norm tests. A non-diagonal metric may invalidate
the current separable soft threshold or cone projection and needs its own prox.
This review does not select or implement such a metric and predicts no speedup.

The first supported contract should reject nonzero Hessians, shifted-coordinate
captures, nonfinite coefficients, changed initial-mass rows or noncausal/different
mass patterns unless separately derived. Retain constant transformed inequalities
or prove them feasible before removing them. Parallel FP64 scans reassociate sums;
original mass equations remain accuracy gates. A deterministic synthetic FP64
probe has maximum mass residual below `1.8e-16` here, which is a probe result,
not a general roundoff guarantee. All six rational forward/reverse/dual checks
pass exactly.
