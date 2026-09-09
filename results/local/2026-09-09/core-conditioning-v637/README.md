# Tangent diagnostic: original objective and residual motion survive projection

The approved diagnostic ran once after the constructed tiny checks passed.
It completed exactly2 banded factors and6 right-hand sides, with12 implied
triangular solves. All six probes passed the fixed FP64/Decimal65 residual,
reconstruction, energy and cross-symmetry checks. There was no optimization,
proximal operation, eigensolver, GPU call, trajectory update or retry. The
original v635/v636 qualifications and objective comparisons are unchanged.

Worker report SHA256:
`94160c263477132572293e974f6c95dd34310bca316478d8c5a8e40a1577137f`.
Ready SHA256:
`a36eca5e60a50b7ae5189df3027f23894f645d227a2e5f504d256367c01fee13`.

## What was distinguished

Physical T spans only0.19–0.475. Lambda10000 appears only in the virtual selector
columns and does not directly reduce the physical steps. The canonical
objective covector is h=-e_terminal_mass+epsilon, with both exact corrections
epsilon543=+2^-62 and epsilon547=-2^-62 retained. All original coefficients,
including the small Gamma contributions to the six other dynamics components,
were retained in E_p and in every multiply and factor.

| Directional quantity | Early | Near-converged |
|---|---:|---:|
| h^T L h | 1.59047092050e-5 | 1.59048043414e-5 |
| h^T T h | 0.316666666667 | 0.316666666667 |
| Canonical objective mobility fraction | 5.02253975e-5 | 5.02256979e-5 |
| Original c_p^T T c_p | 1.59280822717e-5 | 1.59280822717e-5 |
| Tangent mobility / original-c ambient energy | 0.998532588 | 0.998538560 |
| v635 residual tangent-energy fraction | 0.999999999783 | 0.999999999756 |
| v636 residual tangent-energy fraction | 0.999976969126 | 0.999988868959 |
| Objective alignment with v635 tangent residual | 0.0905470 | 0.0905613 |
| Objective alignment with v636 tangent residual | 0.0815236 | 0.0779166 |

The apparently dramatic canonical-terminal-mass ratio is representation-specific.
Adding an equality-normal covector changes its ambient norm but leaves its
tangent action unchanged. The original c_p used by the algorithm has ambient
energy1.59281e-5, and projection preserves about99.853% of that energy. Therefore
these probes do **not** show that equality projection throttles the actual
objective gradient by20,000 times. `check_covector_scale.py` supplies this
planned reporting comparison by saved coefficient sums only, without another
factor or solve. Neither ratio is a condition number.

The remaining stationarity residual is also almost entirely tangent; its large
Gamma entries cannot be dismissed as equality-normal components. Its alignment
with the objective direction is weak. Rescaling equality rows alone leaves L
unchanged. The current evidence does not establish an equality-tangent
conditioning bottleneck that warrants changing the metric.

At the actual retained v636 dual-projection site, both inputs have1930 strictly
negative scalar arguments,44 strictly positive scalar arguments,37 SOC middle
branches and38 SOC interior branches. There are no nondifferentiable boundary
arguments in these saved sites. The smallest absolute strict scalar argument
is about1.07e-4. Exact squared SOC branch margins are retained in the report.

| Quadratic coupling quotient on tangent probe | Early | Near-converged |
|---|---:|---:|
| Full C, canonical objective | 0.902499998818 | 0.902499998841 |
| Responding cone map, canonical objective | 0.243259686036 | 0.243233459035 |
| Full C, v635 stationarity | 0.902499840743 | 0.902499840773 |
| Responding cone map, v635 stationarity | 0.0142089896510 | 0.0142080221870 |
| Full C, v636 stationarity | 0.902499868644 | 0.902499865309 |
| Responding cone map, v636 stationarity | 0.000221509368262 | 0.0000996019547769 |

The full-C safety metric is nearly saturated along these directions, but the
current cone projection responds weakly to the remaining tangent residuals.
The v635 residual is evaluated at that same v636 site; the actual v635 last
projection argument was not retained, so this is not a matched active-set
history. The v636 site is infeasible and is not asserted to be an optimum face.
These observations support a weak local cone-response tail. They do not establish a global convergence rate or
attribute all slow convergence to one condition number.

In particular, the small responding-cone quotient is not a safe global norm
bound. It cannot justify larger steps, removing the1930 currently inactive
rows, freezing a face, or tuning from a known solution. Any later metric must
act on equality-feasible directions while preserving all original rows and
must receive a new global coefficient-based safety argument. No such metric
has been selected or implemented here. More Halpern iterations, dual-only
polishing and tighter inner gates are not recommended by this diagnostic.

## Numerical reliability and charged work

The largest original relative linear/equality check is below2.54e-16. The
canonical objective projection has a roughly5.2–5.6e-15 equality cross term
against1.59e-5 energy; that term is explicitly retained. Decimal65 energy
identity discrepancies are below5.75e-21, and cross-symmetry differences below
7.02e-23. No negative result was clipped and no inconclusive probe was hidden.
These are backward/consistency checks, not a forward-error condition estimate.

The worker retains full covectors, original-row auxiliary multipliers,
displacements, ordered RHS/Hmu/equality products, metric arrays, factor
diagonals/scales and six weighted C products. The multipliers and displacements
are labeled diagnostics; no corrected original result is constructed.

Actual work also includes2 Gram products,24 E-vector products,12 E-transpose
vector products,6 H-vector products,6 C-vector products,4 original residual
formations and6 Decimal65 probe verifications. Exact sparse-support and branch
arithmetic, parsing and all retention are charged to walltime. The two factors
took about0.124ms and0.074ms; the three-RHS solves0.092ms and0.054ms. These are
nested measurements, not solver throughput or speedup evidence.

Complete worker-process time was0.569185466s, worker body0.345977736s and
supervisor0.696981419s. Supervisor8803/worker8814 exited0 and were reaped. The
30-second deadline and10/30-second cleanup bounds were not approached. Runtime
is the pinned WSL Python3.12.13, NumPy2.5.2, SciPy1.18.1, one BLAS thread and
GPU hidden. Tiny validation used three independent exact projector covectors,
the mass-chain epsilon identity, scalar/SOC branch and boundary cases,
rank-failure cost retention, duplicate-call rejection and three invalid-input
checks; no captured matrix was used in that stage.

`index.json` is the earlier four-file proposal seal. `ready.json` binds the
tiny-tested runnable source. `result-index.json` binds the final complete
saved evidence. Neither earlier seal nor any v635/v636 evidence was rewritten.

## Portable package

Archive paths are repository-relative. This package includes the exact four saved
points, two coefficient captures and metrics, source/runtime pins, tiny records,
full diagnostic vectors and completed reports. Previous full experiments remain
in the linked v635/v636 sibling packages; their index/archive pins are included.
External runtime binaries are identified by hash and omitted. No compiled binary,
private key, cache or Git state is present. The historical proposal and readiness
seals remain unchanged.

Run `python audit_package.py --package . --index-sha256 INDEX_SHA` from a copied
package. The stdlib checker verifies member sizes/hashes/safe paths and the saved
source/input/runtime/status/vector bindings. It does not execute archived source,
recompute numerical results, factor, solve, project, propagate or run a GPU.
