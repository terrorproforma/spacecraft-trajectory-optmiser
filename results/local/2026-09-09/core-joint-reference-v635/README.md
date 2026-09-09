# Joint equality/L1 reference: stable merit, still 0/2 qualified

The single fixed CPU experiment completed 10,000 outer updates on each of the
two frozen inputs, with **0/2 original numerical qualifications**. Both original
long-double and 65-digit audits agree on all eight retained points. Neither a
native solver nor a GPU ran. These are two changed stages of one synthetic
fixture, not independent mission instances or a matched speed benchmark.

The two numerical changes from v633 are a trial's unchanged proximal KKT stop
before Armijo, and stable dual decrease using the original sparse E transpose
and the actual represented multiplier delta. An unchanged signed threshold
branch uses the active quadratic identity; crossings use the full Gram bound.
FP64 arithmetic makes this a practical screen, not an outward certificate.
Original zero Q, c, lambda, equalities, scalar/SOC constraints, T/Sigma_C,
1e-11 inner tests, rank screens, 32 directions and 12 halvings remain fixed.
The cold initial joint prox is charged. No old L1 dual is double-counted.

## Final original-coordinate results

| Measure | Early | Near-converged | Limit |
|---|---:|---:|---:|
| Relative primal | 7.31481e-12 | 7.25860e-12 | 1e-9 |
| Relative stationarity | 2.06982e-8 | 2.07014e-8 | 1e-9 |
| Objective gap | 0.00126134376 | 0.00126156925 | 1e-9 |
| Primal cone violation | 1.13431e-9 | 1.12668e-9 | 1e-8 |
| Maximum normalized block complementarity | 8.54139e-13 | 7.91964e-13 | 1e-9 |
| Original virtual L1 cost / pair Fenchel defect | 0 / 0 | 0 / 0 | original gates |
| Qualification | Fail | Fail | all original gates |

Dual cone violations are below 2.11e-19. Both final primal, cone and
complementarity gates pass; stationarity is about 20.7 times its limit and the
gap about 1.26 million times its limit. The proximal stationarity residual is
about 7e-16 absolute, far smaller than the remaining original 2.07e-4 absolute
stationarity defect. This does not support tightening the inner tolerance.

The saved Decimal identity is
g = x^T*r_dual - y^T*(Ax-b) - z^T*(Gx+s-h) + z^T*s.
For early, x^T*r_dual is -0.00126134376158; the equality term is 6.53e-15
and z^T*s is 2.89e-12. Near is analogous. Gamma and thrust-vector coordinates
contribute about 98.7% of the gap; worst stationarity is original Gamma
variable611, node19. All 525 virtual/epigraph/slack pairs are exactly zero,
actual pair sums equal lambda, and the q box has more than 9999.01 margin.

At 100/1,000/5,000/10,000 updates, the early gap is approximately
0.040295/0.024865/0.001830/0.001261; near follows the same progression. The
gap is not globally monotone, and no unsaved iterate or extrapolated convergence
is claimed. v631 reached its caps with virtual complementarity dominating
gaps0.09146/0.08159; v633 stopped after0/73 outer updates on merit failures.
v635 removes that stopping failure and the final virtual penalty, while its
remaining outer dual/gap tail still prevents qualification.

## Work and timing

There were 20,002 joint calls including both starts, 15,469 accepted inner
directions, and two full plus two active band factors. No factor rejection,
fallback, halving, inner limit or numerical inner failure occurred. Of the
trial exits, 15,460 passed KKT before merit; only nine merit products were
needed, all on unchanged branches. The other 4,542 calls passed at their base.
All counters, scalar attempts and final/checkpoint trial operands are retained.

Complete worker-process time was **19.76466773s**, worker body19.57450895s,
and supervisor19.95689143s. This includes imports, coefficient/metric checks,
Gram construction/factors, initial prox, updates, diagnostic copies, roughly
47.9MB of raw run evidence, original gates and teardown. Nested timers overlap
and must not be summed. The separate saved-vector audit took1.12149550s in its
body. No GPU throughput, production speedup or matched timing advantage follows.

The final points now satisfy the prerequisites for a bounded fixed-primal dual
correction experiment. That experiment has not run, and feasibility of the
constrained correction is not established. Any later handoff must preserve
x/s/retained z, actual L1 boxes and original gates, keep CPU-reference provenance
distinct from native status, and charge the preceding solve plus correction.

## Portable evidence

The archive contains the complete isolated source, exact two inputs, unchanged
auditors and metric certificates, tiny/runtime/preflight records, full original
checkpoints, complete inner traces, both audits and the saved gap decomposition.
The nine-file v634 tiny evidence is included. Selected immutable prior audit
records support the historical comparisons; full earlier experiments remain
linked in [v631](../core-primal-reference-v631/README.md) and
[v633](../core-joint-reference-v633/README.md). Their index/archive hashes are
recorded in index.json. Frozen preparation plans are historical pre-run records.

Runtime is the pinned WSL Python3.12.13 with NumPy2.5.2/SciPy1.18.1 and63-bit
long-double mantissa. Exact runtime identities are recorded; external runtime
binaries are not copied. No compiled binary, private key, Git state or cache is
included. Publication does not require executing scientific or archived code.

Run `python audit_package.py --package . --index-sha256 INDEX_SHA` from a copied
package. This stdlib checker validates every member and its size/hash, source
and runtime bindings, native-status distinctions, all eight vector hashes and
saved gate verdicts, and complete work accounting. It performs no propagation,
factorization, proximal solve, optimization or new original numerical audit.
