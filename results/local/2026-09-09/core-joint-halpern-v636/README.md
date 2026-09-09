# Reflected-Halpern joint-prox reference: 0/2, feasibility regression

One fixed CPU experiment completed 10,000 outer maps for each of two changed
stages of the same synthetic fixture. **Neither final point passed the original
gates.** The unchanged long-double and Decimal65 auditors agree on all eight
retained points. The modestly lower objective values are infeasible and are
not qualified progress. This variant is not being ported or given a larger cap.

The sole outer change from v635 is primal-first joint PDHG wrapped in reflected
Halpern averaging and the pinned restart predicate. It uses the original
zero-Q objective, equalities, L1 penalties, scalar/SOC constraints, coefficient
T/Sigma_C, 1e-11 practical inner tests, 32 directions and 12 Newton halvings.
There is no PI update, metric adjustment, multiplier clipping or objective
reference in the algorithm. Every cold initial joint prox is charged.

Only the actual proximal P output is exported, with its own equality multiplier
and L1 subgradient. Reflected/averaged private working duals may be nonconic.
The saved analysis verifies the three post-initial P readouts per input bitwise
against their public primal and retained dual arrays. It also independently
checks every scalar restart decision. These checks do not reapply a proximal
operator, factor a matrix or execute the outer method.

| Final original measure | Early | Near-converged | Limit |
|---|---:|---:|---:|
| Relative primal equation residual | 2.18402e-16 | 2.49499e-16 | 1e-9 |
| Relative stationarity | 1.96601e-8 | 1.96118e-8 | 1e-9 |
| Objective gap | 0.00113299668 | 0.00114946669 | 1e-9 |
| Primal cone violation | 1.01151e-5 | 5.90706e-6 | 1e-8 |
| Maximum normalized complementarity | 1.39203e-8 | 1.25766e-8 | 1e-9 |
| Virtual-control L1 cost / pair Fenchel defect | 0 / 0 | 0 / 0 | original gates |
| Qualification | Fail | Fail | all original gates |

Dual cone violations remain below 3.21e-19. Original stationarity still fails
by about19.6 times, gap by about1.13–1.15 million times, primal cone feasibility
by591–1012 times, and complementarity by12.6–13.9 times. The worst SOC is cone74,
starting at original G row3320. All525 virtual/epigraph pairs remain exactly
zero. Both final worst stationarity coordinates remain Gamma variable611.
In v635, both primal-cone and complementarity gates passed; this regression
prevents interpreting the lower scalar costs as better solutions.

| Saved same-input cost comparison | Early | Near-converged |
|---|---:|---:|
| v636 final c^T x | 0.022645104863056159 | 0.022644950612063814 |
| v635 final c^T x | 0.022669895273641100 | 0.022669874567925268 |
| Qualified saved QOCO c^T x | 0.021875586368818126 | 0.021875559685933571 |
| v636 minus v635 | -2.47904106e-5 | -2.49239559e-5 |
| v636 excess over QOCO | 3.51770454% | 3.51712567% |

The QOCO references passed both saved original audits, with native status1 for
early retained trial1 and status2 for the near cold comparison. They are
approximate qualified references, not proofs of the exact optimum. Their
source/input/result bindings remain in the published
[v636 decision](../core-primal-decision-v636/README.md), with the exact prior
cost findings included here. Reference costs were opened only after both
trajectory iteration loops and stopping decisions. Exact Decimal65 comparisons
are reporting evidence, never additional stopping tolerances.

The signed-gap identity is retained in `saved-analysis/findings.json`. The
stationarity-dot-primal term still accounts for nearly all of the gap;
equality terms are around1e-15 and complementarity sums around2–4e-8. Improving
equality residuals alone does not resolve the original cone and dual tail.
Conditioning within the equality-feasible directions is a possible next
mathematical question; no new metric, restart, implementation or run is claimed.

## Work, outcomes and timing

The run used20,002 joint calls and20,010 accepted inner directions. Two full
and two active band factors succeeded; all later active factors reused their
mask. There were no inner limits, factor rejections, fallbacks, halvings or
numerical failures. Both cases stopped at their original outer cap. The only
failed preflight was a caller's relative output path, rejected by the unchanged
containment guard before any child or numerical work; its record is preserved.

Each case recorded8 restarts and58 fixed-point metrics. Restart iterations were
200,400,800,1400,2200,3600,5800,9200, with epoch references taken at the next
actual map. There were20,002 trial-KKT exits and8 Armijo exits, all using the
unchanged signed-branch merit identity. Complete counts, attempts, actual
products, scalar traces, fixed-point terms and original checkpoint arrays are
retained. Eight initial/final/intermediate points were independently audited.

Complete worker-process time was **24.076937979s**; worker body23.868277908s
and supervisor24.300589266s. These scopes include setup, initial prox, all
iterations, original gates, diagnostic copies, evidence writes and teardown.
Nested timers overlap and must not be summed. The additional saved-data audit
and analysis are separately timed. The previous v635 complete process was
19.76466773s, but these single CPU executions are not matched latency samples;
both are unqualified and this run records additional outer diagnostics. There
is no native solver, GPU throughput, mission-score or performance-win claim.
Audit sentinel0 is preserved and is not relabeled as native termination.

## Portable evidence

All archive names begin with their repository-relative paths. The primary kit
is `build/performance/core-joint-halpern-v636/`: source, exact inputs and metrics,
runtime/source pins, tiny/preflight records, complete run outputs, original
auditors and saved analyses. The new tiny suite covers exact Fraction/KKT maps,
both M identities, minus-cross mutation, nonconic working versus conic P states,
mu/q ownership, alpha/restart boundaries, counter/reference order and five
failure-retention paths. Unchanged v635 tiny validation is included as saved
evidence, not executed again. Prior complete evidence is linked in
[v635](../core-joint-reference-v635/README.md); dependency index/archive hashes
are recorded. Frozen design and launch documents are historical pre-run plans.

Runtime was pinned Python3.12.13/NumPy2.5.2/SciPy1.18.1 in WSL,63-bit long-double
mantissa, one BLAS thread and GPU hidden. External runtime binary identities
are recorded; their bytes, compiled binaries, keys, caches and Git state are
excluded. No shared production source was changed.

Run `python audit_package.py --package . --index-sha256 INDEX_SHA` from a copied
package. This stdlib verifier checks every archived member and size/hash,
safe paths, tested/runtime/source bindings, eight vector hashes, saved original
gate decisions, counts, P ownership, restart records and cost-report bindings.
It does not execute archived numerical code, recompute original numerical gates,
propagate, factor, project, optimize or invoke a GPU.
