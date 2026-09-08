# Automatic-grid cold comparison v610

Four actual cold calls use the unchanged v609d core and v609e executable, with
native automatic grid selection, 100,000 iterations and a 30-second deadline each.
All finish at the iteration cap and fail the unchanged original-equation gate.
The optional common-KKT policy follows the same numerical trajectory to FP64
rounding differences. This is an acceptance-overhead diagnostic, not a qualified
throughput result, cold-convergence improvement, mission score or SOTA claim.

| Capture | Policy | Iterations | Native solve seconds | Original normalized gap |
|---|---|---:|---:|---:|
| conditioning-natural-auto | natural | 100000 | 5.035395020 | 0.999924305755 |
| conditioning-common-auto | common KKT | 100000 | 5.413709961 | 0.999924305755 |
| difficult-natural-auto | natural | 100000 | 5.078668945 | 0.0112270504689 |
| difficult-common-auto | common KKT | 100000 | 5.470209473 | 0.0112270504693 |

No supplied-point bootstraps, retries or recovery calls occurred. The first solve
completed before the original results-parser rejected a null grid-count field.
The resumed runner reuses that exact hashed output and executes only the remaining
three calls. The executable records a null requested grid when automatic selection
is used; it does not export the effective grid count. The original failure report,
both runner versions and all four raw outputs are retained.

The separate CPU analysis re-evaluates prior archived cold vectors and the frozen
scaling implementation. Numerical spectral estimates put the two zero-Q stability
products below one; this is not a formal general spectral certificate. Its gap
decomposition identifies substantive equality/stationarity/complementarity errors,
and an exact L1 epigraph-prox reformulation hypothesis for future work.

The native binaries and source lineage are identified in run/report.json and the
[v609 evidence](../persistent-common-kkt-v609/README.md). Physics and score remain
at the incumbent; these conic captures do not certify nonlinear mission dynamics.
