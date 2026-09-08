# v606 result audit

The finite **local RTX 5090** run completed successfully as an experiment, with **no certified score improvement**. H100 transfer was rejected before export and was not retried. All four route calls were executed locally; there must be no H100 duplicate of this four-route budget.

`audit.py` reads the frozen preparation and execution evidence in `../truth-set-v606/`, verifies all 256 indexed preparation files and both runtime library hashes, and writes only this separate audit directory. The original kit and output were not changed by this audit.

Both original controls reproduced through actual CUDA-seeded SCvx solves and passed both independent and official full-fleet checkers. Both fixed-cargo replacement probes certified every earlier leg, including all four changed incident transfers, then failed at their unchanged final Earth-return leg. Exact prescribed epochs and cargo were retained. Neither failed return received a leg certificate or a full-fleet promotion.

| Measured quantity | Ship 2 replacement | Ship 7 replacement |
| --- | ---: | ---: |
| Earlier certified legs | 15 / 15 | 16 / 16 |
| Extra certified prefix fuel versus control | 178.700067 kg | 95.392567 kg |
| Propellant available at start of return | 13.723155 kg | 91.543915 kg |
| Same scheduled control return's measured burn | 189.744659 kg | 183.794239 kg |
| Failed return's virtual-control residual | 0.002306424 | 0.002189442 |

The control's return burn is measured at its own heavier initial mass. It is **not** a proved minimum or an exact requirement for the probe. The correct conclusion is that these fixed schedules did not yield a certified gain; they did not demonstrate a proxy false negative. A failed nonconvex SCvx solve does not prove physical infeasibility, and there is no certified complete-route propellant measurement for either failed probe.

The run used 66 native leg calls: 64 converged and two failed returns. It recorded 362 SCvx iterations, 301 accepted iterations, 362 QOCO subproblem reports and 15,217 QOCO inner iterations. Total campaign time was 110.373791 s. The three full-fleet checker calls took 62.008025 s, the four route refinements took 47.774765 s, and timed native-wrapper calls within refinement took 42.383195 s. Native-wrapper wall time includes host marshalling; it is not isolated GPU kernel time. CPU verification remains a substantial part of this diagnostic run.

The retained fleet remains **14,051.854893908598 raw kg / 12,810.135953048577 fixed-bonus weighted kg**, with 23 ships and both baseline checkers passing. These are different quantities; no cargo was sacrificed to make a probe feasible.

A distinct next hypothesis to consider is **return-window rescue at fixed cargo**: use a saved certified probe prefix, allow waiting after its final collection, and screen legal Earth-arrival times before refining a bounded number of return legs. Ship 7 has the larger remaining reserve and is the better initial probe. Any such future trial must preserve cargo, all mission limits and both final fleet checkers. It would test return phasing rather than repeat the same four routes or relax physics tolerances. No such new search or GPU solve was launched here.
