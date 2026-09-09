# Bonus-weighted beam construction: no usable improvement

Applying the official bonus weights during beam construction and collection scheduling produced different routes, but no better feasible fleet replacement in these two families. The verified incumbent remains **13,526.96124117307 weighted kg and 14,915.0444900762 raw kg, with 24 ships and 208 asteroids** (Result `1f420928…`). This experiment performed zero refinements, trajectory certificates or fleet checks.

| Arm | Ship | Complete proxy plans | Unique physical requests | Best weighted cargo, kg | Search seconds |
|---|---:|---:|---:|---:|---:|
| Unit control | 10 | 1,020 | 1,020 | 579.157224452 | 26.997474207 |
| Unit control | 21 | 940 | 940 | 539.207632594 | 22.638949567 |
| Bonus weights | 10 | 851 | 851 | 534.733710782 | 32.200803551 |
| Bonus weights | 21 | 885 | 885 | 516.524087244 | 26.053269710 |

The bonus arm introduced **745 and 775 new physical prescriptions**, respectively, and **355 and 289 new deployment orders**. A physical prescription is the exact sequence of flight endpoints/epochs, deployment and collection epochs, and cargo. Proxy delta-v, inflation, mass estimates and scores are excluded from that identity. There were 369 and 331 new full itinerary orders, including collection order. Whole-plan hash changes alone were not counted as new physical requests.

Neither bonus pool contains a standalone weighted improvement, and every bonus candidate also fails the current fleet's raw ship-count budget as an individual replacement. Exhaustive saved-ledger pairing of compatible ship 10/21 choices, including either incumbent, finds no positive bonus-arm combination. This is arithmetic over saved proposed cargo, not a new mission search or certificate. All bonus plans avoid ship 24's nine asteroids; 24 ship-10 plans and 25 ship-21 plans conflict with other currently retained ships. Those conflicts are recorded separately from the budget failures.

The unit arm retains the known ship-10 request `dcfc0573…`, with a proposed +5.528863602 weighted kg and +5.749486653 raw kg. This exact return request has already failed earlier trajectory refinements. It is not a newly discovered gain. The best unit pairing keeps ship 21 unchanged; no new bonus prescription improves that choice.

The depth distribution also changed. Unit ship 10 retained 114 nine-miner and 13 ten-miner plans; its bonus arm retained 11 nine-miner plans and no ten-miner plan. Unit ship 21 retained 44 nine-miner plans, versus six under bonus weighting. These are saved outcomes, not proof of which intermediate prefix was rejected.

The frozen source explains why replacing unit weights is more than a final-output sort. `search.py` `_select` rewards expected weighted mining and subtracts the unchanged **0.15** proxy-propellant penalty. `plan_score` uses the same penalty when choosing a completion. Collection DP receives the bonus weights with its unchanged **1.0** propellant penalty (and the existing additional 0.15 arm). The bonus coefficients are all at most one, while those penalties remain fixed. This changes relative reward/cost tradeoffs as well as asteroid preference. The width-128 beam and two-variants-per-deployed-set admission then retain a different set of prefixes. Cluster bonus, time/lookahead penalties and chain-tour scoring were zero/off here. These facts support a changed search tradeoff; the saved outputs do not isolate one penalty or pruning decision as the cause of lower quality.

Keep the productive unit arm. A future controlled policy can retain both raw-mass and weighted-score alternatives and make the fleet's raw budget explicit during admission. This result does not justify blindly switching the default to bonus weighting or widening these same two families again. No such follow-up ran here.

All four searches reached depth 10 with 1,025 parent expansions each. Their 3,696 complete proxy plans required 134,783,148 recorded Lambert branch requests; these are work counters, not certified solutions. Search timers sum to 107.890497035 seconds. The two workers total 110.467825316 seconds, including the preserved initial abort and continuation overhead. There is no paired speedup claim: the objective arms perform different work and each was measured once.

The original v636 worker stopped after unit ship 10 because the harness compared integer-keyed in-memory summary dictionaries against string-keyed decoded JSON. The first mismatch was `/0/deploy_epochs`. Source-backed restoration of those key types exactly reproduces the captured failing digest. The saved 1,020 plans and 122,872 scalar values already equal the archived v835 control exactly, including physical-request order. The continuation compares the actual saved JSON on both sides without tolerance or coercion, reuses that control and executes only unit ship 21 plus the two bonus searches. Both original and continuation owners were reaped with empty final GPU inventories. The original abort, all readiness revisions and all raw outputs remain retained.

`evidence.zip` contains both harnesses, complete generated pools, telemetry, failures, source/input manifests and saved-only audit scripts. `package-map.json` links the unchanged v841 host/native package and v835 source inputs to their existing archives by exact member/hash, avoiding duplicate binaries or source trees. It also binds the current Result to the published fleet-addition-v633 artifact. Preparation README/ready files inside the archive are historical; this README reports the terminal outcome.

Verify the package with the index SHA supplied at publication:

```text
python -B verify_package.py --index-sha256 <index-sha256> --dependencies --roundtrip
```

The optional roundtrip reruns only the standard-library saved-cargo/request analysis in a temporary directory. It performs no GPU, geometry, optimization, refinement or certification work.
