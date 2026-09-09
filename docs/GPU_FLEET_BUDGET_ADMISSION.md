# Fleet-budget admission and the v630 prefix test

The production search/refinement interface can now evaluate a replacement
against the actual verified fleet's weighted objective and raw-mass ship rule.
A replacement may lose raw cargo if its weighted score improves and the complete
fleet still has enough raw mass. The incumbent context copies the ship cargo and
asteroid ledgers from the exact Result accepted by both checkers. Missing bonus
weights, mismatched checker bindings, dependent miner inventories and asteroid
conflicts are rejected.

The bounded completion-admission queue retains immutable candidate prescriptions
and their original native completion observations. Its executor uses fixed cargo
and epochs. `price_cluster` also has an explicit one-ship replacement mode that
retains certified alternatives within the existing finite attempt budget.
Shortlisting first covers distinct Earth/depth groups, then fills remaining
slots with different schedules or asteroid sets sharing an Earth seed. It no
longer discards every alternative merely because its Earth leg is repeated.

Each proposal is evaluated against the same incumbent budget. Independently
admitting several replacements does not authorize combining their raw-mass
deficits. Final fleet selection and both complete-fleet checkers remain required.
The production changes pass 116 focused CPU tests. The frozen mission harness
has a further 24 CPU orchestration tests.

## Actual finite mission experiment

Three archived candidates were selected through the new production policy and
passed to the existing native fixed-cargo refiner, using the corrected endpoint
controller and the [new CUDA boundary ephemerides](GPU_ROUTE_EPHEMERIDES.md).
Their previous failures used older controller sources. Together they prescribed
28.595948 weighted kg of improvement, before trajectory certification.

**None of the three complete routes certified.** The current fleet remains
**13,023.704901 weighted kg / 14,291.006160 raw kg**, with 23 ships and 199
asteroids. No candidate was promoted and no new fleet file was emitted.

| Candidate | First rejected transfer (MJD) | Completed native leg calls | Rejected leg outcome |
| --- | --- | ---: | --- |
| Ship 22, v794 rank 0 | 9443 → 32613, 65093 → 65273 | 2 | Stationary penalized point with nonzero defects; 13 SCvx iterations, 6 accepted |
| Ship 10, v794 rank 0 | 17126 → Earth, 69263 → 69713 | 18 | Nonzero virtual control remains; 33 iterations, 3 accepted |
| Ship 19, v802 rank 0 | 51417 → 22760, 65498 → 65648 | 4 | Nonzero virtual control remains; 36 iterations, 4 accepted |

The latter two retain the optimizer's `infeasible` status label in the evidence.
That label is not a mathematical infeasibility certificate for the underlying
trajectory. These outcomes do not establish that a different initialization,
mesh or schedule cannot produce a feasible transfer. Physics gates were not
relaxed and failed attempts remain visible.

The saved native trajectories meet their requested final position nodes to
within 1.60e-9 km on all three failed legs. Their maximum dynamics/virtual
defects remain 2.136e-3, 2.782e-3 and 1.496e-3 in native model units, respectively.
Qualified conic subproblems occur within each run, alongside later unqualified
ones. These are failures of the complete nonlinear refinement, and the failed
trajectories are never sent through a successful-certificate path. The endpoint
merit correction alone therefore does not resolve these cases.

The complete worker takes **20.406 seconds** in this one local observation:
24 native leg calls, 157 SCvx iterations, 21 flight-certificate calls and one
waiting-coast certificate. All three first legs use archived Earth trajectories;
the other 21 leg attempts start from the native cold initializer. Three CUDA
batches prepare 56 route boundary states with zero CPU orbital-state calls in that step.
Two CPU state calls remain in the waiting-event callback. No search, Lambert
generation, retiming or cargo reduction occurs in this experiment. No final
fleet selection or checker pair is needed because no complete route qualifies.

This is a production-policy and GPU-integration test with preserved failure
evidence, not a speedup or score gain. The next mission experiment should change
initialization or the coupled itinerary around the recorded difficult transfers;
simply admitting and rerunning the same prescriptions cannot establish progress.

The retained verified fleet and loading instructions remain in
[the v629 frontier record](GPU_MASS_BUDGETED_FRONTIER.md). The
[portable evidence package](../results/local/2026-09-09/fleet-budgeted-prefix-v630/README.md)
contains frozen inputs, source, native readbacks, controls, certificates and all
failures, with a standalone hash and saved-output validator.
