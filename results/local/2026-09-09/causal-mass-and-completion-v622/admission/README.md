# Bounded refinement admission v622 — CPU evidence

Fully costed final-mass proxy rejections can now be retained for a finite, opt-in
fixed-cargo refinement queue. Admission is **not** a feasible route, a certificate,
or a fleet score. This tranche executed no GPU job or trajectory refinement and
did not change the incumbent.

The frozen control split contains five exposed development routes (ships 1, 4,
7, 10, 23) and 17 additional independent routes. All asteroid footprints are
disjoint. Ship 3 is excluded because its inventory is not independently closed.
These are historical v595/v616 controls, not the current fleet. “Held-out” means
excluded from development of this queue policy; their results appeared in earlier
audits, and they are not asserted to be held out from the historical inflation fit.
The existing model coefficients were not changed or refitted.

Exactly 34 scalar `_finish_cpu` evaluations ran once on the 17 held-out routes,
with the archived measured deployment prefix and two fixed models. They used
saved Lambert DVs and existing cached feature values. The other 20 development
readbacks were reused from v619/v621. There were zero new geometry, feature
arithmetic, Lambert, SCvx, native/GPU, certification, or fleet-promotion calls.

| Control group | Model | Prefix | Proxy passes | Complete mass rejections | Selected regular / uncertain |
| --- | --- | --- | ---: | ---: | ---: |
| Development, 5 routes | Flat | Flat proxy | 1 | 4 | 1 / 2 |
| Development, 5 routes | Flat | Measured | 0 | 5 | 0 / 2 |
| Development, 5 routes | Existing fit | Flat proxy | 1 | 4 | 1 / 2 |
| Development, 5 routes | Existing fit | Measured | 0 | 5 | 0 / 2 |
| Queue held-out, 17 routes | Flat | Measured | 8 | 9 | 2 / 2 |
| Queue held-out, 17 routes | Existing fit | Measured | 3 | 14 | 2 / 2 |

Every control passed the earlier completion gates. Their archived full-fleet and
leg certificate evidence remains positive, so these mass rejections are concrete
proxy counterexamples. An archived certificate does not prove that a fresh cold
start with today's solver will succeed. No such claim is made here.

The policy retains at most 48 immutable prescriptions, reserves at most two
uncertain requests within four total refinement claims, and ranks uncertainty by
ascending predicted deficit, then descending weighted gain and prescription hash.
It has no learned deficit cutoff and no special handling of ship IDs. Exact
prescriptions are deduplicated across proxy models. Claimed failures consume
budget. The held-out replay uses equal-score control replacements; those replay
queues cannot claim or launch jobs. Live candidates require positive weighted
gain estimates and a separate raw-mass fleet ship-count check.

Production additions include an explicit fixed-cargo refiner extracted from the
previously exercised v606/v611 wrapper, and a queue executor that accepts captured
summary bytes. It preserves one pass, exact epochs/cargo, native leg settings,
mass events, independent leg certification and the route master. Direct calls now
also check flight endpoint continuity before creating a runner, and leg telemetry
retains the actual certification backend. The caller must
supply full-fleet emission plus **both** final checkers. Request identity, Result
identity, exact returned cargo/events, full route certification, weighted score
improvement and raw ship-count admissibility are mandatory. Normal bundle
behavior is unchanged. See [HOOK_CONTRACT.md](HOOK_CONTRACT.md).

Evidence:

- [selection.json](selection.json), [preparation.json](preparation.json), and
  [input-provenance.json](input-provenance.json) bind the pre-outcome split,
  source, 23 original route summaries, historical Result and checker reports.
- [output/report.json](output/report.json), 54 per-request files,
  [counterexamples](output/counterexamples.json), and
  [fuel residuals](output/fuel-residuals.json) retain all outcomes.
- [readback-audit.json](readback-audit.json) verifies all 54 identities and 608
  saved mass/event bookkeeping comparisons without another model evaluation.
  `ordered_shortlists` records priority order; the raw report's `selected` lists
  are membership lists in input order.
- [validation/attempt-05](validation/attempt-05) records 76 focused CPU tests and
  clean Ruff checks, including the real `GpuCompletion.run` reconstruction loop
  with only its native evaluator stubbed. No native library was loaded.

The first saved-output audit incorrectly treated archived solver-reported fuel
as identical to independently propagated mass loss. Their difference is about
0.8–1.1 milligrams per route. The failed audit/script are retained; the completed
audit records both quantities and checks the sequential identity using certified
mass loss. No numerical gate or raw control result was changed.

`source/` is the frozen v621 host plus the original queue helper used for the 34
CPU completions. The final helper adds only a `retained()` getter for pruning
external envelope storage; the control outcomes were not rerun. Final production
integration files are separately pinned in the implementation snapshot. Root's
compact native-model work has its own build/GPU evidence and is not claimed as
part of this CPU control run.
