# CPU incumbent-admission correction, v619

The finishing bridge now preserves the DP's configured hop-cost model, prices it and the generic Earth return at the actual sequential mass, and records the inflation actually spent in returned `PlannedLeg` objects. `_plan_from_tour` no longer computes redundant guessed-mass estimates before that pass. A certified return-cell override retains its measured inflation. Existing authority, mining-stay, cargo and final mass gates remain unchanged. This is a CPU bridge correction; it makes no new claim of a fully GPU-native pipeline or a measured GPU speedup.

The preselected controls are target ship 23 and ships 1, 4, 7 and 10, each with independent inventory and cargo matching the retained Result. Selection was frozen before replay. Each uses its exact archived schedule, asteroid footprint, saved Lambert DVs and cargo. The two prefix cases use either the unchanged flat deployment proxy or the measured deployment-prefix mass. The two cost configurations use the v616 no-fit settings or the existing, unmodified `hop_inflation_fit.json`. These are deterministic controls, not asserted statistical holdouts from the historical fit; its original training/holdout source metadata is in `selection.json`.

The matched comparison performs 40 CPU finishing-bridge calls: five ships × two prefix masses × two fixed models × old/new implementation. It performs no candidate generation, DP solve, Lambert solve, refinement, physics certification, fleet emission or promotion. Native library loading is blocked during both replays. Geometry for the existing fit is calculated from the pinned catalogue at the saved departure epochs.

| Ship 23 case | Baseline final cargo margin, kg | Corrected margin, kg |
| --- | ---: | ---: |
| Flat deployment proxy, no fit | -49.003614 | -23.796634 |
| Measured deployment prefix, no fit | -35.563593 | -9.587670 |
| Flat deployment proxy, existing fit | -49.003614 | -19.178123 |
| Measured deployment prefix, existing fit | -35.563593 | -6.385482 |

Baseline accepts 0/20 cases; the corrected bridge accepts 2/20. Both accepted cases are ship 4 with a flat-proxy deployment prefix, one per cost model. Every measured-prefix case still fails. All 40 controls pass all individual authority checks; every rejection is the final dry-plus-cargo proxy gate. The two scalar admissions are not new certified trajectories. All routes retain their original cargo, and the verified fleet score is unchanged. The five archived route certificates were not recomputed.

The all-23 independent return-component calculation also agrees with the root audit: builder-style guessed mass exceeds measured return departure mass by a median 707.493615 kg. Holding the actual burn mass fixed isolates a median 22.811826 kg excess predicted return fuel from that stale mass argument alone (maximum 33.180272 kg). The generic return model evaluated at actual mass still overpredicts 20/23 archived returns, with median error +18.364126 kg. These component numbers are not a full sequential route replay or an infeasibility proof.

This correction is therefore necessary for internally consistent pricing, but insufficient to reproduce every retained incumbent. It does not justify loosening gates, lowering cargo or refitting to these five routes. The next admission investigation should separate remaining sequential model error from generation-grid/pruning losses before spending another generation budget.

## Evidence

- `selection.json`: prespecified routes, hashes, model/prefix arms and historical fit metadata.
- `source-sha256.json`: exact 190-file f8b2ac7a baseline and the initial candidate overlay; `source/candidate` was frozen but never replayed.
- `final-source-sha256.json`: exact 190-file final source, differing from the baseline only in `search.py`.
- `final-source-provenance.json`: final code/test hashes, script lineage and the removal of redundant builder pricing after review.
- `output/baseline.json`, `output/candidate-final.json`: all control outcomes and complete per-flight mass, authority, spent inflation and propellant traces, including failures.
- `output/audit.json`: paired comparison and independent all-23 component recomputation with every original input hash checked.
- `validation-final-02/report.json`: 45 tests passed in 28.44 seconds; Ruff check and format check passed. The focused file has 11 behavioral cases, including once-per-flight evaluation, configured-model separation, immutable input legs, certified-cell override retention, and unchanged rejection gates.
- `validation/report.json`: retained first broad pass, 43 passed and two harness/baseline failures. The f8 test fixture lacked `_resident_table`; the current committed three-line fixture correction is recorded in `validation-final-02/committed-fixture.diff`. The original blanket CDLL block also rejected the heap test's `libc.so.6`; final validation allows only that exact library name and rejects every GPU/solver library. No production code changed between these two passes.

The source and data lineage derives from v616. Its GPU library/profile metadata is retained only as provenance and was not used in this experiment. The fit's checked-out raw hash differs from the frozen source copy because of line endings; decoded JSON equality was required before use. No coefficients changed. The first draft of the independent component audit used an additive correction; comparison caught the mismatch before report emission and it was corrected to the frozen model's multiplicative, clamped formula. No route replay was repeated.

The final pinned CPU suite can be repeated without changing stored outputs:

```powershell
wsl -d Ubuntu-22.04 -- env PYTHONDONTWRITEBYTECODE=1 /home/angus/worktrees/spacepdhcg-literature-venv/bin/python -B build/performance/incumbent-admission-v619/validate_final.py --tests
```

The replay scripts create their reports exclusively and refuse to overwrite the retained results. `evidence-manifest.json` indexes every evidence file; its own hash is reported separately.
