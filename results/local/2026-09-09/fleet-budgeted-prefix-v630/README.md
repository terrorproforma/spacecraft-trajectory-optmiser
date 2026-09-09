This finite local experiment produced **no verified fleet improvement**. The current fleet remains [frontier v629](../mass-budgeted-frontier-v629/README.md): 13,023.704900978004 weighted kg, 14,291.006160165 raw kg, 23 ships and 199 asteroids. None of the proposed 28.595948274286 weighted kg was added to that score.

Production admission now derives replacement budgets from the exact independently checked fleet ledger. It permits weighted-positive/raw-negative replacements when the fleet's original raw ship-count rule allows them. The shortlist retains distinct same-Earth depth and timing alternatives within `refine_top`; the optional cluster replacement path uses fixed cargo and epochs. CPU tests also cover the bounded CompletionRecorder/uncertainty queue. That queue remains weighted-only and every eventual promotion still requires both full-fleet checkers.

The mission test admitted three real saved generated requests through this production policy and ran production `refine_fixed` with the corrected endpoint merit and CUDA boundary ephemerides. These were new tests of historically failed prefixes under the corrected solver, not newly generated geometry. Each first leg used the exact current archived Earth control seed; subsequent flown legs used the existing cold initializer.

| Candidate | Native legs attempted | Native outer iterations | First failed leg, zero-based | Remaining maximum defect |
|---|---:|---:|---|---:|
| Ship 22, v794 rank 0 | 2 of 16 | 14 | 1: 9443 → 32613, MJD 65093 → 65273 | 0.0021356523 |
| Ship 10, v794 rank 0 | 18 of 18 | 98 | 17: 17126 → Earth, MJD 69263 → 69713 | 0.0027817927 |
| Ship 19, v802 rank 0 | 4 of 16 | 45 | 3: 51417 → 22760, MJD 65498 → 65648 | 0.0014960263 |

All 24 native calls returned. There were 21 accepted flight certificates, one accepted GPU coast certificate, three CUDA boundary batches covering 56 requested states, and two residual CPU `body_state` callback calls. The worker took 20.405814766 seconds. There was no fleet selection, full-fleet checker call, emitted replacement, or incumbent promotion because all three routes failed before route qualification. The solver backend was native CUDA SCvx with QOCO conic subproblems. These are unpaired workload measurements, not a speedup claim or solutions-per-second benchmark.

The failed transcriptions satisfy their arrival-position node constraints to 0–1.60 × 10⁻⁹ km of numerical residual (at most about 1.6 micrometres), while dynamics and virtual-control defects remain approximately 0.0015–0.0028. The failed legs had 11/13, 14/33 and 17/36 qualified conic reports respectively, with unqualified attempts also retained. The blocker is therefore not simply that every conic solve failed or that endpoint constraints were absent. Native status strings include `failed` and `infeasible`; they are optimizer outcomes, **not mathematical infeasibility certificates**. No failed returned state was passed off as a trajectory certificate. A distinct next diagnostic would isolate the internal dynamics/virtual-control stationary point and conditioning, retaining the current physics gates, rather than repeat these three requests unchanged.

The supervisor exited normally with no surviving owned descendants or remaining GPU compute processes. Original limits were three routes, 50 native calls, 2,200 outer iterations (40 plus four polish per leg), 50 flight certificates, three waits and one final checker pair. Every failed route, raw pre-clamp state/control array, post-clamp array, native report and certificate readback is retained. GPU ephemeris prerequisite tests and archived representation checks passed independently before launch.

The package is lossless and contains no compiled libraries or verifier executables:

- `preparation.tar.gz` retains the exact non-host preparation files, ready map and stopped alias-mismatch attempt. Its README describes the historical prelaunch state.
- `native-source.tar.gz` and `native-manifest.json` retain the exact 741-file compiled source tree, including the 192 host source files. The two changed production policy modules are in `policy/` and reconstruct the actual frozen host overlays.
- `outputs.tar.gz` retains all raw mission outputs; `report.json` and `launch-report.json` are convenient exact copies.
- `audit/` contains a saved-data-only check of identities, sequential certified masses, counts, raw arrays and first failures. It performed no propagation or numerical solve.
- `policy-validation/` records 116 passing CPU integration tests and clean Ruff; the prepared harness has 24 passing CPU orchestration tests with complete subprocess logs.

Verify package hashes and reconstruct every one of the 237 original prepared file identities without extracting or executing the mission:

```powershell
python .\results\local\2026-09-09\fleet-budgeted-prefix-v630\verify_package.py
```

`sha256.json` binds all package files. The compiled core and QOCO runtime hashes remain in the manifest/profile for a separately authorized rebuild or run. No new visualizer dataset was created because no new fleet qualified; the [current v629 display](http://127.0.0.1:4173/?dataset=gtoc12-frontier-v629&ship=8&epoch=69807&preset=oblique&z=1) is retained.
