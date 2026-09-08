# Consistent completion costs, v619

The correction preserves the configured DP hop-cost model, prices every collection/return flight once at its actual sequential mass, retains certified return-cell overrides, and records the inflation actually spent. Authority, mining-stay, cargo and final mass gates are unchanged. This is a CPU finishing-bridge correction, with no measured GPU speedup or fleet score gain.

These are historical positive controls pinned to the v595/v616 incumbent, not the newer fleet published in commit 2ccb93c1. The comparison does not describe the current fleet's score. The matched comparison has 40 CPU bridge calls on preselected ships 23, 1, 4, 7 and 10: old/new source × flat/existing-fit model × flat-proxy/measured deployment prefix. Baseline admits 0/20 cases and corrected source admits 2/20, both ship 4 with an optimistic flat deployment prefix. Every measured-prefix case still fails the final proxy mass gate. For ship 23, measured-prefix margin improves from −35.563593 to −9.587670 kg without the fit, and to −6.385482 kg with the unchanged existing fit. A negative proxy margin is not physical infeasibility: the historical ship 23 trajectory has an archived positive dry-mass margin of 2.861384 kg.

The independent all-23 return-component check confirms median guessed-mass excess 707.493615 kg and median return-fuel overprediction 22.811826 kg from that mass argument alone. Even at actual mass the generic return model overpredicts 20/23 returns, median 18.364126 kg. These component diagnostics are not new trajectory certificates. The 45 CPU regression tests and Ruff pass; the first 43-pass/two-harness-failure run is preserved and explained in the detailed kit README. No tolerances, epochs, cargo, fitting coefficients or incumbent fleet changed.

## Contents and integrity

`kit/README.md` gives the full interpretation in the historical experiment's context. `kit/output/` retains every baseline/final control and failure trace plus the independent audit. `root-audit/` contains the original 23-control diagnostic, its executed script and the exact source bytes it read. All 23 route inputs are included in `inputs.tar.gz`. Archived certification flags and the pre-existing inventory-to-Result binding are retained; this package does not rerun the official/internal physics verifiers, and it does not contain trajectory files sufficient to claim a fresh full-fleet certificate.

The original 612 indexed kit files reconstruct byte-for-byte: the 190-file f8b2ac7a source is stored once in `source-base.tar.gz`, and the initial/final `search.py` overlays are in `source-overlays.tar.gz`. The initial overlay was never executed. `reconstruction.json` maps the 612 entries and all 26 original root-diagnostic files to these archives/raw files. The final production source hash is 9c5d64e37ceea7e5fd8c680556cd41bfbbb1e5a282aacba5d3ede0cbe0797d18. The complete raw kit manifest and all superseded failure evidence are retained. No binaries, credentials, caches or new numerical run outputs were added.

## Package-only use

Python 3.12 is the recorded runtime. `verify`, `extract` and `audit` need only Python's standard library and this package. `bridges` requires NumPy and the existing pinned catalogue/bonus data; `tests` additionally needs pytest/scipy and their dependencies. No command downloads data or loads a GPU/solver library. The test harness allows only libc.so.6 for heap accounting. Reproduction adapters relocate input/data paths in memory; archived files and source hashes stay unchanged.

```sh
python -B reproduce.py verify
python -B reproduce.py extract --work /tmp/consistent-completion-v619-extracted
python -B reproduce.py audit --work /tmp/consistent-completion-v619-audit
python -B reproduce.py bridges --work /tmp/consistent-completion-v619-bridges --data /path/to/pinned/gtoc12/data
python -B reproduce.py tests --work /tmp/consistent-completion-v619-tests --data /path/to/pinned/gtoc12/data
```

Every work directory must be new. `bridges` deliberately repeats the 40 CPU bridge evaluations, then audits them; `audit` recomputes only the saved-output/component checks; `tests` repeats the 45 CPU tests. Packaging validated archive reconstruction only and did not execute these numerical replays again. Runtime dependencies/data are external; the geometry-free saved-output audit is fully self-contained.

## Next bounded proposal, not launched

The next milestone is same-pool native completion/parity and systematic residual-cost attribution, before wider generation or another timing search. Use exactly these five preselected schedules, two prefix masses and two frozen model configurations: 20 native forward evaluations with saved DVs, compared against the 20 corrected CPU outcomes at every leg. There should be zero generation, Lambert search or cargo changes. The current native joint API uses retimer flat/ratio/return models and per-pair calibration; it does not implement the DP five-feature fit used by the −6.39 kg CPU case. A small native finishing adapter must therefore preserve that fit and certified-cell provenance before claiming parity. Silently replacing it with the retimer model would invalidate this comparison.

In parallel, decompose residual error at each archived measured burn mass across all 23 historical controls, separating deployment, collection and return costs without refitting. The saved final 27463→30805 collect hop on ship 23 has 84.296016 kg sequential proxy fuel versus 76.686179 kg archived measured fuel, at masses 1370.877223 versus 1369.983590 kg. That identifies a useful residual to investigate; it is not a guaranteed saving or proof that this one leg causes the entire route deficit. If new solver truth is needed, permit at most two fixed-epoch native leg refinements for that hop: an archived-mass positive control and the exact sequential-proxy mass case, with a hard 120-second budget and unchanged certification gates. A failed positive control stops interpretation of the probe.

Do not run wider generation until measured-prefix incumbent admission is resolved under an explicit validated cost model. Any resulting route must still undergo the complete low-thrust refinement and both final fleet checkers before promotion. This is a proposal only; the 20 native parity evaluations, residual analysis and two leg refinements have not been executed or prepared as a runnable experiment here.
