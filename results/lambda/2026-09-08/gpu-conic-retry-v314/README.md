# GPU conic-retry investigation — 8 September 2026

Development candidate; not merged into main. The full comparison passes in both
modes, so a reduced campaign failure rate has not been established. The underlying
identical-QP qualification variability remains under investigation.

The CUDA controller permits two cold retries of an unqualified inaccurate QP at
the unchanged trust region. Every attempt consumes the existing budget; acceptance
still requires the original conic and nonlinear physics gates. No CPU numerical
fallback or new trajectory transfer is introduced.

## Results

- 24/24 exact departure replays pass independent physics certification on each
  GPU. `history-audit.json` checks all accepted QPs against their requested gate
  and confirms unchanged trust values after each of the 62 recorded retries.
- Both GPUs pass 20 GPU SCvx tests, controller/graph/deadline/input-guard tests,
  and four sanitizer checks of the controller with zero errors/hazards.
- All eight complete H100 campaigns return 548.255 kg and pass both checkers.
  Four runs per controller give medians of 60.362 seconds for the previous
  controller and 60.656 seconds for the retry controller: about 0.5% slower.
  These results establish neither a speedup nor a failure-rate improvement.
- Fleet incumbent remains 12,805.194 weighted kg. These one-ship performance
  replays are not new fleet incumbents or official leaderboard submissions.

`comparison/summary.json` contains complete-run timings and physical residuals.
`v316` through `v323` contain complete outputs, exact commands and raw logs.
`diagnostics/v307` contains eight captured QPs. v308/v309/v311/v312 retain the
regularisation, refinement, Ruiz and objective-rescaling sweeps. v324 varies
primal/equality/cone regularisation separately; none of its eight settings passes
all 16 original-QP audits. Their compact
samples include first/last full primal-dual vectors; SHA-256 and absolute Lambda
paths identify the full vector logs retained remotely. Reported sweep pass counts
come from the independent long-double original-equation audit, not solver status.
v309's `baseline` uses an exploratory 1e-9 internal stopping tolerance; production
uses 1e-11. The external qualification gate remains 1e-9 in every diagnostic.

`rtx5090/rejected-precise-ir-v310` preserves a separate compensated-residual
prototype that failed all 32 candidate replays; it is not in the code candidate.
The initial prototype configuration failure is retained as build evidence, not
a numerical solver result.

`diagnostics/v314/source` identifies the frozen H100 implementation;
`final-source` includes the final Python rejection-label correction. The four
native source files are identical. Frozen histories can label a physically
rejected qualified candidate as `unqualified`; use each solver report's
`qualified` field. The final formatter distinguishes this case correctly.

Both downloaded archives are byte-hash verified before extraction. `sha256.json`
pins all files except itself, and `final-source-sha256.json` pins normalized-LF
source contents. Frozen binary hashes and build commands are in the per-GPU
manifests and reports. `helpers` contains the launch, retrieval and audit recipes.
