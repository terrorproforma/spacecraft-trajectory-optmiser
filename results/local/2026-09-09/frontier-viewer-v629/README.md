# Frontier display verification

The current dataset is `gtoc12-frontier-v629`, binding Result
`765cb7ef97d38926317dbbf4ae8700626dffec06c48af3d6f68dae47c154f3da`.
The existing HTTP server returned the exact local bytes of all ten checked
assets, including the gap-aware renderer and current fleet manifest. The
browser accessibility state confirms the selected fleet and active WebGL2;
this is not a pixel-screenshot review.

The gap/import/geometry test selection passed 16 tests with zero failures.
One historical import fixture was skipped because its source export was not
available. The installed-dataset schema check passed. An independent read-only
review approved all marked gaps, saved sample labels, 142 source/data hashes,
the unchanged 21 histories and both checker bindings.

`http-and-validation.json` binds the actual source bytes and separates the
HTTP checks from preceding test and browser observations. `verify_http.py` is
the exact historical checker copied from `build/performance`; it refuses to
overwrite the original output directory. No new solver, numerical propagation
or GPU work was performed by this display verification.

The renderer and tests are committed alongside this package under
`results/lambda/2026-09-06/visualiser`. Mission exports and exact sample provenance
are in the [mission package](../mass-budgeted-frontier-v629/README.md).
[Full result path and copy-paste loading instructions](../../../../docs/GPU_MASS_BUDGETED_FRONTIER.md).
