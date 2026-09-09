# Verified fleet frontier v629

**13,023.704900978004 weighted kg and 14,291.006160165 raw kg**, with 23 ships and 199 asteroids. Both original complete-fleet checkers pass on the exact top-level `Result.txt`, SHA `765cb7ef97d38926317dbbf4ae8700626dffec06c48af3d6f68dae47c154f3da`.

The gain over the preceding verified union is **23.880057595485 weighted kg** and **11.718001369200 raw kg**. Only ships 8 and 19 change; the other 21 Result sections are byte-identical. Weighted score and physical cargo mass are reported separately.

The finite local RTX 5090 batch refined four previously generated fixed-cargo route alternatives using the CUDA SCvx outer loop and QOCO. All four qualified: 72 flight solves, 286 outer iterations, 72 flight certificates and 6 wait certificates. Its 43.832-second worker includes 21.431 seconds for its CPU full-fleet checker. A separate CPU-only pair qualified the composition into the newer fleet in 22.804 seconds. These are unpaired measurements, not a speedup comparison. See `RESULTS.md` for route-level outcomes and the shared fleet raw-mass budget that admits a weighted-positive/raw-negative ship 8.

The prelaunch README inside `campaign.tar.gz` is historical preparation, including its original v628 baseline. The archived `ready.json` and all 82 indexed preparation files are preserved exactly. The final composition and these results supersede its prepared-only status; no old inputs were changed to make the run pass.

## Evidence layout

- Top-level Result and raw checker reports provide the final fleet, exact score, original violations/tolerances and binding hashes. The official display rounds mass; the independent report retains full precision.
- `campaign.tar.gz` preserves the exact four requests, archived seeds, orchestration/tests/attempts, all four outcomes, native pre-clamp and post-clamp arrays, every flight/wait certificate input/readback, one GPU selection and the first checker pair.
- `composition.tar.gz` preserves the newer-fleet section splice and its single fresh original checker pair, including the 21-section identity proof.
- `viewer.tar.gz` preserves the export scripts, source audits, exact saved-node NPZs, installed dataset and HTTP checks. Changed ships use saved native nodes and certificate endpoints, not independently integrated dense histories; three waits are explicit gaps with no inserted samples. Other 21 histories are reused exactly. The import also compares existing orbital-context samples against the pinned Kepler elements; it performs no new spacecraft propagation.
- `frozen-prerequisites.tar.gz` preserves the frozen host, prior helper code, original source identities, relevant baseline inputs, and the 388-file native source archive. Runtime binaries are represented by hashes in `external-runtime.json`, not copied here. The older published v799 generation archive remains a hash-bound external dependency for reproducing the complete candidate-pool inventory.
- `archive-manifest.json` indexes every archive member and its original workspace-relative path. `omitted-artifacts.json` records hashes of copied verifier executables and duplicate public catalogues omitted from the archives. All raw numerical readbacks are retained.

Run this package-only audit with ordinary Python; it uses no project runtime, GPU or numerical propagation:

```powershell
python verify_package.py
```

It verifies every indexed byte and archive member, preservation of the original 82 ready-file hashes, the final checker binding and exact retained/replaced ship sections. Archives retain workspace-relative paths for reconstruction in an isolated workspace. Numerical re-execution requires the pinned runtime/data and a new separately tracked run directory; the preserved launch markers intentionally prevent overwriting the original run.

The additive viewer dataset is `gtoc12-frontier-v629` in the existing multi-run visualiser:

```text
http://127.0.0.1:4173/?dataset=gtoc12-frontier-v629&ship=8&epoch=69807&preset=oblique&z=1
```

If its server needs starting from the repository root:

```powershell
cd results/lambda/2026-09-06/visualiser
node scripts/serve.mjs --port=4173
```

Dataset: `results/lambda/2026-09-06/visualiser/data/gtoc12-frontier-v629`. The app registration, gap-aware rendering and default selection are published separately by the parent task.
