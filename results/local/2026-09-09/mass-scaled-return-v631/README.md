The archived return control passed; the fixed-cargo candidate return failed. **No score improvement or new fleet Result was produced.** The retained fleet remains 13,023.704900978 weighted kg and 14,291.006160165 raw kg, with 23 ships and 199 asteroids.

This was one bounded local RTX 5090 run using the native CUDA SCvx/QOCO path. It made two return solves, 24 total SCvx iterations and one fresh GPU flight-certification call in 10.787373 seconds of worker time. No new prefix/wait propagation, Lambert search, whole-route rerun or full-fleet checker call occurred. There is no overall speedup claim.

| Outcome | Original-mass control | Candidate-mass return |
|---|---:|---:|
| Initial mass, kg | 1472.1728669809 | 1438.8883038351994 |
| SCvx iterations / accepted | 2 / 2 | 22 / 0 |
| Native seconds | 0.719377 | 9.093305 |
| Max normalized dynamics defect | 1.9984e-13 | 2.1322e-13 |
| Fresh independent certificate | Passed | None |

The candidate retained its scaled seed. Its unverified final node mass was 1166.9174106318044 kg, 10.2899815653 kg below the fixed 1177.2073921971253 kg requirement. All four qualified conic trial steps were rejected. Native `infeasible` / `virtual control remains inf` are retained optimizer diagnostics, not a mathematical infeasibility certificate. The original control's independent final mass was 1193.9107054861288 kg, with a 0.637092 km position error. Raw native controls and normal post-pipeline controls are both retained; the failed candidate was returned before clamping or certification.

Both calls uploaded the exact same archived initial-state and thrust bytes. The new CUDA initializer performed mass/thrust scaling before rollout; the candidate factor was 0.9773908595299954, and the control factor was 1. Each trajectory payload was 5,480 bytes, excluding the time vector and other transfers. The source is the already verified ship 10 return from asteroid 17126 to Earth, MJD 69263–69713, with 226 nodes, 106 two-day burns and 119 coast intervals. No interpolation or retiming was used.

The frozen preparation contains 25 passing CPU construction/emission tests and its earlier retained failures. The native build's 92 passing CPU tests, 27 intentionally skipped GPU cases, exact sources and original/scaled/controller/memcheck logs are retained. Unit validation and the mission experiment are separate measurements.

Files are preserved as follows:

- `frozen-kit.tar.gz`: all 245 sealed preparation files plus the original ready manifest, including the exact 192-file Python host. Its README describes the prelaunch state.
- `external-evidence.tar.gz`: all 216 bound external inputs, including the entire saved v630 candidate prefix and failed cold return, baseline fleet/checker binding, and native source archives/manifests. `external-map.json` maps original relative dependencies to their archived repository paths. This permits assessment of the reused certificates without relying on mutable workspace files.
- `raw-output.tar.gz`: all 31 actual output files, including both seed payloads, raw/post-pipeline solutions, every conic report and the one independent certificate readback.
- `native-validation.tar.gz`: the completed native unit/memcheck logs and supervisor evidence. No compiled library or executable is included.
- `saved-audit.json`: saved-only input, mass, array and counter comparisons. `audit_saved.py` records that audit implementation; it requires NumPy and the original kit layout.
- `sha256.json` and `verify_package.py`: portable byte/manifest/source/counter verification. Every archive also has its member hash/size index.

From this directory, run `python verify_package.py`. This uses only Python's standard library and makes no solver, verifier, propagation or GPU call. It verifies the source archives against their build manifests and binds every prepared input and raw output. It is not a fresh physical certification.

The current verified fleet remains [mass-budgeted-frontier-v629](../mass-budgeted-frontier-v629/README.md). The [implementation and interpretation note](../../../../docs/GPU_MASS_SCALED_INITIALIZATION.md) explains the additive API and remaining mass-admission issue. A successful initialization does not establish that the candidate is feasible, and no incumbent was overwritten.
