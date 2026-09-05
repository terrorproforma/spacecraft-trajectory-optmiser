# H100 deferred GPU validation sweep - triage (v2 candidate 3373988)

Host: Lambda H100 80GB HBM3 (sm_90), driver 580.105.08, CUDA 12.8, Ubuntu 22.04.5, Python 3.12.14.
Manifest: `benchmarks/gpu_deferred_validation_v2.json` / `docs/GPU_DEFERRED_VALIDATION_V2.md`.
QOCO: cuda-algebra 09f0495 + `qoco_absolute_kkt_stopping.patch` (`v1/build-current-head-qoco/libqoco.so`,
the same library the H100 G1-G3 reseal of v1 9e75b47 passed with).

Every item in this directory was executed for the first time on real GPU hardware: the manifest records that
both CUDA trees were "configured and built for sm_120 but never executed" on the RTX 5090 host because the G4
claim-core campaign owned the device. The "expected" lines in the manifest were therefore predictions.

## Verdicts

| item | verdict | root cause class |
|---|---|---|
| gate-gpu-preflight | PASS | - |
| planner-cuda-ctest-subset (release, debug) | PASS 10/10 each | - |
| gtoc12-gpu-lambert-parity (ctest + pytest) | PASS | - |
| planner-gpu-examples pd6, low_thrust | PASS (certified on GPU) | - |
| planner-gpu-examples hcw | FAIL exit 2 `converged_not_certified` | candidate numerics: independent replay defect 1.386e-6 (limit 1e-6), replay parity 1.386e-6 (limit 1e-9); device residuals all pass |
| planner-gpu-examples pd3 | FAIL exit 2 `converged_not_certified` | candidate numerics: canonical residual 2.52e-3 (limit 1e-6) after 14 QOCO numeric updates; objective 0.5522 vs 0.4927 quoted in the manifest |
| planner-gpu-pytest | FAIL 4/9 (hcw, pd3, pd6, pdhcg-selection) | hcw N=20 pure_qoco -> maximum_iterations; pd3 canonical_residual; pd6 N=10 **CPU reference** fails canonical_residual (hardware-independent); pdhcg backend on hcw -> not converged |
| planner-memcheck/initcheck/synccheck/racecheck hcw | FAIL exit 2 | sanitizer output is clean (`ERROR SUMMARY: 0 errors`, 0 bytes leaked); the exit code is the executable's not-certified status above |
| planner-memcheck pd3 (manifest command) | FAIL exit 255 | manifest defect: the raw example carries `maximum_tilt: 30.0` degrees; the executable expects radians (`'maximum_tilt' must lie strictly inside (0, pi/2)`). Re-run on the CLI-normalised `native-request.json` in `supplement/`: sanitizer clean, exit 2 (canonical_residual) |
| *-viewer-check (all four) | FAIL | candidate bug: 3d41a33/3373988 made `web/trajectory-viewer/scripts/check.mjs` read `gtoc12.js`, `webgl.js`, `kepler.js`, `camera.js`, but `_VIEWER_FILES` in `src/spacepdhcg/planner/viewer_export.py` still copies only index.html/app.js/math.js/styles.css/package.json/README.md. `tests/test_planner_viewer_export.py::test_viewer_check_accepts_the_export` fails on CPU too |
| literature-device-time-dilated-ctest (release, debug) + 4 sanitizers | FAIL | pd6_fft device A/B/z coefficient parity **0.980** (limit 2e-9) for substeps 1 and 4; sigma column 6.2e-7 / 4.0e-5 (limit 2e-9); reconstruction 1.4e-14 and quaternion tangency 1.4e-17 pass. pd3_fft parity 1.8e-15 / 2.4e-15 passes. Magnitudes from gdb (`supplement/pd6-parity-magnitude-gdb.log`). A 0.98 gap is not sm_90 rounding. Localisation still open: the chari pure-QOCO pd6_fft GPU batch (below) converges on 81/81 trajectories with independent host replay defects <= 9.9e-7, so either the test compares against a differently parameterised host linearisation, or the literature path does not exercise the same coefficient block |
| literature-gpu-run | PASS (exit 2 as expected) | acikmese-ploen-2007-pd3: GPU SCvx converged, 399.367 kg (published 399.5, CPU 399.361) -> reproduced. blackmore-2010-pd3-case1: GPU leg recorded `deferred` because the preflight saw the gpu-run process's own PID as a foreign holder (fix 5aabbfc). chari-2024-pd6-monte-carlo stays `gap` by design |
| literature-rerun-5aabbfc (blackmore + chari on the fixed HEAD) | PASS (exit 2 as expected) | blackmore-2010-pd3-case1: GPU SCvx converged, 398.845 kg (published 399.4, CPU 398.84) -> reproduced. chari-2024-pd6-monte-carlo: `pure_qoco_native_pd6_fft` **measured** for batch sizes 1/16/64: 81/81 converged and accepted, max replay defect 9.9e-7, max path violation 8.7e-7, ~26 s per trajectory serial (0.038 accepted trajectories/s; the CPU independent batch on the same host converged 0/81 and accepted 7/17 on sizes 1 and 16); `persistent_device_scvx` stays `blocked` by design. Tracked report twins rewritten (diff carried home as `v2-literature-report-twins.patch`, not committed) |
| full-cuda-ctest (release, debug) | FAIL 67/68 | only `device_time_dilated_test` fails (above) |

## Discriminators (`supplement/`)

* CPU reference on this host certifies all four examples (`cpu-plan-*-status.log`: `certified True`), so the
  hcw/pd3 GPU certification failures are properties of the `spacepdhcg_plan` device path (or of its
  interaction with the H100), not of the examples or the host toolchain.
* The v1 device SCvx driver on the same host and QOCO library passed G3 (`hcw_displaced` terminal residual
  2.9e-8, `pure_qoco_displaced` P1-C-pd3 canonical residual 1.3e-11), so the QOCO GPU IPM itself is healthy.
* Whether the hcw/pd3 planner results reproduce on the RTX 5090 is unknown: they have never run there.

## Source fixes made on the instance

* `5aabbfc fix(literature): GPU preflight must not count its own process as a foreign device holder`
  (environment-revealed: native-Linux nvidia-smi lists compute apps, WSL never does).

## Not fixed here (candidate defects to take back to the v2 branch)

1. pd6_fft device coefficient kernel (parity 0.98).
2. planner viewer export file list vs `check.mjs`.
3. pd3 GPU canonical residual / hcw GPU replay defect and hcw N=20 non-convergence; pd6 N=10 CPU reference
   not certified (the pytest fixture's short horizon).
4. Manifest `planner-memcheck` pd3 command should use a radians document (or go through the CLI).
