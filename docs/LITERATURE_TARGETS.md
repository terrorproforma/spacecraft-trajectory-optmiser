# Literature reproduction track (campaign Phase 0-1)

Status date: **2026-09-03**. Branch: `feat/literature-targets`.

This track implements Phase 0 (freeze inputs) and Phase 1 (reference reproduction) of
`docs/COMPARATIVE_SOLVER_CAMPAIGN.md` as runnable, tested targets. It is a separate track from
the `single-gpu-v1` gate sequence: it produces reference-reproduction evidence and pinned inputs,
not G4/G5 performance claims.

## Artefacts

| Artefact | Purpose |
| --- | --- |
| `benchmarks/literature/targets.json` | machine-readable target registry (id, family, runner, profile, support, expected labels) |
| `benchmarks/literature/profiles/*.json` | frozen literature profiles (constants, boundary conditions, published values, envelopes) |
| `benchmarks/literature/provenance.json` | provenance record for every literature value (schema `experiments/schema/literature_provenance.schema.json`) |
| `benchmarks/literature/external_sources.json` | pinned external downloads (URL, SHA-256, size, licence, revision) |
| `benchmarks/literature/tops_selection.json` | frozen TOPS metadata selection at revision `24fe8849` |
| `benchmarks/literature/gtoc_reduced_subsets.json` | frozen GTOC5/9/12 reduced subsets (metadata rules only) |
| `benchmarks/literature/chari_2024_initial_positions.json` | committed seeded Monte Carlo samples |
| `benchmarks/literature/reference_reproduction.json` | machine-readable reproduction report |
| `docs/REFERENCE_REPRODUCTION_REPORT.md` | rendered reproduction report |
| `results/literature/<target>.json` | per-target details (iteration histories, per-sample rows) |

Code lives in `src/spacepdhcg/literature/`:

- `provenance.py` - schema validation plus semantic rules (digit preservation, unique ids, label
  vocabulary, descriptive-only reasons, profile cross-references);
- `pinned_values.py` and `scripts/literature/build_provenance.py` - curated records and the
  deterministic regeneration of the store (a test freezes the committed file);
- `external_sources.py` - cached, checksum-verified fetches (`SPACEPDHCG_LITERATURE_CACHE`,
  downloads only with `SPACEPDHCG_LITERATURE_ONLINE=1` or `spacepdhcg literature fetch`);
- `scvx_core.py` - independent CPU successive convexification with **free final time** (time
  dilation `sigma`, FOH control, RK4 state-transition/forced-response integration, virtual
  control, quadratic soft trust regions) used as a `measured-local` reference;
- `pd3_acikmese_ploen.py` - P1-C: lossless-convexification SOCP, repository CPU SCvx, optional
  pure-QOCO GPU SCvx, independent nonlinear replay;
- `pd6_szmuk_2018.py` - P1-D: Szmuk-Acikmese 2018 vehicle model (thrust-arm torque, gimbal,
  tilt, glide slope, rate limits) with analytic Jacobians and free final time;
- `pd6_szmuk_2018_native.py` - P1-D through the native `pd6_fft` free-final-time transcription
  (`src/spacepdhcg/native/free_time.py` is the ctypes bridge plus the outer loop);
- `pd6_monte_carlo.py` - P1-D-MC: seeded Chari et al. dispersion, independent CPU batch, and a
  deferred pure-QOCO GPU batch through `pd6_fft` (`spacepdhcg literature gpu-run`);
- `gpu_preflight.py` - refuses GPU legs while a `--g4-session`/`--g4-server` process owns the
  device (`spacepdhcg literature gpu-preflight`);
- `low_thrust.py` - P1-E and TOPS Cartesian two-body transfers (fixed or free time);
- `low_thrust_mee.py` - modified-equinoctial-element SCvx with revolution bookkeeping, spiral
  seeded guess, trust-weight continuation and Cartesian replay (Dionysus, TOPS P3/P1);
- `tops.py`, `gtopx.py`, `gtoc.py` - suite ingestion, evaluator wrappers, reduced subsets;
- `report.py`, `cli.py` - report generation and the `spacepdhcg literature` command.

The frozen inputs (`benchmarks/literature/targets.json`, `provenance.json`,
`external_sources.json`, the profiles, `chari_2024_initial_positions.json`,
`benchmarks/literature_baselines.json`, `experiments/schema/literature_provenance.schema.json`)
are located through `spacepdhcg.resources`: `$SPACEPDHCG_BENCHMARKS_DIR` when set, else the source
checkout, else the byte-identical copies packaged in the wheel (`spacepdhcg/_data/`), so
`spacepdhcg literature list/status/provenance/run` work from an installed package.  Generated
outputs (`reference_reproduction.json`, `docs/REFERENCE_REPRODUCTION_REPORT.md`,
`results/literature/`) are written below the checkout, or below the working directory when the
package is installed.

## Commands

```bash
python -m pip install -e '.[dev]'
export SPACEPDHCG_LITERATURE_CACHE=$HOME/.cache/spacepdhcg-literature   # optional
spacepdhcg literature list
spacepdhcg literature fetch                 # download + verify every pinned artifact
spacepdhcg literature status
spacepdhcg literature run acikmese-ploen-2007-pd3
spacepdhcg literature run all
spacepdhcg literature provenance
python scripts/literature/build_provenance.py --check
SPACEPDHCG_QOCO_LIBRARY=/path/to/libqoco.so spacepdhcg literature run acikmese-ploen-2007-pd3
spacepdhcg literature gpu-preflight          # exit 0 only when no G4 process owns the device
spacepdhcg literature gpu-run acikmese-ploen-2007-pd3 chari-2024-pd6-monte-carlo  # deferred legs
```

`gpu-run` exits with code 3 and touches nothing when the preflight refuses (a
`device_scvx_integration_test --g4-session|--g4-server` process, or any other compute process
unless `--allow-shared`, is on the device).

Tests: `pytest tests/test_literature_*.py`. Network-dependent tests skip with an explicit reason
when the pinned artifact is not cached.

## Planner mapping

The planner CLI branch (`feat/planner-cli`) had no problem-schema commits when this track was
frozen. Profiles are therefore plain JSON under `benchmarks/literature/profiles/` with a
documented per-family `parameters` block, and every runner exposes a
`profile_from_document()`/`problem_from_document()`/`parameters_from_document()` adapter. When
`spacepdhcg plan --literature <id>` lands it should call the registry (`load_target_registry()`),
load the profile, and hand the adapter output to its own problem object. The umbrella command
`spacepdhcg` (`src/spacepdhcg/cli.py`) exposes `register()` for the planner sub-command.

## Free-final-time status

- Implemented in the independent CPU core (`scvx_core.py`) for the 6-DoF and low-thrust
  families; validated by discretisation tests (matrix-exponential agreement on a linear system,
  sigma-column sensitivity against finite differences) and by the Szmuk 2018 reproduction.
- Implemented natively as **new topologies** `pd3_fft` and `pd6_fft`
  (`cpp/include/spacepdhcg/transcription/powered_descent_{3dof,6dof}_free_time.hpp`, shared
  helpers in `free_time_common.hpp`, sigma-augmented variational RK4 in
  `time_dilated_flow_linearisation.hpp`, C API `spacepdhcg_pd3_fft_*`/`spacepdhcg_pd6_fft_*`).
  The frozen fixed-time headers, the P1-C/P1-D CSC fingerprints and the G4 policy hashes are
  untouched. Decision vector `[states | controls | sigma | virtual | epigraphs]`; every dynamics
  row carries a sigma column `S_k = d x_{k+1} / d sigma`; the 6-DoF variant applies the
  unit-quaternion tangent projection to `A_k`, `B_k` and `S_k`, and encodes the Szmuk thrust-arm
  torque, linearised thrust-magnitude bound, gimbal, rate, glide-slope and attitude-tilt cones.
- CPU validation (Release and Debug `-Werror`, 47/47 ctest): affine reconstruction of the RK4 map
  through the CQP rows, `A`/`B`/`S` columns against central differences, quaternion tangency,
  boundary-mask row vacuity; Python (`tests/test_native_free_time.py`) repeats the oracle checks
  through the ctypes bridge and runs the outer loop on Szmuk 2018: native `pd6_fft`
  `t_f = 3.39254 UT` against the FOH Python core's `3.39008 UT` (gap 0.0025 UT, declared
  ZOH-vs-FOH envelope 0.01 UT). Without the tilt cone the native optimum is a different, shorter
  problem (`t_f ~ 2.97 UT`), which is recorded as evidence that the bound is active.
- CUDA kernels (`time_dilated_variational_kernel`, `fill_time_dilated_sigma_csc_kernel`, C API
  `spacepdhcg_cuda_time_dilated_variational_rk4_async` /
  `spacepdhcg_cuda_fill_time_dilated_csc_async`) compile for `sm_120` with nvcc 12.8.
  `cpp/cuda/tests/device_time_dilated_test.cu` (CPU/GPU coefficient parity for `pd3_fft` and
  `pd6_fft` at 1/2/4 substeps, sigma-column finite differences, quaternion tangency, CSC sigma
  fill) is **deferred**: it must not run while the G4 measured campaign owns the device.
  Deferred GPU commands, to be run serially once `spacepdhcg literature gpu-preflight` passes:
  `ctest -R device_time_dilated_test` in the CUDA build, one
  `compute-sanitizer --tool memcheck` / `racecheck` pass over the same executable, and the two
  literature legs below.
- The published P1-D comparison stays `descriptive-only` for the converged time of flight
  (figure-only in the paper) and `measured-local` for both of our values.

## GPU status

The RTX 5090 has been owned by `device_scvx_integration_test --g4-session` / `--g4-server`
processes for every window of this track, so no GPU workload was launched. The pure-QOCO GPU leg
of P1-C and the pure-QOCO `pd6_fft` batch for P1-D-MC are now **deferred** rather than blocked:
`spacepdhcg literature gpu-run <target>` runs the preflight, refuses with exit code 3 while a G4
process owns the device, and otherwise executes the leg and updates the report. The persistent
device SCvx batch for P1-D-MC stays `blocked` (the frozen G4 fixture families have no entry point
for arbitrary initial states or the Szmuk/Chari thrust-arm model).
