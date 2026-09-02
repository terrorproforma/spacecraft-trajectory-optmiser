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
- `pd6_monte_carlo.py` - P1-D-MC: seeded Chari et al. dispersion, independent CPU batch;
- `low_thrust.py` - P1-E and TOPS Cartesian two-body transfers (fixed or free time);
- `tops.py`, `gtopx.py`, `gtoc.py` - suite ingestion, evaluator wrappers, reduced subsets;
- `report.py`, `cli.py` - report generation and the `spacepdhcg literature` command.

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
```

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
- **Not** implemented in the native C++/CUDA transcription or the device coefficient kernels
  (`cpp/include/spacepdhcg/transcription/powered_descent_6dof.hpp`,
  `cpp/cuda/src/device_scvx.cu`). Those remain fixed-final-time. Adding `sigma` there changes the
  frozen CSC topology (an extra decision column in every dynamics row, an extra objective term,
  a trust-region entry) and therefore the G4 policy hashes and the sealed evidence contracts of
  `integration/single-gpu-v1`; it was deliberately not attempted while the seal worker owned
  that worktree and the GPU. The published P1-D comparison is recorded as `descriptive-only` for
  the converged time of flight (figure-only in the paper) and `measured-local` for our value.

## GPU status during this freeze

The RTX 5090 was owned by a `device_scvx_integration_test --g4-session` process for the whole
window, so the pure-QOCO GPU leg of P1-C and any GPU batch for P1-D-MC were not executed
(`blocked`, exact reasons in the report). The P1-C GPU leg is one command once the device is free
(see above); the P1-D-MC GPU batch additionally needs a device entry point that accepts arbitrary
initial states for the 6-DoF fixture and the Szmuk/Chari thrust-arm dynamics.
