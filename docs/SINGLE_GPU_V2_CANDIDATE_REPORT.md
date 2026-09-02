# `integration/single-gpu-v2-candidate` - consolidation report

Worktree `/home/angus/worktrees/spacepdhcg-single-gpu-v2`, created 2026-09-03 from
`integration/single-gpu-v1` at **63271d5** (the branch tip at that moment; it contains the requested
9678134 plus the five G4 campaign commits 2e34d30 → 63271d5, so promotion stays a fast-forward). The G4
claim-core worktree/branch (`spacepdhcg-single-gpu-integration`, capability `e546583b…`, source
9a4cbea) was never checked out, committed to or modified; the GPU was never used; the lock file
`/home/angus/.spacepdhcg-gpu.lock` was never created.

## Integrated commits

| Branch | Tip used | Commits (base → tip) | Merge commit |
| --- | --- | --- | --- |
| `feat/planner-cli` (base b6afb49) | c74fdb7 | 27569ad, c74fdb7 | 2e7548b |
| `feat/literature-targets` (base b6afb49) | f6e8140 | 715c1db, 196b3ba, 4f92133, 2d1c52f, 9dd243b, d81c528, 57cee5c, 1fa99ae, 8e18b93, f6e8140 | da0a96b |
| `feat/gtoc12-asteroid-mining` (base 9678134) | **fa91b43** (tip at start; the worktree carried uncommitted `clusters.py`/`retiming.py` work that was left alone) | 20d999b, 38a9fd7, 550b3ec, c22149f, 9a8ec44, 4ab97d3, ebe26f7, accc5df, cd92871, 5a06d98, 6a35a00, fa91b43 | 2f4be21 |

`git cherry HEAD <tip>` is empty for all three tips and `git merge-base --is-ancestor` holds for
`integration/single-gpu-v1`, both branch tips and fa91b43.

### Dedupes

- `benchmarks/literature_baselines.json` and `docs/COMPARATIVE_SOLVER_CAMPAIGN.md`: the user's spec
  was imported twice (715c1db on literature, 20d999b on gtoc12) with byte-identical content; git
  merged them clean, one copy remains. Patch-ids differ only because 715c1db also carries the matrix,
  protocol, outline and manifest-test updates and 20d999b the `.gitignore`/`pyproject` lines.
- `src/spacepdhcg/__main__.py`: added on the literature merge with the exact gtoc12 content so the
  gtoc12 merge saw no conflict.
- `pyproject.toml`: three definitions of the `spacepdhcg` console script collapsed into one
  (`spacepdhcg.cli:main`).

## Conflicts and resolutions

| File | Sides | Resolution |
| --- | --- | --- |
| `.cursor/memory/AGENT_SCRATCHPAD.md`, `DEVLOG.md` | all three merges | additive: both sides' entries kept in order (script kept outside the repo) |
| `cpp/include/spacepdhcg/c_api.h`, `cpp/src/c_api.cpp` | planner ↔ literature | both appended independent ABI blocks at the same anchors (planner transcription ABI; free-final-time pd3_fft/pd6_fft ABI); three-way union keeps both in full, deduplicated the shared `<memory>`/`<vector>` includes; `g++ -std=c++20 -fsyntax-only -Werror` clean, then all host builds |
| `pyproject.toml` | planner (`planner.cli:main`) ↔ literature (`cli:main`) ↔ gtoc12 (`cli:main`, duplicate key) | single `spacepdhcg = "spacepdhcg.cli:main"` |
| `src/spacepdhcg/cli.py` | literature ↔ gtoc12 (add/add) | unified dispatcher: planner `plan/validate/capabilities/defaults/summary` at top level via new `planner.cli.add_commands()`, `literature …` and `gtoc12 …` groups, `register()` hook kept, `dispatch()` accepts `func` (planner/literature) and `function` (gtoc12) handlers; `planner.cli.main` unchanged for its tests; new `tests/test_cli_dispatch.py` (9 tests) |
| `docs/COMPARATIVE_SOLVER_CAMPAIGN.md` | literature ↔ gtoc12 (add/add) | kept the literature implementation-status appendix, added the GTOC12 track pointer |
| `cpp/cuda/src/device_scvx.cu` | 9a4cbea ↔ literature | auto-merged; compiled in both CUDA trees |
| `README.md` | all | auto-merged (Planner section, literature/GTOC12 doc links) |

## CPU verification matrix (all on the candidate, `CUDA_VISIBLE_DEVICES=''`)

Evidence: `build-v2-verification/` (ignored) - `summary.tsv`, `summary-py.tsv`, `summary-py2.tsv`,
`summary-web.tsv`, `commands.txt`, one log per step, `versions.txt`, `artifact-sha256*.txt`.

| Step | Result |
| --- | --- |
| Ruff `check .` / `format --check .` | clean (259 files) |
| `cpp` RelWithDebInfo / Debug / Debug+ASan+UBSan, C API, Werror | build OK; CTest 49/49 each (incl. `planner_c_api_smoke`, `planner_problem_smoke`, `powered_descent_free_time_transcription_smoke`, `time_dilated_flow_smoke`) |
| `cpp/native` RelWithDebInfo / Debug / ASan+UBSan | build OK; CTest 8/8 each |
| Schema generation `--check` (`generate_orbitweaver_g7_schemas.py`, `generate_g4_policy_header.py`) | pass |
| Full Python suite (`python -S -m pytest`, source tree, fresh RelWithDebInfo library, pinned **CPU** QOCO 09f0495+abstol patch, GTOC12 data) | **490 passed, 23 skipped** (skips: GPU-gated planner/G4 session, offline literature artefacts, node) |
| `tests/test_planner_viewer_export.py` with Linux node 20 on PATH | 4/4 (planner export and archive both accepted by `check.mjs`) |
| G4 contract/capability hash tests (`test_g4_policy`, `test_g4_execution_contract`, `test_g4_executor_contract`, `test_g4_claim_core_decision`, `test_g4_decisions`, `test_g4_qualification`, `test_g4_scheduler`) | 67 passed; claim-core digest `40dc2174…`, policy `9ab3b444…`, applicability `1c4e0d51…` unchanged (blob-identical to 63271d5) |
| G6 synthetic reproducibility / freeze refusal (`test_paper1_g6`, `test_paper1_result`) | 37 passed |
| GTOC12 (`test_gtoc12_verifier` incl. official-binary reproduction, `_rules_and_data`, `_pipeline`, `_search2`, `_ephemeris_format`) | 39 passed, 0 skipped |
| Literature + free-time + manifests + G7 contracts | 81 passed, 11 skipped (offline artefacts) |
| Wheel + sdist (`python -m build`) | OK; wheel `6fb48889…`, sdist `55bd5862…` |
| Fresh consumer venv: import, `c_api_version()==1`, packaged library exports `spacepdhcg_planner_create`, `spacepdhcg_pd3_fft_create`, `spacepdhcg_pd6_fft_create` | pass |
| Consumer CLI: `spacepdhcg --help`, `validate`, `defaults hcw`, `plan --backend cpu_reference` (HCW certified), `gtoc12 --help`, `python -m spacepdhcg --help`, GPU-unavailable exit 66 | pass |
| Consumer CLI: `literature list/provenance`, `gtoc12 reduced-instance` **from the installed wheel** | FAIL (pre-existing on both branches: `REPOSITORY_ROOT = Path(__file__).parents[3]` needs the source checkout; passes from the worktree) - follow-up, not an integration defect |
| CMake install + `cpp/package-smoke` consumer | build OK, CTest 1/1 |
| CUDA `sm_120` Release and Debug configure + **build only** (all 175 targets incl. `spacepdhcg_plan`, `device_time_dilated_test`, `device_scvx_integration_test`, `recovery_test`; pinned PDHCG 167c8b7) | OK; CTest inventory 68 tests (not executed) |
| Web viewer `npm run check` | pass (Linux node 20 and Windows node 24): `Validated 5 archive trajectories`, data SHA `b160734e…` |
| Web viewer `npm test` | 6/6 on Windows node 24 (the suite's server root uses `pathname.slice(1)` and the archive source is a `\\wsl.localhost` path, so it is Windows-specific; on Linux 4/6 with those two environment failures; `web/` differs from 63271d5 only in `scripts/check.mjs`) |
| Tree clean before/after every phase; GPU lock never created | pass |

## Follow-ups (not integration defects)

- Wheel consumers cannot run `spacepdhcg literature …` / `spacepdhcg gtoc12 …` because both
  packages resolve `benchmarks/` relative to the source tree; either package the registries or
  fail with a clear message.
- `docs/REFERENCE_REPRODUCTION_REPORT.md` (generated) carries trailing whitespace flagged by
  `git diff --check`; regenerate through the report writer if that matters.
- GPU-deferred work: `docs/GPU_DEFERRED_VALIDATION_V2.md` / `benchmarks/gpu_deferred_validation_v2.json`.

## Promotion

```bash
cd /home/angus/worktrees/spacepdhcg-single-gpu-integration \
  && test -z "$(git status --porcelain=v1)" \
  && test "$(git rev-parse HEAD)" = 63271d58b78df343c1ae694525e460db31696f5d \
  && git merge --ff-only integration/single-gpu-v2-candidate
```

Run only after the claim-core finish script has sealed `results/gpu/g4/claim-core-9a4cbea/` and no
`run_g4_campaign.py` / `--g4-server` / `--g4-session` process remains.
