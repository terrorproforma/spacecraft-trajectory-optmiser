# Installed-wheel asset resolution fix

Branch `integration/single-gpu-v2-candidate` (worktree `/home/angus/worktrees/spacepdhcg-single-gpu-v2`),
base 2c8c651. CPU only; the G4 integration worktree and the GPU were not touched
(`CUDA_VISIBLE_DEVICES=""` throughout, `/home/angus/.spacepdhcg-gpu.lock` never created).
Evidence: `build-v2-wheel-fix/` (`summary.tsv`, `summary-py.tsv`, `commands.txt`, one log per step).

## Defect

`spacepdhcg literature …`, `spacepdhcg gtoc12 …` (and `spacepdhcg-orbitweaver-g7 validate-matrix`
without an argument) failed from an installed wheel: `src/spacepdhcg/literature/*.py` and
`src/spacepdhcg/gtoc12/*.py` located `benchmarks/` through `Path(__file__).resolve().parents[3]`,
which is `<site-packages>/../..` once installed
(`FileNotFoundError: .../lib/python3.12/benchmarks/literature/targets.json`).

## Resolver: `spacepdhcg.resources`

Every frozen asset is addressed by its repository-relative POSIX path
(`benchmarks/literature/targets.json`, `experiments/schema/literature_provenance.schema.json`, …)
and `resources.asset_path(name)` resolves it in one fixed order:

| # | Source | Rule |
| --- | --- | --- |
| 1 | `$SPACEPDHCG_BENCHMARKS_DIR` | Explicit `benchmarks/` directory. When set it is **authoritative** for every `benchmarks/...` asset: `benchmarks/<rest>` must exist as `$SPACEPDHCG_BENCHMARKS_DIR/<rest>`; a missing file raises `AssetNotFound` naming the variable and the path instead of silently falling back. Does not apply to `experiments/schema/...`. |
| 2 | source checkout | `<root>/src/spacepdhcg/resources.py` with `<root>/pyproject.toml` and `<root>/benchmarks` beside it (the historical `parents[3]` behaviour: dev trees, editable installs). |
| 3 | packaged data | `spacepdhcg/_data/<name>` inside the wheel. |

`AssetNotFound` (a `FileNotFoundError`) lists every location that was searched and says whether the
asset is part of `PACKAGED_ASSETS` ("installation is incomplete") or only exists in a checkout.
Companions: `locate_directory()` (run-time directories such as `benchmarks/gtoc12/data`: override
or checkout, else `None`), `cache_root()` (`$SPACEPDHCG_CACHE_DIR`, `$XDG_CACHE_HOME/spacepdhcg`,
`~/.cache/spacepdhcg`), `output_root()` (checkout, else the working directory, for generated
reports), `repository_root()`, `load_json_asset()`, `compare_packaged_assets()`.

Behaviour per module (no `parents[n]` repository-root assumption remains in `src/`):

- `literature/registry.py`, `external_sources.py`, `provenance.py`, `pd6_monte_carlo.py`: module
  path constants became `registry_path()`, `manifest_path()`, `schema_path()/store_path()/baselines_path()`,
  `samples_path()`; loaders take `path=None`. A custom registry file still resolves its profiles two
  levels above itself (mirrors `benchmarks/literature/targets.json`). `write_provenance_store()` writes
  below `output_root()` — never into the wheel.
- `literature/report.py`, `literature/cli.py`: `report_json_path()`, `report_markdown_path()`,
  `details_dir()` under `output_root()`; git commit/dirty state become `unknown`/`True` without a checkout.
- `gtoc12/data.py`: `pins_path()`, `rules_path()`, new `load_rules()`; `data_directory()` =
  `$SPACEPDHCG_GTOC12_DATA` → pinned `benchmarks/gtoc12/data` of the override/checkout →
  `cache_root()/gtoc12`. Missing-data message now points at `spacepdhcg gtoc12 fetch`.
- `gtoc12/fetch.py` (new): the fetch logic moved out of `scripts/gtoc12/fetch_gtoc12_data.py` so
  `spacepdhcg gtoc12 fetch [--only … --skip-optional --timeout]` works from a wheel; the script is a
  thin wrapper.
- `gtoc12/reduced_instance.py`: `default_rule_path()`. `gtoc12/lambert.py`:
  `resolve_native_library_path()` = explicit path → `$SPACEPDHCG_GTOC12_C_API` → checkout
  `build/gtoc12` (compiled on demand) → packaged `libspacepdhcg` (verified: it exports the Lambert ABI).
  `gtoc12/cli.py`: commit lookup through `resources.repository_root()`.
- `planner/viewer_export.py`: viewer sources from `$SPACEPDHCG_VIEWER_SOURCE` or the checkout's
  `web/trajectory-viewer`; `None` when installed (viewer not packaged, export omits the static files).
- `orbitweaver/cli.py`: `validate-matrix` default resolves `benchmarks/paper2_matrix.json` through the
  resolver instead of the working directory.
- Unchanged by design: `paper1/cli.py --repository` (explicit argument for freeze/clean-clone),
  `orbitweaver/instances.py::load_paper2_instance_contract(repository)` (explicit argument), G4
  loaders (`load_policy(path, …)` take explicit paths). Tests keep their own `Path(__file__).parents[1]`.

## Packaged assets (`src/spacepdhcg/_data`, 34 files, 270,583 bytes + README.md)

Mirror maintained by `python scripts/sync_packaged_assets.py` (`--check` exits 1 on any missing,
differing, or stray file); `tests/test_resources.py` compares every copy to the repository original
by SHA-256 and checks the three hash locks against both copies. `wheel.packages = ["src/spacepdhcg"]`
ships the directory verbatim; `benchmarks/gtoc12/data/` and every pinned download stay external.

| Asset | Bytes |
| --- | ---: |
| `benchmarks/campaign_scopes/full-multi-gpu-v1.json` | 793 |
| `benchmarks/campaign_scopes/single-gpu-v1.json` | 1,969 |
| `benchmarks/g4_applicability.json` / `.sha256` (`1c4e0d51…`) | 8,873 / 88 |
| `benchmarks/g4_h5_h6_claim_core.json` / `.sha256` (`40dc2174…`) | 2,613 / 91 |
| `benchmarks/g4_policy.json` / `.sha256` (`9ab3b444…`) | 7,080 / 81 |
| `benchmarks/gtoc12/gtoc12_rules.json` | 3,759 |
| `benchmarks/gtoc12/pins.json` | 6,873 |
| `benchmarks/gtoc12/reduced_instance_v1.json` | 1,651 |
| `benchmarks/gtoc12/reference_reproductions.json` | 5,614 |
| `benchmarks/literature/chari_2024_initial_positions.json` | 29,093 |
| `benchmarks/literature/external_sources.json` | 16,623 |
| `benchmarks/literature/gtoc_reduced_subsets.json` | 6,491 |
| `benchmarks/literature/profiles/*.json` (11 profiles) | 14,460 |
| `benchmarks/literature/provenance.json` | 112,740 |
| `benchmarks/literature/targets.json` | 7,597 |
| `benchmarks/literature/tops_selection.json` | 13,880 |
| `benchmarks/literature_baselines.json` | 11,329 |
| `benchmarks/paper1_matrix.json` | 8,433 |
| `benchmarks/paper2_instances.json` | 1,865 |
| `benchmarks/paper2_matrix.json` | 3,796 |
| `experiments/schema/literature_provenance.schema.json` | 4,791 |

Not packaged on purpose: `benchmarks/literature/reference_reproduction.json` and
`docs/REFERENCE_REPRODUCTION_REPORT.md` (generated outputs), `benchmarks/gpu_deferred_validation_v2.json`,
`benchmarks/g5_*.json` (not read by any command), `web/trajectory-viewer` (static viewer, optional).

## Validation (all CPU)

| Check | Result |
| --- | --- |
| `ruff check .` / `ruff format --check .` | clean (267 files) |
| Schema `--check` (G7, G4 policy header), `build_provenance.py --check`, `sync_packaged_assets.py --check` | pass |
| Full Python suite (`python -S -m pytest`, RelWithDebInfo library, pinned CPU QOCO, GTOC12 data) | **527 passed, 23 skipped** (550 collected: 30 new resolver tests in `tests/test_resources.py`; the same 23 GPU/offline/node skips as the candidate report) |
| G4 contract/capability hash tests | 67 passed; policy `9ab3b444…`, applicability `1c4e0d51…`, claim core `40dc2174…` unchanged (`sha256sum -c` OK) |
| G6 freeze (`test_paper1_g6`, `test_paper1_result`) | 37 passed |
| Topology/handback/G7/G5 hash tests (`test_qoco_handback`, `test_orbitweaver_g7[_records]`, `test_g4_native_session`, `test_g5_campaign`) | 85 passed, 1 skipped (CUDA session executable) |
| GTOC12 / literature / planner+CLI+resources subsets | 39 passed / 81 passed, 11 offline skips / 92 passed, 11 GPU+node skips |
| `git diff HEAD -- benchmarks/ experiments/ cpp/ papers/ web/` | empty (frozen trees untouched) |
| Wheel + sdist (`python -m build`) | wheel `e3f51456…`, sdist `26dd1a1e…`; wheel audit: 34 assets byte-identical to the repository, no `gtoc12/data`, native library present |
| Wheel built from the extracted sdist (`--no-isolation`) | same 35 `_data` entries, identical digests |
| Fresh `uv` venv (Python 3.12, cwd `/tmp`, `PYTHONPATH`/repo env unset): `spacepdhcg --help`, `capabilities`, `validate`, `defaults hcw`, `plan --backend cpu_reference` (HCW certified), `python -m spacepdhcg --help`, `spacepdhcg-paper1 --help` | pass |
| Consumer `literature list`, `literature status`, `literature provenance` (126 records / 10 profiles), `literature report` without records → exit 1 with message | pass |
| Consumer `gtoc12 --help`, `gtoc12 fetch --help`, rules unit path (`load_rules() == constants.rules_payload()`, reduced rule sha `718dd7e7…`, data dir → `~/.cache/spacepdhcg/gtoc12`), missing-data message, `gtoc12 reduced-instance` with `SPACEPDHCG_GTOC12_DATA` | pass |
| Consumer `NativeLambert()` through the packaged `libspacepdhcg.so`; `tests/test_gtoc12_rules_and_data.py` + `tests/test_gtoc12_verifier.py` (incl. official-binary reproduction) run against the **installed** package | 20 passed |
| Consumer `spacepdhcg-orbitweaver-g7 validate-matrix` (packaged matrix) | pass |
| Consumer `SPACEPDHCG_BENCHMARKS_DIR` override: resolves to the override; a bogus registry there fails with `RegistryError` (no silent fallback) | pass |
| Installed `_data` vs repository (`sync_packaged_assets.py --check --packaged-dir <site-packages>/spacepdhcg/_data`) | 34 assets byte-identical |
