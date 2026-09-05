# Packaged assets

Generated, byte-identical copies of frozen repository files so `spacepdhcg literature ...`,
`spacepdhcg gtoc12 ...` and the other tools work from an installed wheel.

Do not edit anything here. The originals live under `benchmarks/` and `experiments/schema/`;
after changing one of them run `python scripts/sync_packaged_assets.py` (`--check` verifies the
mirror and `tests/test_resources.py` enforces it by SHA-256). The list of mirrored files is
`spacepdhcg.resources.PACKAGED_ASSETS`; the lookup order (explicit `SPACEPDHCG_BENCHMARKS_DIR`,
source checkout, then this directory) is documented in `spacepdhcg.resources`.

Large pinned downloads (GTOC12 official data, literature artefacts) are never mirrored here.
