"""Locate frozen benchmark and specification assets from an override, a checkout, or the wheel.

Every small, frozen JSON asset the command-line tools read -- the literature target registry,
provenance store, external-source pins and profiles, the GTOC12 rules/pins/reduced-instance rule,
the G4 policy/applicability/claim-core contracts with their hash locks, the campaign scopes, the
paper matrices and the provenance JSON schema -- is addressed by its repository-relative POSIX
path, for example ``benchmarks/literature/targets.json``.  :func:`asset_path` resolves such a
name in one fixed order:

1. ``$SPACEPDHCG_BENCHMARKS_DIR`` -- an explicit ``benchmarks/`` directory.  When it is set it is
   authoritative for every ``benchmarks/...`` asset: ``benchmarks/<rest>`` must exist as
   ``$SPACEPDHCG_BENCHMARKS_DIR/<rest>`` and no other location is consulted, so a misconfigured
   override fails loudly instead of silently reading another tree.
2. The source checkout that contains this module (``<root>/src/spacepdhcg/resources.py`` with
   ``<root>/pyproject.toml`` and ``<root>/benchmarks`` beside it).  This is the historical
   ``Path(__file__).resolve().parents[3]`` behaviour, so development trees and editable installs
   keep reading the tracked files directly.
3. The copies shipped inside the wheel under ``spacepdhcg/_data/`` (mirrored from the repository
   by ``scripts/sync_packaged_assets.py``; ``tests/test_resources.py`` proves every copy is
   byte-identical to its original).

Large pinned downloads -- the GTOC12 official data, literature artefacts -- are never packaged.
They are fetched into a data or cache directory (see :func:`cache_root`), exactly as before.
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib import resources as _importlib_resources
from pathlib import Path, PurePosixPath
from typing import Any

BENCHMARKS_DIR_ENV = "SPACEPDHCG_BENCHMARKS_DIR"
CACHE_DIR_ENV = "SPACEPDHCG_CACHE_DIR"
PACKAGE_DATA_DIRECTORY = "_data"

#: Repository-relative paths of every frozen asset shipped inside the wheel.  Keep this list
#: small and explicit: pinned downloads are fetched into :func:`cache_root`, never packaged.
#: ``scripts/sync_packaged_assets.py`` copies these files into ``src/spacepdhcg/_data`` and
#: ``tests/test_resources.py`` fails when a copy drifts from the repository original.
PACKAGED_ASSETS: tuple[str, ...] = (
    "benchmarks/campaign_scopes/full-multi-gpu-v1.json",
    "benchmarks/campaign_scopes/single-gpu-v1.json",
    "benchmarks/g4_applicability.json",
    "benchmarks/g4_applicability.sha256",
    "benchmarks/g4_h5_h6_claim_core.json",
    "benchmarks/g4_h5_h6_claim_core.sha256",
    "benchmarks/g4_policy.json",
    "benchmarks/g4_policy.sha256",
    "benchmarks/gtoc12/gtoc12_rules.json",
    "benchmarks/gtoc12/pins.json",
    "benchmarks/gtoc12/reduced_instance_v1.json",
    "benchmarks/gtoc12/reference_reproductions.json",
    "benchmarks/literature/chari_2024_initial_positions.json",
    "benchmarks/literature/external_sources.json",
    "benchmarks/literature/gtoc_reduced_subsets.json",
    "benchmarks/literature/profiles/acikmese-ploen-2007-pd3.json",
    "benchmarks/literature/profiles/blackmore-2010-pd3-case1.json",
    "benchmarks/literature/profiles/chari-2024-pd6-monte-carlo.json",
    "benchmarks/literature/profiles/esa-tops-2026.json",
    "benchmarks/literature/profiles/gtoc12-official-verifier.json",
    "benchmarks/literature/profiles/gtoc5-data-pin.json",
    "benchmarks/literature/profiles/gtoc9-example-validation.json",
    "benchmarks/literature/profiles/gtopx-2021.json",
    "benchmarks/literature/profiles/szmuk-acikmese-2018-pd6-2d.json",
    "benchmarks/literature/profiles/tafazzol-taheri-earth-dionysus.json",
    "benchmarks/literature/profiles/tafazzol-taheri-earth-mars.json",
    "benchmarks/literature/provenance.json",
    "benchmarks/literature/targets.json",
    "benchmarks/literature/tops_selection.json",
    "benchmarks/literature_baselines.json",
    "benchmarks/paper1_matrix.json",
    "benchmarks/paper2_instances.json",
    "benchmarks/paper2_matrix.json",
    "experiments/schema/literature_provenance.schema.json",
)

#: Files that may live inside ``spacepdhcg/_data`` without being assets.
PACKAGED_DATA_EXTRA_FILES: frozenset[str] = frozenset({"README.md"})


class AssetNotFound(FileNotFoundError):
    """Raised when a repository-relative asset exists in none of the searched locations."""


def _relative(asset: str) -> PurePosixPath:
    path = PurePosixPath(asset)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"asset names must be repository-relative POSIX paths, got {asset!r}")
    return path


def repository_root() -> Path | None:
    """Return the source checkout containing this module, or ``None`` for an installed package.

    The checkout is recognised by its layout: this file at ``<root>/src/spacepdhcg/resources.py``
    with ``<root>/pyproject.toml`` and ``<root>/benchmarks`` beside it -- the tree the historical
    ``Path(__file__).resolve().parents[3]`` module constants pointed at.
    """

    here = Path(__file__).resolve()
    try:
        root = here.parents[2]
    except IndexError:
        return None
    if here.parents[1].name != "src" or here.parents[0].name != "spacepdhcg":
        return None
    if not (root / "pyproject.toml").is_file() or not (root / "benchmarks").is_dir():
        return None
    return root


def benchmarks_override() -> Path | None:
    """The explicit ``$SPACEPDHCG_BENCHMARKS_DIR`` directory, or ``None`` when unset."""

    value = os.environ.get(BENCHMARKS_DIR_ENV, "")
    if not value.strip():
        return None
    return Path(value).expanduser()


def packaged_data_root() -> Path:
    """Directory of the copies shipped inside the package (``spacepdhcg/_data``)."""

    return Path(str(_importlib_resources.files("spacepdhcg").joinpath(PACKAGE_DATA_DIRECTORY)))


def asset_candidates(asset: str) -> list[tuple[str, Path]]:
    """Return the ``(source, path)`` pairs :func:`asset_path` consults, in resolution order."""

    relative = _relative(asset)
    override = benchmarks_override()
    if override is not None and relative.parts[0] == "benchmarks":
        # The override is authoritative for the benchmarks subtree: nothing else is consulted.
        return [(BENCHMARKS_DIR_ENV, override.joinpath(*relative.parts[1:]))]
    candidates: list[tuple[str, Path]] = []
    root = repository_root()
    if root is not None:
        candidates.append(("source checkout", root.joinpath(*relative.parts)))
    candidates.append(("packaged data", packaged_data_root().joinpath(*relative.parts)))
    return candidates


def _describe_miss(asset: str, candidates: list[tuple[str, Path]]) -> str:
    searched = "; ".join(f"{source}: {path}" for source, path in candidates)
    if candidates and candidates[0][0] == BENCHMARKS_DIR_ENV:
        return (
            f"asset {asset!r} not found: {BENCHMARKS_DIR_ENV} is set, so only "
            f"{candidates[0][1]} was consulted and it does not exist (unset the variable to use "
            "the source checkout or the packaged copy)"
        )
    hint = (
        "it is part of the packaged set, so the installation is incomplete"
        if asset in PACKAGED_ASSETS
        else "it is not part of spacepdhcg.resources.PACKAGED_ASSETS, so it is only available "
        f"from a source checkout or via {BENCHMARKS_DIR_ENV}"
    )
    return f"asset {asset!r} not found; searched {searched}; {hint}"


def asset_path(asset: str) -> Path:
    """Resolve a repository-relative asset (see the module docstring for the order)."""

    candidates = asset_candidates(asset)
    for _, path in candidates:
        if path.is_file():
            return path
    raise AssetNotFound(_describe_miss(asset, candidates))


def load_json_asset(asset: str) -> Any:
    """Parse a JSON asset located by :func:`asset_path`."""

    with asset_path(asset).open(encoding="utf-8") as handle:
        return json.load(handle)


def locate_directory(asset_directory: str) -> Path | None:
    """Where a repository-relative *directory* lives for an override or a checkout.

    Used for directories that are populated at run time (the GTOC12 data directory
    ``benchmarks/gtoc12/data``), so existence is not required.  Returns ``None`` for an
    installed package without an override; callers then fall back to :func:`cache_root`.
    """

    relative = _relative(asset_directory)
    override = benchmarks_override()
    if override is not None and relative.parts[0] == "benchmarks":
        return override.joinpath(*relative.parts[1:])
    root = repository_root()
    if root is not None:
        return root.joinpath(*relative.parts)
    return None


def cache_root() -> Path:
    """Writable cache for pinned downloads: ``$SPACEPDHCG_CACHE_DIR``, XDG, or ``~/.cache``."""

    override = os.environ.get(CACHE_DIR_ENV, "")
    if override.strip():
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    base = Path(xdg).expanduser() if xdg.strip() else Path.home() / ".cache"
    return base / "spacepdhcg"


def output_root() -> Path:
    """Root for generated reports: the checkout when there is one, else the working directory."""

    root = repository_root()
    return root if root is not None else Path.cwd()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packaged_asset_files(packaged_root: Path | None = None) -> list[Path]:
    """Every regular file below the packaged data directory (excluding the README)."""

    root = packaged_data_root() if packaged_root is None else packaged_root
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in PACKAGED_DATA_EXTRA_FILES
        and "__pycache__" not in path.parts
    )


def compare_packaged_assets(
    repository: Path, packaged_root: Path | None = None
) -> dict[str, list[str]]:
    """Compare packaged copies against the repository originals by SHA-256.

    Returns ``{"missing": [...], "different": [...], "stray": [...]}``; every list is empty
    when the mirror is complete, byte-identical, and contains nothing else.
    """

    root = packaged_data_root() if packaged_root is None else packaged_root
    missing: list[str] = []
    different: list[str] = []
    for asset in PACKAGED_ASSETS:
        original = repository.joinpath(*_relative(asset).parts)
        copy = root.joinpath(*_relative(asset).parts)
        if not copy.is_file():
            missing.append(asset)
        elif not original.is_file():
            different.append(f"{asset} (missing from the repository)")
        elif file_sha256(original) != file_sha256(copy):
            different.append(asset)
    expected = set(PACKAGED_ASSETS)
    stray = [
        path.relative_to(root).as_posix()
        for path in packaged_asset_files(root)
        if path.relative_to(root).as_posix() not in expected
    ]
    return {"missing": missing, "different": different, "stray": stray}


__all__ = [
    "BENCHMARKS_DIR_ENV",
    "CACHE_DIR_ENV",
    "PACKAGED_ASSETS",
    "AssetNotFound",
    "asset_candidates",
    "asset_path",
    "benchmarks_override",
    "cache_root",
    "compare_packaged_assets",
    "file_sha256",
    "load_json_asset",
    "locate_directory",
    "output_root",
    "packaged_asset_files",
    "packaged_data_root",
    "repository_root",
]
