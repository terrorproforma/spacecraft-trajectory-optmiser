"""Asset resolver: lookup order, error messages, and byte-identical packaged copies.

The installed-wheel defect this guards against: ``spacepdhcg literature ...`` and
``spacepdhcg gtoc12 ...`` used to resolve ``benchmarks/`` through ``Path(__file__).parents[3]``,
which only exists in a source checkout.  ``installed_layout`` below simulates the wheel by hiding
the checkout so every command must succeed from the copies under ``spacepdhcg/_data``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from spacepdhcg import cli, resources
from spacepdhcg.gtoc12 import constants as gtoc12_constants
from spacepdhcg.gtoc12 import data as gtoc12_data
from spacepdhcg.gtoc12 import lambert, reduced_instance
from spacepdhcg.literature import external_sources, provenance, report
from spacepdhcg.literature.registry import load_target_registry
from spacepdhcg.orbitweaver import cli as orbitweaver_cli
from spacepdhcg.planner import viewer_export

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = ROOT / "src" / "spacepdhcg" / "_data"
REGISTRY = "benchmarks/literature/targets.json"
SCHEMA = "experiments/schema/literature_provenance.schema.json"
SYNC_SCRIPT = ROOT / "scripts" / "sync_packaged_assets.py"
REDUCED_RULE_SHA256 = "718dd7e76f8f09295ae53de58b56626c5d8eb42fa397a27ab190b6511b39bd25"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clear_caches() -> None:
    gtoc12_data.load_pins.cache_clear()
    gtoc12_data.load_rules.cache_clear()


@pytest.fixture
def no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(resources.BENCHMARKS_DIR_ENV, raising=False)


@pytest.fixture
def installed_layout(monkeypatch: pytest.MonkeyPatch, no_override: None) -> None:
    """Pretend the package is installed: no checkout is detected, only packaged copies remain."""

    monkeypatch.setattr(resources, "repository_root", lambda: None)
    _clear_caches()
    yield
    _clear_caches()


# --- resolution order -----------------------------------------------------------------------


def test_repository_root_is_the_source_checkout() -> None:
    assert resources.repository_root() == ROOT
    assert resources.packaged_data_root() == PACKAGED


def test_checkout_wins_without_an_override(no_override: None) -> None:
    assert resources.asset_path(REGISTRY) == ROOT / REGISTRY
    assert resources.asset_path(SCHEMA) == ROOT / SCHEMA
    assert [source for source, _ in resources.asset_candidates(REGISTRY)] == [
        "source checkout",
        "packaged data",
    ]


def test_explicit_override_wins_over_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "custom-benchmarks"
    target = override / "literature" / "targets.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"schema_version": "custom"}', encoding="utf-8")
    monkeypatch.setenv(resources.BENCHMARKS_DIR_ENV, str(override))

    assert resources.asset_path(REGISTRY) == target
    assert resources.load_json_asset(REGISTRY) == {"schema_version": "custom"}
    assert resources.asset_candidates(REGISTRY) == [(resources.BENCHMARKS_DIR_ENV, target)]
    assert resources.locate_directory("benchmarks/gtoc12/data") == override / "gtoc12" / "data"


def test_override_is_authoritative_for_benchmarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(resources.BENCHMARKS_DIR_ENV, str(tmp_path))
    with pytest.raises(resources.AssetNotFound) as error:
        resources.asset_path(REGISTRY)
    message = str(error.value)
    assert resources.BENCHMARKS_DIR_ENV in message
    assert str(tmp_path / "literature" / "targets.json") in message
    assert "unset the variable" in message
    # a checkout copy exists, but the explicit override must never be bypassed silently
    assert (ROOT / REGISTRY).is_file()


def test_override_does_not_apply_outside_the_benchmarks_subtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(resources.BENCHMARKS_DIR_ENV, str(tmp_path))
    assert resources.asset_path(SCHEMA) == ROOT / SCHEMA


def test_blank_override_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(resources.BENCHMARKS_DIR_ENV, "   ")
    assert resources.benchmarks_override() is None
    assert resources.asset_path(REGISTRY) == ROOT / REGISTRY


def test_packaged_copy_is_used_without_a_checkout(installed_layout: None) -> None:
    path = resources.asset_path(REGISTRY)
    assert path == PACKAGED / REGISTRY
    assert path.read_bytes() == (ROOT / REGISTRY).read_bytes()
    assert resources.asset_candidates(REGISTRY) == [("packaged data", PACKAGED / REGISTRY)]
    assert resources.locate_directory("benchmarks/gtoc12/data") is None


def test_missing_asset_error_lists_every_searched_location(installed_layout: None) -> None:
    with pytest.raises(resources.AssetNotFound) as error:
        resources.asset_path("benchmarks/does_not_exist.json")
    message = str(error.value)
    assert "packaged data: " in message
    assert str(PACKAGED / "benchmarks" / "does_not_exist.json") in message
    assert "not part of spacepdhcg.resources.PACKAGED_ASSETS" in message
    assert resources.BENCHMARKS_DIR_ENV in message


def test_missing_packaged_asset_reports_an_incomplete_installation(
    installed_layout: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resources, "packaged_data_root", lambda: tmp_path / "empty")
    with pytest.raises(resources.AssetNotFound, match="installation is incomplete"):
        resources.asset_path(REGISTRY)


def test_missing_asset_in_checkout_names_both_locations(
    no_override: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resources, "packaged_data_root", lambda: tmp_path / "empty")
    with pytest.raises(resources.AssetNotFound) as error:
        resources.asset_path("benchmarks/no_such_manifest.json")
    message = str(error.value)
    assert "source checkout: " in message and "packaged data: " in message


@pytest.mark.parametrize("name", ["/etc/passwd", "../benchmarks/x.json", "benchmarks/../x", ""])
def test_asset_names_must_be_repository_relative(name: str) -> None:
    with pytest.raises(ValueError):
        resources.asset_path(name)


def test_asset_names_are_normalised(no_override: None) -> None:
    assert resources.asset_path("benchmarks/./literature/targets.json") == ROOT / REGISTRY


def test_cache_root_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(resources.CACHE_DIR_ENV, str(tmp_path / "explicit"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert resources.cache_root() == tmp_path / "explicit"
    monkeypatch.delenv(resources.CACHE_DIR_ENV)
    assert resources.cache_root() == tmp_path / "xdg" / "spacepdhcg"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert resources.cache_root() == Path.home() / ".cache" / "spacepdhcg"


# --- packaged copies ------------------------------------------------------------------------


def test_packaged_assets_are_byte_identical_to_the_repository() -> None:
    report_ = resources.compare_packaged_assets(ROOT, PACKAGED)
    assert report_ == {"missing": [], "different": [], "stray": []}, (
        "run python scripts/sync_packaged_assets.py"
    )
    for asset in resources.PACKAGED_ASSETS:
        assert _sha256(ROOT / asset) == _sha256(PACKAGED / asset), asset


def test_packaged_hash_locks_match_their_frozen_files() -> None:
    locks = [asset for asset in resources.PACKAGED_ASSETS if asset.endswith(".sha256")]
    assert {Path(lock).name for lock in locks} == {
        "g4_applicability.sha256",
        "g4_h5_h6_claim_core.sha256",
        "g4_policy.sha256",
    }
    for lock in locks:
        frozen = lock[: -len(".sha256")] + ".json"
        assert frozen in resources.PACKAGED_ASSETS
        recorded = (PACKAGED / lock).read_text(encoding="utf-8").split()[0]
        assert recorded == _sha256(PACKAGED / frozen) == _sha256(ROOT / frozen), lock


def test_packaged_assets_stay_small_and_exclude_pinned_downloads() -> None:
    files = resources.packaged_asset_files(PACKAGED)
    assert {path.relative_to(PACKAGED).as_posix() for path in files} == set(
        resources.PACKAGED_ASSETS
    )
    assert sum(path.stat().st_size for path in files) < 1 << 20
    for asset in resources.PACKAGED_ASSETS:
        assert asset.endswith((".json", ".sha256")), asset
        assert "gtoc12/data" not in asset and "results/" not in asset, asset
    assert not (PACKAGED / "benchmarks" / "gtoc12" / "data").exists()


def test_sync_script_check_passes_on_the_committed_mirror() -> None:
    completed = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "byte-identical" in completed.stdout


def test_sync_script_detects_drift_and_repairs_it(tmp_path: Path) -> None:
    mirror = tmp_path / "_data"
    shutil.copytree(PACKAGED, mirror)
    (mirror / "benchmarks" / "g4_policy.json").write_text("{}", encoding="utf-8")
    (mirror / "benchmarks" / "gtoc12" / "pins.json").unlink()
    (mirror / "benchmarks" / "stray.json").write_text("{}", encoding="utf-8")

    check = [sys.executable, str(SYNC_SCRIPT), "--check", "--packaged-dir", str(mirror)]
    drift = subprocess.run(check, capture_output=True, text=True, cwd=ROOT, check=False)
    assert drift.returncode == 1
    assert "different: benchmarks/g4_policy.json" in drift.stderr
    assert "missing: benchmarks/gtoc12/pins.json" in drift.stderr
    assert "stray: benchmarks/stray.json" in drift.stderr

    repair = [sys.executable, str(SYNC_SCRIPT), "--packaged-dir", str(mirror)]
    assert subprocess.run(repair, capture_output=True, text=True, cwd=ROOT).returncode == 0
    assert subprocess.run(check, capture_output=True, text=True, cwd=ROOT).returncode == 0
    assert resources.compare_packaged_assets(ROOT, mirror) == {
        "missing": [],
        "different": [],
        "stray": [],
    }


# --- commands from the installed layout -----------------------------------------------------


def test_literature_commands_work_from_packaged_copies(
    installed_layout: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(external_sources.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.delenv(external_sources.ONLINE_ENV, raising=False)

    assert cli.main(["literature", "list"]) == 0
    listed = capsys.readouterr().out
    assert "acikmese-ploen-2007-pd3" in listed and "gtoc12-official-verifier" in listed

    assert cli.main(["literature", "status"]) == 0
    status = capsys.readouterr().out
    assert "missing" in status and "tops.twobody" in status

    assert cli.main(["literature", "provenance"]) == 0
    assert "records across" in capsys.readouterr().out

    registry = load_target_registry()
    for target in registry.targets:
        assert target.load_profile()["id"] == target.id
    assert provenance.schema_path() == PACKAGED / SCHEMA
    assert external_sources.manifest_path() == PACKAGED / external_sources.MANIFEST_ASSET


def test_gtoc12_rules_and_rule_files_come_from_packaged_copies(
    installed_layout: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert gtoc12_data.rules_path() == PACKAGED / gtoc12_data.RULES_ASSET
    assert gtoc12_data.load_rules() == gtoc12_constants.rules_payload()
    assert gtoc12_data.pins_path() == PACKAGED / gtoc12_data.PINS_ASSET
    assert reduced_instance.default_rule_path() == PACKAGED / reduced_instance.DEFAULT_RULE_ASSET
    assert reduced_instance.load_rule()[1] == REDUCED_RULE_SHA256

    monkeypatch.delenv(gtoc12_data.DATA_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setenv(resources.CACHE_DIR_ENV, str(tmp_path / "cache"))
    assert gtoc12_data.data_directory() == tmp_path / "cache" / "gtoc12"
    with pytest.raises(gtoc12_data.Gtoc12DataError, match="spacepdhcg gtoc12 fetch"):
        gtoc12_data.verified_path("GTOC12_Asteroids_Data.txt")
    assert gtoc12_data.data_available() is False

    monkeypatch.setenv(gtoc12_data.DATA_ENVIRONMENT_VARIABLE, str(tmp_path / "explicit"))
    assert gtoc12_data.data_directory() == (tmp_path / "explicit").resolve()


def test_gtoc12_help_and_parsers_from_packaged_layout(installed_layout: None) -> None:
    parser = cli.build_parser()
    arguments = parser.parse_args(["gtoc12", "fetch", "--skip-optional", "--only", "pins"])
    assert arguments.skip_optional is True and arguments.only == ["pins"]
    with pytest.raises(SystemExit) as exit_:
        parser.parse_args(["gtoc12", "--help"])
    assert exit_.value.code == 0


def test_lambert_library_resolution(
    installed_layout: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from spacepdhcg.native import NativeLibraryError, packaged_library_path

    monkeypatch.delenv(lambert.LIBRARY_ENVIRONMENT_VARIABLE, raising=False)
    assert lambert.default_native_library_path() is None
    try:
        packaged = packaged_library_path()
    except NativeLibraryError:
        # no packaged library next to src/spacepdhcg/native: the wheel fallback must say so
        with pytest.raises(NativeLibraryError):
            lambert.resolve_native_library_path()
    else:
        assert lambert.resolve_native_library_path() == packaged
    with pytest.raises(RuntimeError, match="source checkout"):
        lambert.compile_native_library(tmp_path / "never.so")

    explicit = tmp_path / "explicit.so"
    explicit.write_bytes(b"")
    monkeypatch.setenv(lambert.LIBRARY_ENVIRONMENT_VARIABLE, str(explicit))
    assert lambert.resolve_native_library_path() == explicit
    assert lambert.resolve_native_library_path(tmp_path / "explicit.so") == explicit


def test_lambert_default_path_in_checkout(no_override: None) -> None:
    assert lambert.default_native_library_path() == (
        ROOT / "build" / "gtoc12" / "libspacepdhcg_c_api.so"
    )


def test_viewer_source_depends_on_the_checkout(
    installed_layout: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(viewer_export.VIEWER_SOURCE_ENVIRONMENT, raising=False)
    assert viewer_export.default_viewer_source() is None
    monkeypatch.setattr(resources, "repository_root", lambda: ROOT)
    assert viewer_export.default_viewer_source() == ROOT / "web" / "trajectory-viewer"


def test_report_paths_follow_the_output_root(
    installed_layout: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert resources.output_root() == tmp_path
    assert report.report_json_path() == tmp_path / report.REPORT_JSON_RELATIVE
    assert report.report_markdown_path() == tmp_path / report.REPORT_MD_RELATIVE
    assert report.details_dir() == tmp_path / report.DETAILS_DIR_RELATIVE
    assert report._git_commit() == "unknown" and report._dirty() is True
    monkeypatch.setattr(resources, "repository_root", lambda: ROOT)
    assert report.report_json_path() == ROOT / report.REPORT_JSON_RELATIVE


def test_orbitweaver_matrix_default_resolves_from_packaged_copy(
    installed_layout: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert orbitweaver_cli.validate_matrix(argparse.Namespace(matrix=None)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True and "P2-A" in payload["families"]


def test_provenance_store_default_write_targets_the_output_root(
    installed_layout: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    store = provenance.load_provenance_store(known_profiles=load_target_registry().ids())
    provenance.write_provenance_store(store)
    written = tmp_path / provenance.STORE_ASSET
    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8")) == store.as_dict()
    # the packaged copy is read-only input and must never be rewritten
    assert _sha256(PACKAGED / provenance.STORE_ASSET) == _sha256(ROOT / provenance.STORE_ASSET)
