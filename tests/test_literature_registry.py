"""Target registry, profiles, and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacepdhcg.cli import build_parser
from spacepdhcg.literature import external_sources
from spacepdhcg.literature.registry import (
    RESULT_STATUSES,
    SUPPORT_LEVELS,
    RegistryError,
    load_target_registry,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TARGETS = {
    "acikmese-ploen-2007-pd3",
    "blackmore-2010-pd3-case1",
    "szmuk-acikmese-2018-pd6-2d",
    "chari-2024-pd6-monte-carlo",
    "tafazzol-taheri-earth-mars",
    "tafazzol-taheri-earth-dionysus",
    "esa-tops-2026",
    "gtopx-2021",
    "gtoc12-official-verifier",
    "gtoc9-example-validation",
    "gtoc5-data-pin",
}


def test_registry_lists_every_campaign_target() -> None:
    registry = load_target_registry()
    assert set(registry.ids()) == EXPECTED_TARGETS
    for target in registry.targets:
        assert target.support in SUPPORT_LEVELS
        assert target.runner.split(":")[0].startswith("spacepdhcg.literature.")
        assert callable(target.resolve_runner())
        profile = target.load_profile()
        assert profile["id"] == target.id
        for label in target.expected_labels:
            assert label in {
                "analytic",
                "published-reference",
                "reproduced-external",
                "measured-local",
                "descriptive-only",
            }


def test_registry_artifacts_are_pinned() -> None:
    registry = load_target_registry()
    manifest = external_sources.load_manifest()
    for target in registry.targets:
        for artifact in target.requires_artifacts:
            assert artifact in manifest, f"{target.id} requires unpinned artifact {artifact}"


def test_families_match_campaign_matrix() -> None:
    registry = load_target_registry()
    paper1 = json.loads((ROOT / "benchmarks" / "paper1_matrix.json").read_text(encoding="utf-8"))
    paper2 = json.loads((ROOT / "benchmarks" / "paper2_matrix.json").read_text(encoding="utf-8"))
    families = {f["id"] for f in paper1["families"]} | {f["id"] for f in paper2["families"]}
    for target in registry.targets:
        if target.family is not None:
            assert target.family in families, target.id


def test_unsupported_targets_carry_reasons(tmp_path: Path) -> None:
    document = json.loads((ROOT / "benchmarks" / "literature" / "targets.json").read_text())
    broken = next(t for t in document["targets"] if t["support"] == "unsupported")
    broken = dict(broken)
    broken.pop("unsupported_reason")
    document["targets"] = [broken]
    path = tmp_path / "benchmarks" / "literature" / "targets.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document))
    with pytest.raises(RegistryError, match="unsupported_reason"):
        load_target_registry(path)


def test_result_statuses_are_the_report_vocabulary() -> None:
    assert set(RESULT_STATUSES) == {
        "reproduced",
        "gap",
        "descriptive-only",
        "unsupported",
        "blocked",
    }


def test_cli_lists_targets(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    arguments = parser.parse_args(["literature", "list"])
    assert arguments.func(arguments) == 0
    output = capsys.readouterr().out
    for target in EXPECTED_TARGETS:
        assert target in output


def test_cli_validates_provenance(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    arguments = parser.parse_args(["literature", "provenance"])
    assert arguments.func(arguments) == 0
    assert "records across" in capsys.readouterr().out


def test_external_manifest_is_well_formed() -> None:
    manifest = external_sources.load_manifest()
    assert len(manifest) >= 30
    paths = [a.relative_path for a in manifest.values()]
    assert len(paths) == len(set(paths))
    for artifact in manifest.values():
        assert len(artifact.sha256) == 64
        assert artifact.size_bytes > 0
        assert artifact.licence
        assert artifact.url.startswith("http")


def test_fetch_offline_raises_explicit_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(external_sources.CACHE_ENV, str(tmp_path))
    monkeypatch.delenv(external_sources.ONLINE_ENV, raising=False)
    with pytest.raises(external_sources.ArtifactUnavailable, match="SPACEPDHCG_LITERATURE_ONLINE"):
        external_sources.fetch("tops.twobody")


def test_checksum_mismatch_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(external_sources.CACHE_ENV, str(tmp_path))
    artifact = external_sources.load_manifest()["tops.twobody"]
    path = tmp_path / artifact.relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not the pinned content")
    with pytest.raises(external_sources.ChecksumMismatch):
        external_sources.fetch("tops.twobody")
