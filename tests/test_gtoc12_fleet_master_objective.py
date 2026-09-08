"""Archive fleet results and stdout must report the selected, verified objective."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.cli import build_parser
from spacepdhcg.gtoc12 import (
    archive,
    bundles,
    cli,
    cooperative,
    data,
    fleet,
    official,
    solution,
    verifier,
    viewer_export,
)


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("independent_ok,official_ok", [(True, True), (False, True), (True, False)])
def test_master_reporting_distinguishes_score_cargo_and_failure(
    tmp_path, monkeypatch, capsys, weighted, independent_ok, official_ok
):
    catalogue = SimpleNamespace(ids=np.array([1]))
    monkeypatch.setattr(data, "load_catalogue", lambda: catalogue)
    monkeypatch.setattr(
        data, "load_bonus_table", lambda: SimpleNamespace(coefficient=np.array([0.5]))
    )
    monkeypatch.setattr(cli, "_commit", lambda *_: "synthetic-fixture")
    monkeypatch.setattr(cli, "_peak_rss_mb", lambda: 0.0)
    monkeypatch.setattr(archive, "discover_archives", lambda *a: [])
    monkeypatch.setattr(
        archive, "recertify_archives", lambda *a, **kw: [SimpleNamespace(ships=[], summary=dict)]
    )
    monkeypatch.setattr(bundles, "bundle_columns", lambda *a, **kw: [object()])
    route = SimpleNamespace(plan=SimpleNamespace(asteroids=[1]), total_collected_kg=600.0)

    def master(*a, weights, **kw):
        assert weights == ({1: 0.5} if weighted else None)
        return SimpleNamespace(routes=lambda: [route], summary=dict, exhaustive=True)

    monkeypatch.setattr(cooperative, "solve_fleet_master", master)
    monkeypatch.setattr(
        fleet,
        "assemble_fleet",
        lambda *a: SimpleNamespace(write=lambda path: path.write_text("synthetic fleet")),
    )
    independent = verifier.VerificationReport(
        independent_ok, 1, [], [], {}, 600.0, 100.0, weighted_score_fixed_bonus_kg=300.0
    )
    monkeypatch.setattr(
        verifier,
        "Gtoc12Verifier",
        lambda *a, **kw: SimpleNamespace(verify_file=lambda *a: independent),
    )
    monkeypatch.setattr(official, "official_verifier_available", lambda: True)
    monkeypatch.setattr(
        official,
        "run_official_verifier",
        lambda *a: official.OfficialVerification(official_ok, 1, 1, 600.0, "", "", 0),
    )
    monkeypatch.setattr(solution.Solution, "read", lambda *a: None)
    monkeypatch.setattr(viewer_export, "write_viewer_dataset", lambda *a, **kw: {})
    args = build_parser().parse_args(
        [
            "gtoc12",
            "fleet-master",
            "--run-id",
            "master-objective",
            "--output",
            str(tmp_path),
            "--source",
            "synthetic-archive",
            "--workers",
            "1",
            *([] if weighted else ["--no-bonus-weights"]),
        ]
    )
    ok = independent_ok and official_ok
    assert cli.cmd_fleet_master(args) == (0 if ok else 1)
    report = json.loads((tmp_path / "run_report.json").read_text())
    expected_score = (300.0 if weighted else 600.0) if ok else None
    assert report["final_fleet"]["score_kg"] == expected_score
    assert report["final_fleet"]["total_mass_kg"] == 600.0
    assert report["final_fleet"]["weighted_score_fixed_bonus_kg"] == 300.0
    assert report["score_kind"] == (
        "weighted_score_fixed_bonus_kg" if weighted else "total_mass_kg"
    )
    assert report["status"] == ("scored" if ok else "fleet_failed_verification")
    assert (report["best"] is not None) is ok
    text = capsys.readouterr().out.strip()
    while text:
        printed, end = json.JSONDecoder().raw_decode(text)
        text = text[end:].lstrip()
    assert printed["score_kg"] == expected_score
    assert printed["total_mass_kg"] == 600.0
    assert printed["weighted_score_fixed_bonus_kg"] == 300.0
