"""A route check cannot substitute for checking the final assembled fleet."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.cli import build_parser
from spacepdhcg.gtoc12 import (
    cli,
    cooperative,
    data,
    fleet,
    official,
    pipeline,
    search,
    solution,
    verifier,
    viewer_export,
)


@pytest.mark.parametrize("independent_ok,official_ok", [(True, True), (False, True), (True, False)])
def test_final_fleet_verification_controls_status_and_exit(
    tmp_path, monkeypatch, independent_ok, official_ok
):
    catalogue = SimpleNamespace(ids=np.array([57530]))
    monkeypatch.setattr(data, "load_catalogue", lambda: catalogue)
    monkeypatch.setattr(data, "load_bonus_table", lambda: None)
    monkeypatch.setattr(cli, "catalogue_pool", lambda *_: catalogue.ids)
    plan = SimpleNamespace(asteroids=[57530], summary=lambda: {})
    route = SimpleNamespace(
        plan=plan,
        certified=True,
        refined_arc_count=1,
        collected_mass={57530: 600.0},
        total_collected_kg=600.0,
        summary=lambda: {"certified": True},
    )
    result = SimpleNamespace(
        candidates=[plan],
        expansions=1,
        lambert_evaluations=1,
        wall_seconds=0.0,
        failures=[],
        depth_reached=1,
        best_by_depth={},
    )
    monkeypatch.setattr(search, "RouteSearch", lambda *a, **kw: SimpleNamespace(run=lambda: result))
    monkeypatch.setattr(pipeline, "refine_route", lambda *a, **kw: route)
    monkeypatch.setattr(
        pipeline, "write_route_artifacts", lambda *a: {"solution": str(tmp_path / "route.txt")}
    )
    monkeypatch.setattr(solution.Solution, "read", lambda *a: None)
    monkeypatch.setattr(viewer_export, "write_viewer_dataset", lambda *a, **kw: {})
    monkeypatch.setattr(
        fleet,
        "assemble_fleet",
        lambda *a: SimpleNamespace(write=lambda path: path.write_text("synthetic test artifact")),
    )
    monkeypatch.setattr(
        cooperative,
        "MinerPool",
        lambda: SimpleNamespace(
            touched=lambda: set(),
            register=lambda *a: None,
            summary=lambda: {},
        ),
    )
    monkeypatch.setattr(
        cooperative.FleetColumn,
        "from_plan",
        lambda *a, **kw: SimpleNamespace(summary=lambda: {}),
    )
    monkeypatch.setattr(
        cooperative,
        "solve_fleet_master",
        lambda *a, **kw: SimpleNamespace(
            selected=[1],
            collected_kg=600.0,
            objective=600.0,
            summary=lambda: {},
            routes=lambda: [route],
        ),
    )

    def check(path, ok):
        # Every individual route passes; only the assembled fleet can fail.
        accepted = ok if Path(path).parent.name == "fleet" else True
        return SimpleNamespace(
            summary=lambda: {"ok": accepted, "total_mass_kg": 600.0},
            total_mass_kg=600.0,
            scored_masses={57530: 600.0},
            score_data={57530: 600.0},
        )

    monkeypatch.setattr(
        verifier,
        "Gtoc12Verifier",
        lambda *a, **kw: SimpleNamespace(verify_file=lambda path: check(path, independent_ok)),
    )
    monkeypatch.setattr(official, "official_verifier_available", lambda: True)
    monkeypatch.setattr(official, "run_official_verifier", lambda path: check(path, official_ok))
    args = build_parser().parse_args(
        [
            "gtoc12",
            "run",
            "--run-id",
            "final-check",
            "--output",
            str(tmp_path),
            "--full-catalogue",
            "--ships",
            "1",
            "--refine-top",
            "1",
            "--no-retime",
            "--no-cooperative",
            "--no-bonus-weights",
        ]
    )
    accepted = independent_ok and official_ok
    assert cli.cmd_run(args) == (0 if accepted else 1)
    report = json.loads((tmp_path / "run_report.json").read_text())
    assert report["ships"][0]["status"] == "scored"
    assert report["status"] == ("scored" if accepted else "fleet_failed_verification")
    assert report["best"]["accepted"] is accepted
    assert report["best"]["score_kg"] == (600.0 if accepted else None)
    assert (tmp_path / "fleet" / "Result.txt").is_file()
