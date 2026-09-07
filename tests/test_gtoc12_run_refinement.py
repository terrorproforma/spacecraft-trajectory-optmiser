"""Search must not mistake local solver failure for proven route infeasibility."""

import json
from types import SimpleNamespace

import numpy as np

from spacepdhcg.cli import build_parser
from spacepdhcg.gtoc12 import cli, data, pipeline, search


def test_failed_leg_does_not_suppress_later_candidates(tmp_path, monkeypatch):
    catalogue = SimpleNamespace(ids=np.array([57530]))
    monkeypatch.setattr(data, "load_catalogue", lambda: catalogue)
    monkeypatch.setattr(data, "load_bonus_table", lambda: None)
    monkeypatch.setattr(cli, "catalogue_pool", lambda *_: catalogue.ids)
    plans = []
    # Same bodies/departure: first a repeat, then a different arrival. Both
    # used to be skipped after the first local solver failure.
    for arrival in (64828.0, 64828.0, 64928.0, 65028.0):
        leg = SimpleNamespace(
            from_id=0, to_id=57530, departure_epoch=64328.0, arrival_epoch=arrival
        )
        plans.append(SimpleNamespace(legs=[leg], summary=lambda: {}))
    result = SimpleNamespace(
        candidates=plans,
        expansions=1,
        lambert_evaluations=4,
        wall_seconds=0.0,
        failures=[],
        depth_reached=1,
        best_by_depth={},
    )
    monkeypatch.setattr(search, "RouteSearch", lambda *a, **kw: SimpleNamespace(run=lambda: result))
    attempted = []

    def refine(plan, *args, **kwargs):
        attempted.append(plan)
        return SimpleNamespace(
            certified=False,
            refined_arc_count=1,
            legs=[SimpleNamespace(planned=plan.legs[0], certified=False)],
            summary=lambda: {"certified": False},
        )

    monkeypatch.setattr(pipeline, "refine_route", refine)
    args = build_parser().parse_args(
        [
            "gtoc12",
            "run",
            "--run-id",
            "retry-regression",
            "--output",
            str(tmp_path),
            "--full-catalogue",
            "--ships",
            "1",
            "--refine-top",
            "3",
            "--no-bonus-weights",
            "--no-retime",
            "--no-cooperative",
        ]
    )
    assert cli.cmd_run(args) == 0
    assert [id(plan) for plan in attempted] == [id(plan) for plan in plans[:3]]
    report = json.loads((tmp_path / "run_report.json").read_text())
    assert report["status"] == "no_certified_route"
    assert [row["rank"] for row in report["ships"][0]["refinements"]] == [0, 1, 2]
