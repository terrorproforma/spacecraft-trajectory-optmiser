"""Campaign promotion must match the verified objective, including the final pass.

Synthetic search/checker boundaries isolate command behavior without launching a
trajectory search or claiming that these fixtures constitute physics validation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.cli import build_parser
from spacepdhcg.gtoc12 import (
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


@dataclass(frozen=True)
class Candidate:
    name: str
    asteroid: int
    raw_kg: float
    independent_ok: bool = True
    official_ok: bool = True


@pytest.fixture
def run_campaign(tmp_path, monkeypatch):
    catalogue = SimpleNamespace(ids=np.array([1, 2]))
    bonus = SimpleNamespace(coefficient=np.array([1.0, 0.5]))
    monkeypatch.setattr(data, "load_catalogue", lambda: catalogue)
    monkeypatch.setattr(data, "load_bonus_table", lambda: bonus)
    monkeypatch.setattr(cli, "catalogue_pool", lambda *_: catalogue.ids)
    monkeypatch.setattr(cli, "_commit", lambda *_: "synthetic-fixture")
    monkeypatch.setattr(cli, "_peak_rss_mb", lambda: 0.0)
    memory = SimpleNamespace(peak_mb=0.0, samples=0, stop=lambda: None)
    monkeypatch.setattr(cli, "_MemorySampler", lambda: SimpleNamespace(start=lambda: memory))
    monkeypatch.setattr(
        bundles,
        "family_partitions",
        lambda *a, **kw: [(0, catalogue.ids, {"partition": "synthetic"})],
    )
    monkeypatch.setattr(bundles, "bundle_columns", lambda *a: [object()])
    monkeypatch.setattr(solution.Solution, "read", lambda *a: None)
    monkeypatch.setattr(viewer_export, "write_viewer_dataset", lambda *a, **kw: {})
    monkeypatch.setattr(official, "official_verifier_available", lambda: True)

    def run(candidates, *, weighted=True):
        by_name = {candidate.name: candidate for candidate in candidates}
        masters = iter(candidates)
        objective_weights = {1: 1.0, 2: 0.5} if weighted else None

        def master(*a, **kw):
            assert kw["weights"] == objective_weights
            candidate = next(masters)
            route = SimpleNamespace(
                plan=SimpleNamespace(asteroids=[candidate.asteroid]),
                total_collected_kg=candidate.raw_kg,
                candidate=candidate,
            )
            return SimpleNamespace(
                selected=[object()],
                routes=lambda: [route],
                ships=1,
                collected_kg=candidate.raw_kg,
                # A planner estimate is not a substitute for the checker result.
                objective=99999.0,
                exhaustive=False,
                summary=lambda: {"objective": 99999.0},
            )

        monkeypatch.setattr(cooperative, "solve_fleet_master", master)

        def price(*a, on_result, **kw):
            for index in range(len(candidates) - 1):
                on_result(
                    SimpleNamespace(
                        consistent=lambda: None,
                        label=index,
                        members=catalogue.ids,
                        ships=[],
                        peak_rss_mb=0.0,
                        wall_seconds=0.0,
                        summary=lambda: {
                            "rejected": [],
                            "earth_legs": {},
                            "repairs": [],
                            "cooperative": {},
                        },
                    )
                )

        monkeypatch.setattr(bundles, "price_clusters", price)
        monkeypatch.setattr(
            fleet,
            "assemble_fleet",
            lambda plan, cat: SimpleNamespace(
                write=lambda path: path.write_text(plan.routes[0].candidate.name)
            ),
        )

        def check(path):
            candidate = by_name[Path(path).read_text()]
            return verifier.VerificationReport(
                ok=candidate.independent_ok,
                ship_count=1,
                violations=[],
                legs=[],
                mined={},
                total_mass_kg=candidate.raw_kg,
                ship_limit=100.0,
                weighted_score_fixed_bonus_kg=float(
                    candidate.raw_kg * bonus.coefficient[candidate.asteroid - 1]
                ),
            )

        def official_check(path):
            candidate = by_name[Path(path).read_text()]
            return official.OfficialVerification(
                candidate.official_ok, 1, 1, candidate.raw_kg, "", "", 0
            )

        monkeypatch.setattr(
            verifier,
            "Gtoc12Verifier",
            lambda *a, **kw: SimpleNamespace(verify_file=check),
        )
        monkeypatch.setattr(official, "run_official_verifier", official_check)
        args = build_parser().parse_args(
            [
                "gtoc12",
                "cluster-fleet",
                "--run-id",
                "objective-fixture",
                "--output",
                str(tmp_path),
                "--no-lp-duals",
                "--workers",
                "1",
                *([] if weighted else ["--no-bonus-weights"]),
            ]
        )
        status = cli.cmd_cluster_fleet(args)
        report = json.loads((tmp_path / "run_report.json").read_text())
        return status, report

    return run


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
def test_promotion_and_budget_marks_follow_verified_objective(run_campaign, weighted, reverse):
    raw_rich = Candidate("raw-rich", 2, 750.0)  # 375 weighted kg
    weighted_rich = Candidate("weighted-rich", 1, 600.0)  # 600 weighted kg
    candidates = [raw_rich, weighted_rich]
    if reverse:
        candidates.reverse()
    # A final re-evaluation of the last candidate must not erase a better one.
    status, report = run_campaign([*candidates, candidates[-1]], weighted=weighted)
    expected = weighted_rich if weighted else raw_rich
    objective = 600.0 if weighted else 750.0
    assert status == 0 and report["status"] == "scored"
    assert Path(report["best"]["artifacts"]["solution"]).read_text() == expected.name
    assert report["best"]["score_kg"] == objective
    assert report["best"]["total_mass_kg"] == expected.raw_kg
    assert report["score_kind"] == (
        "weighted_score_fixed_bonus_kg" if weighted else "total_mass_kg"
    )
    assert all(mark["score_kg"] == objective for mark in report["budget_marks"].values())
    assert all(mark["total_mass_kg"] == expected.raw_kg for mark in report["budget_marks"].values())
    assert [item["verified_total_mass_kg"] for item in report["timeline"]] == [
        candidate.raw_kg for candidate in candidates
    ]
    assert report["timeline"][-1]["incumbent_score_kg"] == objective


@pytest.mark.parametrize(
    "final,expected_score,expected_name",
    [
        (Candidate("worse-final", 2, 800.0), 600.0, "incumbent"),
        (Candidate("better-final", 1, 700.0), 700.0, "better-final"),
        (Candidate("equal-final", 1, 600.0), 600.0, "incumbent"),
    ],
)
def test_final_candidate_cannot_reduce_verified_best(
    run_campaign, final, expected_score, expected_name
):
    status, report = run_campaign([Candidate("incumbent", 1, 600.0), final])
    assert status == 0
    assert report["best"]["score_kg"] == expected_score
    assert Path(report["best"]["artifacts"]["solution"]).read_text() == expected_name
    assert report["final_fleet"]["ok"]
    assert report["final_fleet"]["total_mass_kg"] == final.raw_kg
    assert all(mark["score_kg"] == expected_score for mark in report["budget_marks"].values())


@pytest.mark.parametrize("independent_ok,official_ok", [(False, True), (True, False)])
@pytest.mark.parametrize("incumbent_ok", [True, False])
def test_failed_final_is_nonzero_and_preserves_only_a_verified_incumbent(
    run_campaign, independent_ok, official_ok, incumbent_ok
):
    incumbent = Candidate("incumbent", 1, 600.0, independent_ok=incumbent_ok)
    final = Candidate("failed-final", 1, 900.0, independent_ok, official_ok)
    status, report = run_campaign([incumbent, final])
    assert status == 1 and report["status"] == "fleet_failed_verification"
    assert not report["final_fleet"]["ok"]
    assert report["final_fleet"]["score_kg"] is None
    assert report["final_fleet"]["total_mass_kg"] == 900.0
    assert report["final_fleet"]["official"]["total_mass_kg"] == 900.0
    if incumbent_ok:
        assert report["best"]["score_kg"] == 600.0
        assert Path(report["best"]["artifacts"]["solution"]).read_text() == "incumbent"
        assert all(mark["score_kg"] == 600.0 for mark in report["budget_marks"].values())
    else:
        assert report["best"] is None
        assert all(mark is None for mark in report["budget_marks"].values())
