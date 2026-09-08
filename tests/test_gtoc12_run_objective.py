"""Route promotion and retiming must preserve the independently verified objective."""

import json
from dataclasses import dataclass
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
    retiming,
    search,
    solution,
    verifier,
    viewer_export,
)


@dataclass(frozen=True)
class RouteCase:
    name: str
    asteroid: int
    mass: float
    independent_ok: bool = True
    official_ok: bool = True
    foreign: bool = False


@pytest.fixture
def run_routes(tmp_path, monkeypatch):
    catalogue = SimpleNamespace(ids=np.array([1, 2, 3]))
    bonus = SimpleNamespace(coefficient=np.array([1.0, 0.5, 1.0]))
    monkeypatch.setattr(data, "load_catalogue", lambda: catalogue)
    monkeypatch.setattr(data, "load_bonus_table", lambda: bonus)
    monkeypatch.setattr(cli, "catalogue_pool", lambda *_: catalogue.ids)
    monkeypatch.setattr(cli, "_commit", lambda *_: "synthetic-fixture")
    monkeypatch.setattr(cli, "_peak_rss_mb", lambda: 0.0)
    monkeypatch.setattr(solution.Solution, "read", lambda *a: None)
    monkeypatch.setattr(
        solution.Solution,
        "write",
        lambda value, path: path.write_text(",".join(ship.name for ship in value.ships)),
    )
    monkeypatch.setattr(
        pipeline,
        "emit_solution",
        lambda route, *a, **kw: SimpleNamespace(ships=[SimpleNamespace(name=route.plan.name)]),
    )
    monkeypatch.setattr(viewer_export, "write_viewer_dataset", lambda *a, **kw: {})
    monkeypatch.setattr(official, "official_verifier_available", lambda: True)
    monkeypatch.setattr(
        cooperative,
        "MinerPool",
        lambda: SimpleNamespace(touched=set, register=lambda *a: None, summary=dict),
    )
    monkeypatch.setattr(
        cooperative.FleetColumn,
        "from_plan",
        lambda *a, route, **kw: SimpleNamespace(slot=a[1], route=route, summary=dict),
    )

    def run(
        cases,
        *,
        weighted=True,
        retimed=(),
        primary=0,
        scores=None,
        prefix=None,
        context_failure=None,
        independent_codes=None,
        official_codes=None,
        final_codes=(),
        empty_master=False,
    ):
        all_cases = {
            case.name: case for case in [*cases, *retimed, *([] if prefix is None else [prefix])]
        }
        routes = {}
        for case in all_cases.values():
            deploys = {} if case.foreign else {case.asteroid: 65000.0}
            if case is prefix:
                deploys.update(
                    {item.asteroid: 65000.0 for item in all_cases.values() if item.foreign}
                )
            plan = SimpleNamespace(
                name=case.name,
                asteroids=[case.asteroid],
                collect_epochs={case.asteroid: 66000.0},
                deploy_epochs=deploys,
                foreign_deploy_epochs={case.asteroid: 65000.0} if case.foreign else {},
                summary=lambda name=case.name: {"name": name},
            )
            routes[case.name] = SimpleNamespace(
                plan=plan,
                certified=True,
                refined_arc_count=1,
                collected_mass={case.asteroid: case.mass},
                total_collected_kg=case.mass,
                final_mass_kg=1500.0,
                summary=lambda: {"certified": True},
            )
        batches = iter([cases] if prefix is None else [[prefix], cases])

        def make_search(*a, **kw):
            result = SimpleNamespace(
                candidates=[routes[case.name].plan for case in next(batches)],
                expansions=1,
                lambert_evaluations=1,
                wall_seconds=0.0,
                failures=[],
                depth_reached=1,
                best_by_depth={},
            )
            return SimpleNamespace(run=lambda: result)

        monkeypatch.setattr(search, "RouteSearch", make_search)
        monkeypatch.setattr(pipeline, "refine_route", lambda plan, *a, **kw: routes[plan.name])
        monkeypatch.setattr(retiming, "Retimer", lambda *a, **kw: None)

        def improve(plan, *a, **kw):
            variants = [] if prefix is not None and plan.name == prefix.name else retimed
            return SimpleNamespace(
                route=routes[variants[primary].name] if variants else None,
                certified_routes=[routes[case.name] for case in variants],
                summary=dict,
                attempts=[],
                wall_seconds=0.0,
            )

        monkeypatch.setattr(retiming, "improve_and_certify", improve)

        def artifacts(route, cat, directory):
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "Result.txt"
            path.write_text(route.plan.name)
            return {"solution": str(path)}

        monkeypatch.setattr(pipeline, "write_route_artifacts", artifacts)
        monkeypatch.setattr(
            fleet,
            "assemble_fleet",
            lambda plan, cat: SimpleNamespace(
                write=lambda path: path.write_text(
                    ",".join(route.plan.name for route in plan.routes)
                )
            ),
        )
        master_inputs = []

        def master(columns, *, weights, **kw):
            assert weights == ({1: 1.0, 2: 0.5, 3: 1.0} if weighted else None)
            master_inputs.append([column.route.plan.name for column in columns])
            selected = [
                max(
                    (column for column in columns if column.slot == slot),
                    key=lambda column: sum(
                        mass * (1.0 if weights is None else weights[asteroid])
                        for asteroid, mass in column.route.collected_mass.items()
                    ),
                )
                for slot in sorted({column.slot for column in columns})
            ]
            if empty_master:
                selected = []
            return SimpleNamespace(
                selected=selected,
                routes=lambda: [column.route for column in selected],
                summary=dict,
                objective=99999.0,
                collected_kg=sum(column.route.total_collected_kg for column in selected),
            )

        monkeypatch.setattr(cooperative, "solve_fleet_master", master)

        def read_cases(path):
            return [all_cases[name] for name in Path(path).read_text().split(",")]

        def dependencies_present(checked):
            return not any(case.foreign for case in checked) or (
                prefix is not None and prefix in checked
            )

        def check(path):
            checked = read_cases(path)
            codes = (
                final_codes
                if Path(path).parent.name == "fleet"
                else (independent_codes or {}).get(checked[-1].name, [])
            )
            weighted_score = sum(
                case.mass * bonus.coefficient[case.asteroid - 1] for case in checked
            )
            if scores and checked[-1].name in scores:
                weighted_score = scores[checked[-1].name]
            if scores and "__final__" in scores and Path(path).parent.name == "fleet":
                weighted_score = scores["__final__"]
            return verifier.VerificationReport(
                ok=all(case.independent_ok for case in checked)
                and dependencies_present(checked)
                and not codes
                and not (
                    context_failure == "independent"
                    and Path(path).parent.name == "verification_context"
                ),
                ship_count=len(checked),
                violations=[
                    verifier.Violation(code, None, None, "injected check") for code in codes
                ],
                legs=[],
                mined={
                    case.asteroid: verifier.MinedAsteroid(
                        case.asteroid, 1, 65000.0, index, 66000.0, case.mass, True, 66100.0
                    )
                    for index, case in enumerate(checked, start=1)
                },
                total_mass_kg=sum(case.mass for case in checked),
                ship_limit=100.0,
                weighted_score_fixed_bonus_kg=weighted_score,
            )

        def checker(*a, **kw):
            assert kw["bonus"] is bonus
            return SimpleNamespace(verify_file=check)

        def official_check(path):
            checked = read_cases(path)
            codes = (
                []
                if Path(path).parent.name == "fleet"
                else (official_codes or {}).get(checked[-1].name, [])
            )
            ok = all(case.official_ok for case in checked) and dependencies_present(checked)
            ok = ok and not codes
            if context_failure == "official" and Path(path).parent.name == "verification_context":
                ok = False
            return official.OfficialVerification(
                ok,
                len(checked),
                len(checked),
                sum(case.mass for case in checked),
                "\n".join(code + ". injected check" for code in codes),
                "",
                0 if not codes else 3 if codes == ["Error301"] else 1,
                {case.asteroid: case.mass for case in checked},
            )

        monkeypatch.setattr(verifier, "Gtoc12Verifier", checker)
        monkeypatch.setattr(official, "run_official_verifier", official_check)
        args = build_parser().parse_args(
            [
                "gtoc12",
                "run",
                "--run-id",
                "objective-fixture",
                "--output",
                str(tmp_path),
                "--full-catalogue",
                "--ships",
                "1" if prefix is None else "2",
                "--refine-top",
                str(len(cases)),
                "--refine-recovery",
                "0",
                "--no-cooperative",
                *([] if retimed else ["--no-retime"]),
                *([] if weighted else ["--no-bonus-weights"]),
            ]
        )
        status = cli.cmd_run(args)
        report = json.loads((tmp_path / "run_report.json").read_text())
        return status, report, master_inputs

    return run


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
def test_initial_route_choice_follows_verified_objective(run_routes, weighted, reverse):
    cases = [RouteCase("weighted-rich", 1, 600.0), RouteCase("raw-rich", 2, 800.0)]
    if reverse:
        cases.reverse()
    status, report, _ = run_routes(cases, weighted=weighted)
    expected_name = "weighted-rich" if weighted else "raw-rich"
    expected_score = 600.0 if weighted else 800.0
    assert status == 0
    assert report["ships"][0]["best"]["plan"]["name"] == expected_name
    assert report["ships"][0]["best"]["score_kg"] == expected_score
    assert report["best"]["score_kg"] == expected_score
    assert report["best"]["score_kind"] == (
        "weighted_score_fixed_bonus_kg" if weighted else "total_mass_kg"
    )
    assert report["best"]["total_mass_kg"] == expected_score


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize(
    "original,replacement",
    [
        (RouteCase("original", 1, 600.0), RouteCase("retimed", 2, 800.0)),
        (RouteCase("original", 2, 800.0), RouteCase("retimed", 1, 600.0)),
    ],
)
def test_retimed_promotion_uses_verified_objective(run_routes, weighted, original, replacement):
    status, report, master_inputs = run_routes([original], weighted=weighted, retimed=[replacement])
    expected = next(
        case for case in (original, replacement) if case.asteroid == (1 if weighted else 2)
    )
    assert status == 0
    assert report["ships"][0]["best"]["plan"]["name"] == expected.name
    assert report["best"]["score_kg"] == expected.mass
    assert master_inputs[-1] == ["original", "retimed"]
    after = report["ships"][0]["retiming"]["after"]
    assert after["total_mass_kg"] == replacement.mass
    assert after["weighted_score_fixed_bonus_kg"] == replacement.mass * (
        1.0 if replacement.asteroid == 1 else 0.5
    )


@pytest.mark.parametrize("independent_ok,official_ok", [(False, True), (True, False)])
def test_failed_retimed_primary_cannot_enter_master_or_hide_another_verified_variant(
    run_routes, independent_ok, official_ok
):
    retimed = [
        RouteCase("rejected-primary", 1, 1000.0, independent_ok, official_ok),
        RouteCase("accepted-variant", 1, 700.0),
    ]
    status, report, master_inputs = run_routes([RouteCase("original", 1, 600.0)], retimed=retimed)
    assert status == 0
    assert report["ships"][0]["best"]["plan"]["name"] == "accepted-variant"
    assert report["best"]["score_kg"] == 700.0
    assert master_inputs[-1] == ["original", "accepted-variant"]
    rejected = report["ships"][0]["retiming"]["verifications"][0]
    assert not rejected["accepted"] and rejected["score_kg"] is None
    assert rejected["total_mass_kg"] == 1000.0


@pytest.mark.parametrize("missing_score", [None, float("nan"), float("inf")])
@pytest.mark.parametrize("phase", ["initial", "retimed"])
def test_selected_weighted_objective_never_falls_back_to_raw_mass(run_routes, missing_score, phase):
    original = RouteCase("original", 1, 600.0)
    unscored = RouteCase("unscored", 1, 1000.0)
    cases = [original, unscored] if phase == "initial" else [original]
    retimed = [unscored] if phase == "retimed" else []
    status, report, master_inputs = run_routes(
        cases, retimed=retimed, scores={"unscored": missing_score}
    )
    assert status == 0 and report["best"]["score_kg"] == 600.0
    assert report["ships"][0]["best"]["plan"]["name"] == "original"
    assert master_inputs[-1] == ["original"]
    if phase == "initial":
        rejected = report["ships"][0]["refinements"][1]
    else:
        rejected = report["ships"][0]["retiming"]["verifications"][0]
    assert not rejected["accepted"] and rejected["score_kg"] is None
    assert rejected["total_mass_kg"] == 1000.0
    assert "weighted_score_fixed_bonus_kg" in rejected["score_error"]


@pytest.mark.parametrize("missing_score", [None, float("nan"), float("inf")])
def test_missing_final_weighted_score_fails_after_a_valid_route(run_routes, missing_score):
    status, report, _ = run_routes(
        [RouteCase("original", 1, 600.0)], scores={"__final__": missing_score}
    )
    assert report["ships"][0]["best"]["accepted"]
    assert status == 1 and report["status"] == "fleet_failed_verification"
    assert not report["best"]["accepted"] and report["best"]["score_kg"] is None
    assert report["best"]["total_mass_kg"] == 600.0


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("phase", ["initial", "retimed"])
@pytest.mark.parametrize("foreign_asteroid", [2, 3])
def test_cooperative_candidate_checks_fleet_context_but_ranks_its_own_cargo(
    run_routes, weighted, phase, foreign_asteroid
):
    prefix = RouteCase("deployer", 1, 1000.0)
    masses = {2: 800.0, 3: 600.0}
    own_asteroid = 5 - foreign_asteroid
    own = RouteCase("own", own_asteroid, masses[own_asteroid])
    foreign = RouteCase("foreign", foreign_asteroid, masses[foreign_asteroid], foreign=True)
    status, report, master_inputs = run_routes(
        [own, foreign] if phase == "initial" else [own],
        retimed=[foreign] if phase == "retimed" else [],
        prefix=prefix,
        weighted=weighted,
    )
    assert status == 0
    expected = next(case for case in (own, foreign) if case.asteroid == (3 if weighted else 2))
    ship = report["ships"][1]
    assert ship["best"]["plan"]["name"] == expected.name
    assert ship["best"]["score_kg"] == expected.mass
    assert report["best"]["score_kg"] == prefix.mass + expected.mass
    assert master_inputs[-1] == ["deployer", "own", "foreign"]
    entry = ship["refinements"][1] if phase == "initial" else ship["retiming"]["verifications"][0]
    assert entry["accepted"]
    assert entry["verification_scope"] == "fleet_with_candidate"
    assert entry["objective_scope"] == "candidate_returned_mass"
    assert entry["total_mass_kg"] == foreign.mass
    assert entry["weighted_score_fixed_bonus_kg"] == foreign.mass * (
        0.5 if foreign.asteroid == 2 else 1.0
    )
    assert entry["independent"]["total_mass_kg"] == prefix.mass + foreign.mass
    assert entry["official"]["total_mass_kg"] == prefix.mass + foreign.mass
    assert entry["candidate_scored_masses"] == {str(foreign.asteroid): foreign.mass}
    assert Path(entry["artifacts"]["solution"]).read_text() == "foreign"
    context_path = Path(entry["verification_artifacts"]["solution"])
    assert context_path.parent.name == "verification_context"
    assert context_path.read_text() == "deployer,foreign"


@pytest.mark.parametrize("phase", ["initial", "retimed"])
@pytest.mark.parametrize("context_failure", ["independent", "official"])
def test_failed_cooperative_context_cannot_promote_or_enter_master(
    run_routes, phase, context_failure
):
    own = RouteCase("own", 3, 600.0)
    foreign = RouteCase("foreign", 2, 1500.0, foreign=True)
    status, report, master_inputs = run_routes(
        [own, foreign] if phase == "initial" else [own],
        retimed=[foreign] if phase == "retimed" else [],
        prefix=RouteCase("deployer", 1, 1000.0),
        context_failure=context_failure,
    )
    assert status == 0
    ship = report["ships"][1]
    assert ship["best"]["plan"]["name"] == "own"
    assert report["best"]["score_kg"] == 1600.0
    assert master_inputs[-1] == ["deployer", "own"]
    rejected = (
        ship["refinements"][1] if phase == "initial" else ship["retiming"]["verifications"][0]
    )
    assert not rejected["accepted"] and rejected["score_kg"] is None
    assert rejected["verification_scope"] == "fleet_with_candidate"
    assert not rejected[context_failure]["ok"]


def supporting_route(name, deploys, foreign=None, masses=None):
    masses = masses or {}
    return SimpleNamespace(
        plan=SimpleNamespace(
            name=name,
            deploy_epochs=deploys,
            foreign_deploy_epochs=foreign or {},
            collect_epochs={asteroid: 66000.0 for asteroid in masses},
        ),
        collected_mass=masses,
        certified=True,
    )


def test_deployer_closure_recurses_and_matches_epochs():
    stale = supporting_route("stale", {20: 64900.0})
    first = supporting_route("first", {10: 65000.0})
    second = supporting_route("second", {20: 65010.0}, {10: 65000.0})
    unrelated = supporting_route("unrelated", {30: 65000.0})
    candidate = supporting_route("candidate", {}, {20: 65010.0})
    selected, missing = cli._candidate_support_routes([stale, first, unrelated, second], candidate)
    assert selected == [first, second, candidate] and not missing
    changed = supporting_route("changed", {}, {20: 65010.01})
    selected, missing = cli._candidate_support_routes([stale, first, second], changed)
    assert selected == [changed]
    assert missing == [{"asteroid": 20, "deploy_epoch": 65010.01}]


def test_invalid_support_context_does_not_prove_the_column_unusable():
    first = supporting_route("first", {1: 65000.0, 10: 65000.0}, masses={1: 100.0})
    second = supporting_route("second", {2: 65000.0, 20: 65000.0}, masses={2: 0.0})
    filler = supporting_route("high-cargo", {3: 65000.0}, masses={3: 600.0})
    unrelated = [supporting_route(f"low-{a}", {a: 65000.0}, masses={a: 0.0}) for a in range(4, 9)]
    candidate = supporting_route("candidate", {}, {10: 65000.0, 20: 65000.0}, {10: 50.0, 20: 50.0})
    previous = [first, second, filler, *unrelated]
    selected, missing = cli._candidate_support_routes(previous, candidate)
    assert selected == [first, second, candidate] and not missing
    columns = [
        cooperative.FleetColumn.from_plan(
            index,
            index,
            route.plan.name,
            route.plan,
            route.collected_mass,
            certified=True,
            route=route,
        )
        for index, route in enumerate([*previous, candidate])
    ]
    assert "exceed the limit" in cooperative.fleet_feasible(columns)
    closure = [column for column in columns if any(column.route is route for route in selected)]
    assert "exceed the limit" in cooperative.fleet_feasible(closure)
    master = cooperative.solve_fleet_master(columns, max_ships=10, lp_bound=False)
    assert any(route is candidate for route in master.routes())
    assert len(master.routes()) == 4 and master.objective == 800.0
    assert cooperative.fleet_feasible(master.selected) == ""


@pytest.mark.parametrize("phase", ["initial", "retimed"])
@pytest.mark.parametrize("final_failure", [False, True])
def test_ship_rule_only_certificate_reaches_final_master_without_becoming_a_scored_fleet(
    run_routes, phase, final_failure
):
    provisional = RouteCase("provisional", 1, 700.0)
    cases = [provisional] if phase == "initial" else [RouteCase("original", 2, 800.0)]
    status, report, master_inputs = run_routes(
        cases,
        retimed=[provisional] if phase == "retimed" else [],
        independent_codes={"provisional": ["Error301"]},
        official_codes={"provisional": ["Error301"]},
        final_codes=["Error203"] if final_failure else [],
    )
    ship = report["ships"][0]
    assert ship["status"] == "route_certified_pending_fleet"
    entry = ship["best"]
    assert not entry["accepted"] and not entry["ok"] and entry["score_kg"] is None
    assert entry["candidate_eligible"] and entry["candidate_score_kg"] == 700.0
    assert entry["candidate_certification"] == "independent_physics_pending_fleet"
    assert entry["requires_final_fleet_verification"]
    assert "provisional" in master_inputs[-1]
    assert status == (1 if final_failure else 0)
    assert report["status"] == ("fleet_failed_verification" if final_failure else "scored")
    assert report["best"]["score_kg"] == (None if final_failure else 700.0)


def test_provisional_columns_with_no_feasible_master_are_not_reported_as_a_scored_empty_fleet(
    run_routes,
):
    status, report, master_inputs = run_routes(
        [RouteCase("provisional", 1, 700.0)],
        independent_codes={"provisional": ["Error301"]},
        official_codes={"provisional": ["Error301"]},
        empty_master=True,
    )
    assert status == 0
    assert master_inputs[-1] == ["provisional"]
    assert report["best"] is None and report["status"] == "no_verified_fleet"


@pytest.mark.parametrize(
    "code",
    [
        "Error001",
        "Error002",
        "Error003",
        "Error004",
        "Error005",
        "Error101",
        "Error201",
        "Error202",
        "Error203",
        "Error401",
        "Error501",
        "Error502",
        "Error503",
        "Error504",
        "Error505",
        "Error506",
        "Error507",
        "Error601",
        "Error602",
        "Error603",
        "Error604",
        "Error605",
        "Error607",
        "Error701",
        "Error702",
        "Error703",
        "Error704",
        "Error705",
        "Error801",
        "Error802",
        "Error803",
        "Error804",
        "Error805",
        "Error806",
        "Error807",
        "Error901",
    ],
)
def test_other_independent_violations_cannot_become_provisional_columns(run_routes, code):
    status, report, master_inputs = run_routes(
        [RouteCase("original", 1, 600.0), RouteCase("rejected", 1, 700.0)],
        independent_codes={"rejected": ["Error301", code]},
        official_codes={"rejected": ["Error301"]},
    )
    assert status == 0 and report["best"]["score_kg"] == 600.0
    assert master_inputs[-1] == ["original"]
    entry = report["ships"][0]["refinements"][1]
    assert not entry["candidate_eligible"] and entry["candidate_score_kg"] is None
    assert not entry["accepted"] and entry["score_kg"] is None


@pytest.mark.parametrize("official_failure", [[], ["Error203"], ["Error301", "Error901"]])
def test_other_official_failures_cannot_become_provisional_columns(run_routes, official_failure):
    rejected = RouteCase("rejected", 1, 700.0, official_ok=False)
    status, report, master_inputs = run_routes(
        [RouteCase("original", 1, 600.0), rejected],
        independent_codes={"rejected": ["Error301"]},
        official_codes={"rejected": official_failure},
    )
    assert status == 0 and report["best"]["score_kg"] == 600.0
    assert master_inputs[-1] == ["original"]
    assert not report["ships"][0]["refinements"][1]["candidate_eligible"]
