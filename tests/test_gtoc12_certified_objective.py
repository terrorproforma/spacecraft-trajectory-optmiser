"""Certified route selection must retain the planner's objective after payload sizing."""

from types import SimpleNamespace

import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12 import retiming
from spacepdhcg.gtoc12.pipeline import RefinedRoute
from spacepdhcg.gtoc12.search import RoutePlan


def plan(masses, *, orphan=False):
    deploy = {a: C.MISSION_START_MJD for a in masses}
    if orphan:
        deploy[3] = C.MISSION_START_MJD
    return RoutePlan((), deploy, {a: C.MISSION_END_MJD - 500 for a in masses}, masses, 0.0, 3000.0)


@pytest.mark.parametrize(
    "weights,actual,orphan_credit,expected",
    [
        ({1: 1.0, 2: 0.5}, ({1: 100.0}, {2: 150.0}), 0.0, 0),
        (None, ({1: 100.0}, {2: 150.0}), 0.0, 1),
        # Proxy ranking favours the second route, but payload sizing reverses it.
        ({1: 1.0, 2: 0.5}, ({1: 100.0}, {2: 190.0}), 0.0, 0),
        # Equal objectives retain the first certified route deterministically.
        ({1: 1.0, 2: 0.5}, ({1: 100.0}, {2: 200.0}), 0.0, 0),
        # The configured value of a deployed, uncollected miner remains part of
        # the planning objective; it is not reported as already collected mass.
        ({1: 1.0, 2: 0.5}, ({1: 100.0}, {2: 150.0}), 0.5, 1),
    ],
)
def test_certified_choice_uses_actual_planning_objective(
    monkeypatch, weights, actual, orphan_credit, expected
):
    original = plan({1: 1.0})
    candidates = [plan({1: 110.0}), plan({2: 250.0}, orphan=orphan_credit > 0)]
    routes = [
        RefinedRoute(candidate, [], mass, 600.0, True, True, 1, 0.0, {})
        for candidate, mass in zip(candidates, actual, strict=True)
    ]
    improvements = iter(candidates)
    refinements = iter(routes)
    monkeypatch.setattr(
        retiming,
        "improve_plan",
        lambda *args, **kwargs: SimpleNamespace(plan=next(improvements), summary=lambda: {}),
    )
    timer = SimpleNamespace(
        weights=weights,
        settings=retiming.RetimeSettings(orphan_credit=orphan_credit),
        bans={},
        inflations={},
    )
    result = retiming.improve_and_certify(
        original, None, timer, None, max_attempts=2, refine=lambda candidate: next(refinements)
    )
    assert result.route is routes[expected]
    assert result.certified_routes == routes
    first, second = (a["refined"]["objective_kg"] for a in result.attempts)
    assert (second > first) == (expected == 1)
    assert result.attempts[1]["refined"]["collected_kg"] == sum(actual[1].values())
