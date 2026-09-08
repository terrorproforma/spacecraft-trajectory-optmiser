"""CPU-only checks for substitution construction, fleet inventory and promotion."""

import ctypes
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("asteroid_substitution_v599", HERE / "run.py")
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_viewer_failure_retains_new_verified_score_and_immutable_older_result(tmp_path):
    first = tmp_path / "first.txt"
    first.write_text("first verified solution")
    report = {"best": {"score_kg": 10.0}, "promotions": []}

    def checked(score):
        return {
            "ok": True,
            "independent": {"ok": True},
            "official": {"ok": True},
            "score_kg": score,
            "total_mass_kg": driver.MIN_FLEET_RAW_KG + 1,
        }

    assert driver.publish_verified_candidate(
        tmp_path,
        first,
        checked(11),
        {"ship": 2},
        report,
        lambda destination: {"directory": str(destination)},
    )
    previous_path = Path(report["best"]["solution"])
    second = tmp_path / "second.txt"
    second.write_text("second verified solution")

    def failed_viewer(_destination):
        raise RuntimeError("viewer write failed")

    assert driver.publish_verified_candidate(
        tmp_path, second, checked(12), {"ship": 7}, report, failed_viewer
    )
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["best"]["score_kg"] == 12
    assert saved["best"]["case"] == {"ship": 7}
    assert "viewer write failed" in saved["best"]["viewer_export_error"]
    assert Path(saved["best"]["solution"]).read_text() == "second verified solution"
    assert previous_path.read_text() == "first verified solution"
    rejected = checked(13) | {"official": {"ok": False}}
    assert not driver.publish_verified_candidate(
        tmp_path, first, rejected, {"ship": 9}, report, failed_viewer
    )
    assert json.loads((tmp_path / "report.json").read_text())["best"]["score_kg"] == 12


@pytest.fixture(scope="module", autouse=True)
def no_native_execution():
    """Fail if a future change tries to load a native library during these tests."""
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "SPACEPDHCG_GTOC12_GPU_TESTS": "0",
        "SPACEPDHCG_TEST_GTOC12_JOINT_BATCH": "0",
    }
    with (
        patch.dict(os.environ, environment),
        patch.object(
            ctypes, "CDLL", side_effect=AssertionError("CPU tests must not load native libraries")
        ),
        patch.object(
            ctypes.cdll,
            "LoadLibrary",
            side_effect=AssertionError("CPU tests must not load native libraries"),
        ),
    ):
        yield


@dataclass(frozen=True)
class Visit:
    body: int
    deploy: bool
    collect: bool
    role_out: str
    foreign_deploy_epoch: float | None = None
    pinned_arrival: float | None = None


@pytest.fixture
def paired():
    visits = [
        Visit(0, False, False, "earth_out"),
        Visit(1, True, False, "deploy_hop"),
        Visit(2, True, False, "deploy_hop"),
        Visit(3, True, False, "collect_hop"),
        Visit(1, False, True, "collect_hop"),
        Visit(2, False, True, "collect_hop"),
        Visit(3, False, True, "earth_return"),
        Visit(0, False, False, ""),
    ]
    arrivals = 64328.0 + np.array([0, 500, 700, 900, 1500, 1700, 1900, 2400])
    return visits, arrivals, arrivals.copy()


@pytest.fixture
def camp():
    visits = [
        Visit(0, False, False, "earth_out"),
        Visit(1, True, False, "deploy_hop"),
        Visit(2, True, True, "collect_hop"),
        Visit(1, False, True, "earth_return"),
        Visit(0, False, False, ""),
    ]
    arrivals = 64328.0 + np.array([0, 500, 700, 1900, 2400])
    departures = arrivals.copy()
    departures[2] += 800
    return visits, arrivals, departures


@pytest.mark.parametrize("layout", ["paired", "camp"])
def test_replacement_preserves_actions_epochs_and_original(request, layout):
    visits, arrivals, departures = request.getfixturevalue(layout)
    original = deepcopy(visits)
    old_arr, old_dep = arrivals.copy(), departures.copy()
    changed = driver.replacement_visits(visits, 2, 99, {7, 8})
    assert visits == original
    assert changed is not visits and len(changed) == len(visits)
    assert Counter((v.deploy, v.collect, v.role_out) for v in changed) == Counter(
        (v.deploy, v.collect, v.role_out) for v in original
    )
    assert [v.body for v in changed] == [99 if v.body == 2 else v.body for v in original]
    assert sum(v.deploy for v in changed if v.body == 99) == 1
    assert sum(v.collect for v in changed if v.body == 99) == 1
    assert sum(v.body == 99 for v in changed) == (1 if layout == "camp" else 2)
    np.testing.assert_array_equal(arrivals, old_arr)
    np.testing.assert_array_equal(departures, old_dep)
    # Modifying the returned container cannot alter the original visit container.
    changed[0] = replace(changed[0], body=999)
    assert visits == original


@pytest.mark.parametrize(
    "old,new,excluded",
    [
        (2, 2, set()),
        (2, 0, set()),
        (2, -1, set()),
        (2, 99, {99}),
        (2, 3, set()),
        (88, 99, set()),
        (1, 99, set()),
        (3, 99, set()),
    ],
    ids=[
        "unchanged",
        "earth",
        "negative",
        "occupied",
        "duplicate",
        "absent",
        "earth-out-endpoint",
        "earth-return-endpoint",
    ],
)
def test_replacement_rejects_invalid_footprint_or_endpoint(paired, old, new, excluded):
    with pytest.raises(ValueError, match=r"footprint|Earth endpoints"):
        driver.replacement_visits(paired[0], old, new, excluded)


@pytest.mark.parametrize("invalid", ["orphan", "collect-only", "double-deploy", "double-collect"])
def test_replacement_requires_one_own_deployment_and_collection(paired, invalid):
    visits = list(paired[0])
    if invalid == "orphan":
        visits[5] = replace(visits[5], collect=False)
    elif invalid == "collect-only":
        visits[2] = replace(visits[2], deploy=False)
    elif invalid == "double-deploy":
        visits[5] = replace(visits[5], deploy=True)
    else:
        visits[2] = replace(visits[2], collect=True)
    with pytest.raises(ValueError, match="exactly one own deployment and collection"):
        driver.replacement_visits(visits, 2, 99, set())


@pytest.mark.parametrize(
    "field,value", [("foreign_deploy_epoch", 64328.0), ("pinned_arrival", 65028.0)]
)
@pytest.mark.parametrize("index", [2, 4], ids=["replacement-visit", "other-visit"])
def test_replacement_rejects_cooperative_or_pinned_route(paired, field, value, index):
    visits = list(paired[0])
    visits[index] = replace(visits[index], **{field: value})
    with pytest.raises(ValueError, match="cooperative or pinned"):
        driver.replacement_visits(visits, 2, 99, set())


def mining_stays(visits, arrivals, departures):
    deployed = {v.body: arrivals[i] for i, v in enumerate(visits) if v.deploy}
    return {v.body: departures[i] - deployed[v.body] for i, v in enumerate(visits) if v.collect}


@pytest.mark.parametrize("layout", ["paired", "camp"])
def test_epoch_seeds_move_only_the_intended_mining_event(request, layout):
    visits, arrivals, departures = request.getfixturevalue(layout)
    visits = driver.replacement_visits(visits, 2, 99, set())
    original_arr, original_dep = arrivals.copy(), departures.copy()
    baseline_stays = mining_stays(visits, arrivals, departures)
    seeds = driver.epoch_seeds(visits, arrivals, departures, 99)
    assert [mode for mode, _, _ in seeds] == [
        "unchanged_epochs",
        "deploy_10_days_earlier",
        "collect_10_days_later",
    ]
    for mode, arr, dep in seeds:
        np.testing.assert_array_equal(arr[[0, -1]], arrivals[[0, -1]])
        np.testing.assert_array_equal(dep[[0, -1]], departures[[0, -1]])
        unaffected = [i for i, v in enumerate(visits) if v.body != 99]
        np.testing.assert_array_equal(arr[unaffected], arrivals[unaffected])
        np.testing.assert_array_equal(dep[unaffected], departures[unaffected])
        stays = mining_stays(visits, arr, dep)
        assert stays[99] == baseline_stays[99] + (0 if mode == "unchanged_epochs" else 10)
        assert {a: t for a, t in stays.items() if a != 99} == {
            a: t for a, t in baseline_stays.items() if a != 99
        }
        if layout == "paired":
            np.testing.assert_array_equal(dep, arr)
        elif mode == "deploy_10_days_earlier":
            assert dep[2] == departures[2] and arr[2] == arrivals[2] - 10
        elif mode == "collect_10_days_later":
            assert arr[2] == arrivals[2] and dep[2] == departures[2] + 10
        assert np.all(arr[1:] > dep[:-1]) and np.all(dep >= arr)
        assert not np.shares_memory(arr, arrivals) and not np.shares_memory(dep, departures)
    seeds[0][1][0] -= 1
    assert seeds[1][1][0] == original_arr[0]
    np.testing.assert_array_equal(arrivals, original_arr)
    np.testing.assert_array_equal(departures, original_dep)


@pytest.mark.parametrize("blocked", ["deploy", "collect"])
def test_epoch_seeds_exclude_zero_duration_transfer(paired, blocked):
    visits, arr, dep = paired
    visits = driver.replacement_visits(visits, 2, 99, set())
    if blocked == "deploy":
        arr[2] = dep[2] = dep[1] + 10
        rejected_mode = "deploy_10_days_earlier"
    else:
        arr[6] = dep[6] = dep[5] + 10
        rejected_mode = "collect_10_days_later"
    modes = [mode for mode, _, _ in driver.epoch_seeds(visits, arr, dep, 99)]
    assert "unchanged_epochs" in modes and rejected_mode not in modes


@pytest.mark.parametrize("index,epoch", [(0, 64327.0), (-1, 69808.0)])
def test_epoch_seeds_reject_outside_mission_window(paired, index, epoch):
    visits, arr, dep = paired
    arr[index] = dep[index] = epoch
    assert driver.epoch_seeds(visits, arr, dep, 2) == []


def test_objective_gate_allows_raw_loss_only_with_exact_fleet_slack():
    # Invert n = 2 exp(0.004 * total_mass / n) at exactly n = 23.
    boundary = 23 * math.log(23 / 2) / 0.004
    assert driver.MIN_FLEET_RAW_KG == pytest.approx(boundary, abs=1e-10)
    fleet = boundary + 12.0
    assert driver.objective_gate(599, 501, 600, 500, fleet)
    assert driver.objective_gate(588.000002, 501, 600, 500, fleet)
    assert not driver.objective_gate(588, 501, 600, 500, fleet)
    assert not driver.objective_gate(587, 501, 600, 500, fleet)


@pytest.mark.parametrize("weighted", [499.0, 500.0, 500.5])
def test_objective_gate_requires_strict_weighted_gain(weighted):
    assert not driver.objective_gate(620, weighted, 600, 500, driver.MIN_FLEET_RAW_KG + 20)


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_objective_gate_rejects_nonfinite_inputs(nonfinite):
    names = ["raw", "weighted", "before_raw", "before_weighted", "fleet_raw", "minimum_gain"]
    valid = dict(
        raw=599,
        weighted=501,
        before_raw=600,
        before_weighted=500,
        fleet_raw=driver.MIN_FLEET_RAW_KG + 12,
        minimum_gain=0.5,
    )
    for field in names:
        assert not driver.objective_gate(**(valid | {field: nonfinite})), field
    assert not driver.objective_gate(**(valid | {"minimum_gain": -0.5}))


def accepted_check():
    return {
        "ok": True,
        "independent": {"ok": True},
        "official": {"ok": True},
        "score_kg": 12811.0,
        "total_mass_kg": driver.MIN_FLEET_RAW_KG + 1,
    }


def test_verified_promotion_requires_weighted_gain_and_feasible_ship_count():
    checked = accepted_check()
    assert driver.verified_promotion(checked, 12810.0)
    assert not driver.verified_promotion(checked, 12811.0)
    assert not driver.verified_promotion(checked, 12812.0)
    assert not driver.verified_promotion(
        checked | {"total_mass_kg": driver.MIN_FLEET_RAW_KG - 0.01}, 12810.0
    )


@pytest.mark.parametrize("failed", ["ok", "independent", "official"])
def test_verified_promotion_rejects_either_failed_checker(failed):
    checked = accepted_check()
    checked[failed] = False if failed == "ok" else {"ok": False}
    assert not driver.verified_promotion(checked, 12810.0)


@pytest.mark.parametrize("field", ["ok", "independent", "official", "score_kg", "total_mass_kg"])
def test_verified_promotion_rejects_missing_evidence(field):
    checked = accepted_check()
    del checked[field]
    assert not driver.verified_promotion(checked, 12810.0)


@pytest.mark.parametrize("value", [None, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["score_kg", "total_mass_kg"])
def test_verified_promotion_rejects_nonfinite_scores_and_mass(field, value):
    assert not driver.verified_promotion(accepted_check() | {field: value}, 12810.0)


@pytest.fixture(scope="module")
def frozen_inventory(no_native_execution):
    bonus_path = HERE.parent / "family-result-audit-v592" / "bonus_coefficients.txt"
    if not bonus_path.is_file() or not (HERE / "inputs/fresh-verification.json").is_file():
        pytest.skip("optional pinned local fleet audit and bonus fixture unavailable")
    assert driver.sha256(bonus_path) == driver.BONUS_SHA
    summaries, audit, _ = driver.load_inputs()
    # Import only the solution parser from the frozen source, never the working tree.
    with (
        patch.object(sys, "path", list(sys.path)),
        patch.object(sys, "meta_path", list(sys.meta_path)),
    ):
        driver.activate_source()
        from spacepdhcg.gtoc12.solution import Solution

        module_path = Path(sys.modules[Solution.__module__].__file__).resolve()
        assert module_path.is_relative_to((HERE / "source/src").resolve())
        solution = Solution.read(HERE / "inputs/Result.txt")
    coefficients = np.loadtxt(bonus_path, usecols=0)
    assert len(coefficients) == 60000
    weights = {i + 1: float(value) for i, value in enumerate(coefficients)}
    return summaries, audit, weights, solution


def test_frozen_23_ship_inventory_matches_independent_scoring(frozen_inventory):
    summaries, audit, weights, solution = frozen_inventory
    rows, footprints = driver.inventory(summaries, audit, weights, solution)
    assert [row["ship"] for row in rows] == list(range(1, 24))
    assert (
        sum(len(ids) for ids in footprints.values()) == len(set.union(*footprints.values())) == 196
    )
    assert sum(row["collect_count"] for row in rows) == 195
    assert sum(row["verified_raw_kg"] for row in rows) == pytest.approx(
        14051.854893908598, abs=1e-7
    )
    assert sum(row["verified_weighted_kg"] for row in rows) == pytest.approx(
        12810.135953048577, abs=1e-7
    )
    ship15 = next(row for row in rows if row["ship"] == 15)
    assert ship15["deploy_count"] == ship15["collect_count"] == 9
    for row in rows:
        assert row["fixed_earth_departure_target"] not in row["replaceable"]
        assert row["fixed_earth_return_source"] not in row["replaceable"]
    slack = audit["independent"]["total_mass_kg"] - driver.MIN_FLEET_RAW_KG
    ship2 = next(row for row in rows if row["ship"] == 2)
    args = (
        ship2["verified_weighted_kg"] + 1,
        ship2["verified_raw_kg"],
        ship2["verified_weighted_kg"],
        audit["independent"]["total_mass_kg"],
    )
    assert slack > 0 and driver.objective_gate(ship2["verified_raw_kg"] - slack / 2, *args)
    assert not driver.objective_gate(ship2["verified_raw_kg"] - slack - 0.001, *args)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("deploy_epoch", "does not describe"),
        ("collect_epoch", "does not describe"),
        ("foreign", "cooperative incumbent"),
        ("collected_mass", "independent scored masses"),
        ("missing_collection_mass", "independent scored masses"),
        ("uncertified", "uncertified"),
    ],
)
def test_frozen_inventory_rejects_mutated_summary(frozen_inventory, mutation, reason):
    summaries, audit, weights, solution = frozen_inventory
    changed = deepcopy(summaries)
    summary = changed[2]
    body = "41045"
    if mutation in ("deploy_epoch", "collect_epoch"):
        phase = "deploy_epochs" if mutation == "deploy_epoch" else "collect_epochs"
        summary["plan"][phase][body] += 0.5
    elif mutation == "foreign":
        summary["plan"]["foreign_deploy_epochs"] = {body: 64328.0}
    elif mutation == "collected_mass":
        summary["collected_mass_kg"][body] += 1
    elif mutation == "missing_collection_mass":
        del summary["collected_mass_kg"][body]
    else:
        summary["certified"] = False
    with pytest.raises(ValueError, match=reason):
        driver.inventory(changed, audit, weights, solution)


@pytest.mark.parametrize("bad_weight", [None, 0.0, -1.0, math.nan, math.inf])
def test_frozen_inventory_requires_present_finite_positive_bonus(frozen_inventory, bad_weight):
    summaries, audit, weights, solution = frozen_inventory
    changed = dict(weights)
    if bad_weight is None:
        del changed[41045]
    else:
        changed[41045] = bad_weight
    with pytest.raises(ValueError, match="fixed-bonus coefficients"):
        driver.inventory(summaries, audit, changed, solution)
