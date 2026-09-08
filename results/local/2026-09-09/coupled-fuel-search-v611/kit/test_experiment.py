"""CPU behavior checks for finite search controls and actual incumbent fixtures."""

# Frozen activation precedes production imports.
# ruff: noqa: E402

import ctypes
import math
import os
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import common
import pytest

common.activate()
from domain import bounds, incumbent_joint, search_order, shortlist
from fixed_refine import refine_fixed, validate_prescription
from freeze_incumbents import ship_bytes

from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
from spacepdhcg.gtoc12.jointopt import route_from_summary
from spacepdhcg.gtoc12.low_thrust import ScvxSettings
from spacepdhcg.gtoc12.solution import Solution


@pytest.fixture(scope="module", autouse=True)
def no_native():
    with patch.object(
        ctypes, "CDLL", side_effect=AssertionError("CPU tests prohibit native loading")
    ):
        yield


@pytest.fixture(scope="module")
def data():
    os.environ["SPACEPDHCG_GTOC12_DATA"] = common.read(common.ROOT / "profile.json")["data"]
    os.environ["SPACEPDHCG_TEST_GTOC12_JOINT_BATCH"] = "0"
    catalogue, bonus = load_catalogue(), load_bonus_table()
    weights = {int(a): float(bonus.coefficient[int(a) - 1]) for a in catalogue.ids}
    return catalogue, weights


@pytest.mark.parametrize("ship", common.SHIPS)
def test_actual_incumbent_start_is_feasible_measured_and_independent(data, ship):
    catalogue, weights = data
    joint, route, visits, arr, dep = incumbent_joint(catalogue, weights, ship, 0.05)
    checked = joint.evaluate(visits, arr, dep)
    assert checked.feasible and checked.measured_legs == 17
    assert len(visits) == 18 and len(route.legs) == 17
    validate_prescription(checked.plan, checked.plan.collected_mass)
    fleet = Solution.read(common.ROOT / "inputs/Result.txt")
    other = {e.event_id for s in fleet.ships if s.ship_id != ship for e in s.asteroid_visits()}
    assert not set(route.plan.asteroids) & other
    assert ship_bytes(common.ROOT / "inputs/Result.txt", ship) == ship_bytes(
        common.ROOT / "inputs/archived-control-fleet.txt", ship
    )


@pytest.mark.parametrize("price", common.PRICES)
def test_fuel_price_changes_only_proxy_objective_not_cargo_or_physics(data, price):
    catalogue, weights = data
    joint, route, visits, arr, dep = incumbent_joint(catalogue, weights, 7, price)
    evaluation = joint.evaluate(visits, arr, dep)
    assert evaluation.feasible and evaluation.collected_kg == pytest.approx(
        route.total_collected_kg
    )
    assert evaluation.objective == pytest.approx(
        evaluation.weighted_kg + price * evaluation.spare_kg
    )
    assert evaluation.measured_legs == 17


def test_equal_arm_caps_and_exact_native_row_budget():
    order = search_order()
    assert len(order) == 12 and len(set(order)) == 12
    assert all(sum(p == price for _, p in order) == 4 for price in common.PRICES)
    cap = bounds()
    assert cap["max_itinerary_rows"] == 95628
    assert cap["max_lambert_direction_requests"] == 3251352
    assert cap["max_native_leg_solves"] == 68


def candidate(price, gain, identity, ship=7, eligible=True, spare=0):
    return {
        "margin_price": price,
        "weighted_gain_kg": gain,
        "plan_sha256": identity,
        "ship": ship,
        "objective_eligible": eligible,
        "spare_kg": spare,
    }


def test_shortlist_prioritizes_weighted_gain_and_deduplicates_arms():
    rows = [
        candidate(0.05, 3, "same"),
        candidate(0.25, 5, "same"),
        candidate(0.25, 4, "other", ship=20),
        candidate(1.0, 100, "invalid", eligible=False),
        candidate(1.0, 2, "last", spare=500),
    ]
    assert [r["plan_sha256"] for r in shortlist(rows)] == ["same", "other", "last"]
    assert shortlist(list(reversed(rows))) == shortlist(rows)


@pytest.mark.parametrize(
    "raw,weighted", [(math.nan, 601), (601, math.nan), (601, math.inf), (500, 601), (610, 599)]
)
def test_shortlist_gate_rejects_nonfinite_or_wrong_objective(raw, weighted):
    baseline = {"raw_kg": 14051.854893908598}
    original = {"raw_kg": 614, "weighted_kg": 600}
    assert not common.eligible(raw, weighted, baseline, original)


def test_weighted_gain_can_admit_small_raw_loss_within_actual_fleet_slack():
    baseline = {"raw_kg": 14051.854893908598}
    original = {"raw_kg": 614, "weighted_kg": 600}
    assert common.eligible(610, 601, baseline, original)
    assert not common.eligible(600, 650, baseline, original)


@pytest.mark.parametrize(
    "failure", ["official", "independent", "raw", "score", "wrong_payload", "nongain"]
)
def test_promotion_requires_both_checks_exact_payload_and_weighted_gain(failure):
    baseline = {"raw_kg": 14051.85, "weighted_kg": 12810.13}
    expected = {"raw_kg": 14053.08, "weighted_kg": 12820.04}
    checked = {
        "ok": True,
        "official": {"ok": True},
        "independent": {"ok": True},
        "total_mass_kg": expected["raw_kg"],
        "score_kg": expected["weighted_kg"],
    }
    assert common.fleet_accept(checked, baseline, expected)
    if failure in ("official", "independent"):
        checked[failure]["ok"] = False
    elif failure == "raw":
        checked["total_mass_kg"] = 14000
    elif failure == "score":
        checked["score_kg"] = math.nan
    elif failure == "wrong_payload":
        checked["score_kg"] += 1
    else:
        baseline["weighted_kg"] = checked["score_kg"]
    assert not common.fleet_accept(checked, baseline, expected)


def test_failed_fixed_cargo_refinement_keeps_prescription_and_stops_after_first_leg(data):
    catalogue, _ = data
    route = route_from_summary(common.read(common.ROOT / "inputs/ship-07.json"))
    plan, cargo = route.plan, deepcopy(route.collected_mass)
    original_plan, original_cargo = plan.summary(), deepcopy(cargo)
    calls, retained = [], []

    class Runner:
        def __init__(self, settings):
            self.telemetry = {"mock_CPU_only": True}

        def solve(self, leg, boundary, index):
            calls.append(index)
            return SimpleNamespace(certified=False, mass_after_leg=math.nan), {
                "leg": index,
                "reason": "controlled failure",
            }

        def close(self):
            pass

    result = refine_fixed(
        plan,
        catalogue,
        cargo,
        settings=ScvxSettings(),
        runner_factory=Runner,
        on_leg=lambda index, item, record: retained.append(record),
    )
    assert not result.certified and calls == [0] and len(retained) == 1
    assert plan.summary() == original_plan and cargo == original_cargo


def test_native_failure_counts_against_hard_limit_and_is_retained(tmp_path, monkeypatch):
    from run import count_native

    from spacepdhcg.gtoc12 import gpu_scvx

    def fail(*args):
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(gpu_scvx, "solve_native", fail)
    report = {"native_legs_started": 67, "native_legs_finished": 67, "active_case": "test"}
    boundary = SimpleNamespace(
        departure_epoch=1, arrival_epoch=2, initial_mass=3000, minimum_final_mass=500
    )
    with count_native(report, tmp_path):
        with pytest.raises(RuntimeError, match="controlled failure"):
            gpu_scvx.solve_native(boundary)
        with pytest.raises(RuntimeError, match="Hard68"):
            gpu_scvx.solve_native(boundary)
    assert report["native_legs_started"] == report["native_legs_finished"] == 68
    assert len((tmp_path / "native-legs.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("state", ["busy", "compute_process", "observation_timeout", "idle"])
def test_foreground_supervisor_lock_and_single_launch(tmp_path, monkeypatch, state):
    import fcntl
    import subprocess

    import launch

    (tmp_path / "run.py").write_text("# CPU mock\n")
    (tmp_path / "source").mkdir()
    lock_path = tmp_path / "shared.lock"
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(
        common, "environment", lambda: ({"python": "mock", "lock": str(lock_path)}, {})
    )
    monkeypatch.setattr(common, "ready_check", lambda: {})
    monkeypatch.setattr(common, "runtime_check", lambda env: {})
    launched = []

    def observe(command, **kwargs):
        if state == "observation_timeout":
            raise subprocess.TimeoutExpired(command, 15)
        return SimpleNamespace(stdout="123,busy,100MiB" if state == "compute_process" else "")

    def create(command, **kwargs):
        launched.append(command)
        assert kwargs["start_new_session"] and len(kwargs["pass_fds"]) == 1

        def wait(timeout=None):
            assert timeout == 1830
            with lock_path.open("a+") as contender, pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return 0

        return SimpleNamespace(pid=12345, wait=wait)

    monkeypatch.setattr(subprocess, "run", observe)
    monkeypatch.setattr(subprocess, "Popen", create)
    args = SimpleNamespace(execute=True, output=tmp_path / "output", wall_seconds=1800)
    with lock_path.open("a+") as holder:
        if state == "busy":
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state == "idle":
            assert launch.main(args) == 0
        else:
            with pytest.raises((BlockingIOError, RuntimeError, subprocess.TimeoutExpired)):
                launch.main(args)
    assert len(launched) == int(state == "idle")
    with pytest.raises(FileExistsError):
        launch.main(args)


def test_archived_control_is_not_equivalent_after_any_raw_ship_change(tmp_path):
    source = common.ROOT / "inputs/Result.txt"
    changed = tmp_path / "Result.txt"
    data = source.read_bytes()
    lines = data.splitlines()
    i = next(i for i, line in enumerate(lines) if int(line.split()[0]) == 7)
    lines[i] += b" "
    changed.write_bytes(b"\n".join(lines))
    assert ship_bytes(source, 7) != ship_bytes(changed, 7)


def test_combining_individual_gains_still_respects_fleet_mass_and_unique_ships():
    checked = {"ok": True, "official": {"ok": True}, "independent": {"ok": True}}
    baseline = {"raw_kg": common.RAW_FLOOR + 8, "weighted_kg": 1000}
    rows = [
        {"ship": 7, "raw_gain": -6, "weighted_gain": 10, "checked": checked},
        {"ship": 10, "raw_gain": -6, "weighted_gain": 9, "checked": checked},
        {"ship": 11, "raw_gain": 1, "weighted_gain": 2, "checked": checked},
    ]
    subset, score = common.best_combination(rows, baseline)
    assert [r["ship"] for r in subset] == [7, 11] and score == 1012
    rows[1]["ship"], rows[1]["raw_gain"] = 7, 5
    subset, score = common.best_combination(rows, baseline)
    assert len({r["ship"] for r in subset}) == len(subset)
    assert score == 1012


def test_failed_fleet_or_nonfinite_gain_cannot_enter_combination():
    bad = {
        "ship": 7,
        "raw_gain": 1,
        "weighted_gain": 100,
        "checked": {"ok": False, "official": {"ok": True}, "independent": {"ok": False}},
    }
    baseline = {"raw_kg": common.RAW_FLOOR + 1, "weighted_kg": 1000}
    assert common.best_combination([bad], baseline) == ([], 1000)
    bad["checked"] = {"ok": True, "official": {"ok": True}, "independent": {"ok": True}}
    bad["weighted_gain"] = math.nan
    assert common.best_combination([bad], baseline) == ([], 1000)


def test_supervisor_timeout_reaps_only_owned_child_preserving_raw_partial_report(tmp_path):
    import subprocess
    import sys

    from launch import retain_timeout_state, wait_owned_child

    report_path = tmp_path / "report.json"
    common.write(report_path, {"native_legs_started": 2, "native_legs_finished": 1})
    original_bytes = report_path.read_bytes()
    record = {}
    child = subprocess.Popen(
        [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        assert wait_owned_child(child, record, timeout=0.05, grace=0.2) == 124
        assert child.poll() is not None and record["timed_out"]
        assert os.getpid() != child.pid
        retain_timeout_state(tmp_path, record)
        state = common.read(tmp_path / "supervisor-timeout.json")
        assert not state["counts_complete"] and not state["automatic_retry"]
        assert state["preserved_report"]["native_legs_started"] == 2
        assert state["preserved_report"]["native_legs_finished"] == 1
        assert report_path.read_bytes() == original_bytes
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
