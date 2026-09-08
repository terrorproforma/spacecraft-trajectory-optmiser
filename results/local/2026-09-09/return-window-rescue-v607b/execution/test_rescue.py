"""CPU behavioral checks; no native library may be loaded."""

# Frozen source activation must precede every production import.
# ruff: noqa: E402

import ctypes
import dataclasses
import math
import os
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import support

support.activate()
from prefix import assert_prefix_emission, certify_wait, load_prefix, p, return_plan, solve_return
from screen import coarse_windows, fine_windows, screen_windows, select_basins

from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.solution import format_solution


@pytest.fixture(scope="session", autouse=True)
def forbid_native_loading():
    with patch.object(
        ctypes, "CDLL", side_effect=AssertionError("CPU validation cannot load native libraries")
    ):
        yield


@pytest.fixture(scope="module")
def data():
    os.environ["SPACEPDHCG_GTOC12_DATA"] = support.environment({})[0]["data"]
    catalogue = load_catalogue()
    return catalogue


@pytest.fixture(scope="module")
def prefixes(data):
    return {name: load_prefix(name, data) for name in ("control", "probe")}


@pytest.fixture(scope="module")
def archived_control(data):
    return load_prefix("control", data, include_archived_return=True)


def complete_control(full):
    summary = support.read(support.ROOT / "inputs/control/refinement.json")
    return p.RefinedRoute(
        full.plan,
        list(full.legs),
        dict(full.cargo),
        summary["final_mass_kg"],
        True,
        True,
        1,
        0,
        {},
        [],
    )


def test_exact_full_control_Result_roundtrip_with_real_saved_arrays(archived_control, data):
    emitted = p.emit_solution(complete_control(archived_control), data)
    assert (
        format_solution(emitted).encode()
        == (support.ROOT / "inputs/control/Result.txt").read_bytes()
    )
    assert len(archived_control.legs) == 17
    assert all(c.position_error_km < 500 for c in archived_control.fresh_certificates)


def test_mass_ledger_and_read_only_probe_prefix(prefixes):
    probe = prefixes["probe"]
    assert len(probe.legs) == 16
    assert probe.after_mass == 1207.2044215842654
    assert sum(probe.cargo.values()) == 615.6605065023956
    assert probe.after_mass - 500 - sum(probe.cargo.values()) == pytest.approx(91.54391508186995)
    with pytest.raises(ValueError):
        probe.legs[0].solution.thrust_n[0, 0] = 10


@pytest.mark.parametrize("wait", [0, 1, 79, 300, 588])
def test_positive_wait_preserves_mass_and_matches_asteroid_with_analytic_path_bound(
    prefixes, data, wait
):
    probe = prefixes["probe"]
    before = probe.fingerprint()
    checked = certify_wait(probe, data, 69218 + wait)
    assert checked["accepted"]
    assert checked["mass_preserved_kg"] == probe.after_mass
    assert checked["additional_asteroid_events"] == 0
    assert checked["position_error_km"] < 1e-6
    assert checked["velocity_error_km_s"] < 1e-12
    assert checked["analytic_perihelion_lower_bound_au"] == pytest.approx(2.7467748)
    assert probe.fingerprint() == before


@pytest.mark.parametrize(
    "departure,arrival",
    [(69217, 69800), (69218, 69218), (69807, 69808), (69300, 69808), (math.nan, 69728)],
)
def test_invalid_return_windows_rejected_before_solver(prefixes, departure, arrival):
    with pytest.raises(ValueError):
        return_plan(prefixes["probe"], departure, arrival)


def test_waiting_plan_keeps_every_original_collection_epoch_and_cargo(prefixes):
    prefix = prefixes["probe"]
    plan, last = return_plan(prefix, 69297, 69807)
    assert plan.deploy_epochs == prefix.plan.deploy_epochs
    assert plan.collect_epochs == prefix.plan.collect_epochs
    assert plan.collected_mass == prefix.cargo
    assert plan.legs[:-2] == prefix.plan.legs[:-1]
    assert plan.legs[-2].role == "camp" and last.departure_epoch == 69297
    assert prefix.plan.legs[-1].departure_epoch == 69218


def test_single_return_uses_real_adapter_and_known_prefix_certification(
    prefixes, archived_control, data, monkeypatch
):
    calls = []
    old = archived_control.legs[-1].solution

    def solve(boundary, settings):
        calls.append(boundary)
        return dataclasses.replace(old, boundary=boundary, thrust_n=old.thrust_n.copy())

    monkeypatch.setattr(p, "solve_leg", solve)
    settings = p.ScvxSettings(**support.read(support.ROOT / "preparation.json")["scvx_settings"])
    original = prefixes["control"].fingerprint()
    route, result = solve_return(prefixes["control"], data, 69218, 69728, settings)
    assert len(calls) == 1 and route.certified and route.master_certified
    assert calls[0].initial_mass == prefixes["control"].after_mass
    emitted = p.emit_solution(route, data)
    assert_prefix_emission(prefixes["control"], emitted)
    assert (
        format_solution(emitted).encode()
        == (support.ROOT / "inputs/control/Result.txt").read_bytes()
    )
    assert prefixes["control"].fingerprint() == original
    assert result["status"] == "return_and_route_certified"


@pytest.mark.parametrize(
    "mutation", ["position", "velocity", "mass", "epoch", "cargo", "duplicate"]
)
def test_emitter_guard_rejects_any_prefix_event_or_inventory_mutation(
    prefixes, archived_control, data, mutation
):
    solution = p.emit_solution(complete_control(archived_control), data)
    prefix = prefixes["control"]
    assert_prefix_emission(prefix, solution)
    event = next(e for e in solution.ships[0].events if e.is_asteroid)
    if mutation == "position":
        event.after.position[0] += 1
    elif mutation == "velocity":
        event.after.velocity[0] += 1
    elif mutation in ("mass", "cargo", "epoch"):
        field, increment = ("epoch", 1) if mutation == "epoch" else ("mass", 0.1)
        altered = dataclasses.replace(
            event.after, **{field: getattr(event.after, field) + increment}
        )
        index = next(i for i, item in enumerate(solution.ships[0].items) if item is event)
        solution.ships[0].items[index] = dataclasses.replace(event, after=altered)
    else:
        solution.ships[0].items.insert(1, deepcopy(event))
    with pytest.raises(ValueError):
        assert_prefix_emission(prefix, solution)


def test_waiting_emission_keeps_exact_prefix_and_has_no_third_visit(
    prefixes, archived_control, data
):
    prefix = prefixes["control"]
    plan, planned = return_plan(prefix, 69297, 69807)
    old = archived_control.legs[-1]
    # Geometry is deliberately not claimed here: this is an emitter construction fixture.
    times = np.linspace(69297, 69807, len(old.solution.node_epochs_mjd))
    r0, v0 = p.body_state(data, 59653, 69297)
    rf, vf = p.body_state(data, 0, 69807)
    boundary = p.LegBoundary(
        69297,
        r0,
        v0,
        69807,
        rf,
        vf,
        prefix.after_mass,
        free_arrival_vinf=True,
        minimum_final_mass=500 + sum(prefix.cargo.values()),
    )
    solved = dataclasses.replace(old.solution, boundary=boundary, node_epochs_mjd=times)
    last = dataclasses.replace(old, planned=planned, solution=solved)
    route = p.RefinedRoute(
        plan, [*prefix.legs, last], dict(prefix.cargo), 503, True, True, 1, 0, {}, []
    )
    emitted = p.emit_solution(route, data)
    checked = assert_prefix_emission(prefix, emitted)
    assert checked["asteroid_events"] == 16
    last_collection = [event for event in emitted.ships[0].events if event.event_id == 59653][-1]
    assert last_collection.epoch == 69218
    returns = [arc for arc in emitted.ships[0].burns if arc.start >= 69218]
    assert min(arc.start for arc in returns) >= 69297
    assert certify_wait(prefix, data, 69297)["accepted"]


@pytest.mark.parametrize("bad", ["flag", "mass", "array_hash", "certificate"])
def test_import_rejects_uncertified_or_inconsistent_prefix(data, monkeypatch, bad):
    original = support.read

    def read(path):
        value = original(path)
        if str(path).endswith("probe/leg-00.json"):
            value = deepcopy(value)
            if bad == "flag":
                value["certified"] = False
            elif bad == "mass":
                value["initial_mass_kg"] -= 1
            elif bad == "array_hash":
                value["solution_arrays"]["sha256"] = "invalid"
            else:
                value["certificate"]["final_mass_kg"] -= 1
        return value

    monkeypatch.setattr(support, "read", read)
    with pytest.raises(ValueError):
        load_prefix("probe", data)


def test_failed_native_return_retains_result_and_never_mutates_prefix(
    prefixes, archived_control, data, monkeypatch
):
    old = archived_control.legs[-1].solution
    seen = []

    def fail(boundary, settings):
        return dataclasses.replace(
            old, status="infeasible", boundary=boundary, diagnostic="controlled failed return"
        )

    monkeypatch.setattr(p, "solve_leg", fail)
    prefix = prefixes["probe"]
    before = prefix.fingerprint()
    route, detail = solve_return(
        prefix,
        data,
        69218,
        69728,
        p.ScvxSettings(),
        on_result=lambda item, d: seen.append((item, d)),
    )
    assert route is None and detail["status"] == "native_return_uncertified"
    assert len(seen) == 1 and seen[0][0].solution is not None
    assert "controlled failed return" in seen[0][1]["diagnostic"]
    assert prefix.fingerprint() == before


def test_missing_prefix_certificate_cannot_be_silently_certified(prefixes, data, monkeypatch):
    prefix = deepcopy(prefixes["probe"])
    prefix.fresh_certificates[0] = dataclasses.replace(
        prefix.fresh_certificates[0], position_error_km=1e9
    )
    calls = []
    monkeypatch.setattr(p, "solve_leg", lambda *args: calls.append(args))
    route, detail = solve_return(prefix, data, 69218, 69728, p.ScvxSettings())
    assert route is None and not calls
    assert detail["status"] == "return_path_exception"


def test_legal_grid_and_fine_budget_are_deterministic():
    coarse = coarse_windows()
    assert len(coarse) == len(set(coarse)) == 7022
    assert (69218, 69728) in coarse and (69218, 69807) in coarse
    centres = [{"departure": d, "arrival": a} for d, a in coarse[::220]][:32]
    fine = fine_windows(centres, set(coarse))
    assert len(fine) <= 2592 and len(coarse) + len(fine) <= 9614
    assert not set(fine) & set(coarse)
    assert fine == sorted(set(fine))
    assert all(69218 <= d < a <= 69807 for d, a in coarse + fine)


def row(dep, arr, fuel, feasible=True):
    return {
        "departure": dep,
        "arrival": arr,
        "tof_days": arr - dep,
        "predicted_propellant_kg": fuel,
        "authority_ratio": 0.2,
        "geometry_feasible": feasible,
    }


def test_diverse_selection_rejects_bad_geometry_but_keeps_best_near_miss():
    rows = [
        row(69218, 69728, 100),
        row(69219, 69729, 101),
        row(69250, 69760, 102),
        row(69300, 69800, 103),
        row(69280, 69780, 104),
        row(69218, 69807, 1, False),
        row(69218, 69729, math.nan),
    ]
    selected = select_basins(rows, 4)
    assert [item["predicted_propellant_kg"] for item in selected] == [100, 102, 103, 104]
    assert select_basins(list(reversed(rows)), 4) == selected


def test_screening_retains_all_raw_rows_with_mocked_geometry(prefixes, data, tmp_path):
    class Engine:
        def __init__(self, **kwargs):
            self.telemetry = {"mock_CPU_only": True}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def paired_hops(self, catalogue, source, target, departure, tof):
            assert source == 59653 and target == 0
            dv = 3 + ((departure - 69218) / 500) ** 2 + ((departure + tof - 69790) / 500) ** 2
            return SimpleNamespace(
                total_delta_v=dv,
                tof_days=tof,
                departure_delta_v=dv,
                arrival_delta_v=np.zeros(len(tof)),
                departure_velocity=np.zeros((len(tof), 3)),
                arrival_velocity=np.zeros((len(tof), 3)),
                feasible=np.ones(len(tof), dtype=bool),
            )

    output = tmp_path / "screening"
    result = screen_windows(prefixes["probe"], data, output, engine_factory=Engine)
    assert result["coarse_rows"] == 7022 and result["total_rows"] <= 9614
    assert len(result["selected"]) == 4
    for name, count in (("coarse", result["coarse_rows"]), ("fine", result["fine_rows"])):
        if not count:
            continue
        metadata = support.read(output / (name + ".json"))
        assert len(metadata["rows"]) == count
        assert metadata["raw_sha256"] == support.sha(output / (name + ".npz"))


@pytest.mark.parametrize(
    "bad", ["control", "official", "independent", "cargo", "raw", "score", "dry"]
)
def test_no_promotion_without_all_verification_and_cargo_gates(bad):
    checked = {
        "ok": True,
        "official": {"ok": True},
        "independent": {"ok": True},
        "prescribed_cargo_verified": True,
        "total_mass_kg": 14053.08,
        "score_kg": 12820.04,
    }
    best, control, dry = {"score_kg": 12810.13}, True, 503
    assert support.better(checked, best, dry, control)
    if bad == "control":
        control = False
    elif bad in ("official", "independent"):
        checked[bad]["ok"] = False
    elif bad == "cargo":
        checked["prescribed_cargo_verified"] = False
    elif bad == "raw":
        checked["total_mass_kg"] = 14000
    elif bad == "score":
        checked["score_kg"] = math.nan
    else:
        dry = 499
    assert not support.better(checked, best, dry, control)


def test_equal_verified_cargo_keeps_better_fuel_margin():
    checked = {
        "ok": True,
        "official": {"ok": True},
        "independent": {"ok": True},
        "prescribed_cargo_verified": True,
        "total_mass_kg": 14053.08,
        "score_kg": 12820.04,
    }
    best = {"score_kg": 12820.04, "ship7_final_dry_mass_kg": 503}
    assert support.better(checked, best, 504, True)
    assert not support.better(checked, best, 502, True)


def test_native_hard_budget_and_failed_call_accounting(tmp_path, monkeypatch):
    from run import observe_native

    from spacepdhcg.gtoc12 import gpu_scvx

    calls = []
    monkeypatch.setattr(
        gpu_scvx,
        "solve_native",
        lambda *args, **kwargs: (
            calls.append(args)
            or SimpleNamespace(status="converged", iterations=3, accepted_iterations=2)
        ),
    )
    report = {"native_returns_started": 0, "native_returns_completed": 0, "active_case": "test"}
    boundary = SimpleNamespace(
        departure_epoch=69218, arrival_epoch=69728, initial_mass=1207, minimum_final_mass=1115
    )
    with observe_native(tmp_path, report):
        for _ in range(5):
            gpu_scvx.solve_native(boundary)
        with pytest.raises(RuntimeError, match="five native"):
            gpu_scvx.solve_native(boundary)
    assert len(calls) == report["native_returns_started"] == report["native_returns_completed"] == 5
    assert len((tmp_path / "native-returns.jsonl").read_text().splitlines()) == 5


def test_native_exception_is_retained_and_consumes_budget(tmp_path, monkeypatch):
    from run import observe_native

    from spacepdhcg.gtoc12 import gpu_scvx

    def failing_native(*args, **kwargs):
        raise RuntimeError("controlled native failure")

    monkeypatch.setattr(gpu_scvx, "solve_native", failing_native)
    report = {"native_returns_started": 0, "native_returns_completed": 0, "active_case": "test"}
    boundary = SimpleNamespace(
        departure_epoch=69218, arrival_epoch=69728, initial_mass=1207, minimum_final_mass=1115
    )
    with observe_native(tmp_path, report), pytest.raises(RuntimeError, match="controlled native"):
        gpu_scvx.solve_native(boundary)
    assert report["native_returns_started"] == report["native_returns_completed"] == 1
    retained = support.read(tmp_path / "native-returns.jsonl")
    assert "controlled native failure" in retained["error"]
    assert gpu_scvx.solve_native is failing_native


@pytest.mark.parametrize("state", ["busy", "compute_process", "observation_timeout", "idle"])
def test_supervisor_refuses_unsafe_state_and_launches_only_once(tmp_path, monkeypatch, state):
    import fcntl
    import subprocess

    import launch

    (tmp_path / "run.py").write_text("# isolated CPU fixture\n")
    (tmp_path / "source").mkdir()
    lock_path = tmp_path / "shared.lock"
    monkeypatch.setattr(support, "ROOT", tmp_path)
    monkeypatch.setattr(
        support, "environment", lambda: ({"python": "mock-python", "lock": str(lock_path)}, {})
    )
    monkeypatch.setattr(support, "validate_ready", lambda: {})
    monkeypatch.setattr(support, "validate_runtime", lambda env: {})
    observations, launched = [], []

    def observe(command, **kwargs):
        observations.append(command)
        if state == "observation_timeout":
            raise subprocess.TimeoutExpired(command, 15)
        return SimpleNamespace(
            stdout="777, another-job, 100 MiB" if state == "compute_process" else ""
        )

    def create(command, **kwargs):
        launched.append((command, kwargs))
        assert kwargs["start_new_session"] and len(kwargs["pass_fds"]) == 1

        def wait():
            assert support.read(tmp_path / "launch.json")["pid"] == 12345
            # The supervisor must still own the shared lock during its foreground wait.
            with lock_path.open("a+") as contender, pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return 2

        return SimpleNamespace(pid=12345, wait=wait)

    monkeypatch.setattr(subprocess, "run", observe)
    monkeypatch.setattr(subprocess, "Popen", create)
    args = SimpleNamespace(execute=True, output=tmp_path / "output", wall_seconds=1800)
    with lock_path.open("a+") as holder:
        if state == "busy":
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state == "idle":
            assert launch.main(args) == 2
        else:
            with pytest.raises((BlockingIOError, RuntimeError, subprocess.TimeoutExpired)):
                launch.main(args)
    record = support.read(tmp_path / "launch.json")
    assert record["launched"] == (state == "idle")
    assert len(launched) == int(state == "idle")
    if state == "busy":
        assert observations == []
    if state != "idle":
        assert record["status"] == "refused_before_GPU_launch"
    else:
        assert record["returncode"] == 2
    with pytest.raises(FileExistsError):
        launch.main(args)
    assert len(launched) == int(state == "idle")
