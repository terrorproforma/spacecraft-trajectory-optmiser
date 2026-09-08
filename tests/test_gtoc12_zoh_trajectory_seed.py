"""CPU-only archival-seed validation, routing and physical-pointer ABI checks."""

from __future__ import annotations

import ctypes as ct
import dataclasses
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.gtoc12 import fixed_refinement, gpu_scvx, low_thrust, pipeline
from spacepdhcg.gtoc12.low_thrust import LegBoundary, ScvxSettings, ZohTrajectorySeed


@pytest.fixture(autouse=True)
def no_native_or_ballistic(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "CPU seed tests cannot invoke native code or generate a ballistic seed"
        )

    monkeypatch.setattr(ct, "CDLL", forbidden)
    monkeypatch.setattr(low_thrust, "_ballistic_reference", forbidden)


def settings(**changes):
    return dataclasses.replace(
        ScvxSettings(
            discretisation_backend="cuda",
            assembly_backend="cuda",
            convex_solver_backend="qoco",
            outer_loop_backend="cuda",
        ),
        **changes,
    )


def boundary(*, free=False):
    return LegBoundary(
        64400.0,
        np.array([1.5e8, 0.0, 0.0]),
        np.array([0.0, 29.0, 0.0]),
        64406.0,
        np.array([1.49e8, 1.0e7, 0.0]),
        np.array([-2.0, 28.0, 0.0]),
        3000.0,
        free_departure_vinf=free,
    )


def seed_values(bnd=None):
    bnd = bnd or boundary()
    return dict(
        node_epochs_mjd=np.array([64400.0, 64402.0, 64404.0, 64406.0]),
        initial_state=np.r_[bnd.departure_position, bnd.departure_velocity, bnd.initial_mass],
        thrust_n=np.array([[0.3, -0.0, 0.0], [0.0, 0.2, 0.0], [-0.1, 0.0, 0.0], [0.0, -0.0, 0.0]]),
        source_sha256="a" * 64,
    )


def test_seed_is_defensive_immutable_snapshot_and_keeps_signed_zero():
    values = seed_values()
    original = {
        key: value.tobytes() for key, value in values.items() if isinstance(value, np.ndarray)
    }
    seed = ZohTrajectorySeed(**values)
    values["thrust_n"][:] = 0.6
    for key, raw in original.items():
        array = getattr(seed, key)
        assert array.tobytes() == raw and array.flags.c_contiguous
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        seed.source_sha256 = "b" * 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_epochs_mjd", [1.0]),
        ("node_epochs_mjd", [64400.0, 64402.0, 64402.0, 64406.0]),
        ("node_epochs_mjd", [64400.0, 64402.0, 64404.0, float("nan")]),
        ("initial_state", np.zeros(6)),
        ("initial_state", [1.5e8, 0, 0, 0, 29, 0, 0]),
        ("initial_state", [1.5e8, 0, 0, 0, 29, 0, -1]),
        ("initial_state", [1.5e8, 0, 0, 0, float("inf"), 0, 3000]),
        ("thrust_n", np.zeros((4, 4))),
        ("thrust_n", np.full((4, 3), float("nan"))),
        ("thrust_n", np.ones((4, 3))),
        ("source_sha256", "not-a-digest"),
        ("source_sha256", "A" * 64),
    ],
)
def test_malformed_seed_rejected(field, value):
    values = seed_values()
    values[field] = value
    with pytest.raises(ValueError):
        ZohTrajectorySeed(**values)


@pytest.mark.parametrize(
    "change", ["grid", "position", "velocity", "mass", "hold", "backend", "numpy"]
)
def test_seed_boundary_and_mode_mismatch_rejected_before_native(change):
    bnd, config, values = boundary(), settings(), seed_values()
    if change == "grid":
        values["node_epochs_mjd"][1] = np.nextafter(values["node_epochs_mjd"][1], np.inf)
    elif change == "position":
        values["initial_state"][0] = np.nextafter(values["initial_state"][0], np.inf)
    elif change == "velocity":
        values["initial_state"][4] = np.nextafter(values["initial_state"][4], np.inf)
    elif change == "mass":
        values["initial_state"][6] = np.nextafter(values["initial_state"][6], np.inf)
    elif change == "hold":
        config = settings(hold="lagrange")
    elif change == "backend":
        config = ScvxSettings()
    else:
        config = settings(seed_backend="numpy")
    with pytest.raises(ValueError):
        low_thrust.solve_leg(bnd, config, seed=ZohTrajectorySeed(**values))


def test_exact_seed_reaches_native_without_host_rollout(monkeypatch):
    bnd = boundary(free=True)
    values = seed_values(bnd)
    values["initial_state"][3] = 6.000000002323666
    seed = ZohTrajectorySeed(**values)
    seen = []

    def observed(*args, **kwargs):
        assert kwargs == {"seed": seed}
        assert args[7] is args[8] is None
        assert np.array_equal(bnd.departure_epoch + args[4], seed.node_epochs_mjd)
        seen.append(seed.initial_state.tobytes())
        return "native-result-sentinel"

    monkeypatch.setattr(gpu_scvx, "solve_native", observed)
    assert low_thrust.solve_leg(bnd, settings(), seed=seed) == "native-result-sentinel"
    assert seen == [values["initial_state"].tobytes()]


class StubSolve:
    """A Python ctypes-call sink, not a numerical solver or native library."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        n = args[0] + 1
        seed_x = None if args[9] is None else np.ctypeslib.as_array(args[9], shape=(7,)).copy()
        seed_u = (
            None if args[10] is None else np.ctypeslib.as_array(args[10], shape=(n * 3,)).copy()
        )
        self.calls.append((args, seed_x, seed_u))
        states = np.ctypeslib.as_array(args[13], shape=(n * 7,)).reshape(n, 7)
        controls = np.ctypeslib.as_array(args[14], shape=(n * 4,)).reshape(n, 4)
        states[:] = 0.0
        states[:, 6] = 1.0
        controls[:] = 0.0
        result = ct.cast(args[17], ct.POINTER(gpu_scvx._Result)).contents
        result.status = 4
        result.virtual_inf = float("inf")
        result.departure_vinf[0] = 0.125
        return 0


def native_args(bnd, config, internal=None):
    days = np.array([0.0, 2.0, 4.0, 6.0])
    model = low_thrust._Model(bnd.initial_mass)
    bvalues = {
        "r0": bnd.departure_position / low_thrust.DU_KM,
        "v0": bnd.departure_velocity / low_thrust.VU_KM_S,
        "rf": bnd.arrival_position / low_thrust.DU_KM,
        "vf": bnd.arrival_velocity / low_thrust.VU_KM_S,
    }
    x, u = (None, None) if internal is None else internal
    return (
        bnd,
        config,
        model,
        days * low_thrust.C.DAY_S / low_thrust.TU_S,
        days,
        bvalues,
        np.zeros(4),
        x,
        u,
        time.perf_counter(),
    )


def library_environment(monkeypatch, tmp_path, library):
    path = tmp_path / "stub-library"
    path.write_bytes(b"CPU test sentinel; never loaded")
    monkeypatch.setenv("SPACEPDHCG_GTOC12_CUDA_LIBRARY", str(path))
    monkeypatch.setenv("SPACEPDHCG_QOCO_LIBRARY", str(path))
    monkeypatch.setattr(ct, "CDLL", lambda _: library)


def test_new_ctypes_entry_receives_exact_physical_arrays(monkeypatch, tmp_path):
    bnd = boundary(free=True)
    values = seed_values(bnd)
    values["initial_state"][3] = 6.000000002323666
    seed = ZohTrajectorySeed(**values)
    new, old = StubSolve(), StubSolve()
    library_environment(
        monkeypatch,
        tmp_path,
        SimpleNamespace(
            spacepdhcg_gtoc12_scvx_solve_zoh_seed_host=new,
            spacepdhcg_gtoc12_scvx_solve_host=old,
        ),
    )
    result = gpu_scvx.solve_native(*native_args(bnd, settings()), seed=seed)
    assert len(new.calls) == 1 and not old.calls
    args, x, u = new.calls[0]
    assert len(args) == len(new.argtypes) == 18
    assert x.tobytes() == seed.initial_state.tobytes()
    assert u.tobytes() == seed.thrust_n.tobytes()
    assert args[1] == 0 and args[2] == 1
    assert result.seed_backend == "cuda_zoh_replay"
    assert result.status == "timeout" and result.iterations == 0
    assert result.departure_vinf_km_s[0] == 0.125 * low_thrust.VU_KM_S


@pytest.mark.parametrize("internal", [False, True])
def test_existing_native_seed_selection_unchanged(monkeypatch, tmp_path, internal):
    old, new = StubSolve(), StubSolve()
    library_environment(
        monkeypatch,
        tmp_path,
        SimpleNamespace(
            spacepdhcg_gtoc12_scvx_solve_host=old,
            spacepdhcg_gtoc12_scvx_solve_zoh_seed_host=new,
            spacepdhcg_gtoc12_seed_evaluate_host=object(),
        ),
    )
    arrays = (np.ones((4, 7)), np.zeros((4, 4))) if internal else None
    result = gpu_scvx.solve_native(*native_args(boundary(), settings(), arrays))
    assert len(old.calls) == 1 and not new.calls
    assert result.seed_backend == ("numpy" if internal else "cuda")
    if not internal:
        assert old.calls[0][0][9] is old.calls[0][0][10] is None


def test_requested_seed_never_falls_back_to_legacy_entry(monkeypatch, tmp_path):
    old = StubSolve()
    library_environment(
        monkeypatch, tmp_path, SimpleNamespace(spacepdhcg_gtoc12_scvx_solve_host=old)
    )
    with pytest.raises(RuntimeError, match="ZOH replay seed extension; no fallback"):
        gpu_scvx.solve_native(
            *native_args(boundary(), settings()), seed=ZohTrajectorySeed(**seed_values())
        )
    assert not old.calls


def test_direct_bridge_rejects_two_initial_references_before_loading():
    with pytest.raises(ValueError, match="combined with internal"):
        gpu_scvx.solve_native(
            *native_args(boundary(), settings(), (np.ones((4, 7)), np.zeros((4, 4)))),
            seed=ZohTrajectorySeed(**seed_values()),
        )


@pytest.mark.parametrize("with_seed", [False, True])
def test_driver_passes_explicit_seed_without_generic_warm_import(monkeypatch, with_seed):
    bnd = boundary()
    seed = ZohTrajectorySeed(**seed_values()) if with_seed else None
    registry = pipeline.LegRegistry()
    request = SimpleNamespace(deterministic_id=7)
    registry.register(request, bnd, seed=seed)
    seen = []

    def fake_solve(boundary_arg, config, **kwargs):
        seen.append((boundary_arg, kwargs))
        return SimpleNamespace(status="failed", diagnostic="CPU routing sentinel")

    monkeypatch.setattr(pipeline, "solve_leg", fake_solve)
    driver = pipeline.Gtoc12ScvxDriver(None, None, registry, settings())
    result = driver.solve(request, threading.Event())
    assert result.status == pipeline.G3Status.NUMERICAL_FAILURE
    assert seen == [(bnd, {"seed": seed} if with_seed else {})]
    assert driver.import_warm_state(seed, 3) is False


def test_frozen_runner_registers_seed_before_scheduler(monkeypatch):
    class StopBeforeSolve(Exception):
        pass

    seed = ZohTrajectorySeed(**seed_values())
    runner = fixed_refinement.FrozenLegRunner.__new__(fixed_refinement.FrozenLegRunner)
    runner.registry = pipeline.LegRegistry()

    def stop(requests):
        assert runner.registry.records[requests[0].deterministic_id].seed is seed
        raise StopBeforeSolve

    runner.scheduler = SimpleNamespace(run=stop)
    leg = pipeline.PlannedLeg(0, 1, 64400.0, 64406.0, 0.0, 1.0, "earth_out")
    with pytest.raises(StopBeforeSolve):
        runner.solve(leg, boundary(), 0, seed=seed)
