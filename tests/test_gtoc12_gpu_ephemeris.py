"""Boundary batching, ownership and independent orbital-state comparisons."""

from __future__ import annotations

import ctypes as ct
import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.gtoc12 import gpu_ephemeris as g
from spacepdhcg.gtoc12 import pipeline as p
from spacepdhcg.gtoc12.constants import AU_KM
from spacepdhcg.gtoc12.ephemeris import asteroid_state, earth_state


class Catalogue:
    ids = np.array([1, 2, 3])
    epoch_mjd = np.full(3, 64328.0)
    semi_major_axis_km = AU_KM * np.array([1.8, 2.9, 3.1])
    eccentricity = np.array([0.0, 0.3, 0.81])
    inclination_rad = np.array([0.0, 0.12, 0.6])
    ascending_node_rad = np.array([0.0, 1.1, 5.8])
    argument_of_perihelion_rad = np.array([0.0, 0.8, 3.5])
    mean_anomaly_rad = np.array([0.0, 0.35, 6.1])

    def index_of(self, ids):
        ids = np.asarray(ids)
        if np.any(ids < 1) or np.any(ids > 3):
            raise ValueError("Unknown asteroid")
        return ids - 1


def test_backend_follows_native_outer_and_cpu_ablation_is_explicit():
    assert p.ScvxSettings().selected_ephemeris_backend() == "cpu"
    assert p.ScvxSettings(outer_loop_backend="cuda").selected_ephemeris_backend() == "cuda"
    assert (
        p.ScvxSettings(
            outer_loop_backend="cuda", ephemeris_backend="cpu"
        ).selected_ephemeris_backend()
        == "cpu"
    )
    with pytest.raises(ValueError, match="ephemeris_backend"):
        p.ScvxSettings(ephemeris_backend="silent_fallback").selected_ephemeris_backend()


def test_route_batches_distinct_endpoints_once_and_reuses_shared_epochs(monkeypatch):
    calls = []

    class Workspace:
        def __init__(self, catalogue, ids, capacity):
            calls.append((list(ids), capacity))
            self.telemetry = {"backend": "cuda", "batches": 1, "state_requests": capacity}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            calls.append("closed")

        def states(self, ids, epochs):
            calls.append((list(ids), list(epochs)))
            values = np.arange(3 * len(ids), dtype=float).reshape(-1, 3)
            return values, -values

    monkeypatch.setattr(g, "GpuEphemeris", Workspace)
    monkeypatch.setattr(p, "body_state", lambda *_: pytest.fail("CPU ephemeris called"))
    legs = [
        SimpleNamespace(from_id=0, to_id=1, departure_epoch=64328, arrival_epoch=64628),
        SimpleNamespace(from_id=1, to_id=0, departure_epoch=64628, arrival_epoch=65128),
    ]
    states, telemetry = p.prepare_route_states(
        None, legs, p.ScvxSettings(outer_loop_backend="cuda")
    )
    assert list(states) == [(0, 64328), (1, 64628), (0, 65128)]
    assert calls == [([0, 1, 0], 3), ([0, 1, 0], [64328, 64628, 65128]), "closed"]
    assert telemetry["state_requests"] == 3
    np.testing.assert_array_equal(states[(1, 64628)][0], [3, 4, 5])


def test_cpu_route_keeps_lazy_boundary_path(monkeypatch):
    monkeypatch.setattr(g, "GpuEphemeris", lambda *_: pytest.fail("Native workspace created"))
    assert p.prepare_route_states(None, [object()], p.ScvxSettings()) == (
        None,
        {"backend": "cpu", "batches": 0},
    )


def test_native_error_is_not_replaced_with_cpu_ephemeris(monkeypatch):
    def missing(*_):
        raise RuntimeError("No native ephemeris")

    monkeypatch.setattr(g, "GpuEphemeris", missing)
    monkeypatch.setattr(p, "body_state", lambda *_: pytest.fail("CPU fallback"))
    leg = SimpleNamespace(from_id=0, to_id=1, departure_epoch=64328, arrival_epoch=64628)
    with pytest.raises(RuntimeError, match="No native ephemeris"):
        p.prepare_route_states(None, [leg], p.ScvxSettings(outer_loop_backend="cuda"))


def test_wrapper_rejects_ambiguous_integer_input_before_library_load(monkeypatch):
    monkeypatch.setattr(ct, "CDLL", lambda *_: pytest.fail("Native library loaded"))
    for ids, capacity in [([0.5], 2), ([True], 2), ([0], 1.5), ([0], 0)]:
        with pytest.raises(ValueError):
            g.GpuEphemeris(Catalogue(), ids, capacity)


def test_retained_wrapper_returns_owned_arrays_and_checks_owner(monkeypatch):
    # A deterministic native bridge stub tests ownership without solving or
    # pretending that these arrays are independently propagated orbital states.
    class Function:
        def __init__(self, body):
            self.body = body

        def __call__(self, *args):
            return self.body(*args)

    count = 0

    def create(*args):
        args[-1]._obj.value = 123
        return 0

    def evaluate(handle, request_ptr, n, output_ptr):
        nonlocal count
        count += 1
        raw = (ct.c_char * (n * g.RESULT.itemsize)).from_address(output_ptr)
        out = np.frombuffer(raw, dtype=g.RESULT)
        out["position"] = count
        out["velocity"] = -count
        out["status"] = out["reserved"] = 0
        return 0

    def destroy(handle):
        handle._obj.value = None
        return 0

    library = SimpleNamespace(
        spacepdhcg_gtoc12_ephemeris_create=Function(create),
        spacepdhcg_gtoc12_ephemeris_host=Function(evaluate),
        spacepdhcg_gtoc12_ephemeris_destroy=Function(destroy),
    )
    monkeypatch.setenv("SPACEPDHCG_GTOC12_CUDA_LIBRARY", __file__)
    monkeypatch.setattr(ct, "CDLL", lambda *_: library)
    with g.GpuEphemeris(Catalogue(), [0, 1], 2) as workspace:
        position, _ = workspace.states([1], [64328])
        workspace.states([0, 1], [64328, 64329])
        np.testing.assert_array_equal(position, np.ones((1, 3)))
        for ids, epochs in [([2], [64328]), ([1], [float("nan")]), ([1, 1, 1], [1, 2, 3])]:
            with pytest.raises(ValueError):
                workspace.states(ids, epochs)
        failures = []

        def other_thread():
            try:
                workspace.states([0], [64328])
            except RuntimeError as error:
                failures.append(str(error))

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        assert failures == ["CUDA ephemeris workspace belongs to another thread"]
        assert count == 2
    with pytest.raises(RuntimeError, match="closed"):
        workspace.states([0], [64328])


@pytest.mark.skipif(
    not os.environ.get("SPACEPDHCG_REQUIRE_CUDA_EPHEMERIS"),
    reason="Explicit CUDA ephemeris test run required",
)
def test_cuda_states_match_independent_cpu_equations():
    catalogue = Catalogue()
    ids = [body for epoch in (-1200.0, 64328.0, 64628.25, 70000.0) for body in (0, 1, 2, 3)]
    epochs = [epoch for epoch in (-1200.0, 64328.0, 64628.25, 70000.0) for body in (0, 1, 2, 3)]
    with g.GpuEphemeris(catalogue, ids, len(ids)) as workspace:
        actual_r, actual_v = workspace.states(ids, epochs)
        for i, (body, epoch) in enumerate(zip(ids, epochs, strict=True)):
            expected_r, expected_v = (
                earth_state(epoch) if body == 0 else asteroid_state(catalogue, body, epoch)
            )
            assert np.linalg.norm(actual_r[i] - expected_r) < 1e-3
            assert np.linalg.norm(actual_v[i] - expected_v) < 1e-9
        workspace.states([0], [64328.0])
        assert workspace.telemetry["batches"] == 2
        assert workspace.telemetry["state_requests"] == 17
