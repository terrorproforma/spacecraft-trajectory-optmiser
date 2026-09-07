"""CUDA forward masses preserve CPU accounting, failures and cache policy."""

import os
from dataclasses import replace

import numpy as np
import pytest
from test_gtoc12_gpu_retime import fixture

from spacepdhcg.gtoc12 import lambert

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


def compare_forward(actual, expected):
    assert actual[2] == expected[2]
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=1e-10)
    if actual[0] is None:
        assert expected[0] is None
        return
    a, b = actual[0], expected[0]
    assert a.deploy_epochs == b.deploy_epochs and a.collect_epochs == b.collect_epochs
    assert a.foreign_deploy_epochs == b.foreign_deploy_epochs
    assert a.collected_mass == b.collected_mass
    assert abs(a.propellant_proxy_kg - b.propellant_proxy_kg) < 1e-10
    assert abs(a.final_mass_proxy_kg - b.final_mass_proxy_kg) < 1e-10
    assert len(a.legs) == len(b.legs)
    for x, y in zip(a.legs, b.legs, strict=True):
        assert (x.from_id, x.to_id, x.departure_epoch, x.arrival_epoch, x.role) == (
            y.from_id,
            y.to_id,
            y.departure_epoch,
            y.arrival_epoch,
            y.role,
        )
        assert x.delta_v_proxy_km_s == y.delta_v_proxy_km_s
        assert abs(x.inflation - y.inflation) < 1e-13


@pytest.mark.parametrize("cooperative", [False, True])
@pytest.mark.parametrize("initial_mass", [600.0, 3000.0])
@pytest.mark.parametrize("mass_scale", [0.3, 1.0])
def test_forward_and_mass_rounds_match_cpu(monkeypatch, cooperative, initial_mass, mass_scale):
    timer, visits = fixture(monkeypatch, 4, cooperative)
    timer.search_settings = replace(timer.search_settings, initial_mass=initial_mass)
    masses = [initial_mass * mass_scale] * (len(visits) - 1)
    with lambert.using_lambert_backend("cuda") as gpu:
        for price in [0.0, 0.15, 1.0]:
            dp = gpu.retime_dp(timer, visits, masses, price, True)
            assert dp is not None and dp is not NotImplemented
            a, d = timer.lattice.epochs[dp[0]].tolist(), timer.lattice.epochs[dp[1]].tolist()
            actual = timer._forward(visits, a, d)
            assert gpu.retime_workspace.last_forward is not None
            with monkeypatch.context() as cpu:
                cpu.setattr(lambert, "cuda_retime_forward_result", lambda *args: None)
                expected = timer._forward(visits, a, d)
            compare_forward(actual, expected)
            gpu.retime_cuda_forward = False
            expected_rounds = timer._solve_at_price(visits, masses, price)
            gpu.retime_cuda_forward = True
            actual_rounds = timer._solve_at_price(visits, masses, price)
            assert actual_rounds[1:3] == expected_rounds[1:3]
            np.testing.assert_allclose(actual_rounds[3], expected_rounds[3], rtol=0, atol=1e-10)
            if actual_rounds[0] is not None:
                compare_forward((actual_rounds[0], [], ""), (expected_rounds[0], [], ""))
        assert gpu.telemetry["completed_retime_forward_calls"] > 0


@pytest.mark.parametrize("failure", ["collect_without_deploy", "stay_too_short"])
def test_forward_rejects_invalid_mining_schedule(monkeypatch, failure):
    timer, visits = fixture(monkeypatch, 2, tied=True)
    if failure == "collect_without_deploy":
        visits[1] = replace(visits[1], deploy=False)
    else:
        visits[1] = replace(visits[1], pinned_arrival=timer.lattice.epochs[100])
        visits[2] = replace(
            visits[2], deploy=False, collect=False, pinned_arrival=timer.lattice.epochs[102]
        )
        visits[3] = replace(visits[3], pinned_arrival=timer.lattice.epochs[104])
    with lambert.using_lambert_backend("cuda") as gpu:
        dp = gpu.retime_dp(timer, visits, [2500.0] * 4, 0.0, True)
        assert dp is not None
        a, d = timer.lattice.epochs[dp[0]].tolist(), timer.lattice.epochs[dp[1]].tolist()
        actual = timer._forward(visits, a, d)
        assert actual == (None, [], failure)
        monkeypatch.setattr(lambert, "cuda_retime_forward_result", lambda *args: None)
        assert timer._forward(visits, a, d) == actual


def test_forward_cache_rejects_changed_policy(monkeypatch):
    timer, visits = fixture(monkeypatch, 3)
    with lambert.using_lambert_backend("cuda") as gpu:
        dp = gpu.retime_dp(timer, visits, [2500.0] * 4, 0.15, True)
        assert dp is not None
        a, d = timer.lattice.epochs[dp[0]].tolist(), timer.lattice.epochs[dp[1]].tolist()
        ws = gpu.retime_workspace
        assert ws.forward_result(timer, visits, a, d) is not None
        timer.ban(1, 2, 0.01)
        assert ws.forward_result(timer, visits, a, d) is None
        dp = gpu.retime_dp(timer, visits, [2500.0] * 4, 0.15, True)
        if dp is not None:
            a, d = timer.lattice.epochs[dp[0]].tolist(), timer.lattice.epochs[dp[1]].tolist()
            timer.inflations[(1, 2)] = 1.7
            assert ws.forward_result(timer, visits, a, d) is None


def test_missing_forward_entry_point_requires_explicit_cpu_mode(monkeypatch):
    timer, visits = fixture(monkeypatch, 1)
    with lambert.using_lambert_backend("cuda") as gpu:
        timer._dp(visits, [2500.0] * 4, 0.15)
        monkeypatch.setattr(gpu.retime_workspace, "evaluate_forward", None)
        with pytest.raises(RuntimeError, match="rebuilt native library"):
            gpu.retime_dp(timer, visits, [2500.0] * 4, 0.15, True)
        gpu.retime_cuda_forward = False
        assert gpu.retime_dp(timer, visits, [2500.0] * 4, 0.15, True) is NotImplemented
