"""Device price/mass control matches the reference driver and weighted choice."""

import os
from dataclasses import replace

import numpy as np
import pytest
from test_gtoc12_gpu_retime import fixture
from test_gtoc12_retime_forward import compare_forward

from spacepdhcg.gtoc12 import lambert
from spacepdhcg.gtoc12.retiming import plan_value

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("cooperative", [False, True])
@pytest.mark.parametrize("initial_mass", [600.0, 3000.0])
def test_device_driver_matches_price_search(monkeypatch, seed, cooperative, initial_mass):
    timer, visits = fixture(monkeypatch, seed, cooperative)
    timer.search_settings = replace(timer.search_settings, initial_mass=initial_mass)
    deploy = [v.body for v in visits if v.deploy]
    collect = [v.body for v in visits if v.collect]
    foreign = {v.body: v.foreign_deploy_epoch for v in visits if v.foreign_deploy_epoch is not None}
    profile = [initial_mass * (0.3 if seed % 2 else 1.0)] * (len(visits) - 1)
    with lambert.using_lambert_backend("cuda") as gpu:
        for max_prices, max_masses in [(1, 1), (4, 2), (8, 4)]:
            timer.settings = replace(
                timer.settings, max_price_rounds=max_prices, max_mass_rounds=max_masses
            )
            gpu.retime_cuda_driver = False
            final_profiles = []
            reference = timer._solve_at_price

            def observe(*args, reference=reference, final_profiles=final_profiles):
                result = reference(*args)
                final_profiles.append(result[3])
                return result

            with monkeypatch.context() as cpu:
                cpu.setattr(timer, "_solve_at_price", observe)
                expected = timer.retime_order(deploy, collect, profile, foreign=foreign)
            before = gpu.retime_workspace.calls
            gpu.retime_cuda_driver = True
            actual = timer.retime_order(deploy, collect, profile, foreign=foreign)
            assert actual.failure == expected.failure
            assert actual.price == expected.price
            assert actual.price_rounds == expected.price_rounds
            assert actual.mass_rounds == expected.mass_rounds
            assert actual.objective_after == pytest.approx(
                expected.objective_after, abs=1e-9, rel=0
            )
            assert (actual.plan is None) == (expected.plan is None)
            if actual.plan is not None:
                compare_forward((actual.plan, [], ""), (expected.plan, [], ""))
                assert actual.objective_after == pytest.approx(
                    plan_value(actual.plan, timer), abs=1e-9, rel=0
                )
            metadata, native_profile = gpu.retime_workspace.last_driver
            np.testing.assert_allclose(native_profile, final_profiles[-1], atol=1e-10, rtol=0)
            assert metadata["evaluations"] == gpu.retime_workspace.calls - before
        assert gpu.telemetry["completed_retime_driver_calls"] == 3


def test_driver_requires_new_library_and_explicit_reference_toggle(monkeypatch):
    timer, visits = fixture(monkeypatch, 1)
    with lambert.using_lambert_backend("cuda") as gpu:
        timer._dp(visits, [2500.0] * 4, 0.15)
        monkeypatch.setattr(gpu.retime_workspace, "evaluate_order", None)
        with pytest.raises(RuntimeError, match="price/mass driver requires a rebuilt"):
            lambert.cuda_retime_order(timer, visits, [2500.0] * 4)
        gpu.retime_cuda_driver = False
        assert lambert.cuda_retime_order(timer, visits, [2500.0] * 4) is NotImplemented


@pytest.mark.parametrize("enabled", [False, True])
def test_overflowing_price_is_rejected_like_reference(monkeypatch, enabled):
    timer, _visits = fixture(monkeypatch, 0, tied=True)
    timer.search_settings = replace(timer.search_settings, initial_mass=400.0)
    timer.settings = replace(timer.settings, price_growth=1e308)
    with lambert.using_lambert_backend("cuda") as gpu:
        gpu.retime_cuda_driver = enabled
        with pytest.raises(RuntimeError, match="native status 1"):
            timer.retime_order([1, 2], [2, 1], [400.0] * 4)


def test_driver_reuses_graph_with_new_weights_pins_and_first_ties(monkeypatch):
    timer, _visits = fixture(monkeypatch, 0, cooperative=True, tied=True)
    with lambert.using_lambert_backend("cuda") as gpu:
        for weights, credit, margin in [
            ({1: 0.7, 2: 1.2}, 0.4, 400.0),
            (None, 0.0, 5000.0),
            ({1: 2.0, 2: 0.0}, 0.8, 600.0),
        ]:
            timer.weights = weights
            timer.settings = replace(
                timer.settings, orphan_credit=credit, orphan_margin_days=margin
            )
            results = []
            for enabled in [False, True]:
                gpu.retime_cuda_driver = enabled
                results.append(
                    timer.retime_order(
                        [1],
                        [2],
                        [2500.0] * 3,
                        foreign={2: 65000.0},
                        pinned={1: float(timer.lattice.epochs[20])},
                    )
                )
            expected, actual = results
            assert actual.plan is not None
            assert actual.price == expected.price == timer.settings.propellant_price
            assert actual.objective_after == pytest.approx(
                expected.objective_after, abs=1e-9, rel=0
            )
            assert actual.price_rounds == expected.price_rounds
            compare_forward((actual.plan, [], ""), (expected.plan, [], ""))
