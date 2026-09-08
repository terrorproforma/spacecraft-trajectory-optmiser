"""GPU-built ephemerides preserve screening and fixed-order retiming decisions."""

import os

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import AsteroidCatalogue, load_catalogue
from spacepdhcg.gtoc12.ephemeris import asteroid_state, earth_state
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.retiming import Retimer
from spacepdhcg.gtoc12.search import SearchSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("source,target", [(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)])
def test_elements_match_cpu_ephemeris_screening(source, target):
    # Circular, inclined, high-eccentricity and retrograde orbits, including
    # epochs on both sides of the element reference and partial final batches.
    cat = AsteroidCatalogue(
        np.arange(1, 4),
        np.full(3, 64328.0),
        np.array([2.2, 2.8, 3.1]) * C.AU_KM,
        np.array([0.0, 0.2, 0.85]),
        np.array([0.0, 0.6, 2.9]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.3, 0.7]),
        np.array([0.0, 2.0, 5.0]),
        "synthetic",
    )
    departures = np.array([62000.125, 64328.3, 66000.999, 68000.01])
    tofs = np.array([60.375, 180.125, 400.0, 700.0, 900.0])
    dep, tof = np.repeat(departures, len(tofs)), np.tile(tofs, len(departures))

    def state(body, epochs):
        return earth_state(epochs) if body == 0 else asteroid_state(cat, body, epochs)

    r1, v1 = state(source, dep)
    r2, v2 = state(target, dep + tof)
    with using_lambert_backend("cuda", maximum_batch_size=7) as gpu:
        reference = gpu.screen_hops(
            r1,
            v1,
            r2,
            v2,
            dep,
            tof,
            departure_allowance_km_s=6.0 if source == 0 else 0.0,
            arrival_allowance_km_s=6.0 if target == 0 else 0.0,
        )
        actual, feasible = gpu.leg_table(cat, source, target, departures, tofs)
        np.testing.assert_array_equal(feasible.ravel(), reference.feasible)
        np.testing.assert_allclose(actual.ravel(), reference.total_delta_v, rtol=2e-9, atol=2e-8)
        assert gpu.telemetry["completed_element_hops"] == len(dep)
        paired = gpu.paired_hops(cat, source, target, dep, tof)
        np.testing.assert_array_equal(paired.feasible, reference.feasible)
        np.testing.assert_allclose(
            paired.total_delta_v, reference.total_delta_v, rtol=2e-9, atol=2e-8
        )
        from spacepdhcg.gtoc12.ephemeris import propagate_kepler

        valid = paired.feasible
        position, velocity = propagate_kepler(
            r1[valid], paired.departure_velocity[valid], tof[valid] * C.DAY_S
        )
        assert np.max(np.linalg.norm(position - r2[valid], axis=1)) < 0.01
        assert np.max(np.linalg.norm(velocity - paired.arrival_velocity[valid], axis=1)) < 1e-8
        assert gpu.telemetry["completed_element_hops"] == 2 * len(dep)
        # Scratch-buffer reuse must leave the original interface reusable.
        repeat = gpu.screen_hops(r1, v1, r2, v2, dep, tof)
        assert np.all(repeat.feasible == reference.feasible)


def test_retimer_does_not_compute_cpu_states(monkeypatch):
    cat = load_catalogue()
    with using_lambert_backend("cuda") as gpu:
        retimer = Retimer(cat, SearchSettings())
        monkeypatch.setattr(retimer, "_state", lambda *args: pytest.fail("CPU ephemeris used"))
        dv, feasible = retimer.leg_table(45738, 25792, "deploy_hop")
        assert np.count_nonzero(feasible) > 0
        assert gpu.telemetry["completed_element_hops"] == dv.size
        count = gpu.evaluations
        assert retimer.leg_table(45738, 25792, "deploy_hop")[0] is dv
        assert gpu.evaluations == count


def test_invalid_elements_and_times_recover():
    cat = load_catalogue()
    with using_lambert_backend("cuda") as gpu:
        dv, feasible = gpu.leg_table(cat, 1, 2, [np.nan], [100.0, -1.0, 0.0])
        assert not feasible.any() and np.isinf(dv).all()
        with pytest.raises(ValueError, match="vectors"):
            gpu.leg_table(cat, 1, 2, [[64328.0]], [100.0])
        dv, feasible = gpu.leg_table(cat, 1, 2, [64328.0], [100.0])
        assert feasible.any() and np.isfinite(dv).all()


def test_paired_search_uses_cuda_ephemerides_and_retains_options(monkeypatch):
    from spacepdhcg.gtoc12 import search

    cat = load_catalogue()
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_COMPACT_OPTIONS", "0")
    runner = search.RouteSearch(cat, np.array([45738, 25792]), SearchSettings())
    with using_lambert_backend("cuda", maximum_batch_size=97) as gpu:
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_PAIRED_EPHEMERIDES", "0")
        expected_return = runner._return_options(45738, C.MISSION_END_MJD)
        expected_collect = runner._collect_hop_options(45738, 25792, C.MISSION_END_MJD - 500)
        runner._collect_cache.clear()
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_PAIRED_EPHEMERIDES", "1")
        monkeypatch.setattr(
            search, "asteroid_state", lambda *a: pytest.fail("CPU asteroid ephemeris used")
        )
        monkeypatch.setattr(
            search, "earth_state", lambda *a: pytest.fail("CPU Earth ephemeris used")
        )
        actual_return = runner._return_options(45738, C.MISSION_END_MJD)
        actual_collect = runner._collect_hop_options(45738, 25792, C.MISSION_END_MJD - 500)
        for actual, expected in [
            (actual_return, expected_return),
            (actual_collect, expected_collect),
        ]:
            assert len(actual) == len(expected) > 0
            np.testing.assert_array_equal(np.asarray(actual)[:, 1:], np.asarray(expected)[:, 1:])
            np.testing.assert_allclose(
                np.asarray(actual)[:, 0], np.asarray(expected)[:, 0], rtol=2e-9, atol=2e-8
            )
        assert gpu.telemetry["completed_element_hops"] > 0
        empty = gpu.paired_hops(cat, 1, 2, [], [])
        assert len(empty.feasible) == 0
        with pytest.raises(ValueError, match="matching vectors"):
            gpu.paired_hops(cat, 1, 2, [64328.0], [100.0, 200.0])
        invalid = gpu.paired_hops(cat, 1, 2, [np.nan, 64328.0, 64328.0], [100.0, 0.0, -1.0])
        assert not invalid.feasible.any()


@pytest.mark.parametrize("source,target", [(0, 45738), (45738, 0), (45738, 25792)])
def test_compact_options_match_detailed_hops_exactly(source, target):
    cat = load_catalogue()
    with using_lambert_backend("cuda", maximum_batch_size=31) as gpu:
        # Grow and shrink retained scratch, crossing several native chunk boundaries.
        for count in (1, 33, 1025, 7):
            indices = np.arange(count)
            departures = 64328.125 + (indices % 101) * 11.3
            tofs = 60.375 + (indices % 23) * 27.1
            tofs[indices % 19 == 0] = 0.0
            departures[indices % 37 == 0] = np.nan
            hops = gpu.paired_hops(cat, source, target, departures, tofs)
            costs = hops.total_delta_v
            expected = [
                (float(costs[i]), float(departures[i]), float(tofs[i]))
                for i in range(count)
                if hops.feasible[i] and np.isfinite(costs[i])
            ]
            for ordered in (False, True):
                reference = (
                    sorted(expected, key=lambda row: (row[0], -row[1])) if ordered else expected
                )
                before = dict(gpu.telemetry)
                actual = gpu.paired_options(
                    cat, source, target, departures, tofs, sort_returns=ordered
                )
                assert actual == reference
                assert (
                    gpu.telemetry["completed_branch_requests"] - before["completed_branch_requests"]
                    == 2 * count
                )
                assert gpu.telemetry["completed_compact_options"] - before.get(
                    "completed_compact_options", 0
                ) == len(actual)
                assert (
                    gpu.telemetry["compact_option_download_bytes"]
                    - before.get("compact_option_download_bytes", 0)
                    == 24 * len(actual) + 4
                )
        assert gpu.paired_options(cat, source, target, [], [], sort_returns=True) == []
        with pytest.raises(ValueError, match="matching vectors"):
            gpu.paired_options(cat, source, target, [64328.0], [100.0, 200.0], sort_returns=True)
        with pytest.raises(ValueError, match="matching vectors"):
            gpu.paired_options(cat, source, target, [[64328.0]], [[100.0]], sort_returns=False)


def test_compact_search_bypasses_detailed_host_screening(monkeypatch):
    from spacepdhcg.gtoc12 import search
    from spacepdhcg.gtoc12.screening import LambertHop

    cat = load_catalogue()
    runner = search.RouteSearch(cat, np.array([45738, 25792]), SearchSettings())
    with using_lambert_backend("cuda", maximum_batch_size=97) as gpu:
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_COMPACT_OPTIONS", "0")
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_PAIRED_EPHEMERIDES", "1")
        expected_return = runner._return_options(45738, C.MISSION_END_MJD)
        expected_collect = runner._collect_hop_options(45738, 25792, C.MISSION_END_MJD - 500)
        assert expected_return and expected_collect
        runner._collect_cache.clear()
        before = runner.lambert_evaluations
        monkeypatch.delenv("SPACEPDHCG_TEST_GTOC12_COMPACT_OPTIONS")

        def forbidden(*args, **kwargs):
            pytest.fail(
                "Compact search fell back to detailed host screening or CPU cost arithmetic"
            )

        monkeypatch.setattr(gpu, "paired_hops", forbidden)
        monkeypatch.setattr(search, "lambert_hops", forbidden)
        monkeypatch.setattr(search, "asteroid_state", forbidden)
        monkeypatch.setattr(search, "earth_state", forbidden)
        monkeypatch.setattr(LambertHop, "total_delta_v", property(forbidden))
        assert list(runner._return_options(45738, C.MISSION_END_MJD)) == expected_return
        actual_collect = runner._collect_hop_options(45738, 25792, C.MISSION_END_MJD - 500)
        assert list(actual_collect) == expected_collect
        assert runner.lambert_evaluations == 2 * before
        evaluated = gpu.evaluations
        assert runner._collect_hop_options(45738, 25792, C.MISSION_END_MJD - 500) is actual_collect
        assert gpu.evaluations == evaluated
