"""Collection DP kernels preserve exact-mass CPU choices and reconstruction."""

import ctypes as ct
import dataclasses
import os

import numpy as np
import pytest
from test_gtoc12_collectdp import T0, _FakeTable

from spacepdhcg.gtoc12.collectdp import plan_collect_tour
from spacepdhcg.gtoc12.hopcalib import InflationFit
from spacepdhcg.gtoc12.lambert import using_lambert_backend

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("kind", ["ties", "varying", "fit", "phase", "sweep", "infeasible"])
@pytest.mark.parametrize("k,n", [(1, 17), (2, 33), (3, 65), (5, 129)])
def test_collection_cuda_matches_uncached_cpu(kind, k, n):
    ids = list(range(11, 11 + k))
    rng = np.random.default_rng(208 + k)
    costs = {
        (a, b): np.zeros(n) if kind == "ties" else rng.uniform(0.5, 4.0, n)
        for a in ids
        for b in ids
        if a != b
    }
    if kind == "infeasible":
        costs = {pair: np.full(n, np.inf) for pair in costs}
    table = _FakeTable(
        ids,
        costs,
        n_t=n,
        tofs=(180.0, 60.0, 120.0, 120.0),
        return_dv=np.inf if kind == "infeasible" else 1.0,
    )
    table.settings = dataclasses.replace(
        table.settings, fraction_cache_entries=0, hop_inflation_slope=0.65
    )
    table.pair_geometry = lambda a, b, epochs: (
        0.015 * (b - a),
        0.4 + np.arange(len(epochs)) * 0.001,
    )
    table.phase_deg = lambda a, b, epochs: np.full(len(epochs), abs(b - a) * 2.0)
    if kind == "fit":
        table.settings = dataclasses.replace(
            table.settings, inflation_fit=InflationFit((1.1, 0.4, -0.03, 0.02, 0.05), 0.65, 0.85)
        )
    if kind == "phase":
        table.settings = dataclasses.replace(table.settings, harvest_phase=True)
        table.phase_penalty = lambda a, b, epochs: abs(b - a) + np.arange(len(epochs)) * 0.01
    if kind == "sweep":
        for body in ids:
            mask = np.zeros((n, 1), dtype=bool)
            mask[n // 2 :, 0] = True
            table._return_overrides[body] = (np.full((n, 1), 1.12), mask)
    deployed = [(a, T0 + 60 * i) for i, a in enumerate(ids)]
    banned = {(ids[0], ids[-1])} if k > 2 and kind == "varying" else set()
    weights = {a: 0.7 + 0.04 * i for i, a in enumerate(ids)}
    with using_lambert_backend("cuda") as gpu:
        gpu.collect_dp_cuda = False
        expected = plan_collect_tour(
            table, deployed, ids[-1], table.epochs[0], 1800.0, weights=weights, banned_pairs=banned
        )
        gpu.collect_dp_cuda = True
        actual = plan_collect_tour(
            table, deployed, ids[-1], table.epochs[0], 1800.0, weights=weights, banned_pairs=banned
        )
        assert gpu.telemetry["completed_collection_dp_passes"] >= 1
        assert (actual is None) == (expected is None)
        if actual is None:
            return
        assert actual.order == expected.order
        assert actual.collect_epochs == expected.collect_epochs
        assert actual.reposition == expected.reposition
        assert actual.return_departure == expected.return_departure
        assert actual.return_tof == expected.return_tof
        assert actual.dp_states == expected.dp_states
        np.testing.assert_array_equal(actual.hops, expected.hops)
        for field in [
            "objective_kg",
            "propellant_proxy_kg",
            "collected_proxy_kg",
            "hop_propellant_kg",
            "phase_penalty_kg",
            "hop_phase_deg",
        ]:
            np.testing.assert_allclose(
                getattr(actual, field), getattr(expected, field), rtol=0, atol=1e-8
            )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1.0, 0.0])
def test_collection_invalid_mass_is_rejected_without_poisoning_workspace(invalid):
    from spacepdhcg.gtoc12.gpu_collect_dp import Result

    table = _FakeTable([11, 12], {(11, 12): 1.0, (12, 11): 1.0})
    with using_lambert_backend("cuda") as gpu:
        plan_collect_tour(table, [(11, T0), (12, T0)], 12, table.epochs[0], 1800.0)
        ws = gpu.collect_dp_workspace
        masses = np.full(4, 1800.0)
        masses[1] = invalid
        result = Result()
        assert ws.solve_native(ws.handle, masses.ctypes.data, 1800.0, 1.0, ct.byref(result)) == 1
        masses[1] = 1800.0
        assert ws.solve_native(ws.handle, masses.ctypes.data, 1800.0, 1.0, ct.byref(result)) == 0
