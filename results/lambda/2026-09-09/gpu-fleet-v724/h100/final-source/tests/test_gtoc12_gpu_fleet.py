"""CUDA fleet search versus exhaustive subsets, including cooperative closures."""

import itertools
import json
import os

import numpy as np
import pytest

from spacepdhcg.gtoc12.cooperative import (
    FleetColumn,
    FleetMasterResult,
    fleet_feasible,
    solve_fleet_master,
)
from spacepdhcg.gtoc12.gpu_fleet import pack_columns, solve_fleet_cuda

GPU = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1",
    reason="requires serialized native CUDA runtime",
)


def column(i, deploy, collect, mass, foreign=None):
    return FleetColumn(
        i,
        i,
        str(i),
        dict.fromkeys(deploy, 65000.0),
        dict.fromkeys(collect, 66000.0),
        foreign or {},
        dict.fromkeys(collect, mass / max(1, len(collect))),
        True,
    )


def exact(columns, weights=None, max_ships=100):
    best = 0.0
    for bits in itertools.product((False, True), repeat=len(columns)):
        chosen = tuple(c for c, take in zip(columns, bits, strict=True) if take)
        if sum(c.ships for c in chosen) <= max_ships and not fleet_feasible(chosen):
            best = max(best, sum(c.value(weights) for c in chosen))
    return best


def test_dependency_packing_is_epoch_specific_and_conflicts_symmetric():
    rows = [column(0, [1], [], 0), column(1, [1], [], 0), column(2, [2], [1], 900, {1: 65000.0})]
    rows[1].deploys[1] += 1
    _, co, ci, ro, po, pi, _ = pack_columns(rows, None, None)
    assert ci[co[0] : co[1]].tolist() == [1]
    assert ci[co[1] : co[2]].tolist() == [0]
    assert ro.tolist() == [0, 0, 0, 1] and po.tolist() == [0, 1] and pi.tolist() == [0]


def test_uncomputed_fleet_bounds_are_valid_json_nulls():
    result = FleetMasterResult((), 0.0, float("inf"), 0, False)
    result.lp_relaxations = {1: float("-inf"), 2: 10.0}
    summary = json.loads(json.dumps(result.summary(), allow_nan=False))
    for key in ("upper_bound_kg", "gap_kg", "lp_bound_kg", "lp_gap_kg", "root_bound_kg"):
        assert summary[key] is None
    assert summary["lp_relaxations_kg"] == {"1": None, "2": 10.0}
    assert not summary["proven_optimal"]


@GPU
@pytest.mark.parametrize("seed", range(12))
def test_random_conflicts_match_exhaustive_subsets(seed):
    rng = np.random.default_rng(seed)
    columns = []
    for i in range(11):
        deploy = rng.choice(15, size=2, replace=False).tolist()
        columns.append(column(i, deploy, deploy, float(rng.uniform(30, 1200))))
    weights = {a: float(rng.uniform(0.1, 1.0)) for a in range(15)}
    expected = exact(columns, weights, 7)
    result = solve_fleet_cuda(columns, weights=weights, max_ships=7, node_cap=1_000_000)
    assert result.exhaustive and result.proven
    assert abs(result.objective - expected) < 1e-8
    assert result.upper_bound == result.objective
    assert not fleet_feasible(result.selected)


@GPU
def test_foreign_cycles_negative_provider_bundles_and_empty():
    cycle = [
        column(1, [1], [2], 700, {2: 65000.0}),
        column(2, [2], [1], 800, {1: 65000.0}),
        column(3, [1], [1], 900),
    ]
    negative = [column(1, [1, 3], [3], 1), column(2, [2], [1], 800, {1: 65000.0})]
    a, b = column(10, [10], [10], 700), column(11, [11], [11], 700)
    bundles = [FleetColumn.from_bundle(12, "bundle", [a, b]), a, b, column(13, [10], [10], 1500)]
    for columns, weights, ships in [
        (cycle, None, 3),
        (negative, {3: -1.0}, 2),
        (bundles, None, 2),
        ([], None, 0),
    ]:
        result = solve_fleet_master(columns, weights=weights, max_ships=ships, backend="cuda")
        assert result.exhaustive and not fleet_feasible(result.selected)
        assert abs(result.objective - exact(columns, weights, ships)) < 1e-8
        assert result.backend == "cuda"


@GPU
@pytest.mark.parametrize("cap", [0, 1, 9, 64, 513])
def test_budget_retains_valid_incumbent_and_conservative_bound(cap):
    columns = [column(i, [i], [i], 400 + i * 30) for i in range(14)]
    warm = tuple(columns[-4:])
    result = solve_fleet_cuda(columns, node_cap=cap, incumbent=warm, max_ships=8)
    assert result.nodes <= cap and result.objective >= sum(c.collected_kg for c in warm) - 1e-9
    assert not fleet_feasible(result.selected)
    optimum = exact(columns, max_ships=8)
    assert result.upper_bound >= optimum - 1e-8
    if result.exhaustive:
        assert abs(result.objective - optimum) < 1e-8


@GPU
def test_native_rejects_nonfinite_and_duplicate_column_identity():
    with pytest.raises(ValueError, match="unique"):
        solve_fleet_cuda([column(1, [1], [1], 500), column(1, [2], [2], 500)])
    with pytest.raises(RuntimeError, match="native status 1"):
        solve_fleet_cuda([column(1, [1], [1], np.nan)])
