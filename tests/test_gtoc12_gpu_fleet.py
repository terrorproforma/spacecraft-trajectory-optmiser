"""CUDA fleet search versus exhaustive subsets, including cooperative closures."""

import ctypes as ct
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
from spacepdhcg.gtoc12.gpu_fleet import CudaFleetWorkspace, pack_columns, solve_fleet_cuda

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


def csr_reference(columns, weights=None, incumbent=None, cap=200000, rounds=16):
    """Keep the CPU topology + legacy native ABI as an independent setup oracle."""
    from spacepdhcg.gtoc12.cooperative import usable_columns
    from spacepdhcg.gtoc12.gpu_fleet import EXCHANGE_REPORT, REPORT

    usable, _ = usable_columns(columns, weights)
    arrays = pack_columns(usable, weights, incumbent)
    library = ct.CDLL(os.environ["SPACEPDHCG_GTOC12_CUDA_LIBRARY"])
    native = library.spacepdhcg_gtoc12_fleet_search_v2_host
    native.argtypes = (
        [ct.c_int32] * 3 + [ct.c_uint64] + [ct.c_void_p] * 9 + [ct.c_int32, ct.c_void_p]
    )
    native.restype = ct.c_int
    selected = np.empty(len(usable), dtype=np.uint8)
    report = np.zeros(1, dtype=REPORT)
    exchanges = np.zeros(1, dtype=EXCHANGE_REPORT)
    status = native(
        len(usable),
        100,
        3,
        cap,
        *(a.ctypes.data for a in arrays),
        selected.ctypes.data,
        report.ctypes.data,
        rounds,
        exchanges.ctypes.data,
    )
    assert status == 0
    return (
        sorted(c.identifier for i, c in enumerate(usable) if selected[i]),
        report[0],
        exchanges[0],
    )


@GPU
@pytest.mark.parametrize("seed", range(6))
def test_cuda_route_setup_matches_csr_oracle_including_filtered_dependencies(seed):
    from dataclasses import replace

    rng = np.random.default_rng(seed)
    columns = [
        column(
            30 - i,
            (a := rng.choice(20, 3, replace=False).tolist()),
            a,
            float(rng.integers(1, 20)) * 100,
        )
        for i in range(14)
    ]
    # The provider is itself unsupported. Single-pass eligibility keeps its user,
    # but the final provider CSR must not include the filtered-out column.
    columns += [
        column(50, [50], [51], 700, {99: 65000.0}),
        column(51, [51], [50], 900, {50: 65000.0}),
        replace(column(52, [99], [99], float("nan")), certified=False),
        column(53, [53], [54], 700, {54: 65000.0}),
        column(54, [54], [53], 900, {53: 65000.0}),
        column(55, [55], [56], 400, {56: 65000.0 + 2e-6}),
        column(56, [56], [56], 500),
    ]
    weights = {i: float(rng.choice([-0.5, 0, 0.5, 1, 2])) for i in range(60)}
    warm = tuple(columns[-2:])
    for cap, rounds in [(0, 16), (50000, 0), (100, 1)]:
        ids, report, exchanges = csr_reference(columns, weights, warm, cap, rounds)
        result = solve_fleet_cuda(
            columns,
            weights=weights,
            incumbent=warm,
            node_cap=cap,
            exchange_rounds=rounds,
            prefix_bits=3,
        )
        assert [c.identifier for c in result.selected] == ids
        for name in ("objective", "upper_bound", "nodes", "exhaustive"):
            assert getattr(result, name) == report[name]
        for name in ("moves", "rounds", "proposals"):
            assert getattr(result, "exchange_" + name) == exchanges[name]


@GPU
def test_cuda_workspace_creation_does_not_use_cpu_scoring_or_topology(monkeypatch):
    from spacepdhcg.gtoc12 import cooperative, gpu_fleet

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU numerical setup was called")

    columns = [column(2, [2], [2], 800), column(1, [1], [1], 700)]
    with monkeypatch.context() as patch:
        patch.setattr(cooperative, "usable_columns", forbidden)
        patch.setattr(gpu_fleet, "pack_columns", forbidden)
        patch.setattr(FleetColumn, "value", forbidden)
        workspace = CudaFleetWorkspace(columns, weights={1: 2.0})
    with workspace:
        result = workspace.solve()
        assert result.objective == 2200
        assert [c.identifier for c in result.selected] == [1, 2]


@GPU
def test_cuda_scoring_preserves_small_terms_during_weighted_cancellation():
    c = column(1, [1, 2, 3, 4], [1, 2, 3, 4], 4.0)
    weights = {1: 1e16, 2: 1.0, 3: 1.0, 4: -1e16}
    result = solve_fleet_cuda([c], weights=weights)
    assert result.objective == 2.0


@GPU
def test_cuda_setup_handles_more_requirements_than_columns_and_an_empty_usable_pool():
    from dataclasses import replace

    provider = column(1, list(reversed(range(1, 97))), [], 0)
    consumer = column(2, [200], list(range(1, 97)), 2000, dict.fromkeys(range(1, 97), 65000.0))
    for columns in (
        [consumer, provider],
        [consumer],
        [replace(provider, certified=False), consumer],
    ):
        ids, report, _ = csr_reference(columns)
        with CudaFleetWorkspace(columns) as workspace:
            result = workspace.solve()
            assert [c.identifier for c in result.selected] == ids
            assert result.objective == report["objective"]


@GPU
@pytest.mark.parametrize("seed", range(3))
def test_retained_workspace_resets_search_state_and_matches_one_shot(seed):
    rng = np.random.default_rng(seed)
    columns = [
        column(i, (a := rng.choice(10, 2, replace=False).tolist()), a, float(rng.uniform(50, 1400)))
        for i in range(9)
    ]
    columns += [
        column(20, [20], [21], 850, {21: 65000.0}),
        column(21, [21], [20], 900, {20: 65000.0}),
    ]
    with CudaFleetWorkspace(columns, prefix_bits=3) as workspace:
        incumbent = None
        for cap, rounds, ships in [
            (0, 16, 5),
            (100000, 0, 7),
            (1, 1, 2),
            (100000, 16, 6),
            (0, 0, 0),
            (100000, 16, 7),
        ]:
            expected = solve_fleet_cuda(
                columns,
                incumbent=incumbent,
                node_cap=cap,
                exchange_rounds=rounds,
                max_ships=ships,
                prefix_bits=3,
            )
            result = workspace.solve(
                incumbent=incumbent, node_cap=cap, exchange_rounds=rounds, max_ships=ships
            )
            for name in (
                "objective",
                "upper_bound",
                "nodes",
                "exhaustive",
                "device_tasks",
                "exchange_moves",
                "exchange_proposals",
                "exchange_rounds",
            ):
                assert getattr(result, name) == getattr(expected, name)
            assert [c.identifier for c in result.selected] == [
                c.identifier for c in expected.selected
            ]
            if result.exhaustive:
                assert result.objective == pytest.approx(exact(columns, max_ships=ships))
            incumbent = result.selected
        assert workspace.setup_seconds > 0
    workspace.close()
    with pytest.raises(RuntimeError, match="closed"):
        workspace.solve()


@GPU
def test_retained_workspace_owns_packing_snapshot_and_recovers_after_invalid_input():
    a, b = column(1, [1], [1], 700), column(2, [2], [2], 800)
    columns = [FleetColumn.from_bundle(3, "pair", [a, b])]
    weights = {1: 2.0, 2: 1.0}
    with CudaFleetWorkspace(columns, weights=weights) as workspace:
        weights[1] = -10
        a.collected_mass[1] = -1
        columns[0].collected_mass[1] = -100
        first = workspace.solve()
        assert first.objective == 2200
        first.selected[0].collected_mass[1] = -200
        first.selected[0].members[0].deploys.clear()
        with pytest.raises(ValueError):
            workspace.solve(max_ships=-1)
        import ctypes as ct

        from spacepdhcg.gtoc12.gpu_fleet import EXCHANGE_REPORT, REPORT

        warm = np.array([2], dtype=np.uint8)
        selected = np.empty(1, dtype=np.uint8)
        report = np.zeros(1, dtype=REPORT)
        exchanges = np.zeros(1, dtype=EXCHANGE_REPORT)
        assert (
            workspace._solve(
                workspace._handle,
                100,
                100,
                warm.ctypes.data,
                selected.ctypes.data,
                report.ctypes.data,
                16,
                exchanges.ctypes.data,
            )
            == 1
        )
        assert (
            workspace._solve(
                ct.c_void_p(),
                100,
                0,
                warm.ctypes.data,
                selected.ctypes.data,
                report.ctypes.data,
                0,
                exchanges.ctypes.data,
            )
            == 1
        )
        again = workspace.solve()
        assert again.objective == 2200 and not fleet_feasible(again.selected)
        assert again.selected[0].members[0].deploys == {1: 65000.0}


@GPU
def test_retained_workspace_empty_repeated_lifetime_and_failed_create():
    for _ in range(5):
        with CudaFleetWorkspace([]) as workspace:
            for rounds in (16, 0):
                result = workspace.solve(node_cap=0, exchange_rounds=rounds)
                assert result.objective == 0 and result.exhaustive and not result.selected
    with pytest.raises(RuntimeError, match="native status 1"):
        CudaFleetWorkspace([column(1, [1], [1], np.nan)])


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


@GPU
@pytest.mark.parametrize("rounds,expected", [(0, 1800), (1, 2500), (2, 2600), (16, 2600)])
def test_exchanges_repair_greedy_ship_rule_discard_without_tree_search(rounds, expected):
    columns = [column(i, [i], [i], mass) for i, mass in enumerate([20, 20, 200, 500])]
    weights = {0: 50.0, 1: 45.0, 2: 4.0, 3: 1.4}
    result = solve_fleet_cuda(
        columns, weights=weights, max_ships=3, node_cap=0, exchange_rounds=rounds
    )
    assert result.objective == pytest.approx(expected)
    assert result.nodes == 0 and not result.exhaustive
    assert not fleet_feasible(result.selected)
    assert result.exchange_moves <= result.exchange_rounds <= rounds
    assert result.exchange_proposals == 0 if rounds == 0 else result.exchange_proposals > 0


@GPU
@pytest.mark.parametrize("seed", range(12))
def test_exchange_termination_matches_cpu_one_swap_local_optimum(seed):
    rng = np.random.default_rng(seed)
    columns = [
        column(i, (a := rng.choice(13, 2, replace=False).tolist()), a, float(rng.uniform(10, 1100)))
        for i in range(11)
    ]
    weights = {a: float(rng.uniform(0.05, 2)) for a in range(13)}
    result = solve_fleet_cuda(
        columns, weights=weights, max_ships=6, node_cap=0, exchange_rounds=100
    )
    assert result.exchange_rounds < 100 and not fleet_feasible(result.selected)
    for remove in [None, *result.selected]:
        kept = tuple(c for c in result.selected if c is not remove)
        for add in [None, *(c for c in columns if c not in result.selected)]:
            trial = kept + (() if add is None else (add,))
            if sum(c.ships for c in trial) <= 6 and not fleet_feasible(trial):
                assert sum(c.value(weights) for c in trial) <= result.objective + 1e-8


@GPU
@pytest.mark.parametrize("cycle", [False, True])
def test_profitable_exchange_cannot_strand_another_ships_miner(cycle):
    if cycle:
        columns = [
            column(0, [1], [2], 700, {2: 65000.0}),
            column(1, [2], [1], 800, {1: 65000.0}),
            column(2, [3], [3], 1000),
        ]
        weights = None
    else:
        columns = [
            column(0, [1, 3], [3], 1),
            column(1, [2], [1], 800, {1: 65000.0}),
            column(2, [4], [4], 700),
        ]
        weights = {3: -50.0}
    warm = tuple(columns[:2])
    result = solve_fleet_cuda(columns, weights=weights, incumbent=warm, max_ships=2, node_cap=0)
    assert {c.identifier for c in result.selected} == {0, 1}
    assert result.exchange_moves == 0 and not fleet_feasible(result.selected)
    assert result.objective == pytest.approx(exact(columns, weights, 2))
