"""Device expansion preserves scalar admission, original gates and stable ties."""

import ctypes as ct
import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.gpu_expansion import (
    DEPLOY,
    OPTION,
    PARENT,
    POLICY,
    RESULT,
    GpuExpansion,
    expand_and_select,
    pack,
)
from spacepdhcg.gtoc12.lambert import _GPU_BACKEND
from spacepdhcg.gtoc12.search import PlannedLeg, RouteSearch, SearchSettings, _Partial

requires_gpu = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="explicit serialized GPU tests"
)


def fixture(*, ratio=False, tie=False, weighted=False):
    search = object.__new__(RouteSearch)
    search.settings = replace(
        SearchSettings(),
        beam_width=16,
        collect_dp=False,
        deploy_wait_days=(0.0, 15.0),
        reserve_fraction=0.0,
        return_reserve_kg=0.0,
        max_per_deployed_set=2,
        max_per_first=20,
        hop_inflation_slope=0.4 if ratio else None,
    )
    search.weights = {a: (float(a % 7) * 0.1 + 0.5 if weighted else 1.0) for a in range(1, 260)}
    search.asteroid_prices = {a: float(a % 3) * 0.03 for a in range(1, 260)} if weighted else {}
    search._clusters = None
    search.banned_pairs = {(2, 5), (3, 6)}
    search._return_feasible = lambda *_: True
    search.hops_from = lambda *_: dict(
        target_ids=np.arange(1, 258, dtype=np.int64),
        tofs_days=np.array([90.0, 180.0, 360.0]),
        feasible=np.ones((257, 3), dtype=bool),
        total_delta_v=np.zeros((257, 3)) if tie else np.arange(771).reshape(257, 3) * 0.001,
    )
    parents = []
    for source in (3, 2):
        leg = PlannedLeg(0, 1, 59000.0, 59500.0, 1.0, 1.0, "earth_out")
        parents.append(
            _Partial([leg], source, 60000.0, 1800.0, [(1, 59500.0), (source, 60000.0)], 10.0)
        )
    return search, parents


class Gpu:
    def __init__(self):
        self.library = ct.CDLL(os.environ["SPACEPDHCG_GTOC12_CUDA_LIBRARY"])
        self.device_id = 0
        self.telemetry = {}
        self.expansion_workspace = None

    def _owned(self):
        pass


def test_layout_and_original_topology():
    assert [a.itemsize for a in (POLICY, PARENT, DEPLOY, OPTION, RESULT)] == [144, 48, 32, 72, 88]
    search, current = fixture()
    _, parents, deploys, options = pack(search, current)
    assert len(options) == 2 * 2 * 257 * 3
    assert parents["begin"].tolist() == [0, 2]
    assert deploys["body"].tolist() == [1, 3, 1, 2]
    assert options["parent"].tolist() == [0] * (2 * 257 * 3) + [1] * (2 * 257 * 3)


def test_empty_screening_grid_packs_without_candidates():
    search, current = fixture()
    search.hops_from = lambda *_: dict(
        target_ids=np.empty(0, np.int64),
        tofs_days=np.array([90.0]),
        feasible=np.empty((0, 1), bool),
        total_delta_v=np.empty((0, 1)),
    )
    assert len(pack(search, current)[3]) == 0


@requires_gpu
def test_cluster_prior_and_unreachable_lookahead_match_original(monkeypatch):
    search, parents = fixture(weighted=True)
    search.settings = replace(search.settings, cluster_bonus_kg=120.0, collect_lookahead_weight=0.4)
    search._clusters = SimpleNamespace(unvisited_potential=lambda target, seen: int(target % 8))
    search.collect_lookahead = lambda source, targets, epoch, mass: np.where(
        targets % 5 == 0, np.inf, targets * 0.25
    )
    expected = search._select([child for parent in parents for child in search._expand(parent)])
    gpu = Gpu()
    token = _GPU_BACKEND.set(gpu)
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_GPU_EXPANSION", "1")
    try:
        actual = expand_and_select(search, parents)
        assert [p.deployed for p in actual] == [p.deployed for p in expected]
        np.testing.assert_allclose(
            [p.score for p in actual], [p.score for p in expected], rtol=2e-14, atol=2e-11
        )
    finally:
        _GPU_BACKEND.reset(token)
        if gpu.expansion_workspace:
            gpu.expansion_workspace.close()


@requires_gpu
def test_cuda_scope_enables_native_expansion_by_default(monkeypatch):
    monkeypatch.delenv("SPACEPDHCG_TEST_GTOC12_GPU_EXPANSION", raising=False)
    search, parents = fixture()
    gpu = Gpu()
    token = _GPU_BACKEND.set(gpu)
    try:
        assert expand_and_select(search, parents)
        assert gpu.telemetry["expansion_depths"] == 1
        monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_GPU_EXPANSION", "0")
        assert expand_and_select(search, parents) is None
        assert gpu.telemetry["expansion_depths"] == 1
    finally:
        _GPU_BACKEND.reset(token)
        if gpu.expansion_workspace:
            gpu.expansion_workspace.close()


@requires_gpu
@pytest.mark.parametrize(
    "ratio,tie,weighted",
    [(False, False, False), (True, False, True), (False, True, False), (True, True, True)],
)
def test_ranked_beam_matches_scalar_and_reuses_workspace(ratio, tie, weighted, monkeypatch):
    search, parents = fixture(ratio=ratio, tie=tie, weighted=weighted)
    expected = search._select([child for parent in parents for child in search._expand(parent)])
    gpu = Gpu()
    token = _GPU_BACKEND.set(gpu)
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_GPU_EXPANSION", "1")
    try:
        for current in (parents, parents[:1], parents):
            actual = expand_and_select(search, current)
            reference = (
                expected if len(current) == 2 else search._select(search._expand(current[0]))
            )
            assert [p.deployed for p in actual] == [p.deployed for p in reference]
            for a, b in zip(actual, reference, strict=True):
                np.testing.assert_allclose(
                    [a.mass, a.score, a.hop_propellant, a.lookahead_kg],
                    [b.mass, b.score, b.hop_propellant, b.lookahead_kg],
                    rtol=2e-14,
                    atol=2e-11,
                )
                assert [
                    (leg.from_id, leg.to_id, leg.role, leg.departure_epoch, leg.arrival_epoch)
                    for leg in a.legs
                ] == [
                    (leg.from_id, leg.to_id, leg.role, leg.departure_epoch, leg.arrival_epoch)
                    for leg in b.legs
                ]
        assert gpu.telemetry["expansion_depths"] == 3
        assert (
            gpu.telemetry["expansion_materialized_children"]
            < gpu.telemetry["expansion_valid_children"]
        )
    finally:
        _GPU_BACKEND.reset(token)
        if gpu.expansion_workspace:
            gpu.expansion_workspace.close()


@requires_gpu
def test_original_authority_deadline_and_failed_rank_invalidates_previous():
    search, current = fixture()
    inputs = pack(search, current[:1])
    policy, parents, deploys, _ = inputs
    options = np.zeros(7, OPTION)
    authority = (C.THRUST_MAX_N / 1800.0 * 1e-3) * 90.0 * C.DAY_S
    boundary = search.settings.hop_authority_ratio * authority
    for i in range(7):
        options[i] = (0, 1, 20 + i, 60000.0, 90.0, boundary, 0.0, 1.0, 0.0, 0.0)
    options["dv"][1] = np.nextafter(boundary, np.inf)
    options["dv"][2] = np.nextafter(boundary, -np.inf)
    options["departure"][3] = policy["arrival_horizon"][0] - 90.0
    options["departure"][4] = np.nextafter(policy["arrival_horizon"][0], np.inf) - 90.0
    options["target"][5] = 1  # already deployed
    options["allowed"][6] = 0
    workspace = GpuExpansion(Gpu())
    try:
        count = workspace.evaluate(policy, parents, deploys, options)
        assert {row[2] for row in workspace.rows(count)} == {20, 22, 23}
        malformed = options.copy()
        malformed["parent"][0] = 100
        with pytest.raises(RuntimeError, match="native status 1"):
            workspace.evaluate(policy, parents, deploys, malformed)
        output = np.full(1, 7, RESULT)
        before = output.tobytes()
        assert workspace.read(workspace.handle, 0, 1, output.ctypes.data) == 1
        assert output.tobytes() == before
        assert workspace.evaluate(policy, parents, deploys, options[:0]) == 0
        assert list(workspace.rows(0)) == []
        assert workspace.evaluate(policy, parents, deploys, options) == 3
    finally:
        workspace.close()
        workspace.close()
