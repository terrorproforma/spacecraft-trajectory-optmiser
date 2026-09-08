"""Whole device search parity, stop conditions and retained-workspace reuse."""

import math
import os
import time
from copy import deepcopy

import numpy as np
import pytest
import test_gtoc12_gpu_joint as oracle

from spacepdhcg.gtoc12.gpu_joint import evaluate_joint

catalogue_and_bonus = oracle.catalogue_and_bonus
incumbent = oracle.incumbent
gpu = oracle.gpu
FLAG = "SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SEARCH"
pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="serialized CUDA required"
)


@pytest.mark.parametrize("case", ["moves", "empty", "zero", "expired", "ties", "infeasible"])
def test_whole_search_matches_controller(monkeypatch, gpu, incumbent, case):
    template, visits, arr, dep = incumbent
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY", "1")
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_MESH", "1")
    rows = oracle._moves(visits, arr, dep, 3.0)
    values = evaluate_joint(deepcopy(template), visits, *rows)
    worst = min((i for i, v in enumerate(values) if v.feasible), key=lambda i: values[i].objective)
    arr, dep = rows[0][worst].copy(), rows[1][worst].copy()
    mesh, moves, deadline = (3.0, 1.0), 3, math.inf
    if case == "empty":
        mesh = ()
    elif case == "zero":
        moves = 0
    elif case == "expired":
        deadline = time.perf_counter() - 1
    elif case == "ties":
        mesh = (1e-300, 1e-300)
    elif case == "infeasible":
        arr[0] = dep[0] = 0
    monkeypatch.setenv(FLAG, "0")
    baseline = deepcopy(template)
    expected = baseline.optimise_epochs(
        visits, arr, dep, mesh=mesh, max_moves=moves, deadline=deadline
    )
    if case == "moves":
        assert expected[3] > 0
    else:
        assert expected[3] == 0
    monkeypatch.setenv(FLAG, "1")
    for _ in range(2):
        candidate = deepcopy(template)
        before = dict(gpu.telemetry)
        actual = candidate.optimise_epochs(
            visits, arr, dep, mesh=mesh, max_moves=moves, deadline=deadline
        )
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert actual[3] == expected[3]
        oracle._same(actual[2], expected[2], atol=0, rtol=0)
        assert candidate.evaluations == baseline.evaluations
        assert (
            gpu.telemetry["completed_joint_searches"] - before.get("completed_joint_searches", 0)
            == 1
        )
        assert (
            gpu.telemetry["joint_mesh_epoch_upload_bytes"]
            - before.get("joint_mesh_epoch_upload_bytes", 0)
            == arr.nbytes + dep.nbytes
        )
        assert (
            gpu.telemetry["joint_mesh_epoch_download_bytes"]
            - before.get("joint_mesh_epoch_download_bytes", 0)
            == arr.nbytes + dep.nbytes
        )


@pytest.mark.parametrize(
    "mesh,moves,deadline",
    [
        ((0.0,), 3, math.inf),
        ((math.nan,), 3, math.inf),
        ((1.0,), -1, math.inf),
        ((1.0,), 3, math.nan),
    ],
)
def test_invalid_search_settings_fail_before_dispatch(
    monkeypatch, gpu, incumbent, mesh, moves, deadline
):
    joint, visits, arr, dep = incumbent
    monkeypatch.setenv(FLAG, "1")
    before = dict(gpu.telemetry)
    with pytest.raises(ValueError):
        joint.optimise_epochs(visits, arr, dep, mesh=mesh, max_moves=moves, deadline=deadline)
    assert gpu.telemetry == before


def test_device_deadline_stops_between_neighbourhoods(monkeypatch, gpu, incumbent):
    joint, visits, arr, dep = incumbent
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY", "1")
    # Sub-ULP steps make identical rows, so each level deterministically ends
    # without an accepted move. A long schedule must stop on the device timer.
    started = time.perf_counter()
    result, out_arr, out_dep, report = evaluate_joint(
        joint,
        visits,
        arr[None],
        dep[None],
        minimum_objective=-math.inf,
        _mesh_delta=1.0,
        _search_config=(np.full(100000, 1e-300), 1, started + 0.2),
    )
    assert result[1].feasible
    assert report["stop"] == 1
    assert 0 < report["batches"] < 100000
    assert report["moves"] == 0
    assert report["evaluations"] == 1 + report["batches"] * (10 * len(visits) - 14)
    np.testing.assert_array_equal(out_arr[0], arr)
    np.testing.assert_array_equal(out_dep[0], dep)
    assert time.perf_counter() - started < 5


def test_explicit_search_requires_new_core(monkeypatch, gpu, incumbent):
    joint, visits, arr, dep = incumbent
    joint.evaluate(visits, arr, dep)
    monkeypatch.setattr(gpu.joint_workspace, "search", None)
    monkeypatch.setenv(FLAG, "1")
    with pytest.raises(RuntimeError, match="requires spacepdhcg_gtoc12_joint_search_host"):
        joint.optimise_epochs(visits, arr, dep)
