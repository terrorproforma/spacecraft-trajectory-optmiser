"""GPU move ordering, compact winner epochs and real accepted search updates."""

import ctypes as ct
import math
import os
from copy import deepcopy

import numpy as np
import pytest
import test_gtoc12_gpu_joint as oracle

from spacepdhcg.gtoc12.gpu_joint import (
    GEOMETRY_STATS,
    RESULT,
    GpuJoint,
    _geometry_inputs,
    _metadata,
    evaluate_joint,
    evaluate_mesh,
)
from spacepdhcg.gtoc12.jointopt import JointItinerary

catalogue_and_bonus = oracle.catalogue_and_bonus
incumbent = oracle.incumbent
gpu = oracle.gpu
FLAG = "SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_MESH"
RESIDENT = "SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY"
pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="serialized CUDA required"
)


def mesh_rows(visits, arr, dep, delta=3.0):
    # The general parity fixture prepends the incumbent; production moves() does not.
    return tuple(rows[1:] for rows in oracle._moves(visits, arr, dep, delta))


@pytest.mark.parametrize(
    "size,delta,zeros",
    [
        (2, 1.0, False),
        (3, 0.25, False),
        (None, 3.0, False),
        (None, 45.0, False),
        (None, 0.25, True),
    ],
)
def test_native_mesh_epochs_match_every_ordered_move(gpu, incumbent, size, delta, zeros):
    joint, visits, arr, dep = incumbent
    if size:
        indices = [*range(size - 1), len(visits) - 1]
        visits = [visits[i] for i in indices]
        arr = arr[indices]
        dep = dep[indices]
    if zeros:
        arr = np.full_like(arr, -0.0)
        dep = np.full_like(dep, -0.0)
    expected = mesh_rows(visits, arr, dep, delta)
    count, n = expected[0].shape
    native = GpuJoint(gpu, count, n)
    metadata, stages, policy = _metadata(joint, visits)
    elements, records = _geometry_inputs(joint, visits)
    results = np.empty(count, RESULT)
    stats = np.zeros(1, GEOMETRY_STATS)
    output = (np.empty_like(expected[0]), np.empty_like(expected[1]))
    try:
        status = native.mesh(
            native.handle,
            delta,
            *(a.ctypes.data for a in (policy, metadata, stages, arr, dep)),
            ct.addressof(elements),
            records.ctypes.data,
            len(records),
            0.0,
            results.ctypes.data,
            None,
            None,
            None,
            None,
            None,
            *(a.ctypes.data for a in output),
            stats.ctypes.data,
        )
        assert status == 0
        for actual, wanted in zip(output, expected, strict=True):
            np.testing.assert_array_equal(actual.view(np.uint64), wanted.view(np.uint64))
    finally:
        native.close()


@pytest.mark.parametrize("warm", [False, True])
def test_mesh_winner_matches_resident_explicit_epochs(monkeypatch, gpu, incumbent, warm):
    joint, visits, arr, dep = incumbent
    arrivals, departures = mesh_rows(visits, arr, dep)
    if warm:
        oracle._warm_geometry(gpu, joint, visits, arrivals, departures)
    monkeypatch.setenv(RESIDENT, "1")
    expected = evaluate_joint(
        deepcopy(joint), visits, arrivals, departures, minimum_objective=-math.inf
    )
    before = dict(gpu.telemetry)
    actual = evaluate_mesh(deepcopy(joint), visits, arr, dep, 3.0, -math.inf)
    assert actual is not None
    np.testing.assert_array_equal(actual[0], arrivals[expected[0]])
    np.testing.assert_array_equal(actual[1], departures[expected[0]])
    oracle._same(actual[2], expected[1], atol=0, rtol=0)
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
    assert evaluate_mesh(joint, visits, arr, dep, 3.0, actual[2].objective) is None


def test_mesh_controller_accepts_identical_real_moves(monkeypatch, gpu, incumbent):
    template, visits, arr, dep = incumbent
    monkeypatch.setenv(RESIDENT, "1")
    arrivals, departures = mesh_rows(visits, arr, dep)
    values = evaluate_joint(deepcopy(template), visits, arrivals, departures)
    worst = min((i for i, v in enumerate(values) if v.feasible), key=lambda i: values[i].objective)
    arr, dep = arrivals[worst], departures[worst]
    monkeypatch.setenv(FLAG, "0")
    expected = deepcopy(template).optimise_epochs(visits, arr, dep, mesh=(3.0, 1.0), max_moves=3)
    assert expected[3] > 0
    monkeypatch.setenv(FLAG, "1")
    with monkeypatch.context() as guard:

        def forbidden(*args):
            raise AssertionError("Python mesh generation was called")

        guard.setattr(JointItinerary, "moves", staticmethod(forbidden))
        actual = deepcopy(template).optimise_epochs(visits, arr, dep, mesh=(3.0, 1.0), max_moves=3)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[3] == expected[3]
    oracle._same(actual[2], expected[2], atol=0, rtol=0)
