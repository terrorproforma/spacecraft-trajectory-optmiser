"""Resident preflight/cache/geometry parity on complete real-ship neighbourhoods."""

import ctypes as ct
import math
import os
from copy import deepcopy

import numpy as np
import pytest
import test_gtoc12_gpu_joint as oracle

from spacepdhcg.gtoc12.gpu_joint import (
    CACHED_COST,
    GEOMETRY_STATS,
    RESULT,
    _geometry_inputs,
    _metadata,
    evaluate_joint,
)
from spacepdhcg.gtoc12.lambert import using_lambert_backend

catalogue_and_bonus = oracle.catalogue_and_bonus
incumbent = oracle.incumbent
FLAG = "SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY"
pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA"
)


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setenv(oracle.JOINT_FLAG, "1")
    with using_lambert_backend("cuda", maximum_batch_size=97) as backend:
        yield backend


@pytest.mark.parametrize("warm", [False, True])
def test_resident_geometry_matches_staged_arithmetic(monkeypatch, gpu, incumbent, warm):
    template, visits, arr, dep = incumbent
    arrivals, departures = oracle._moves(visits, arr, dep)
    if warm:
        oracle._warm_geometry(gpu, template, visits, arrivals, departures)
    old, new = deepcopy(template), deepcopy(template)
    monkeypatch.setenv(FLAG, "0")
    expected = evaluate_joint(old, visits, arrivals, departures)
    before = dict(gpu.telemetry)
    monkeypatch.setenv(FLAG, "1")
    with monkeypatch.context() as guard:

        def forbidden(*args, **kwargs):
            raise AssertionError("resident geometry must not return through paired_hops")

        guard.setattr(gpu, "paired_hops", forbidden)
        actual = evaluate_joint(new, visits, arrivals, departures)
    for a, e in zip(actual, expected, strict=True):
        oracle._same(a, e, atol=0 if warm else 5e-5, rtol=0 if warm else 2e-8)
    assert gpu.telemetry.get("joint_preflight_download_bytes", 0) == before.get(
        "joint_preflight_download_bytes", 0
    )
    counters = [
        gpu.telemetry["joint_geometry_" + k] - before.get("joint_geometry_" + k, 0)
        for k in ("computed_hops", "cached_hops", "rejected_hops")
    ]
    assert sum(counters) == len(arrivals) * (len(visits) - 1)
    assert (counters[0] == 0) == warm
    assert (
        gpu.telemetry["joint_geometry_stats_download_bytes"]
        - before.get("joint_geometry_stats_download_bytes", 0)
        == 24
    )
    assert new._lambert == template._lambert
    # Exercise resident workspace reuse and compact selection with the same costs.
    index = max(
        (i for i, value in enumerate(actual) if value.feasible), key=lambda i: actual[i].objective
    )
    winner, selected = evaluate_joint(
        new, visits, arrivals, departures, minimum_objective=-math.inf
    )
    assert winner == index
    oracle._same(selected, actual[index], atol=0, rtol=0)
    assert evaluate_joint(
        new, visits, arrivals, departures, minimum_objective=selected.objective
    ) == (None, None)


def test_all_cached_overrides_and_nonfinite_values(monkeypatch, gpu, incumbent):
    joint, visits, arr, dep = incumbent
    arrivals, departures = oracle._moves(visits, arr, dep)
    oracle._warm_geometry(gpu, joint, visits, arrivals, departures)
    # Invalid known costs are authoritative too; measured legs can still use them.
    for i, key in enumerate(joint._lambert):
        if i % 3 == 0:
            joint._lambert[key] = math.inf
        elif i % 5 == 0:
            joint._lambert[key] = math.nan
    monkeypatch.setenv(FLAG, "0")
    expected = evaluate_joint(joint, visits, arrivals, departures)
    monkeypatch.setenv(FLAG, "1")
    actual = evaluate_joint(joint, visits, arrivals, departures)
    for a, e in zip(actual, expected, strict=True):
        oracle._same(a, e, atol=0, rtol=0)
    assert gpu.telemetry["joint_geometry_computed_hops"] == 0


def test_geometry_abi():
    assert CACHED_COST.itemsize == 56 and CACHED_COST.fields["value"][1] == 24
    assert GEOMETRY_STATS.itemsize == 24


def test_native_geometry_input_gates_preserve_outputs(monkeypatch, gpu, incumbent):
    joint, visits, arr, dep = incumbent
    monkeypatch.setenv(FLAG, "0")
    evaluate_joint(joint, visits, arr[None], dep[None])
    native = gpu.joint_workspace
    metadata, stages, policy = _metadata(joint, visits)
    elements, records = _geometry_inputs(joint, visits)
    assert len(records) > 1
    result = np.zeros(1, RESULT)
    stats = np.zeros(1, GEOMETRY_STATS)
    result["failure"] = 41
    stats["computed_hops"] = 99
    for kind, expected in (
        ("duplicate", 1),
        ("unsorted", 1),
        ("stage", 1),
        ("epoch", 1),
        ("cached_flag", 4),
        ("measured_flag", 4),
    ):
        bad = records.copy()
        if kind == "duplicate":
            bad[1] = bad[0]
        elif kind == "unsorted":
            bad[:] = records[::-1]
        elif kind == "stage":
            bad[0]["leg"] = -1
        elif kind == "epoch":
            bad[0]["departure"] = np.nan
        elif kind == "cached_flag":
            bad[0]["cached"] = 2
        else:
            bad[0]["value"]["measured"] = 2
        assert (
            native.geometry(
                native.handle,
                1,
                *(a.ctypes.data for a in (policy, metadata, stages, arr, dep)),
                ct.addressof(elements),
                bad.ctypes.data,
                len(bad),
                0.0,
                result.ctypes.data,
                None,
                None,
                None,
                None,
                None,
                stats.ctypes.data,
            )
            == expected
        )
        assert result["failure"][0] == 41 and stats["computed_hops"][0] == 99
