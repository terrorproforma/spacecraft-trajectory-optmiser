"""Retained DP buffers must behave like fresh problems across topology changes."""

import ctypes as ct
import dataclasses
import os

import numpy as np
import pytest
from test_gtoc12_collectdp import T0, _FakeTable

from spacepdhcg.gtoc12.collectdp import plan_collect_tour
from spacepdhcg.gtoc12.gpu_collect_dp import Inputs, Policy, Result
from spacepdhcg.gtoc12.lambert import using_lambert_backend

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


def problem(k, n, nt, nr, seed):
    rng = np.random.default_rng(seed)
    p = Policy(
        k,
        n,
        nt,
        nr,
        seed % k,
        0,
        0,
        0,
        0.6,
        29.41995,
        0.9,
        1.2,
        1.0,
        0.0,
        0.9,
        1.3,
        1.0,
        (ct.c_double * 5)(),
    )
    data = dict(
        dv=rng.uniform(0.1, 0.9, (k, k, n, nt)),
        returns=rng.uniform(0.2, 1.0, (k, n, nr)),
        tofs=np.arange(1, nt + 1, dtype=np.float64) * 60,
        return_tofs=np.arange(nr, dtype=np.float64) * 120 + 600,
        mined=np.arange(n, dtype=np.float64)[None, :] * rng.uniform(0.1, 1.0, (k, 1)),
        geometry_a=np.zeros((k, k)),
        geometry_l=np.zeros((k, k, n)),
        penalty=rng.uniform(0.0, 0.1, (k, k, n)),
        override_inflation=np.full((k, n, nr), np.nan),
        steps=np.arange(1, nt + 1, dtype=np.int32),
        banned=np.eye(k, dtype=np.int32),
    )
    if seed % 2 and k > 1:
        data["banned"][0, 1] = 1
    return p, data, Inputs(*(data[name].ctypes.data for name, _ in Inputs._fields_))


def api(gpu):
    create = gpu.library.spacepdhcg_collect_create
    update = gpu.library.spacepdhcg_collect_update
    solve = gpu.library.spacepdhcg_collect_solve_v2
    destroy = gpu.library.spacepdhcg_collect_destroy
    create.argtypes = [ct.POINTER(Policy), ct.POINTER(Inputs), ct.POINTER(ct.c_void_p)]
    update.argtypes = [ct.c_void_p, ct.POINTER(Policy), ct.POINTER(Inputs)]
    solve.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_double, ct.c_double, ct.POINTER(Result)]
    destroy.argtypes = [ct.c_void_p]
    for fn in (create, update, solve, destroy):
        fn.restype = ct.c_int
    return create, update, solve, destroy


def solved(fn, handle, k):
    result = Result()
    masses = 1600.0 + np.arange(1 << k, dtype=np.float64)
    assert fn(handle, masses.ctypes.data, 1600.0, 0.03, ct.byref(result)) == 0
    assert result.reserved == 0
    return bytes(result)


def test_rebind_shrinks_restores_and_copies_all_inputs():
    with using_lambert_backend("cuda") as gpu:
        create, update, solve, destroy = api(gpu)
        p, data, inputs = problem(4, 33, 4, 3, 8)
        retained = ct.c_void_p()
        assert create(ct.byref(p), ct.byref(inputs), ct.byref(retained)) == 0
        address = retained.value
        try:
            for shape in [(2, 17, 2, 1, 7), (4, 33, 4, 3, 18), (1, 9, 1, 2, 22), (3, 31, 3, 3, 31)]:
                p, data, inputs = problem(*shape)
                fresh = ct.c_void_p()
                assert create(ct.byref(p), ct.byref(inputs), ct.byref(fresh)) == 0
                try:
                    expected = solved(solve, fresh, p.k)
                finally:
                    assert destroy(fresh) == 0
                assert update(retained, ct.byref(p), ct.byref(inputs)) == 0
                # The native workspace owns copies, including after an update.
                for values in data.values():
                    values.fill(0)
                assert retained.value == address
                assert solved(solve, retained, p.k) == expected
                assert solved(solve, retained, p.k) == expected
        finally:
            assert destroy(retained) == 0


@pytest.mark.parametrize(
    "invalid",
    ["capacity_k", "capacity_n", "capacity_nt", "capacity_nr", "camp", "thrust", "tof", "null"],
)
def test_rejected_update_preserves_the_last_problem(invalid):
    with using_lambert_backend("cuda") as gpu:
        create, update, solve, destroy = api(gpu)
        p, _data, inputs = problem(2, 17, 2, 2, 5)
        retained = ct.c_void_p()
        assert create(ct.byref(p), ct.byref(inputs), ct.byref(retained)) == 0
        try:
            expected = solved(solve, retained, p.k)
            dims = [2, 17, 2, 2]
            if invalid.startswith("capacity_"):
                dims[["k", "n", "nt", "nr"].index(invalid.removeprefix("capacity_"))] += 1
            changed, values, packed = problem(*dims, 11)
            if invalid == "camp":
                changed.camp = changed.k
            if invalid == "thrust":
                changed.thrust = float("nan")
            if invalid == "tof":
                values["tofs"][0] = 0.0
            if invalid == "null":
                packed.mined = None
            assert update(retained, ct.byref(changed), ct.byref(packed)) == (
                4 if invalid.startswith("capacity_") else 1
            )
            assert solved(solve, retained, p.k) == expected
        finally:
            assert destroy(retained) == 0


def test_python_reuses_across_tours_and_rebuilds_when_needed(monkeypatch):
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_REUSE_COLLECT_DP", "1")
    with using_lambert_backend("cuda") as gpu:
        for k, n in [(3, 65), (2, 33), (3, 65), (4, 65), (2, 33)]:
            ids = list(range(11, 11 + k))
            table = _FakeTable(
                ids, {(a, b): 0.3 + 0.01 * b for a in ids for b in ids if a != b}, n_t=n
            )
            table.settings = dataclasses.replace(table.settings, fraction_cache_entries=0)
            args = (table, [(a, T0) for a in ids], ids[-1], table.epochs[0], 1800.0)
            gpu.collect_dp_cuda = False
            expected = plan_collect_tour(*args)
            gpu.collect_dp_cuda = True
            actual = plan_collect_tour(*args)
            assert actual is not None and expected is not None
            assert actual.order == expected.order
            assert actual.collect_epochs == expected.collect_epochs
            np.testing.assert_array_equal(actual.hops, expected.hops)
            assert actual.objective_kg == pytest.approx(expected.objective_kg, rel=0, abs=1e-9)
        assert gpu.telemetry["collect_dp_allocations"] == 2
        assert gpu.telemetry["collect_dp_rebinds"] == 3


@pytest.mark.parametrize("invalid", ["negative_offset", "missing_tables", "missing_returns"])
def test_invalid_resident_rebind_preserves_owned_host_inputs(invalid):
    with using_lambert_backend("cuda") as gpu:
        create, _, solve, destroy = api(gpu)
        p, _data, inputs = problem(2, 17, 2, 2, 6)
        handle = ct.c_void_p()
        assert create(ct.byref(p), ct.byref(inputs), ct.byref(handle)) == 0
        update = gpu.library.spacepdhcg_collect_update_tables
        update.argtypes = [
            ct.c_void_p,
            ct.POINTER(Policy),
            ct.POINTER(Inputs),
            ct.c_void_p,
            ct.c_void_p,
            ct.c_int32,
        ]
        update.restype = ct.c_int
        try:
            expected = solved(solve, handle, p.k)
            pairs = (ct.c_void_p * (p.k * p.k))()
            returns = None if invalid == "missing_returns" else (ct.c_void_p * p.k)()
            offset = -1 if invalid == "negative_offset" else 0
            assert update(handle, ct.byref(p), ct.byref(inputs), pairs, returns, offset) == 1
            assert solved(solve, handle, p.k) == expected
        finally:
            assert destroy(handle) == 0
