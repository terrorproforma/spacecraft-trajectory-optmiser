"""Resident return reuse must preserve exact rows and independent ownership."""

import ctypes as ct
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.gpu_lambert import HopElements, body_elements
from spacepdhcg.gtoc12.gpu_options import GpuResidentOptions
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="serialized CUDA tests"
)


def stats(gpu):
    function = gpu.library.spacepdhcg_orbitweaver_hop_option_cache_stats
    function.argtypes = [ct.c_void_p] + [ct.POINTER(ct.c_uint64)] * 4
    function.restype = ct.c_int
    values = [ct.c_uint64() for _ in range(4)]
    gpu._check(function(gpu.handle, *(ct.byref(v) for v in values)))
    return [v.value for v in values]


def bindings(gpu):
    cached = gpu.library.spacepdhcg_orbitweaver_hop_options_cached_resident
    plain = gpu.library.spacepdhcg_orbitweaver_hop_options_resident
    common = [
        ct.c_void_p,
        ct.POINTER(HopElements),
        ct.c_void_p,
        ct.c_size_t,
        ct.c_int32,
        ct.POINTER(ct.c_void_p),
        ct.POINTER(ct.c_size_t),
    ]
    plain.argtypes = common
    cached.argtypes = [*common, ct.POINTER(ct.c_int32)]
    cached.restype = plain.restype = ct.c_int
    return cached, plain


def query(cat):
    return HopElements(
        body_elements(cat, 45738),
        body_elements(cat, 0),
        C.MU_SUN_KM3_S2,
        0.0,
        C.MAX_VINF_EARTH_KM_S,
    )


def call(gpu, q, times, ordered=1, cached=True):
    functions = bindings(gpu)
    table = GpuResidentOptions(gpu)
    count, hit = ct.c_size_t(), ct.c_int32()
    args = [
        gpu.handle,
        ct.byref(q),
        times.ctypes.data,
        len(times),
        ordered,
        ct.byref(table.handle),
        ct.byref(count),
    ]
    gpu._check(functions[0 if cached else 1](*args, *([ct.byref(hit)] if cached else [])))
    table.count = count.value
    return table, hit.value


def test_reuse_skips_screening_and_survives_producer_replacement(monkeypatch):
    cat = load_catalogue()
    departures = np.arange(20, dtype=float) * 8.0 + 66000.0
    tofs = np.full(20, 480.0)
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_RETAIN_RETURN_OPTIONS", "1")
    with using_lambert_backend("cuda", maximum_batch_size=17) as gpu:
        expected = gpu.paired_options(cat, 45738, 0, departures, tofs, sort_returns=True)
        first = gpu.paired_options(
            cat, 45738, 0, departures, tofs, sort_returns=True, resident=True
        )
        before = dict(gpu.telemetry)
        second = gpu.paired_options(
            cat, 45738, 0, departures, tofs, sort_returns=True, resident=True
        )
        assert first.handle.value != second.handle.value
        assert first.computed_hops == 20 and second.computed_hops == 0
        for key in (
            "completed_batches",
            "completed_branch_requests",
            "compact_option_download_bytes",
            "resident_option_builds",
        ):
            assert gpu.telemetry[key] == before[key]
        assert gpu.telemetry["cached_return_branches"] == 40
        assert stats(gpu)[:3] == [1, 1, 0]
        first.close()
        assert second._host is None
        gpu._prepare(257)  # destroys the producer and its cache; borrowed rows remain owned
        actual = gpu.select_collection(second, 1000.0, 68000.0, SearchSettings(), first=True)
        reference = gpu.select_collection(expected, 1000.0, 68000.0, SearchSettings(), first=True)
        assert actual == reference
        np.testing.assert_array_equal(second.read(), np.asarray(expected).reshape(-1, 3))


@pytest.mark.parametrize("change", ["time", "duration", "elements", "allowance", "sort"])
def test_exact_query_key_matches_fresh_rows(change):
    with using_lambert_backend("cuda") as gpu:
        gpu._prepare(256)
        q = query(load_catalogue())
        times = np.array([[66000.0 + i * 13.0, 500.0] for i in range(5)])
        first, hit = call(gpu, q, times)
        assert hit == 0
        mode = 1
        if change == "time":
            times[0, 0] = np.nextafter(times[0, 0], np.inf)
        elif change == "duration":
            times[0, 1] = np.nextafter(times[0, 1], np.inf)
        elif change == "elements":
            q.departure.mean += 0.01
        elif change == "allowance":
            q.arrival_allowance += 0.25
        else:
            mode = 0
        second, hit = call(gpu, q, times, mode)
        assert hit == 0
        third, hit = call(gpu, q, times, mode)
        assert hit == 1
        plain, _ = call(gpu, q, times, mode, cached=False)
        np.testing.assert_array_equal(second.read(), plain.read())
        np.testing.assert_array_equal(third.read(), plain.read())
        first.close()
        second.close()
        assert stats(gpu)[:3] == [1, 2, 0]


def test_eviction_empty_results_and_native_thread_ownership():
    with using_lambert_backend("cuda", maximum_batch_size=31) as gpu:
        gpu._prepare(256)
        q = query(load_catalogue())
        times = np.array([[66000.0, 500.0]])
        original, _ = call(gpu, q, times)
        expected, _ = call(gpu, q, times, cached=False)
        for i in range(257):
            temporary, _ = call(gpu, q, np.array([[66001.0 + i, 500.0]]))
            temporary.close()
        assert stats(gpu)[2] >= 2
        assert original._host is None
        np.testing.assert_array_equal(original.read(), expected.read())
        rebuilt, hit = call(gpu, q, times)
        assert hit == 0
        np.testing.assert_array_equal(rebuilt.read(), expected.read())
        empty = np.array([[66000.0, 0.0]])
        a, hit = call(gpu, q, empty)
        b, hit2 = call(gpu, q, empty)
        assert (len(a), len(b), hit, hit2) == (0, 0, 0, 1)
        native, _ = bindings(gpu)
        handle, count, hit = ct.c_void_p(), ct.c_size_t(99), ct.c_int32(99)
        with ThreadPoolExecutor(max_workers=1) as pool:
            status = pool.submit(
                native,
                gpu.handle,
                ct.byref(q),
                times.ctypes.data,
                len(times),
                1,
                ct.byref(handle),
                ct.byref(count),
                ct.byref(hit),
            ).result()
            assert status != 0 and not handle.value
            status = pool.submit(gpu.destroy, ct.byref(gpu.handle)).result()
            assert status != 0 and gpu.handle.value
        again, hit = call(gpu, q, empty)
        assert hit == 1 and len(again) == 0


def test_search_evaluation_counter_counts_only_fresh_return_screening(monkeypatch):
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_RETAIN_RETURN_OPTIONS", "1")
    search = RouteSearch(load_catalogue(), [45738], SearchSettings())
    with using_lambert_backend("cuda") as gpu:
        end = C.MISSION_END_MJD - search.settings.end_margin_days
        first = search._return_options(45738, end)
        count = search.lambert_evaluations
        second = search._return_options(45738, end)
        assert count > 0 and search.lambert_evaluations == count
        assert gpu.telemetry["cached_return_branches"] == count
        np.testing.assert_array_equal(first.read(), second.read())
