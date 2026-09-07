"""CUDA neighbour selection preserves filtering, ordering, ties and fallback."""

import ctypes as ct
import os
from dataclasses import replace

import numpy as np
import pytest

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import AsteroidCatalogue, load_catalogue
from spacepdhcg.gtoc12.gpu_neighbours import BODY, Query
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("scale,neighbours", [(0.0, 24), (1.5, 24), (1000.0, 24), (1.5, 1000)])
def test_gpu_neighbour_order_matches_catalogue_cpu(monkeypatch, scale, neighbours):
    catalogue = load_catalogue()
    pool = np.arange(1, 60001, 113, dtype=np.int64)
    settings = SearchSettings(
        neighbours=neighbours, filter_scale=scale, hop_tofs=(60.0, 90.0, 180.0, 480.0)
    )
    cpu = RouteSearch(catalogue, pool, settings)
    cases = [(1, 64328.0), (1000, 65500.0), (50000, 68000.0)]
    expected = [cpu.candidates(source, epoch) for source, epoch in cases]
    with using_lambert_backend("cuda") as gpu:
        search = RouteSearch(catalogue, pool, settings)
        monkeypatch.setattr(
            search, "band_pool", lambda *args: pytest.fail("CPU band filtering used")
        )
        for (source, epoch), ids in zip(cases, expected, strict=True):
            np.testing.assert_array_equal(search.candidates(source, epoch), ids)
        # A different pool replaces the cache; the first search reacquires it.
        smaller = RouteSearch(catalogue, pool[:8], settings)
        assert len(smaller.candidates(1, 64328.0)) <= 7
        np.testing.assert_array_equal(search.candidates(*cases[0]), expected[0])
        assert gpu.telemetry["completed_neighbour_queries"] == 5


def test_gpu_neighbour_ties_invalid_query_and_empty_pool():
    n = 9
    a = np.full(n, 2.8 * C.AU_KM)
    zeros = np.zeros(n)
    catalogue = AsteroidCatalogue(
        np.arange(1, n + 1),
        np.full(n, 64328.0),
        a,
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
        "synthetic",
    )
    settings = SearchSettings(neighbours=4, filter_scale=0, hop_tofs=(90.0, 180.0))
    with using_lambert_backend("cuda"):
        search = RouteSearch(catalogue, catalogue.ids, settings)
        np.testing.assert_array_equal(search.candidates(1, 64328.0), [2, 3, 4, 5])
        with pytest.raises(RuntimeError, match="invalid"):
            search.candidates(1, np.nan)
        np.testing.assert_array_equal(search.candidates(1, 64328.0), [2, 3, 4, 5])
        invalid = RouteSearch(catalogue, catalogue.ids, replace(settings, band_a_au=0))
        with pytest.raises(RuntimeError, match="invalid"):
            invalid.candidates(1, 64328.0)
        assert not len(RouteSearch(catalogue, [], settings).candidates(1, 64328.0))
        assert not len(RouteSearch(catalogue, [1], settings).candidates(1, 64328.0))
    assert BODY.itemsize == 64 and ct.sizeof(Query) == 56
