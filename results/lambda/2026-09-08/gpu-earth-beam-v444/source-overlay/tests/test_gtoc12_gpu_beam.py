"""Initial CUDA beam must retain physics proxies, block limits and tie ordering."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("block,limit", [(1, 7), (3, 7), (1000, 100)])
@pytest.mark.parametrize("bonuses", [False, True])
def test_earth_beam_matches_host_options(monkeypatch, block, limit, bonuses):
    from spacepdhcg.gtoc12 import search

    ids = np.array([57530, 25792, 9595, 4928, 10664, 45738, 1, 2])
    settings = SearchSettings(
        earth_block=block,
        first_level_limit=limit,
        cluster_min_density=0,
        launch_epochs=(64328.125, 64400.25, 64328.125),
        earth_leg_tofs=(400.0, 550.125, 700.0, 550.125),
        cluster_bonus_kg=13.0 if bonuses else 0.0,
        seed_bonus_kg=7.5 if bonuses else 0.0,
    )
    runner = RouteSearch(load_catalogue(), ids, settings)
    if bonuses:
        monkeypatch.setattr(
            RouteSearch,
            "clusters",
            property(lambda self: SimpleNamespace(density_of=lambda a: a % 17)),
        )
    seeded = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]) if bonuses else None
    monkeypatch.setattr(runner, "seeded_mask", lambda pool: seeded)
    monkeypatch.setattr(runner, "_earth_beam_partials", lambda options: options)
    with using_lambert_backend("cuda", maximum_batch_size=31) as gpu:
        for weighted in [False, True]:
            runner.weights = {int(a): 0.4 + (a % 13) / 10 for a in ids} if weighted else {}
            monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_EARTH_BEAM", "0")
            expected = runner._first_level()
            before = gpu.evaluations
            with monkeypatch.context() as patch:
                patch.delenv("SPACEPDHCG_TEST_GTOC12_EARTH_BEAM")
                patch.setattr(
                    search, "screen_earth_to_asteroids", lambda *a: pytest.fail("CPU grid used")
                )
                patch.setattr(
                    search, "propellant_for_delta_v", lambda *a: pytest.fail("CPU cost used")
                )
                patch.setattr(
                    search, "thrust_authority_km_s", lambda *a: pytest.fail("CPU authority used")
                )
                actual = runner._first_level()
            assert len(actual) == len(expected) > 0
            actual, expected = np.asarray(actual), np.asarray(expected)
            np.testing.assert_array_equal(actual[:, 1:4], expected[:, 1:4])
            np.testing.assert_allclose(
                actual[:, [0, 4, 5]], expected[:, [0, 4, 5]], rtol=2e-9, atol=2e-8
            )
            assert gpu.evaluations - before == 2 * len(ids) * 12


def test_earth_beam_empty_invalid_and_reuse(monkeypatch):
    from dataclasses import replace

    cat = load_catalogue()
    runner = RouteSearch(cat, np.array([57530]), SearchSettings(first_level_limit=5))
    monkeypatch.delenv("SPACEPDHCG_TEST_GTOC12_EARTH_BEAM", raising=False)
    with using_lambert_backend("cuda", maximum_batch_size=97):
        original = runner.settings
        for settings in [
            replace(original, launch_epochs=()),
            replace(original, earth_leg_tofs=()),
            replace(original, first_level_limit=0),
        ]:
            runner.settings = settings
            before = runner.lambert_evaluations
            assert runner._first_level() == []
            assert runner.lambert_evaluations == before
        for settings in [
            replace(original, launch_epochs=(float("nan"),)),
            replace(original, earth_leg_tofs=(0.0,)),
            replace(original, earth_block=0),
        ]:
            runner.settings = settings
            with pytest.raises((RuntimeError, ValueError)):
                runner._first_level()
        runner.settings = original
        assert runner._first_level()
        # Reuse across changing dimensions and a larger top-k allocation.
        for limit, epochs in [(1, (64328.0,)), (101, (64328.0, 64400.0)), (3, (64328.0,))]:
            runner.settings = replace(original, first_level_limit=limit, launch_epochs=epochs)
            actual = runner._first_level()
            with monkeypatch.context() as patch:
                patch.setenv("SPACEPDHCG_TEST_GTOC12_EARTH_BEAM", "0")
                expected = runner._first_level()
            assert len(actual) == len(expected) > 0
            for a, b in zip(actual, expected):
                assert a.deployed == b.deployed
                assert abs(a.mass - b.mass) < 1e-7
