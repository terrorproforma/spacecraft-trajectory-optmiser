"""Numerical regression gates for broadcast/cached Lambert screening."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from spacepdhcg.gtoc12 import lambert


def _materialized_residual(z, r1, r2, a_geom, tof, mu, *, stumpff=None):
    """Pre-optimization residual oracle, intentionally materializing the full grid."""
    z, r1, r2, a_geom, tof = (
        np.ascontiguousarray(item, dtype=np.float64)
        for item in np.broadcast_arrays(z, r1, r2, a_geom, tof)
    )
    c, s = lambert._stumpff(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(c) & np.isfinite(s) & (c > 0)
        y = r1 + r2 + a_geom * (z * s - 1) / np.sqrt(np.where(valid, c, 1))
        valid &= np.isfinite(y) & (y >= 0)
        x = np.sqrt(np.where(valid, y, 0) / np.where(valid, c, 1))
        time = (x**3 * s + a_geom * np.sqrt(np.where(valid, y, 0))) / np.sqrt(mu)
    return np.where(valid, time - tof, np.nan), np.where(valid, y, np.nan)


@pytest.mark.parametrize("scan_samples", [127, 8192])
def test_cached_broadcast_solver_is_bitwise_identical(monkeypatch, scan_samples):
    rng = np.random.default_rng(516)
    # Includes both transfer branches, infeasible times, singular endpoints, and
    # enough transfers to cross the default 8192-grid chunk boundary.
    r1 = rng.normal(size=(260, 3))
    r2 = rng.normal(size=(260, 3)) * 2.7
    tof = np.exp(rng.uniform(-4, 5, 260))
    tof[:2] = [0, -1]
    r2[2] = r1[2]
    long_way = rng.random(260) < 0.5
    actual = lambert.lambert_batch(r1, r2, tof, 1.0, long_way=long_way, scan_samples=scan_samples)
    monkeypatch.setattr(lambert, "_residual", _materialized_residual)
    expected = lambert.lambert_batch(r1, r2, tof, 1.0, long_way=long_way, scan_samples=scan_samples)
    assert actual.feasible.any() and not actual.feasible.all()
    for field in fields(actual):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if a.dtype == np.float64:
            a, b = a.view(np.uint64), b.view(np.uint64)
        np.testing.assert_array_equal(a, b)


def test_scan_cache_does_not_cache_mutable_trajectory_data():
    lambert._scan_grid.cache_clear()
    first = lambert._scan_grid(128)
    for item in first:
        assert not item.flags.writeable
    for count in range(129, 141):
        lambert._scan_grid(count)
    assert lambert._scan_grid.cache_info().currsize == 8
    for old, rebuilt in zip(first, lambert._scan_grid(128), strict=True):
        np.testing.assert_array_equal(old, rebuilt)


@pytest.mark.parametrize(
    "r1,r2,days",
    [
        (
            [139395916.52843043, 53066431.26626553, -397.6250211116388],
            [-384725135.8219323, -149064873.93559355, -248707.57559431216],
            472,
        ),
        (
            [-142640460.0303927, 40899626.62993002, -6890.813929957899],
            [-480412602.57008094, 135960926.37426177, -170023.3908161462],
            827,
        ),
    ],
)
@pytest.mark.parametrize("long_way", [False, True])
def test_nearly_aligned_catalogue_endpoints_keep_kepler_closure(r1, r2, days, long_way):
    from spacepdhcg.gtoc12 import constants as C
    from spacepdhcg.gtoc12.ephemeris import propagate_kepler

    result = lambert.lambert_batch([r1], [r2], days * C.DAY_S, long_way=long_way, scan_samples=256)
    assert result.feasible[0]
    r, v = propagate_kepler(np.array([r1]), result.departure_velocity, days * C.DAY_S)
    assert np.linalg.norm(r[0] - r2) < 0.001
    assert np.linalg.norm(v - result.arrival_velocity) < 1e-9
