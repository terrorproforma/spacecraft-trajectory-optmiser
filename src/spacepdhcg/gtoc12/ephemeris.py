"""Heliocentric two-body ephemerides exactly as defined in GTOC12 Appendix 6.1.

Planets, asteroids and coasting ships all follow ``r'' = -mu r / |r|^3`` in the J2000 heliocentric
ecliptic frame.  Bodies are described by six classical elements at ``64328 MJD``; the mean anomaly
advances as ``M(t) = M0 + sqrt(mu/a^3) (t - t0)`` and Kepler's equation ``M = E - e sin E`` is
solved before the position/velocity expressions with the ``P``/``Q`` perifocal vectors.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import (
    DAY_S,
    EARTH,
    MU_SUN_KM3_S2,
    PLANETS,
    PlanetElements,
)
from .data import AsteroidCatalogue

FloatArray = NDArray[np.float64]


def solve_kepler(mean_anomaly: FloatArray, eccentricity: FloatArray, *, tolerance: float = 1e-14):
    """Vectorised Newton solution of ``M = E - e sin E`` for elliptic orbits (radians)."""

    mean = np.mod(np.asarray(mean_anomaly, dtype=np.float64), 2.0 * np.pi)
    ecc = np.asarray(eccentricity, dtype=np.float64)
    if np.any(ecc < 0.0) or np.any(ecc >= 1.0):
        raise ValueError("GTOC12 bodies are elliptic: 0 <= e < 1")
    eccentric = np.where(ecc > 0.8, np.full_like(mean, np.pi), mean).astype(np.float64)
    for _ in range(64):
        residual = eccentric - ecc * np.sin(eccentric) - mean
        step = residual / (1.0 - ecc * np.cos(eccentric))
        eccentric = eccentric - step
        if np.max(np.abs(step)) < tolerance:
            break
    return eccentric


def elements_to_state(
    semi_major_axis_km: FloatArray,
    eccentricity: FloatArray,
    inclination_rad: FloatArray,
    ascending_node_rad: FloatArray,
    argument_of_perihelion_rad: FloatArray,
    mean_anomaly_rad: FloatArray,
    mu: float = MU_SUN_KM3_S2,
) -> tuple[FloatArray, FloatArray]:
    """Official elements -> ``(r, v)`` with ``r`` in km and ``v`` in km/s; leading axis = bodies."""

    a = np.asarray(semi_major_axis_km, dtype=np.float64)
    e = np.asarray(eccentricity, dtype=np.float64)
    i = np.asarray(inclination_rad, dtype=np.float64)
    node = np.asarray(ascending_node_rad, dtype=np.float64)
    peri = np.asarray(argument_of_perihelion_rad, dtype=np.float64)
    eccentric = solve_kepler(mean_anomaly_rad, e)
    # tan(f/2) = sqrt((1+e)/(1-e)) tan(E/2)
    true_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(0.5 * eccentric),
        np.sqrt(1.0 - e) * np.cos(0.5 * eccentric),
    )
    semi_latus = a * (1.0 - e * e)
    radius = semi_latus / (1.0 + e * np.cos(true_anomaly))
    cos_node, sin_node = np.cos(node), np.sin(node)
    cos_peri, sin_peri = np.cos(peri), np.sin(peri)
    cos_inc, sin_inc = np.cos(i), np.sin(i)
    p_vector = np.stack(
        [
            cos_peri * cos_node - sin_peri * sin_node * cos_inc,
            cos_peri * sin_node + sin_peri * cos_node * cos_inc,
            sin_peri * sin_inc,
        ],
        axis=-1,
    )
    q_vector = np.stack(
        [
            -sin_peri * cos_node - cos_peri * sin_node * cos_inc,
            -sin_peri * sin_node + cos_peri * cos_node * cos_inc,
            cos_peri * sin_inc,
        ],
        axis=-1,
    )
    cos_f = np.cos(true_anomaly)[..., None]
    sin_f = np.sin(true_anomaly)[..., None]
    position = radius[..., None] * (p_vector * cos_f + q_vector * sin_f)
    speed_scale = np.sqrt(mu / semi_latus)[..., None]
    velocity = speed_scale * (-p_vector * sin_f + q_vector * (e[..., None] + cos_f))
    return position, velocity


def propagate_elements(
    epoch_mjd: FloatArray,
    mean_anomaly_rad: FloatArray,
    semi_major_axis_km: FloatArray,
    t_mjd: float | FloatArray,
    mu: float = MU_SUN_KM3_S2,
) -> FloatArray:
    """``M(t) = M0 + sqrt(mu/a^3) (t - t0)`` with the epoch difference converted to seconds."""

    dt = (np.asarray(t_mjd, dtype=np.float64) - np.asarray(epoch_mjd, dtype=np.float64)) * DAY_S
    mean_motion = np.sqrt(mu / np.asarray(semi_major_axis_km, dtype=np.float64) ** 3)
    return np.asarray(mean_anomaly_rad, dtype=np.float64) + mean_motion * dt


def planet_state(planet: PlanetElements | int, t_mjd: float | FloatArray):
    """Heliocentric ``(r, v)`` of Venus/Earth/Mars at ``t_mjd`` (km, km/s)."""

    body = PLANETS[planet] if isinstance(planet, int) else planet
    t = np.asarray(t_mjd, dtype=np.float64)
    a = np.full(t.shape, body.semi_major_axis_km)
    mean = propagate_elements(body.epoch_mjd, np.deg2rad(body.mean_anomaly_deg), a, t)
    return elements_to_state(
        a,
        np.full(t.shape, body.eccentricity),
        np.full(t.shape, np.deg2rad(body.inclination_deg)),
        np.full(t.shape, np.deg2rad(body.ascending_node_deg)),
        np.full(t.shape, np.deg2rad(body.argument_of_perihelion_deg)),
        mean,
    )


def earth_state(t_mjd: float | FloatArray):
    return planet_state(EARTH, t_mjd)


def asteroid_state(
    catalogue: AsteroidCatalogue,
    asteroid_ids: int | NDArray[np.int64],
    t_mjd: float | FloatArray,
):
    """Heliocentric ``(r, v)`` of catalogue asteroids; broadcasting IDs against epochs."""

    index = catalogue.index_of(np.asarray(asteroid_ids, dtype=np.int64))
    t = np.asarray(t_mjd, dtype=np.float64)
    index, t = np.broadcast_arrays(index, t)
    mean = propagate_elements(
        catalogue.epoch_mjd[index],
        catalogue.mean_anomaly_rad[index],
        catalogue.semi_major_axis_km[index],
        t,
    )
    return elements_to_state(
        catalogue.semi_major_axis_km[index],
        catalogue.eccentricity[index],
        catalogue.inclination_rad[index],
        catalogue.ascending_node_rad[index],
        catalogue.argument_of_perihelion_rad[index],
        mean,
    )


def all_asteroid_states(catalogue: AsteroidCatalogue, t_mjd: float):
    """States of the entire catalogue at one epoch (two ``(60000, 3)`` arrays)."""

    mean = propagate_elements(
        catalogue.epoch_mjd, catalogue.mean_anomaly_rad, catalogue.semi_major_axis_km, t_mjd
    )
    return elements_to_state(
        catalogue.semi_major_axis_km,
        catalogue.eccentricity,
        catalogue.inclination_rad,
        catalogue.ascending_node_rad,
        catalogue.argument_of_perihelion_rad,
        mean,
    )


def _stumpff(z: FloatArray) -> tuple[FloatArray, FloatArray]:
    z = np.asarray(z, dtype=np.float64)
    c2 = np.empty_like(z)
    c3 = np.empty_like(z)
    positive = z > 1e-8
    negative = z < -1e-8
    small = ~(positive | negative)
    root = np.sqrt(np.abs(z[positive]))
    c2[positive] = (1.0 - np.cos(root)) / z[positive]
    c3[positive] = (root - np.sin(root)) / root**3
    root = np.sqrt(np.abs(z[negative]))
    c2[negative] = (np.cosh(root) - 1.0) / (-z[negative])
    c3[negative] = (np.sinh(root) - root) / root**3
    zs = z[small]
    c2[small] = 0.5 - zs / 24.0 + zs**2 / 720.0 - zs**3 / 40320.0
    c3[small] = 1.0 / 6.0 - zs / 120.0 + zs**2 / 5040.0 - zs**3 / 362880.0
    return c2, c3


def propagate_kepler(
    position: FloatArray,
    velocity: FloatArray,
    dt_s: float | FloatArray,
    mu: float = MU_SUN_KM3_S2,
    *,
    tolerance: float = 1e-15,
) -> tuple[FloatArray, FloatArray]:
    """Universal-variable two-body propagation of Cartesian states (elliptic or hyperbolic).

    Used for ship coast arcs; matches ``elements_to_state`` propagation to round-off.
    """

    r0 = np.atleast_2d(np.asarray(position, dtype=np.float64))
    v0 = np.atleast_2d(np.asarray(velocity, dtype=np.float64))
    dt = np.broadcast_to(np.asarray(dt_s, dtype=np.float64), (r0.shape[0],)).astype(np.float64)
    radius0 = np.linalg.norm(r0, axis=1)
    speed0_sq = np.einsum("ij,ij->i", v0, v0)
    radial = np.einsum("ij,ij->i", r0, v0) / np.sqrt(mu)
    alpha = 2.0 / radius0 - speed0_sq / mu  # 1/a
    sqrt_mu = np.sqrt(mu)
    sign = np.where(dt < 0.0, -1.0, 1.0)

    def time_equation(chi_value: FloatArray) -> tuple[FloatArray, FloatArray]:
        z_value = alpha * chi_value * chi_value
        c2_value, c3_value = _stumpff(z_value)
        f_value = (
            radial * chi_value * chi_value * c2_value
            + (1.0 - alpha * radius0) * chi_value**3 * c3_value
            + radius0 * chi_value
            - sqrt_mu * dt
        )
        df_value = (
            radial * chi_value * (1.0 - z_value * c3_value)
            + (1.0 - alpha * radius0) * chi_value * chi_value * c2_value
            + radius0
        )
        return f_value, df_value

    # F(chi) is monotone increasing (dF/dchi = r > 0): bracket [0, hi] by doubling, then Newton
    # with a bisection safeguard so hyperbolic and near-parabolic arcs cannot diverge.
    lo = np.zeros_like(dt)
    hi = sign * np.where(
        np.abs(alpha) > 1e-12, sqrt_mu * np.abs(alpha) * np.abs(dt), np.sqrt(radius0)
    )
    hi = np.where(hi == 0.0, sign * np.sqrt(radius0), hi)
    for _ in range(200):
        f_hi, _ = time_equation(hi)
        outside = sign * f_hi < 0.0
        if not np.any(outside):
            break
        lo = np.where(outside, hi, lo)
        hi = np.where(outside, 2.0 * hi, hi)
    chi = 0.5 * (lo + hi)
    for _ in range(200):
        f, df = time_equation(chi)
        # shrink the bracket with the current point first, then take a Newton step if it stays
        # inside the *updated* bracket, otherwise bisect it
        positive = sign * f > 0.0
        hi = np.where(positive, chi, hi)
        lo = np.where(positive, lo, chi)
        newton = chi - f / np.where(df > 0.0, df, 1.0)
        inside = (newton - lo) * (newton - hi) < 0.0
        next_chi = np.where(inside & (df > 0.0), newton, 0.5 * (lo + hi))
        step = next_chi - chi
        chi = next_chi
        if np.max(np.abs(step)) <= tolerance * max(1.0, float(np.max(np.abs(chi)))):
            break
    z = alpha * chi * chi
    c2, c3 = _stumpff(z)
    lagrange_f = 1.0 - chi * chi * c2 / radius0
    lagrange_g = dt - chi**3 * c3 / sqrt_mu
    r1 = lagrange_f[:, None] * r0 + lagrange_g[:, None] * v0
    radius1 = np.linalg.norm(r1, axis=1)
    lagrange_fdot = sqrt_mu * chi * (z * c3 - 1.0) / (radius0 * radius1)
    lagrange_gdot = 1.0 - chi * chi * c2 / radius1
    v1 = lagrange_fdot[:, None] * r0 + lagrange_gdot[:, None] * v0
    if np.ndim(position) == 1:
        return r1[0], v1[0]
    return r1, v1


def two_body_acceleration(position: FloatArray, mu: float = MU_SUN_KM3_S2) -> FloatArray:
    radius = np.linalg.norm(position, axis=-1, keepdims=True)
    return -mu * position / radius**3


def hyperbolic_excess(ship_velocity: FloatArray, body_velocity: FloatArray) -> FloatArray:
    return np.asarray(ship_velocity, dtype=np.float64) - np.asarray(body_velocity, dtype=np.float64)
