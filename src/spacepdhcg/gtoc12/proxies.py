"""Low-thrust-aware ΔV proxies for GTOC12 hop screening.

Three estimators, from cheapest to most faithful:

* :func:`phasing_edelbaum_proxy` — Lambert-free.  Edelbaum-style orbit-change ΔV (Δa, Δe, Δi
  between the two element sets) plus a two-impulse phasing ΔV for the residual phase angle that
  remains after drifting for the time of flight.  Vectorised over an entire pool x TOF grid, so it
  is the *cluster-first* ranking used to pick which targets get a Lambert evaluation.
* zero-revolution Lambert (``screening.lambert_hops``) — the reference hops show true low-thrust
  ΔV / Lambert ΔV with median 1.16 and p90 1.34 (``results/gtoc12/proxy_validation.json``).
* :func:`low_thrust_feasible` — mass-consistent time-of-flight check: the inflated ΔV must fit
  inside the thrust authority ``T/m * tof * duty`` with the *mass-averaged* acceleration.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state

FloatArray = NDArray[np.float64]


def phasing_edelbaum_proxy(
    catalogue: AsteroidCatalogue,
    source: int,
    targets: NDArray[np.int64],
    epoch: float,
    tofs_days: FloatArray,
) -> dict[str, FloatArray]:
    """ΔV proxy ``source -> targets`` departing at ``epoch`` for each time of flight.

    Returns ``delta_v`` (targets x tofs, km/s), its per-target minimum ``best_delta_v`` and the
    argmin ``best_tof_days``, plus the departure ``phase_deg`` (positive when the target leads).
    """

    tofs = np.asarray(tofs_days, dtype=float)
    s_index = catalogue.index_of(source)
    t_index = catalogue.index_of(targets)
    a_s = catalogue.semi_major_axis_km[s_index]
    a_t = catalogue.semi_major_axis_km[t_index]
    v_s = np.sqrt(C.MU_SUN_KM3_S2 / a_s)
    n_s = v_s / a_s
    n_t = np.sqrt(C.MU_SUN_KM3_S2 / a_t**3)
    r_s, _ = asteroid_state(catalogue, source, epoch)
    r_t, _ = asteroid_state(catalogue, targets, np.full(targets.shape[0], epoch))
    cross_z = r_s[0] * r_t[:, 1] - r_s[1] * r_t[:, 0]
    phase = np.arctan2(cross_z, r_t @ r_s)
    # orbit-change part: Hohmann-like for Δa, a plane change through the *relative* inclination
    # (inclination-vector difference, so differing nodes count), and a tangential correction for
    # the eccentricity-vector difference (equal e with different perihelia is a different ellipse)
    dv_a = 0.5 * v_s * np.abs(a_t - a_s) / a_s
    inc, node = catalogue.inclination_rad, catalogue.ascending_node_rad
    i_vec = inc[:, None] * np.stack([np.cos(node), np.sin(node)], axis=1)
    dv_i = v_s * np.linalg.norm(i_vec[t_index] - i_vec[s_index], axis=1)
    varpi = node + catalogue.argument_of_perihelion_rad
    e_vec = catalogue.eccentricity[:, None] * np.stack([np.cos(varpi), np.sin(varpi)], axis=1)
    dv_e = 0.5 * v_s * np.linalg.norm(e_vec[t_index] - e_vec[s_index], axis=1)
    dv_orbit = np.sqrt(dv_a**2 + dv_i**2 + dv_e**2)
    # phasing part: residual angle after drifting for ``tof`` closed by a two-impulse phasing
    # manoeuvre (Δn·tof = Δθ  ->  ΔV ≈ (2/3) v Δθ / (n tof))
    tof_s = tofs[None, :] * C.DAY_S
    residual = phase[:, None] + (n_t - n_s)[:, None] * tof_s
    residual = np.arctan2(np.sin(residual), np.cos(residual))
    dv_phase = (2.0 / 3.0) * v_s * np.abs(residual) / (n_s * tof_s)
    delta_v = dv_orbit[:, None] + dv_phase
    best = np.argmin(delta_v, axis=1)
    return {
        "target_ids": np.asarray(targets, dtype=np.int64),
        "tofs_days": tofs,
        "delta_v": delta_v,
        "best_delta_v": delta_v[np.arange(targets.shape[0]), best],
        "best_tof_days": tofs[best],
        "phase_deg": np.rad2deg(phase),
        "dv_orbit": dv_orbit,
    }


def low_thrust_feasible(
    mass_kg: FloatArray,
    delta_v_km_s: FloatArray,
    tof_days: FloatArray,
    *,
    inflation: float = 1.2,
    duty: float = 0.8,
) -> NDArray[np.bool_]:
    """Mass-consistent thrust-authority test for an impulsive proxy ΔV.

    The propellant for the inflated ΔV is removed first so the authority uses the mean of the
    start and end masses; a heavier ship therefore needs a longer flight for the same proxy.
    """

    mass = np.asarray(mass_kg, dtype=float)
    dv = np.asarray(delta_v_km_s, dtype=float) * inflation
    exhaust = C.ISP_S * C.G0_M_S2 * 1e-3
    mass_end = mass * np.exp(-dv / exhaust)
    mean_mass = 0.5 * (mass + mass_end)
    authority = C.THRUST_MAX_N / mean_mass * 1e-3 * np.asarray(tof_days) * C.DAY_S * duty
    return dv <= authority
