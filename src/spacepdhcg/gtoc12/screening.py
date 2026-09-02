"""Candidate screening for GTOC12 legs: analytic Edelbaum proxy and Lambert rendezvous cost.

Screening estimates are *not* trajectories.  They rank candidate hops so the beam search and the
low-thrust refinement only spend effort on physically plausible legs.  Two proxies are used:

* an Edelbaum-style near-circular transfer cost from the orbital elements (no epoch dependence),
  mirroring ``cpp/include/spacepdhcg/orbitweaver/low_thrust_screening.hpp``;
* the zero-revolution Lambert rendezvous cost at explicit epochs (departure and arrival velocity
  matching), with the 6 km/s Earth launch/arrival hyperbolic-excess allowance credited.

A low-thrust leg is accepted when the impulsive cost fits inside the thrust authority
``duty * (T_max / m) * TOF``; ``duty`` absorbs the gravity-loss and finite-thrust penalty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state, earth_state
from .lambert import lambert_batch

FloatArray = NDArray[np.float64]

# The zero-revolution time-of-flight residual is monotone in the universal variable, so a coarse
# bracketing scan suffices for screening; the parity test still uses the kernel's 8192 samples.
SCREENING_SCAN_SAMPLES = 256


def exhaust_velocity_km_s() -> float:
    return C.ISP_S * C.G0_M_S2 * 1e-3


def propellant_for_delta_v(initial_mass_kg: FloatArray, delta_v_km_s: FloatArray) -> FloatArray:
    """Rocket-equation propellant for an impulsive-equivalent Delta-V."""

    return np.asarray(initial_mass_kg) * (
        1.0 - np.exp(-np.asarray(delta_v_km_s) / exhaust_velocity_km_s())
    )


def thrust_authority_km_s(
    mass_kg: FloatArray, tof_days: FloatArray, duty: float = 0.9
) -> FloatArray:
    """Delta-V reachable with continuous thrust ``duty * T_max / m`` over ``tof_days``."""

    acceleration = C.THRUST_MAX_N / np.asarray(mass_kg) * 1e-3  # km/s^2
    return duty * acceleration * np.asarray(tof_days) * C.DAY_S


def edelbaum_proxy(
    a1_km: FloatArray,
    i1_rad: FloatArray,
    a2_km: FloatArray,
    i2_rad: FloatArray,
    mu: float = C.MU_SUN_KM3_S2,
) -> FloatArray:
    """Edelbaum circular-to-circular low-thrust Delta-V proxy (km/s)."""

    v1 = np.sqrt(mu / np.asarray(a1_km))
    v2 = np.sqrt(mu / np.asarray(a2_km))
    coupling = 0.5 * np.pi * np.abs(np.asarray(i2_rad) - np.asarray(i1_rad))
    return np.sqrt(np.maximum(v1 * v1 + v2 * v2 - 2.0 * v1 * v2 * np.cos(coupling), 0.0))


def element_distance_proxy(
    catalogue: AsteroidCatalogue, source_id: int, target_ids: NDArray[np.int64]
) -> FloatArray:
    """Edelbaum proxy plus an eccentricity-vector mismatch penalty between asteroids."""

    s = catalogue.index_of(source_id)
    t = catalogue.index_of(target_ids)
    base = edelbaum_proxy(
        catalogue.semi_major_axis_km[s],
        catalogue.inclination_rad[s],
        catalogue.semi_major_axis_km[t],
        catalogue.inclination_rad[t],
    )
    # eccentricity vector difference expressed as a velocity: e * v_circ
    ex_s = catalogue.eccentricity[s] * np.cos(
        catalogue.ascending_node_rad[s] + catalogue.argument_of_perihelion_rad[s]
    )
    ey_s = catalogue.eccentricity[s] * np.sin(
        catalogue.ascending_node_rad[s] + catalogue.argument_of_perihelion_rad[s]
    )
    ex_t = catalogue.eccentricity[t] * np.cos(
        catalogue.ascending_node_rad[t] + catalogue.argument_of_perihelion_rad[t]
    )
    ey_t = catalogue.eccentricity[t] * np.sin(
        catalogue.ascending_node_rad[t] + catalogue.argument_of_perihelion_rad[t]
    )
    v_circ = np.sqrt(C.MU_SUN_KM3_S2 / catalogue.semi_major_axis_km[t])
    # a plane change of Delta-Omega at inclination i costs ~ v sin(i) dOmega
    node_penalty = (
        v_circ
        * np.abs(np.sin(catalogue.inclination_rad[t]))
        * np.abs(
            np.angle(
                np.exp(1j * (catalogue.ascending_node_rad[t] - catalogue.ascending_node_rad[s]))
            )
        )
    )
    return base + 0.5 * v_circ * np.hypot(ex_t - ex_s, ey_t - ey_s) + node_penalty


@dataclass(slots=True)
class LambertHop:
    """Vectorised Lambert rendezvous screening result for one batch of hops."""

    departure_epoch: FloatArray
    tof_days: FloatArray
    departure_delta_v: FloatArray  # km/s to leave the departure body (after any launch credit)
    arrival_delta_v: FloatArray  # km/s to match the arrival body (after any arrival credit)
    departure_velocity: FloatArray  # heliocentric km/s of the transfer at departure
    arrival_velocity: FloatArray
    feasible: NDArray[np.bool_]

    @property
    def total_delta_v(self) -> FloatArray:
        return self.departure_delta_v + self.arrival_delta_v


def _credit(delta_v: FloatArray, allowance: float) -> FloatArray:
    return np.maximum(delta_v - allowance, 0.0)


def lambert_hops(
    r1: FloatArray,
    v1_body: FloatArray,
    r2: FloatArray,
    v2_body: FloatArray,
    departure_epoch: FloatArray,
    tof_days: FloatArray,
    *,
    departure_allowance_km_s: float = 0.0,
    arrival_allowance_km_s: float = 0.0,
    scan_samples: int = SCREENING_SCAN_SAMPLES,
) -> LambertHop:
    """Evaluate short- and long-way zero-revolution Lambert arcs, keeping the cheaper of the two."""

    tof_s = np.asarray(tof_days, dtype=np.float64) * C.DAY_S
    best_dep = np.full(r1.shape[0], np.inf)
    best_arr = np.full(r1.shape[0], np.inf)
    best_v1 = np.full_like(r1, np.nan)
    best_v2 = np.full_like(r2, np.nan)
    feasible = np.zeros(r1.shape[0], dtype=bool)
    for long_way in (False, True):
        result = lambert_batch(r1, r2, tof_s, long_way=long_way, scan_samples=scan_samples)
        dep = _credit(
            np.linalg.norm(result.departure_velocity - v1_body, axis=1), departure_allowance_km_s
        )
        arr = _credit(
            np.linalg.norm(result.arrival_velocity - v2_body, axis=1), arrival_allowance_km_s
        )
        total = np.where(result.feasible, dep + arr, np.inf)
        better = total < best_dep + best_arr
        best_dep = np.where(better, dep, best_dep)
        best_arr = np.where(better, arr, best_arr)
        best_v1 = np.where(better[:, None], result.departure_velocity, best_v1)
        best_v2 = np.where(better[:, None], result.arrival_velocity, best_v2)
        feasible |= result.feasible
    return LambertHop(
        np.asarray(departure_epoch, dtype=np.float64),
        np.asarray(tof_days, dtype=np.float64),
        best_dep,
        best_arr,
        best_v1,
        best_v2,
        feasible,
    )


def screen_earth_to_asteroids(
    catalogue: AsteroidCatalogue,
    asteroid_ids: NDArray[np.int64],
    departure_epochs: FloatArray,
    tofs_days: FloatArray,
) -> dict[str, FloatArray]:
    """Earth-launch rendezvous grid: arrays shaped ``(asteroids, epochs, tofs)``."""

    ids = np.asarray(asteroid_ids, dtype=np.int64)
    epochs = np.asarray(departure_epochs, dtype=np.float64)
    tofs = np.asarray(tofs_days, dtype=np.float64)
    a_idx, e_idx, t_idx = np.meshgrid(
        np.arange(ids.shape[0]), np.arange(epochs.shape[0]), np.arange(tofs.shape[0]), indexing="ij"
    )
    a_idx, e_idx, t_idx = a_idx.ravel(), e_idx.ravel(), t_idx.ravel()
    t_dep = epochs[e_idx]
    t_arr = t_dep + tofs[t_idx]
    r_e, v_e = earth_state(t_dep)
    r_a, v_a = asteroid_state(catalogue, ids[a_idx], t_arr)
    hop = lambert_hops(
        r_e, v_e, r_a, v_a, t_dep, tofs[t_idx], departure_allowance_km_s=C.MAX_VINF_EARTH_KM_S
    )
    shape = (ids.shape[0], epochs.shape[0], tofs.shape[0])
    return {
        "asteroid_ids": ids,
        "departure_epochs": epochs,
        "tofs_days": tofs,
        "departure_delta_v": hop.departure_delta_v.reshape(shape),
        "arrival_delta_v": hop.arrival_delta_v.reshape(shape),
        "total_delta_v": hop.total_delta_v.reshape(shape),
        "feasible": hop.feasible.reshape(shape),
    }


def screen_asteroid_hops(
    catalogue: AsteroidCatalogue,
    source_id: int,
    target_ids: NDArray[np.int64],
    departure_epoch: float,
    tofs_days: FloatArray,
) -> dict[str, FloatArray]:
    """Rendezvous hops from one asteroid at a fixed epoch to many targets over a TOF grid."""

    ids = np.asarray(target_ids, dtype=np.int64)
    tofs = np.asarray(tofs_days, dtype=np.float64)
    a_idx, t_idx = np.meshgrid(np.arange(ids.shape[0]), np.arange(tofs.shape[0]), indexing="ij")
    a_idx, t_idx = a_idx.ravel(), t_idx.ravel()
    t_arr = departure_epoch + tofs[t_idx]
    r_s, v_s = asteroid_state(
        catalogue, np.full(a_idx.shape[0], source_id), np.full(a_idx.shape[0], departure_epoch)
    )
    r_t, v_t = asteroid_state(catalogue, ids[a_idx], t_arr)
    hop = lambert_hops(r_s, v_s, r_t, v_t, np.full(a_idx.shape[0], departure_epoch), tofs[t_idx])
    shape = (ids.shape[0], tofs.shape[0])
    return {
        "target_ids": ids,
        "tofs_days": tofs,
        "total_delta_v": hop.total_delta_v.reshape(shape),
        "feasible": hop.feasible.reshape(shape),
    }


def screen_asteroid_to_earth(
    catalogue: AsteroidCatalogue,
    source_id: int,
    departure_epochs: FloatArray,
    tofs_days: FloatArray,
) -> dict[str, FloatArray]:
    """Return legs to Earth with the 6 km/s arrival allowance; arrays ``(epochs, tofs)``."""

    epochs = np.asarray(departure_epochs, dtype=np.float64)
    tofs = np.asarray(tofs_days, dtype=np.float64)
    e_idx, t_idx = np.meshgrid(np.arange(epochs.shape[0]), np.arange(tofs.shape[0]), indexing="ij")
    e_idx, t_idx = e_idx.ravel(), t_idx.ravel()
    t_dep = epochs[e_idx]
    t_arr = t_dep + tofs[t_idx]
    r_s, v_s = asteroid_state(catalogue, np.full(e_idx.shape[0], source_id), t_dep)
    r_e, v_e = earth_state(t_arr)
    hop = lambert_hops(
        r_s, v_s, r_e, v_e, t_dep, tofs[t_idx], arrival_allowance_km_s=C.MAX_VINF_EARTH_KM_S
    )
    shape = (epochs.shape[0], tofs.shape[0])
    return {
        "departure_epochs": epochs,
        "tofs_days": tofs,
        "total_delta_v": hop.total_delta_v.reshape(shape),
        "feasible": hop.feasible.reshape(shape),
    }
