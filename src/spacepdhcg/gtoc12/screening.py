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
# Neighbour proxy: penalise relative phase drift accumulated over a typical deploy-to-collect stay.
PHASE_DRIFT_HORIZON_YEARS = 10.0
PHASE_DRIFT_WEIGHT = 0.5


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


# Low-thrust penalty over the impulsive (zero-revolution Lambert) ΔV as a function of the
# authority ratio r = Lambert ΔV / (T_max / m x TOF).  Measured on the 1674 SCvx-certified
# asteroid hops archived by the first three campaigns (results/gtoc12/runs/*): the median
# measured/Lambert ratio is 1.04 at r < 0.1, 1.08 at 0.1-0.2, 1.13 at 0.2-0.3, 1.21 at 0.3-0.4,
# 1.31 at 0.4-0.5 and 1.45 at 0.5-0.6 - a flat 1.2 over-prices slow hops by 10-15 % and
# under-prices hops near the authority limit by 20 %.  ``1.05 + 0.65 r`` sits at the p60-p75 of
# every bin (conservative, so the forward mass check closes), rms residual 0.10.
LOW_THRUST_INFLATION_FLOOR = 1.05
LOW_THRUST_INFLATION_SLOPE = 0.65


def low_thrust_inflation(
    lambert_dv_km_s: FloatArray,
    mass_kg: FloatArray,
    tof_days: FloatArray,
    *,
    floor: float = LOW_THRUST_INFLATION_FLOOR,
    slope: float = LOW_THRUST_INFLATION_SLOPE,
) -> FloatArray:
    """Ratio-dependent propellant inflation ``floor + slope x r`` (see above); vectorised."""

    authority = thrust_authority_km_s(mass_kg, tof_days, 1.0)
    ratio = np.asarray(lambert_dv_km_s) / np.maximum(authority, 1e-12)
    return floor + slope * ratio


# Earth-return inflation (SCvx ΔV / zero-revolution Lambert ΔV with the 6 km/s arrival v∞
# allowance) as a function of the return TOF, measured on the 2455 SCvx-certified returns of
# the archive (results/gtoc12/runs/*, ``hopcalib.certified_returns``).  The zero-revolution
# Lambert ΔV is nearly flat in TOF (6.0-6.4 km/s from 405 to 525 days) while the certified
# low-thrust ΔV falls from 8.3 km/s at 405-435 d to 5.5 km/s at 525-555 d: a 420-day return
# really costs 1.30x Lambert (p65 1.38) and a 540-day one 0.96x.  Pricing every TOF at a flat
# factor (1.0 in the campaigns) therefore under-priced the short returns the re-timer chose by
# 65 kg (median, 133 kg p90) on the v6 fleet and hid that a 120-day longer return is ~70 kg
# cheaper.  Table: bin centre (days) -> p65 of the measured ratio; linear between centres,
# clamped outside.  Within a bin the ratio still matters (405-465 d: 1.22 at r = 0.25, 1.39 at
# r = 0.45), hence the ``RETURN_INFLATION_RATIO_SLOPE`` correction about the median ratio 0.33.
RETURN_INFLATION_TOF_DAYS = (352.0, 420.0, 450.0, 480.0, 510.0, 540.0, 578.0, 630.0, 690.0, 810.0)
RETURN_INFLATION_P65 = (1.323, 1.383, 1.295, 1.195, 1.099, 0.977, 0.885, 0.930, 0.932, 1.014)
RETURN_INFLATION_RATIO_SLOPE = 0.6
RETURN_INFLATION_RATIO_CENTRE = 0.33


def return_inflation_model(
    tof_days: FloatArray | float,
    authority_ratio: FloatArray | float,
    *,
    floor: float = 0.85,
) -> FloatArray:
    """TOF- and ratio-dependent inflation of an Earth return (see the table above); vectorised.

    ``floor`` keeps the model from crediting more than the archive supports at the long-TOF
    end (the p25 of the 555-600 d bin is 0.72; SCvx does beat zero-revolution Lambert there
    because the multi-revolution low-thrust arc is the natural solution, but the forward mass
    check must still close).
    """

    tof = np.asarray(tof_days, dtype=np.float64)
    ratio = np.asarray(authority_ratio, dtype=np.float64)
    base = np.interp(tof, RETURN_INFLATION_TOF_DAYS, RETURN_INFLATION_P65)
    correction = 1.0 + RETURN_INFLATION_RATIO_SLOPE * (ratio - RETURN_INFLATION_RATIO_CENTRE)
    return np.maximum(base * np.clip(correction, 0.85, 1.2), floor)


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
    # Self-cleaning chains revisit the same pair roughly a decade later: bodies with different
    # mean motions drift apart in phase, and re-phasing inside a few-hundred-day hop is expensive.
    mean_motion_s = np.sqrt(C.MU_SUN_KM3_S2 / catalogue.semi_major_axis_km[s] ** 3)
    mean_motion_t = np.sqrt(C.MU_SUN_KM3_S2 / catalogue.semi_major_axis_km[t] ** 3)
    drift = np.abs(mean_motion_t - mean_motion_s) * PHASE_DRIFT_HORIZON_YEARS * C.YEAR_S
    drift_penalty = PHASE_DRIFT_WEIGHT * v_circ * np.minimum(drift, np.pi) / np.pi
    return base + 0.5 * v_circ * np.hypot(ex_t - ex_s, ey_t - ey_s) + node_penalty + drift_penalty


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
