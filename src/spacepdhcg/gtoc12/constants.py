"""Official GTOC12 "Sustainable Asteroid Mining" constants and rules.

Every value is transcribed from the official problem statement (``GTOC12_Problem.pdf``,
released 19 June 2023, Tsinghua University) and the submission-format document.  The
machine-readable copy lives in ``benchmarks/gtoc12/gtoc12_rules.json``; a test asserts the two
never drift apart.  Units follow the statement: km, km/s, kg, N, s, and MJD days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

# --- Table 3: physical constants ------------------------------------------------------------
MU_SUN_KM3_S2: Final = 1.32712440018e11
G0_M_S2: Final = 9.80665
AU_KM: Final = 1.49597870691e8
DAY_S: Final = 86400.0
YEAR_DAYS: Final = 365.25
YEAR_S: Final = YEAR_DAYS * DAY_S

# --- Section 2/4: mission window and ship ---------------------------------------------------
MISSION_START_MJD: Final = 64328.0  # 2035-01-01 00:00:00 TT
MISSION_END_MJD: Final = 69807.0  # 2050-01-01 00:00:00 TT
ISP_S: Final = 4000.0
THRUST_MAX_N: Final = 0.6
DRY_MASS_KG: Final = 500.0
MAX_INITIAL_MASS_KG: Final = 3000.0
MINER_MASS_KG: Final = 40.0
MAX_MINERS_PER_SHIP: Final = 20
MINING_RATE_KG_PER_YEAR: Final = 10.0
MIN_MINING_STAY_YEARS: Final = 1.0
MAX_VINF_EARTH_KM_S: Final = 6.0
MIN_SUN_DISTANCE_AU: Final = 0.3
MAX_SHIPS: Final = 100
SHIP_COUNT_RHO_PER_KG: Final = 0.004

# --- Section 3: bonus coefficient -----------------------------------------------------------
BONUS_INITIAL: Final = 1.0
BONUS_BETA_PER_KG: Final = 0.05
BONUS_GAMMA: Final = -0.1

# --- Section 4.h: verification tolerances ---------------------------------------------------
TOLERANCE_POSITION_KM: Final = 1000.0
TOLERANCE_VELOCITY_KM_S: Final = 1.0e-3  # 1.0 m/s
TOLERANCE_MASS_KG: Final = 1.0e-3

# --- Submission format event identifiers ----------------------------------------------------
EVENT_LAUNCH: Final = 0
EVENT_BURN: Final = -1
EVENT_VENUS_FLYBY: Final = -2
EVENT_EARTH_FLYBY: Final = -3
EVENT_MARS_FLYBY: Final = -4
ASTEROID_COUNT: Final = 60_000
ASTEROID_ELEMENT_EPOCH_MJD: Final = 64328.0
MAX_BURN_SAMPLE_INTERVAL_DAYS: Final = 1.0
THRUST_INTERPOLATION_ORDER: Final = 3  # "third Lagrange interpolating polynomial"

# Derived engine quantity: mass flow per newton of thrust in kg/s.
MASS_FLOW_PER_NEWTON_KG_S: Final = 1.0 / (ISP_S * G0_M_S2)


@dataclass(frozen=True, slots=True)
class PlanetElements:
    """Table 2 planetary parameters at ``MISSION_START_MJD`` (km, deg)."""

    name: str
    event_id: int
    gravitational_parameter_km3_s2: float
    minimum_pericentre_radius_km: float
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    ascending_node_deg: float
    argument_of_perihelion_deg: float
    mean_anomaly_deg: float
    epoch_mjd: float = MISSION_START_MJD


VENUS: Final = PlanetElements(
    "Venus",
    EVENT_VENUS_FLYBY,
    3.24858592000e5,
    6351.0,
    1.08208010521e8,
    6.72988099539e-3,
    3.39439096544,
    7.65796397775e1,
    5.51107191497e1,
    1.11218416921e1,
)
EARTH: Final = PlanetElements(
    "Earth",
    EVENT_EARTH_FLYBY,
    3.98600435436e5,
    6678.0,
    1.49579151285e8,
    1.65519129162e-2,
    4.64389155500e-3,
    1.98956406477e2,
    2.62960364700e2,
    3.58039899470e2,
)
MARS: Final = PlanetElements(
    "Mars",
    EVENT_MARS_FLYBY,
    4.28283752140e4,
    3689.0,
    2.27951663551e8,
    9.33662184095e-2,
    1.84693231241,
    4.94553142513e1,
    2.86731029267e2,
    2.38232037154e2,
)
PLANETS: Final = {planet.event_id: planet for planet in (VENUS, EARTH, MARS)}
PLANETS_BY_NAME: Final = {planet.name.lower(): planet for planet in (VENUS, EARTH, MARS)}


def bonus_coefficient(total_mass_already_mined_kg: float) -> float:
    """Dynamic bonus ``B_i = (1 + 2 (1 + beta * M~_i)^gamma) / 3`` from Section 3.

    ``M~_i`` is the total mass mined from asteroid ``i`` by all leaderboard solutions submitted
    before the scored one.  ``B_i`` equals one for an unmined asteroid and tends to ``1/3``.
    """

    if not math.isfinite(total_mass_already_mined_kg) or total_mass_already_mined_kg < 0.0:
        raise ValueError("already-mined mass must be finite and non-negative")
    return (
        1.0 + 2.0 * (1.0 + BONUS_BETA_PER_KG * total_mass_already_mined_kg) ** BONUS_GAMMA
    ) / 3.0


def maximum_ship_count(average_collected_mass_kg: float) -> float:
    """Real-valued bound ``min(100, 2 exp(rho * M_bar))`` from Section 4.e."""

    if not math.isfinite(average_collected_mass_kg) or average_collected_mass_kg < 0.0:
        raise ValueError("average collected mass must be finite and non-negative")
    return min(float(MAX_SHIPS), 2.0 * math.exp(SHIP_COUNT_RHO_PER_KG * average_collected_mass_kg))


def maximum_collected_mass(stay_days: float) -> float:
    """``M_i <= k (t2 - t1)`` with ``k = 10 kg/yr``; the stay must be at least one year."""

    if not math.isfinite(stay_days) or stay_days < 0.0:
        raise ValueError("stay must be finite and non-negative")
    return MINING_RATE_KG_PER_YEAR * stay_days / YEAR_DAYS


def rules_payload() -> dict[str, object]:
    """Machine-readable rule set mirrored by ``benchmarks/gtoc12/gtoc12_rules.json``."""

    def planet(value: PlanetElements) -> dict[str, object]:
        return {
            "name": value.name,
            "event_id": value.event_id,
            "gravitational_parameter_km3_s2": value.gravitational_parameter_km3_s2,
            "minimum_pericentre_radius_km": value.minimum_pericentre_radius_km,
            "semi_major_axis_km": value.semi_major_axis_km,
            "eccentricity": value.eccentricity,
            "inclination_deg": value.inclination_deg,
            "ascending_node_deg": value.ascending_node_deg,
            "argument_of_perihelion_deg": value.argument_of_perihelion_deg,
            "mean_anomaly_deg": value.mean_anomaly_deg,
            "epoch_mjd": value.epoch_mjd,
        }

    return {
        "schema_version": "1.0.0",
        "problem": "GTOC12 Sustainable Asteroid Mining",
        "source": {
            "statement": "GTOC12_Problem.pdf (released 19 June 2023, Tsinghua University)",
            "format": "GTOC12_Submission_Format.pdf",
            "pins": "benchmarks/gtoc12/pins.json",
        },
        "constants": {
            "mu_sun_km3_s2": MU_SUN_KM3_S2,
            "g0_m_s2": G0_M_S2,
            "au_km": AU_KM,
            "day_s": DAY_S,
            "year_days": YEAR_DAYS,
        },
        "window": {
            "start_mjd": MISSION_START_MJD,
            "end_mjd": MISSION_END_MJD,
            "start_utc_label": "2035-01-01T00:00:00 TT",
            "end_utc_label": "2050-01-01T00:00:00 TT",
        },
        "ship": {
            "specific_impulse_s": ISP_S,
            "maximum_thrust_n": THRUST_MAX_N,
            "dry_mass_kg": DRY_MASS_KG,
            "maximum_initial_mass_kg": MAX_INITIAL_MASS_KG,
            "miner_mass_kg": MINER_MASS_KG,
            "maximum_miners": MAX_MINERS_PER_SHIP,
            "maximum_earth_vinf_km_s": MAX_VINF_EARTH_KM_S,
            "minimum_sun_distance_au": MIN_SUN_DISTANCE_AU,
            "final_mass_rule": "m(t) >= dry mass + collected mass carried at all events",
        },
        "mining": {
            "rate_kg_per_year": MINING_RATE_KG_PER_YEAR,
            "minimum_stay_years": MIN_MINING_STAY_YEARS,
            "max_visits_per_asteroid": 2,
            "first_visit": "deploy one miner: m+ = m- - miner_mass",
            "second_visit": "collect: m+ = m- + M_i with M_i <= rate * (t2 - t1)",
            "unload": "Earth flyby with vinf <= 6 km/s unloads all carried collected mass",
        },
        "fleet": {
            "maximum_ships": MAX_SHIPS,
            "rho_per_kg": SHIP_COUNT_RHO_PER_KG,
            "rule": "N <= min(100, 2 exp(rho * mean collected mass per ship))",
        },
        "merit": {
            "objective": "J = sum_i B_i M_i over asteroids whose mass reached Earth",
            "bonus_initial": BONUS_INITIAL,
            "bonus_beta_per_kg": BONUS_BETA_PER_KG,
            "bonus_gamma": BONUS_GAMMA,
            "bonus_formula": "B_i = (1 + 2 (1 + beta * M~_i)^gamma) / 3",
            "dynamic_vs_fixed": (
                "During the competition B_i changed with every leaderboard update (dynamic). "
                "The archived bonus_coefficients.txt is the frozen end-of-competition state and "
                "gives the fixed post-competition weighted score."
            ),
        },
        "tolerances": {
            "position_km": TOLERANCE_POSITION_KM,
            "velocity_km_s": TOLERANCE_VELOCITY_KM_S,
            "mass_kg": TOLERANCE_MASS_KG,
        },
        "format": {
            "event_launch": EVENT_LAUNCH,
            "event_burn": EVENT_BURN,
            "event_venus_flyby": EVENT_VENUS_FLYBY,
            "event_earth_flyby": EVENT_EARTH_FLYBY,
            "event_mars_flyby": EVENT_MARS_FLYBY,
            "asteroid_ids": [1, ASTEROID_COUNT],
            "asteroid_element_epoch_mjd": ASTEROID_ELEMENT_EPOCH_MJD,
            "burn_sample_interval_days_max": MAX_BURN_SAMPLE_INTERVAL_DAYS,
            "thrust_interpolation": "cubic Lagrange over burn samples",
            "propagation": "numerical integration of state and mass between events",
        },
        "planets": [planet(VENUS), planet(EARTH), planet(MARS)],
    }
