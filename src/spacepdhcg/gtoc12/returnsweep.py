"""SCvx sweep of the Earth return: measured ``asteroid -> Earth`` cost on a (departure, TOF) grid.

Why a sweep and not a proxy.  The Earth return is one leg per ship, it always departs from the
last collected asteroid, and its zero-revolution Lambert ΔV is nearly flat in TOF (6.0-6.4 km/s
from 405 to 525 days) while the certified low-thrust ΔV is not (8.3 km/s at 405-435 d, 5.5 km/s
at 525-555 d on the 2455 certified returns of the archive).  The fleet's returns were flown at
420 days for 279 kg where the references fly 473-486 days for 208-216 kg.  A 1.5-3 s SCvx solve
per grid cell makes the true cost affordable for one leg: ``sweep_return`` flies every cell of a
small lattice-aligned grid around the plan's return, records the measured ΔV (from the certified
final mass) and SCvx's own feasibility, and the re-timer (``Retimer.set_return_sweep``) prices the
return from that table instead of the proxy, so its propellant-price bisection spends the saved
propellant on the hops (earlier deploys, later collects) rather than on a late, short return.

Deterministic: the grid is fixed by the plan and the settings, cells are flown in grid order.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state, earth_state
from .low_thrust import LegBoundary, ScvxSettings, certify_leg, solve_leg
from .pipeline import clamp_thrust
from .screening import exhaust_velocity_km_s, propellant_for_delta_v
from .search import EARTH_ID, RoutePlan

FloatArray = NDArray[np.float64]
CERTIFICATION_TOLERANCE = 0.5  # as ``pipeline.refine_route``: half the official tolerances

# TOFs flown by default (days, multiples of the 15-day lattice): from the fleet's 420 d up to
# the long returns the references use; 45-day spacing keeps a sweep at ~6 x 6 = 36 SCvx solves
DEFAULT_RETURN_TOFS: tuple[float, ...] = (420.0, 465.0, 510.0, 555.0, 600.0, 645.0)


@dataclass(slots=True)
class ReturnSweep:
    """Measured Earth-return costs of one asteroid on a departure x TOF grid."""

    asteroid: int
    mass_kg: float  # departure mass the cells were flown at
    departures: FloatArray  # MJD, on the re-timer lattice
    tofs: FloatArray  # days, lattice multiples
    attempted: NDArray[np.bool_]  # cell flown (arrival inside the window)
    certified: NDArray[np.bool_]  # SCvx converged and the verifier-model rollout closed
    delta_v_km_s: FloatArray  # measured ΔV ``v_e ln(m0 / m_final)``; inf where not certified
    propellant_kg: FloatArray  # at ``mass_kg``; inf where not certified
    solves: int = 0
    wall_seconds: float = 0.0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def cheapest(self) -> tuple[float, float, float] | None:
        """(departure, tof, propellant) of the cheapest certified cell; ``None`` if none."""

        if not np.any(self.certified):
            return None
        flat = int(np.argmin(np.where(self.certified, self.propellant_kg, np.inf)))
        i, j = divmod(flat, self.tofs.shape[0])
        return float(self.departures[i]), float(self.tofs[j]), float(self.propellant_kg[i, j])

    def summary(self) -> dict[str, Any]:
        cheapest = self.cheapest()
        return {
            "asteroid": self.asteroid,
            "mass_kg": self.mass_kg,
            "departures": [float(d) for d in self.departures],
            "tofs": [float(t) for t in self.tofs],
            "attempted": int(self.attempted.sum()),
            "certified": int(self.certified.sum()),
            "solves": self.solves,
            "wall_seconds": self.wall_seconds,
            "cheapest": None
            if cheapest is None
            else {"departure": cheapest[0], "tof_days": cheapest[1], "propellant_kg": cheapest[2]},
            "propellant_kg": [
                [
                    None if not self.certified[i, j] else round(float(self.propellant_kg[i, j]), 1)
                    for j in range(self.tofs.shape[0])
                ]
                for i in range(self.departures.shape[0])
            ],
        }


def return_leg_of(plan: RoutePlan) -> Any | None:
    """The plan's Earth-return leg (``None`` when the plan has none)."""

    for leg in reversed(plan.legs):
        if leg.role == "earth_return" and leg.to_id == EARTH_ID:
            return leg
    return None


def sweep_grid(
    plan: RoutePlan,
    *,
    step_days: float = 15.0,
    back_steps: int = 2,
    forward_steps: int = 4,
    tofs: tuple[float, ...] = DEFAULT_RETURN_TOFS,
    end_margin_days: float = 2.0,
) -> tuple[FloatArray, FloatArray] | None:
    """Lattice-aligned (departures, tofs) around the plan's return departure.

    Departures run from ``back_steps`` before to ``forward_steps`` after the plan's return
    departure (snapped onto the ``step_days`` lattice from the mission start); TOFs are the
    given lattice multiples.  Cells whose arrival misses the window are left to the sweep to
    skip.  ``None`` when the plan has no return.
    """

    leg = return_leg_of(plan)
    if leg is None:
        return None
    for tof in tofs:
        if abs(tof / step_days - round(tof / step_days)) > 1e-9:
            raise ValueError(f"return TOF {tof} is not a multiple of the {step_days} d step")
    k = round((leg.departure_epoch - C.MISSION_START_MJD) / step_days)
    end = C.MISSION_END_MJD - end_margin_days
    ks = np.arange(k - back_steps, k + forward_steps + 1)
    departures = C.MISSION_START_MJD + step_days * ks
    departures = departures[(departures > C.MISSION_START_MJD) & (departures + min(tofs) <= end)]
    return departures, np.asarray(tofs, dtype=np.float64)


def sweep_return(
    catalogue: AsteroidCatalogue,
    asteroid: int,
    mass_kg: float,
    departures: FloatArray,
    tofs: FloatArray,
    *,
    scvx: ScvxSettings | None = None,
    end_margin_days: float = 2.0,
    minimum_final_mass_kg: float = C.DRY_MASS_KG,
    cache: dict[tuple[int, float, float], tuple[bool, float]] | None = None,
) -> ReturnSweep:
    """Fly ``asteroid -> Earth`` with SCvx at every grid cell whose arrival fits the window.

    A cell is certified when SCvx converges and the independent verifier-model rollout of the
    emitted arcs closes on Earth; its ΔV is measured from the certified final mass.  ``cache``
    (keyed by asteroid, departure, TOF) shares results between sweeps of the same ship.
    """

    started = time.perf_counter()
    scvx = scvx or ScvxSettings()
    cache = {} if cache is None else cache
    departures = np.asarray(departures, dtype=np.float64)
    tofs = np.asarray(tofs, dtype=np.float64)
    shape = (departures.shape[0], tofs.shape[0])
    attempted = np.zeros(shape, dtype=bool)
    certified = np.zeros(shape, dtype=bool)
    delta_v = np.full(shape, np.inf)
    propellant = np.full(shape, np.inf)
    diagnostics: list[dict[str, Any]] = []
    solves = 0
    end = C.MISSION_END_MJD - end_margin_days
    for i, departure in enumerate(departures):
        for j, tof in enumerate(tofs):
            arrival = float(departure + tof)
            if arrival > end + 1e-9:
                continue
            attempted[i, j] = True
            key = (int(asteroid), float(departure), float(tof))
            if key in cache:
                ok, dv = cache[key]
            else:
                ok, dv, note = _fly(
                    catalogue,
                    asteroid,
                    mass_kg,
                    float(departure),
                    arrival,
                    scvx,
                    minimum_final_mass_kg,
                )
                solves += 1
                cache[key] = (ok, dv)
                if note:
                    diagnostics.append(
                        {"departure": float(departure), "tof_days": float(tof), "note": note}
                    )
            if ok:
                certified[i, j] = True
                delta_v[i, j] = dv
                propellant[i, j] = float(propellant_for_delta_v(mass_kg, dv))
    return ReturnSweep(
        int(asteroid),
        float(mass_kg),
        departures,
        tofs,
        attempted,
        certified,
        delta_v,
        propellant,
        solves,
        time.perf_counter() - started,
        diagnostics,
    )


@dataclass(slots=True)
class ReturnRetime:
    """Outcome of :func:`retime_return` for one certified route."""

    sweep: ReturnSweep | None
    before_kg: float  # collected mass of the route before
    return_before_kg: float  # SCvx propellant of the flown return before
    improvement: Any  # retiming.CertifiedImprovement or None
    route: Any  # best certified re-timed RefinedRoute (None when nothing better certified)
    skipped: str = ""

    @property
    def after_kg(self) -> float:
        return self.before_kg if self.route is None else float(self.route.total_collected_kg)

    @property
    def return_after_kg(self) -> float:
        if self.route is None:
            return self.return_before_kg
        return _return_propellant(self.route)

    def summary(self) -> dict[str, Any]:
        return {
            "skipped": self.skipped,
            "before_kg": self.before_kg,
            "after_kg": self.after_kg,
            "gain_kg": self.after_kg - self.before_kg,
            "return_before_kg": self.return_before_kg,
            "return_after_kg": self.return_after_kg,
            "sweep": None if self.sweep is None else self.sweep.summary(),
            "improvement": None if self.improvement is None else self.improvement.summary(),
        }


def _return_propellant(route: Any) -> float:
    for leg in reversed(route.legs):
        if leg.planned.role == "earth_return" and leg.solution is not None:
            return float(leg.solution.propellant_kg)
    return math.nan


def neighbourhood(
    catalogue: AsteroidCatalogue, plan: RoutePlan, settings: Any, count: int = 40
) -> NDArray[np.int64]:
    """The plan's asteroids plus up to ``count`` co-moving neighbours of its camp asteroid (the
    beam's own scaled element/phase distance, ``ComovingClusters.neighbours``): the pool a
    re-timing may insert from.  Falls back to the plan's asteroids alone when the camp has no
    neighbours inside the bands."""

    from .clusters import ClusterBands, ComovingClusters

    deploy_order = [a for a, _ in sorted(plan.deploy_epochs.items(), key=lambda kv: kv[1])]
    camp = int(deploy_order[-1])
    bands = getattr(settings, "cluster_bands", None) or ClusterBands.collect_window()
    # coarse element box (3 bands) around the camp first: the phasing-aware clustering is
    # O(pool) KD-tree work and is only needed on the few hundred candidates that can be close
    cat = catalogue
    idx = int(np.searchsorted(cat.ids, camp))
    box = (
        (np.abs(cat.semi_major_axis_km - cat.semi_major_axis_km[idx]) / C.AU_KM <= 3.0 * bands.a_au)
        & (np.abs(cat.eccentricity - cat.eccentricity[idx]) <= 3.0 * bands.e)
        & (np.abs(np.rad2deg(cat.inclination_rad - cat.inclination_rad[idx])) <= 3.0 * bands.i_deg)
    )
    pool = np.unique(np.append(cat.ids[box].astype(np.int64), camp))
    clusters = ComovingClusters(catalogue, pool, bands)
    nearest = [int(a) for a in clusters.neighbours(camp)[:count]]
    members = sorted(set(int(a) for a in plan.asteroids) | set(nearest))
    return np.asarray(members, dtype=np.int64)


def retime_return(
    route: Any,
    catalogue: AsteroidCatalogue,
    *,
    search_settings: Any,
    retime_settings: Any,
    weights: dict[int, float] | None = None,
    scvx: ScvxSettings | None = None,
    refine=None,
    step_days: float | None = None,
    # a ship forced into a late, short return (family 6: 420 d at 278 kg, only 3 cells inside
    # the window) needs the earlier departures to reach the cheap long TOFs; the re-timer's
    # ``extend`` step moves the return later
    back_steps: int = 6,
    forward_steps: int = 6,
    tofs: tuple[float, ...] = DEFAULT_RETURN_TOFS,
    max_attempts: int = 2,
    time_budget_seconds: float = float("inf"),
    excluded: set[int] | None = None,
) -> ReturnRetime:
    """Sweep a certified route's Earth return with SCvx and re-time the route against it.

    The sweep is flown at the route's certified return-departure mass; the re-timer is
    calibrated from the route's certified legs, keeps its Earth-out TOF, prices the return from
    the sweep and re-times the visit order (``improve_and_certify``: proxy re-timing, SCvx
    re-flight, bans, up to ``max_attempts``), also trying to insert a nearby asteroid from
    ``neighbourhood`` when the saved propellant allows.  Only fully certified routes are
    returned; ``excluded`` asteroids (other ships' in the archive group) are never inserted.
    """

    from .retiming import Retimer, calibrate_from_route, improve_and_certify
    from .search import RouteSearch

    plan = route.plan
    leg = return_leg_of(plan)
    before = float(route.total_collected_kg)
    return_before = _return_propellant(route)
    if leg is None:
        return ReturnRetime(None, before, return_before, None, None, "no Earth return")
    if plan.orphaned:
        return ReturnRetime(None, before, return_before, None, None, "leaves orphans")
    step = float(retime_settings.step_days if step_days is None else step_days)
    grid = sweep_grid(
        plan,
        step_days=step,
        back_steps=back_steps,
        forward_steps=forward_steps,
        tofs=tofs,
        end_margin_days=retime_settings.end_margin_days,
    )
    if grid is None or grid[0].shape[0] == 0:
        return ReturnRetime(None, before, return_before, None, None, "no grid inside the window")
    mass = next(
        (
            float(refined.mass_before)
            for refined in route.legs
            if refined.planned.role == "earth_return"
        ),
        float(plan.final_mass_proxy_kg + plan.propellant_proxy_kg),
    )
    sweep = sweep_return(
        catalogue,
        int(leg.from_id),
        mass,
        grid[0],
        grid[1],
        scvx=scvx,
        end_margin_days=retime_settings.end_margin_days,
    )
    if not np.any(sweep.certified):
        return ReturnRetime(sweep, before, return_before, None, None, "no certified return cell")
    retimer = Retimer(catalogue, search_settings, retime_settings, weights)
    calibrate_from_route(retimer, route)
    retimer.protect_earth_leg(plan)
    retimer.set_return_sweep(sweep)
    members = neighbourhood(catalogue, plan, search_settings)
    search = RouteSearch(
        catalogue,
        members,
        search_settings,
        excluded=set(excluded or ()) - set(plan.asteroids),
        weights=weights,
    )
    improvement = improve_and_certify(
        plan,
        search,
        retimer,
        catalogue,
        scvx=scvx,
        max_attempts=max_attempts,
        time_budget_seconds=time_budget_seconds,
        refine=refine,
    )
    best = improvement.route
    if best is not None and best.total_collected_kg <= before + 1e-9:
        best = None
    return ReturnRetime(sweep, before, return_before, improvement, best)


def _fly(
    catalogue: AsteroidCatalogue,
    asteroid: int,
    mass_kg: float,
    departure: float,
    arrival: float,
    scvx: ScvxSettings,
    minimum_final_mass_kg: float,
) -> tuple[bool, float, str]:
    """One SCvx solve + verifier-model rollout; (certified, measured ΔV km/s, diagnostic)."""

    r0, v0 = asteroid_state(catalogue, asteroid, departure)
    rf, vf = earth_state(arrival)
    boundary = LegBoundary(
        departure,
        np.asarray(r0, dtype=np.float64).reshape(3),
        np.asarray(v0, dtype=np.float64).reshape(3),
        arrival,
        np.asarray(rf, dtype=np.float64).reshape(3),
        np.asarray(vf, dtype=np.float64).reshape(3),
        float(mass_kg),
        free_departure_vinf=False,
        free_arrival_vinf=True,
        minimum_final_mass=float(minimum_final_mass_kg),
    )
    try:
        solution = solve_leg(boundary, scvx)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return False, math.inf, f"solve error: {error}"
    if solution.status in ("infeasible", "failed", "timeout"):
        return False, math.inf, f"scvx {solution.status}"
    # same post-processing and acceptance as the route refinement (pipeline): clamp the nodal
    # thrust to T_max, roll the arcs out with the verifier model and accept within half the
    # official tolerances (position, velocity, RK4-vs-DOP853 agreement, path constraints)
    clamp_thrust(solution)
    certificate = certify_leg(solution)
    checks = (
        certificate.position_error_km / C.TOLERANCE_POSITION_KM,
        certificate.velocity_error_km_s / C.TOLERANCE_VELOCITY_KM_S,
        certificate.rk4_vs_dop853_km / C.TOLERANCE_POSITION_KM,
        max(0.0, certificate.maximum_thrust_n - C.THRUST_MAX_N) / C.THRUST_MAX_N
        + max(0.0, C.MIN_SUN_DISTANCE_AU - certificate.minimum_sun_distance_au),
    )
    if max(checks) > CERTIFICATION_TOLERANCE:
        return (
            False,
            math.inf,
            (
                f"rollout {certificate.position_error_km:.1f} km / "
                f"{certificate.velocity_error_km_s * 1e3:.2f} m/s"
            ),
        )
    if certificate.final_mass_kg < minimum_final_mass_kg - 1e-6:
        return False, math.inf, "final mass below the floor"
    measured = exhaust_velocity_km_s() * math.log(mass_kg / certificate.final_mass_kg)
    return True, float(measured), ""
