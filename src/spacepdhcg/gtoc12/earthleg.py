"""Continuous Earth-leg optimisation for the GTOC12 first leg.

The beam used to pick its Earth -> asteroid legs from a discrete ``(launch epoch, TOF)`` grid
(30-day launches x 50-day TOFs).  A grid cell is rarely the cheapest leg to a target: the
launch v∞ direction and the transfer geometry change on a scale of days, and the archived
Earth legs cost a median 484 kg against the references' 447-466 kg.

This module optimises the leg continuously:

* decision variables are the launch epoch ``t0`` and the time of flight ``tof`` (the launch
  v∞ vector is decided by the Lambert surrogate - the departure excess velocity over the
  6 km/s free allowance is what costs propellant - and re-optimised freely inside the 6 km/s
  ball by the SCvx arc that certifies the leg);
* the surrogate objective is the beam's own first-level score, ``mined-mass horizon minus
  propellant_weight x priced propellant`` (so a later arrival is traded against a cheaper
  leg at the mining rate), on the zero-revolution Lambert ΔV inflated by the Earth-leg factor;
* the search is a deterministic multi-start compass (pattern) search with a shrinking step,
  bounded by the official launch constraints (launch inside the mission window, v∞ ≤ 6 km/s
  through the allowance, TOF inside the configured band, ΔV inside the thrust authority);
* the winner is certified by SCvx (``certify``), falling back to the next start if it fails.

Everything is pure numpy on the surrogate; SCvx is called once per candidate we decide to fly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state, earth_state
from .screening import (
    lambert_hops,
    propellant_for_delta_v,
    screen_earth_to_asteroids,
    thrust_authority_km_s,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EarthLegBounds:
    """Box the optimiser searches inside; the defaults are the official launch constraints."""

    launch_min: float = C.MISSION_START_MJD
    launch_max: float = C.MISSION_START_MJD + 3.0 * C.YEAR_DAYS  # the references launch over ~3 y
    tof_min: float = 200.0
    tof_max: float = 1000.0
    latest_arrival: float = C.MISSION_END_MJD - 2.0 * C.YEAR_DAYS  # leave time to mine and return

    def clip(self, launch: float, tof: float) -> tuple[float, float]:
        launch = min(max(launch, self.launch_min), self.launch_max)
        tof = min(max(tof, self.tof_min), self.tof_max)
        tof = min(tof, max(self.latest_arrival - launch, self.tof_min))
        return launch, tof


@dataclass(frozen=True, slots=True)
class EarthLegModel:
    """Surrogate pricing of an Earth leg (beam-consistent)."""

    initial_mass: float = C.MAX_INITIAL_MASS_KG
    inflation: float = 0.9  # reference Earth legs fly at 0.83-0.86x their Lambert ΔV
    authority_ratio: float = 0.95  # Lambert ΔV / full authority admitted
    propellant_weight: float = 0.15  # kg of score per kg of propellant (beam setting)
    horizon: float = C.MISSION_END_MJD - 2.0 * C.YEAR_DAYS  # mined mass counts up to here
    weight: float = 1.0  # target weight (cluster prior)

    def evaluate(
        self, catalogue: AsteroidCatalogue, target: int, launch: FloatArray, tof: FloatArray
    ) -> dict[str, FloatArray]:
        """Vectorised surrogate over paired ``launch``/``tof`` arrays (same shape)."""

        launch = np.atleast_1d(np.asarray(launch, dtype=np.float64))
        tof = np.atleast_1d(np.asarray(tof, dtype=np.float64))
        r_e, v_e = earth_state(launch)
        ids = np.full(launch.shape, target, dtype=np.int64)
        r_a, v_a = asteroid_state(catalogue, ids, launch + tof)
        hop = lambert_hops(
            r_e, v_e, r_a, v_a, launch, tof, departure_allowance_km_s=C.MAX_VINF_EARTH_KM_S
        )
        dv = hop.total_delta_v
        vinf = hop.departure_delta_v
        feasible = hop.feasible
        authority = self.authority_ratio * thrust_authority_km_s(self.initial_mass, tof, 1.0)
        ok = feasible & np.isfinite(dv) & (dv <= authority)
        propellant = propellant_for_delta_v(self.initial_mass, dv * self.inflation)
        mined = C.MINING_RATE_KG_PER_YEAR * np.maximum(self.horizon - launch - tof, 0.0)
        mined /= C.YEAR_DAYS
        score = np.where(ok, self.weight * mined - self.propellant_weight * propellant, -np.inf)
        return {
            "delta_v": dv,
            "departure_excess": vinf,  # departure ΔV charged above the 6 km/s allowance
            "propellant": propellant,
            "feasible": ok,
            "score": score,
            "ratio": dv / thrust_authority_km_s(self.initial_mass, tof, 1.0),
        }


@dataclass(frozen=True, slots=True)
class OptimisedEarthLeg:
    target: int
    launch_epoch: float
    tof_days: float
    lambert_dv_km_s: float
    surrogate_propellant_kg: float
    score: float
    authority_ratio: float
    evaluations: int
    start_launch: float
    start_tof: float

    @property
    def arrival_epoch(self) -> float:
        return self.launch_epoch + self.tof_days

    def summary(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "launch_epoch": self.launch_epoch,
            "tof_days": self.tof_days,
            "lambert_dv_km_s": self.lambert_dv_km_s,
            "surrogate_propellant_kg": self.surrogate_propellant_kg,
            "score": self.score,
            "authority_ratio": self.authority_ratio,
            "evaluations": self.evaluations,
            "start": [self.start_launch, self.start_tof],
        }


def compass_search(
    catalogue: AsteroidCatalogue,
    target: int,
    start_launch: float,
    start_tof: float,
    *,
    model: EarthLegModel,
    bounds: EarthLegBounds,
    initial_step_days: float = 15.0,
    final_step_days: float = 1.0,
    launch_radius_days: float | None = None,
    max_evaluations: int = 400,
) -> OptimisedEarthLeg | None:
    """Deterministic compass search of ``(launch, tof)`` from one start on the surrogate.

    Each round evaluates the four compass moves (±step in launch, ±step in TOF) plus the two
    diagonal launch/TOF trades that keep the arrival fixed; the best improving move is taken,
    otherwise the step halves until ``final_step_days``.  ``launch_radius_days`` confines the
    launch to a window around the start so that distinct grid starts stay distinct legs.
    Epochs are snapped to 0.1 d and TOFs to whole days so the certified plan is reproducible.
    """

    lo_launch = bounds.launch_min
    hi_launch = bounds.launch_max
    if launch_radius_days is not None:
        lo_launch = max(lo_launch, start_launch - launch_radius_days)
        hi_launch = min(hi_launch, start_launch + launch_radius_days)
    local = EarthLegBounds(
        lo_launch, hi_launch, bounds.tof_min, bounds.tof_max, bounds.latest_arrival
    )

    def snap(launch: float, tof: float) -> tuple[float, float]:
        launch, tof = local.clip(launch, tof)
        return round(launch, 1), float(round(tof))

    current = snap(start_launch, start_tof)
    evaluations = 0
    cache: dict[tuple[float, float], dict[str, float]] = {}

    def evaluate(points: list[tuple[float, float]]) -> list[dict[str, float]]:
        nonlocal evaluations
        fresh = [p for p in dict.fromkeys(points) if p not in cache]
        if fresh:
            out = model.evaluate(
                catalogue,
                target,
                np.asarray([p[0] for p in fresh]),
                np.asarray([p[1] for p in fresh]),
            )
            evaluations += len(fresh)
            for k, p in enumerate(fresh):
                cache[p] = {key: float(value[k]) for key, value in out.items()}
        return [cache[p] for p in points]

    best = evaluate([current])[0]
    if not np.isfinite(best["score"]):
        return None
    step = initial_step_days
    while step >= final_step_days and evaluations < max_evaluations:
        launch, tof = current
        moves = [
            snap(launch + step, tof),
            snap(launch - step, tof),
            snap(launch, tof + step),
            snap(launch, tof - step),
            snap(launch + step, tof - step),  # fixed arrival, later launch
            snap(launch - step, tof + step),  # fixed arrival, earlier launch
        ]
        moves = [m for m in moves if m != current]
        results = evaluate(moves)
        scores = [r["score"] for r in results]
        k = int(np.argmax(scores)) if scores else -1
        if k >= 0 and scores[k] > best["score"] + 1e-9:
            current, best = moves[k], results[k]
        else:
            step /= 2.0
    return OptimisedEarthLeg(
        target,
        current[0],
        current[1],
        best["delta_v"],
        best["propellant"],
        best["score"],
        best["ratio"],
        evaluations,
        start_launch,
        start_tof,
    )


def optimise_earth_leg(
    catalogue: AsteroidCatalogue,
    target: int,
    *,
    model: EarthLegModel | None = None,
    bounds: EarthLegBounds | None = None,
    launch_grid_days: float = 30.0,
    tof_grid_days: float = 50.0,
    starts: int = 3,
    launch_radius_days: float | None = None,
    max_evaluations: int = 400,
) -> list[OptimisedEarthLeg]:
    """Multi-start continuous optimisation of the Earth leg to ``target``.

    A coarse grid over the bounds ranks the starts (best ``starts`` cells with distinct launch
    epochs); each is refined by :func:`compass_search`.  Returns the refined legs sorted by
    surrogate score (best first), deduplicated on the snapped ``(launch, tof)``.
    """

    model = model or EarthLegModel()
    bounds = bounds or EarthLegBounds()
    launches = np.arange(bounds.launch_min, bounds.launch_max + 1e-9, launch_grid_days)
    tofs = np.arange(bounds.tof_min, bounds.tof_max + 1e-9, tof_grid_days)
    grid = screen_earth_to_asteroids(catalogue, np.asarray([target]), launches, tofs)
    dv = np.where(grid["feasible"][0], grid["total_delta_v"][0], np.inf)
    authority = model.authority_ratio * thrust_authority_km_s(model.initial_mass, tofs, 1.0)
    ok = np.isfinite(dv) & (dv <= authority[None, :])
    ok &= (launches[:, None] + tofs[None, :]) <= bounds.latest_arrival
    propellant = propellant_for_delta_v(model.initial_mass, dv * model.inflation)
    mined = C.MINING_RATE_KG_PER_YEAR * np.maximum(
        model.horizon - launches[:, None] - tofs[None, :], 0.0
    )
    mined /= C.YEAR_DAYS
    score = np.where(ok, model.weight * mined - model.propellant_weight * propellant, -np.inf)
    best_tof = np.argmax(score, axis=1)
    launch_score = np.take_along_axis(score, best_tof[:, None], axis=1)[:, 0]
    order = np.argsort(-launch_score, kind="stable")
    results: list[OptimisedEarthLeg] = []
    seen: set[tuple[float, float]] = set()
    for e_index in order[:starts]:
        if not np.isfinite(launch_score[e_index]):
            break
        leg = compass_search(
            catalogue,
            target,
            float(launches[e_index]),
            float(tofs[best_tof[e_index]]),
            model=model,
            bounds=bounds,
            launch_radius_days=launch_radius_days,
            max_evaluations=max_evaluations,
        )
        if leg is None:
            continue
        key = (leg.launch_epoch, leg.tof_days)
        if key in seen:
            continue
        seen.add(key)
        results.append(leg)
    results.sort(key=lambda leg: (-leg.score, leg.launch_epoch, leg.tof_days))
    return results


CertifyFn = Callable[[AsteroidCatalogue, int, float, float, float, Any], Any]


@dataclass(slots=True)
class ScvxRefinement:
    """Outcome of :func:`refine_leg_scvx`: the cheapest certified leg found and its trace."""

    leg: Any  # search.EarthLeg (the best certified leg; the start leg if nothing improved)
    start_propellant_kg: float
    evaluations: int  # SCvx calls made (cache hits excluded)
    trace: list[dict[str, float]]

    @property
    def saved_kg(self) -> float:
        return self.start_propellant_kg - float(self.leg.propellant_kg)

    def summary(self) -> dict[str, Any]:
        return {
            "target": int(self.leg.target),
            "launch_epoch": float(self.leg.launch_epoch),
            "tof_days": float(self.leg.tof_days),
            "propellant_kg": float(self.leg.propellant_kg),
            "start_propellant_kg": self.start_propellant_kg,
            "saved_kg": self.saved_kg,
            "scvx_evaluations": self.evaluations,
            "trace": self.trace,
        }


def refine_leg_scvx(
    catalogue: AsteroidCatalogue,
    start_leg: Any,
    certify: CertifyFn,
    scvx: Any,
    *,
    bounds: EarthLegBounds,
    launch_radius_days: float = 15.0,
    step_days: float = 15.0,
    final_step_days: float = 4.0,
    time_weight_kg_per_day: float = 0.22,
    max_evaluations: int = 8,
    cache: dict[tuple[int, float, float], Any] | None = None,
    model: EarthLegModel | None = None,
) -> ScvxRefinement:
    """Compass search of ``(launch, tof)`` around a certified leg with SCvx as the evaluator.

    The zero-revolution Lambert surrogate is not a usable fine-scale objective for Earth legs:
    on the 181 archived certified Earth legs the measured/Lambert ratio scatters 0.86-1.51 at
    the same authority ratio (the low-thrust arc exploits the 6 km/s launch v∞ in directions
    Lambert cannot), so the surrogate steers towards *shorter* legs that then fail or cost more.
    Here every evaluation is a real SCvx certification (``certify`` returns a ``search.EarthLeg``
    or ``None``); the objective is the measured propellant plus ``time_weight_kg_per_day`` x the
    arrival delay relative to the start (later arrival delays the whole deploy chain: about eight
    miners x 10 kg/yr = 0.22 kg per day).  Moves: ±step in launch (inside ``launch_radius_days``
    of the start, so distinct grid legs remain distinct), ±2 x step in TOF (the leg is nearly
    thrust-saturated, so TOF is the strong lever) and the fixed-arrival trade.  The step halves
    when no move improves; the search stops at ``final_step_days`` or ``max_evaluations`` SCvx
    calls.  Deterministic: moves are evaluated in a fixed order, ties keep the incumbent.
    """

    cache = {} if cache is None else cache
    model = model or EarthLegModel()
    target = int(start_leg.target)
    lo_launch = max(bounds.launch_min, start_leg.launch_epoch - launch_radius_days)
    hi_launch = min(bounds.launch_max, start_leg.launch_epoch + launch_radius_days)
    local = EarthLegBounds(
        lo_launch, hi_launch, bounds.tof_min, bounds.tof_max, bounds.latest_arrival
    )

    def snap(launch: float, tof: float) -> tuple[float, float]:
        launch, tof = local.clip(launch, tof)
        return round(launch, 1), float(round(tof))

    start_arrival = start_leg.arrival_epoch

    def objective(leg: Any) -> float:
        if leg is None:
            return float("inf")
        return float(leg.propellant_kg) + time_weight_kg_per_day * (
            leg.arrival_epoch - start_arrival
        )

    current = snap(start_leg.launch_epoch, start_leg.tof_days)
    cache.setdefault((target, current[0], current[1]), start_leg)
    best_leg = start_leg
    best = objective(start_leg)
    evaluations = 0
    trace: list[dict[str, float]] = []

    def probe(point: tuple[float, float]) -> float:
        nonlocal evaluations
        key = (target, point[0], point[1])
        if key not in cache:
            # the leg record keeps the zero-revolution Lambert ΔV of what was flown
            lambert = float(
                model.evaluate(catalogue, target, np.asarray([point[0]]), np.asarray([point[1]]))[
                    "delta_v"
                ][0]
            )
            if not np.isfinite(lambert):
                lambert = float(start_leg.delta_v_km_s)
            cache[key] = certify(catalogue, target, point[0], point[1], lambert, scvx)
            evaluations += 1
        leg = cache[key]
        value = objective(leg)
        trace.append(
            {
                "launch_epoch": point[0],
                "tof_days": point[1],
                "propellant_kg": float("nan") if leg is None else float(leg.propellant_kg),
                "objective": value,
            }
        )
        return value

    step = step_days
    while step >= final_step_days and evaluations < max_evaluations:
        launch, tof = current
        deltas = [
            (0.0, 2.0 * step),
            (0.0, -2.0 * step),
            (step, 0.0),
            (-step, 0.0),
            (step, -step),
            (-step, step),
        ]
        improved = False
        for d_launch, d_tof in deltas:
            move = snap(launch + d_launch, tof + d_tof)
            if move == current or evaluations >= max_evaluations:
                continue
            value = probe(move)
            if value < best - 1e-9:
                best, best_leg, current, improved = value, cache[(target, *move)], move, True
                # line search: keep stepping in the improving direction while it pays
                while evaluations < max_evaluations:
                    nxt = snap(current[0] + d_launch, current[1] + d_tof)
                    if nxt == current:
                        break
                    value = probe(nxt)
                    if value >= best - 1e-9:
                        break
                    best, best_leg, current = value, cache[(target, *nxt)], nxt
                break  # re-evaluate the pattern around the new incumbent
        if not improved:
            step /= 2.0
    return ScvxRefinement(best_leg, float(start_leg.propellant_kg), evaluations, trace)
