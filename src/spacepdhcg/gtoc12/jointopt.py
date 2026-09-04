"""Whole-itinerary joint re-optimisation of one ship (``gtoc12 joint-itinerary``).

The lattice re-timer (:mod:`retiming`) moves a ship's epochs on a 15-day grid through a DP
whose leg costs are the Lambert proxy, and SCvx certifies the result afterwards.  Here every
epoch of the ship's timeline - launch, each visit's arrival and departure (hence each leg's
TOF, each miner's deploy and collect epoch) and the Earth return - is a continuous variable of
*one* problem:

* objective: bonus-weighted collected mass (the mining-rate bookkeeping is exact and
  deterministic: ``10 kg/yr x (collect - deploy)``) plus ``margin_price`` kg per kg of spare
  final-mass margin, the exchange rate at which freed propellant is worth keeping (it pays for
  the extra asteroid the insertion step tries);
* constraints: the official window, one-year minimum stay, non-negative dwell (bounded camps),
  per-role TOF envelopes, the thrust-authority ratio, and the final mass ``>= dry + collected``
  (a schedule whose propellant does not close loads less ore, exactly as ``refine_route``
  sizes collected masses to fit);
* leg propellant: the calibrated pair-cost surrogate (zero-revolution Lambert x the
  ratio-/TOF-dependent inflation model x the pair's SCvx-calibrated residual) for legs SCvx has
  not flown at these epochs, and the *measured* ΔV for legs it has (every certified leg is
  memoised by ``(pair, departure, arrival)`` and the ship mass it flew at).

The outer optimiser is a deterministic pattern search on a shrinking mesh (45 -> 1 days):
single-epoch moves, whole-visit moves and phase shifts (launch..deploy k, collect k..return)
are evaluated on the surrogate and the steepest improving move is taken.  Whenever the
surrogate has found ``min_gain_kg`` more collected mass, the whole itinerary is re-flown with
SCvx (:func:`pipeline.refine_route`, the existing arc refiner) - every changed leg is
re-certified - and only a certified route with more collected mass replaces the incumbent
(monotone acceptance).  A leg that does not fly bans its pair's authority ratio and every
certified leg calibrates its pair, so the surrogate learns from each certification.  When the
joint schedule frees enough propellant and time, one more asteroid from the ship's co-moving
neighbourhood is inserted (deploy and collect positions enumerated, epochs pattern-searched,
certified the same way).  The verifier remains the final arbiter: the accepted route is the
one ``write_route_artifacts`` emits.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state, earth_state
from .low_thrust import ScvxSettings
from .pipeline import RefinedLeg, RefinedRoute, refine_route
from .retiming import Retimer, Visit, calibrate_from_route, visits_of
from .screening import (
    exhaust_velocity_km_s,
    lambert_hops,
    propellant_for_delta_v,
    thrust_authority_km_s,
)
from .search import EARTH_ID, PlannedLeg, RoutePlan

__all__ = [
    "Evaluation",
    "JointItinerary",
    "JointResult",
    "JointSettings",
    "MeasuredLeg",
    "optimise_ship",
    "route_from_summary",
    "weighted_mass",
]

FloatArray = NDArray[np.float64]
EpochKey = tuple[int, int, float, float]


@dataclass(frozen=True, slots=True)
class JointSettings:
    # pattern-search mesh schedule (days); each level runs until no move improves
    mesh_days: tuple[float, ...] = (45.0, 20.0, 8.0, 3.0, 1.0)
    max_moves_per_mesh: int = 60
    # objective kg per kg of spare final-mass margin (propellant freed but not spent)
    margin_price: float = 0.05
    # a certification is only requested (and accepted) for this much more collected mass
    min_gain_kg: float = 0.5
    max_certifications: int = 10
    # a memoised SCvx ΔV is reused while the leg's departure mass stays within this tolerance
    measured_mass_tolerance_kg: float = 60.0
    time_budget_seconds: float = 900.0
    # insertion of one more asteroid once the joint schedule has converged
    insert: bool = True
    insert_neighbours: int = 40
    insert_radius: float = 2.5  # co-moving neighbourhood radius in band units (beam: 1.5)
    insert_trials: int = 3
    insert_mesh_days: tuple[float, ...] = (20.0, 8.0, 3.0)
    # a (surrogate) insertion must beat the incumbent by this much before SCvx is spent on it
    insert_min_gain_kg: float = 5.0


@dataclass(frozen=True, slots=True)
class MeasuredLeg:
    delta_v_km_s: float  # verifier-model ΔV of the certified arc: v_e ln(m_before / m_after)
    mass_before_kg: float
    lambert_km_s: float


@dataclass(slots=True)
class Evaluation:
    plan: RoutePlan | None
    objective: float  # -inf when infeasible
    weighted_kg: float
    collected_kg: float
    spare_kg: float
    propellant_kg: float
    masses: list[float]  # departure mass of every flown leg
    measured_legs: int
    failure: str = ""

    @property
    def feasible(self) -> bool:
        return self.plan is not None


def weighted_mass(collected: dict[int, float], weights: dict[int, float] | None) -> float:
    if weights is None:
        return float(sum(collected.values()))
    return float(sum(weights.get(a, 1.0) * m for a, m in collected.items()))


def route_from_summary(summary: dict[str, Any]) -> RefinedRoute:
    """A lightweight :class:`RefinedRoute` of an archived ``route_summary.json``: the plan,
    the certified legs' masses and ΔV (no SCvx solutions).  Enough to warm-start the joint
    optimiser (measured leg costs, pair calibration) and to serve as the ``before`` route."""

    from .pipeline import plan_from_route_summary

    plan = plan_from_route_summary(summary)
    flown = [leg for leg in plan.legs if leg.role != "camp"]
    records = summary.get("legs") or []
    legs: list[RefinedLeg] = []
    for planned, record in zip(flown, records, strict=False):
        mass_before = float(record.get("mass_before") or math.nan)
        mass_after = float(record.get("mass_after") or math.nan)
        dv = record.get("delta_v_km_s")
        if dv is None and math.isfinite(mass_before) and math.isfinite(mass_after):
            dv = exhaust_velocity_km_s() * math.log(mass_before / mass_after)
        solution = None
        if dv is not None:
            solution = _ArchivedSolution(float(dv), float(record.get("propellant_kg") or 0.0))
        legs.append(
            RefinedLeg(
                planned,
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                solution,  # type: ignore[arg-type]
                None,
                bool(record.get("certified", True)),
                mass_before,
                mass_after,
            )
        )
    return RefinedRoute(
        plan=plan,
        legs=legs,
        collected_mass={int(k): float(v) for k, v in summary["collected_mass_kg"].items()},
        final_mass_kg=float(summary["final_mass_kg"]),
        certified=bool(summary.get("certified")),
        master_certified=bool(summary.get("master_certified", summary.get("certified"))),
        passes=int(summary.get("passes", 1)),
        wall_seconds=float(summary.get("wall_seconds", 0.0)),
        scheduler_telemetry={},
    )


@dataclass(frozen=True, slots=True)
class _ArchivedSolution:
    delta_v_km_s: float
    propellant_kg: float


class JointItinerary:
    """Continuous-epoch evaluator and pattern search for one ship's fixed visit order."""

    def __init__(
        self,
        catalogue: AsteroidCatalogue,
        retimer: Retimer,
        *,
        weights: dict[int, float] | None = None,
        settings: JointSettings | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.retimer = retimer
        self.weights = weights
        self.settings = settings or JointSettings()
        self.measured: dict[EpochKey, MeasuredLeg] = {}
        self._lambert: dict[EpochKey, float] = {}
        self.lambert_evaluations = 0
        self.evaluations = 0

    # -- learning from certifications ----------------------------------------------------

    @staticmethod
    def key(from_body: int, to_body: int, departure: float, arrival: float) -> EpochKey:
        return (int(from_body), int(to_body), round(float(departure), 5), round(float(arrival), 5))

    def learn(self, route: RefinedRoute) -> int:
        """Memoise the measured ΔV of every certified leg of ``route`` and calibrate the
        re-timer's pair inflations from it.  Returns the number of legs memoised."""

        count = 0
        for leg in route.legs:
            if not leg.certified:
                continue
            if not (math.isfinite(leg.mass_before) and math.isfinite(leg.mass_after_leg)):
                continue
            if leg.mass_after_leg <= 0.0 or leg.mass_after_leg > leg.mass_before + 1e-9:
                continue
            p = leg.planned
            dv = exhaust_velocity_km_s() * math.log(leg.mass_before / leg.mass_after_leg)
            self.measured[self.key(p.from_id, p.to_id, p.departure_epoch, p.arrival_epoch)] = (
                MeasuredLeg(float(dv), float(leg.mass_before), float(p.delta_v_proxy_km_s))
            )
            count += 1
        calibrate_from_route(self.retimer, route)
        return count

    # -- leg costs -----------------------------------------------------------------------

    def _state(self, body: int, epoch: float) -> tuple[FloatArray, FloatArray]:
        epochs = np.asarray([epoch], dtype=np.float64)
        if body == EARTH_ID:
            return earth_state(epochs)
        return asteroid_state(self.catalogue, np.asarray([body], dtype=np.int64), epochs)

    def lambert(self, from_body: int, to_body: int, departure: float, arrival: float) -> float:
        """Zero-revolution Lambert ΔV (km/s) of one leg at continuous epochs (memoised)."""

        key = self.key(from_body, to_body, departure, arrival)
        hit = self._lambert.get(key)
        if hit is not None:
            return hit
        tof = arrival - departure
        if tof <= 0.0:
            return math.inf
        r1, v1 = self._state(from_body, departure)
        r2, v2 = self._state(to_body, arrival)
        hop = lambert_hops(
            r1,
            v1,
            r2,
            v2,
            np.asarray([departure], dtype=np.float64),
            np.asarray([tof], dtype=np.float64),
            departure_allowance_km_s=C.MAX_VINF_EARTH_KM_S if from_body == EARTH_ID else 0.0,
            arrival_allowance_km_s=C.MAX_VINF_EARTH_KM_S if to_body == EARTH_ID else 0.0,
        )
        self.lambert_evaluations += 2
        dv = float(hop.total_delta_v[0]) if bool(hop.feasible[0]) else math.inf
        if not math.isfinite(dv):
            dv = math.inf
        if len(self._lambert) > 400_000:
            self._lambert.clear()
        self._lambert[key] = dv
        return dv

    def tof_limits(self, role: str) -> tuple[float, float]:
        s = self.retimer.settings
        lo, hi = s.earth_tof_days if role in ("earth_out", "earth_return") else s.hop_tof_days
        if role == "earth_out" and self.retimer.earth_out_tof_floor is not None:
            lo = max(lo, self.retimer.earth_out_tof_floor)
        return float(lo), float(max(hi, lo))

    def dwell_limit(self, visit: Visit) -> float:
        s = self.retimer.settings
        return s.long_camp_max_days if (visit.deploy and visit.collect) else s.camp_max_days

    # -- evaluation ----------------------------------------------------------------------

    def evaluate(self, visits: list[Visit], arrivals: FloatArray, departures: FloatArray):
        """Exact bookkeeping of one epoch vector for a fixed visit order."""

        self.evaluations += 1
        n = len(visits)
        s = self.retimer.settings
        arr = np.asarray(arrivals, dtype=np.float64)
        dep = np.asarray(departures, dtype=np.float64)
        if arr.shape != (n,) or dep.shape != (n,):
            raise ValueError("one arrival and one departure epoch per visit")
        fail = self._fail
        if arr[0] < C.MISSION_START_MJD - 1e-9:
            return fail("launch_before_window")
        if arr[-1] > C.MISSION_END_MJD - s.end_margin_days + 1e-9:
            return fail("return_after_window")
        if abs(dep[0] - arr[0]) > 1e-9 or abs(dep[-1] - arr[-1]) > 1e-9:
            return fail("earth_visits_have_no_dwell")
        for j in range(1, n - 1):
            dwell = dep[j] - arr[j]
            if dwell < -1e-9:
                return fail("negative_dwell")
            if dwell > self.dwell_limit(visits[j]) + 1e-9:
                return fail("dwell_too_long")
            pinned = visits[j].pinned_arrival
            if pinned is not None and abs(arr[j] - pinned) > 1e-6:
                return fail("pinned_arrival_moved")
        for j in range(n - 1):
            lo, hi = self.tof_limits(visits[j].role_out)
            tof = arr[j + 1] - dep[j]
            if tof < lo - 1e-9 or tof > hi + 1e-9:
                return fail("tof_outside_limits")
        # mining-rate bookkeeping (exact, deterministic)
        deploy: dict[int, float] = {}
        collect: dict[int, float] = {}
        foreign: dict[int, float] = {}
        for visit, a, d in zip(visits, arr, dep, strict=True):
            if visit.deploy:
                if visit.body in deploy:
                    return fail("double_deploy")
                deploy[visit.body] = float(a)
            if visit.collect:
                if visit.body in collect:
                    return fail("double_collect")
                collect[visit.body] = float(d)
                if visit.body not in deploy:
                    if visit.foreign_deploy_epoch is None:
                        return fail("collect_without_deploy")
                    foreign[visit.body] = visit.foreign_deploy_epoch
        minimum_stay = C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS
        collected: dict[int, float] = {}
        for asteroid, epoch in collect.items():
            deployed = deploy[asteroid] if asteroid in deploy else foreign[asteroid]
            stay = epoch - deployed
            if stay < minimum_stay - 1e-6:
                return fail("stay_too_short")
            collected[asteroid] = C.maximum_collected_mass(stay)
        # forward mass flow; a schedule whose propellant does not close loads less ore
        # (proportional scaling, as refine_route sizes the collected masses to fit)
        result = None
        for _round in range(4):
            result = self._forward(visits, arr, dep, deploy, collect, foreign, collected)
            if result.plan is None and result.failure == "mass_below_dry_plus_collected":
                total = sum(collected.values())
                deficit = -result.spare_kg
                if total <= 0.0 or deficit >= total:
                    return fail("mass_below_dry")
                scale = (total - 1.02 * deficit) / total
                collected = {a: m * scale for a, m in collected.items()}
                continue
            break
        assert result is not None
        return result

    def _fail(self, reason: str) -> Evaluation:
        return Evaluation(None, -math.inf, 0.0, 0.0, 0.0, 0.0, [], 0, reason)

    def _forward(
        self,
        visits: list[Visit],
        arr: FloatArray,
        dep: FloatArray,
        deploy: dict[int, float],
        collect: dict[int, float],
        foreign: dict[int, float],
        collected: dict[int, float],
    ) -> Evaluation:
        retimer = self.retimer
        tolerance = self.settings.measured_mass_tolerance_kg
        legs: list[PlannedLeg] = []
        masses: list[float] = []
        mass = float(retimer.search_settings.initial_mass)
        propellant_total = 0.0
        measured_count = 0
        for j in range(len(visits) - 1):
            visit, nxt = visits[j], visits[j + 1]
            if dep[j] > arr[j] + 1e-9:
                legs.append(
                    PlannedLeg(
                        visit.body, visit.body, float(arr[j]), float(dep[j]), 0.0, 1.0, "camp"
                    )
                )
            if visit.collect:
                mass += collected[visit.body]
            role = visit.role_out
            tof = float(arr[j + 1] - dep[j])
            lambert = self.lambert(visit.body, nxt.body, float(dep[j]), float(arr[j + 1]))
            key = self.key(visit.body, nxt.body, dep[j], arr[j + 1])
            measured = self.measured.get(key)
            if measured is not None and abs(measured.mass_before_kg - mass) <= tolerance:
                effective = measured.delta_v_km_s
                inflation = effective / lambert if math.isfinite(lambert) and lambert > 0 else 1.0
                proxy = lambert if math.isfinite(lambert) else effective
                measured_count += 1
            else:
                if not math.isfinite(lambert):
                    return self._fail("leg_infeasible")
                _flat, ratio_limit = retimer._limits(role, visit.body, nxt.body)
                if Retimer.authority_ratio(lambert, mass, tof) > ratio_limit:
                    return self._fail("leg_authority")
                inflation = float(
                    retimer.leg_inflation(role, visit.body, nxt.body, lambert, mass, tof)
                )
                effective = lambert * inflation
                proxy = lambert
            propellant = float(propellant_for_delta_v(mass, effective))
            masses.append(mass)
            mass -= propellant
            propellant_total += propellant
            legs.append(
                PlannedLeg(
                    visit.body, nxt.body, float(dep[j]), float(arr[j + 1]), proxy, inflation, role
                )
            )
            if nxt.deploy:
                mass -= C.MINER_MASS_KG
        spare = mass - (C.DRY_MASS_KG + sum(collected.values()))
        if spare < -1e-9:
            return Evaluation(
                None,
                -math.inf,
                0.0,
                0.0,
                spare,
                propellant_total,
                masses,
                measured_count,
                "mass_below_dry_plus_collected",
            )
        plan = RoutePlan(
            tuple(legs), dict(deploy), dict(collect), collected, propellant_total, mass, foreign
        )
        weighted = weighted_mass(collected, self.weights)
        objective = weighted + self.settings.margin_price * spare
        return Evaluation(
            plan,
            objective,
            weighted,
            float(sum(collected.values())),
            spare,
            propellant_total,
            masses,
            measured_count,
        )

    # -- pattern search ------------------------------------------------------------------

    @staticmethod
    def moves(n: int, delta: float) -> Iterator[dict[int, tuple[float, float]]]:
        """Deterministic move set on ``n`` visits: single epochs, whole visits, phase shifts."""

        for sign in (1.0, -1.0):
            d = sign * delta
            yield {0: (d, d)}  # launch
            yield {n - 1: (d, d)}  # Earth return
            for j in range(1, n - 1):
                yield {j: (d, 0.0)}  # arrival (deploy epoch / hop TOF)
                yield {j: (0.0, d)}  # departure (collect epoch / dwell)
                yield {j: (d, d)}  # the whole visit (dwell kept)
            for j in range(1, n - 1):
                # everything up to and including visit j's arrival (launch and deploy phase)
                prefix = {i: (d, d) for i in range(j)}
                prefix[j] = (d, 0.0)
                yield prefix
                # everything from visit j's departure onwards (collect phase and return)
                suffix = {i: (d, d) for i in range(j + 1, n)}
                suffix[j] = (0.0, d)
                yield suffix
            yield {i: (d, d) for i in range(n)}  # the whole itinerary

    def optimise_epochs(
        self,
        visits: list[Visit],
        arrivals: FloatArray,
        departures: FloatArray,
        *,
        mesh: tuple[float, ...] | None = None,
        max_moves: int | None = None,
        deadline: float = math.inf,
    ) -> tuple[FloatArray, FloatArray, Evaluation, int]:
        """Steepest-ascent pattern search over the epoch vector on a shrinking mesh; returns
        the best epochs, their evaluation and the number of moves taken.  Deterministic."""

        mesh = self.settings.mesh_days if mesh is None else mesh
        max_moves = self.settings.max_moves_per_mesh if max_moves is None else max_moves
        arr = np.array(arrivals, dtype=np.float64)
        dep = np.array(departures, dtype=np.float64)
        best = self.evaluate(visits, arr, dep)
        taken = 0
        if not best.feasible:
            return arr, dep, best, 0
        n = len(visits)
        for delta in mesh:
            moves_here = 0
            while moves_here < max_moves and time.perf_counter() < deadline:
                candidate: tuple[FloatArray, FloatArray, Evaluation] | None = None
                for shift in self.moves(n, delta):
                    a2 = arr.copy()
                    d2 = dep.copy()
                    for j, (da, dd) in shift.items():
                        a2[j] += da
                        d2[j] += dd
                    ev = self.evaluate(visits, a2, d2)
                    if not ev.feasible or ev.objective <= best.objective + 1e-9:
                        continue
                    if candidate is None or ev.objective > candidate[2].objective:
                        candidate = (a2, d2, ev)
                if candidate is None:
                    break
                arr, dep, best = candidate
                moves_here += 1
                taken += 1
        return arr, dep, best, taken

    # -- insertion of one more asteroid --------------------------------------------------

    def insertions(
        self,
        visits: list[Visit],
        arrivals: FloatArray,
        departures: FloatArray,
        candidates: list[int],
    ) -> list[tuple[list[Visit], FloatArray, FloatArray, Evaluation, int]]:
        """Every (asteroid, deploy position, collect position) insertion of a self-cleaning
        ship, evaluated on the surrogate with the new visits' epochs at the midpoints of the
        gaps they split; feasible ones sorted best first."""

        n = len(visits)
        present = {v.body for v in visits}
        camp = next((j for j in range(1, n - 1) if visits[j].deploy and visits[j].collect), None)
        if camp is None:
            return []
        # a new deploy visit between j and j+1 of the deploy phase (never splitting the
        # certified Earth-out leg: its TOF floor is protected), a new collect visit between j
        # and j+1 of the collect phase
        deploy_slots = list(range(1, camp))
        collect_slots = list(range(camp, n - 1))
        results = []
        best_per_asteroid: dict[int, float] = {}
        for asteroid in candidates:
            if asteroid in present:
                continue
            for i in deploy_slots:
                for k in collect_slots:
                    for seed in self._insertion_seeds(
                        visits, arrivals, departures, camp, i, k, asteroid
                    ):
                        ev = self.evaluate(*seed)
                        if not ev.feasible:
                            continue
                        results.append((*seed, ev, asteroid))
                        best_per_asteroid[asteroid] = max(
                            best_per_asteroid.get(asteroid, -math.inf), ev.objective
                        )
        results.sort(key=lambda item: (-item[3].objective, item[4]))
        return results

    @staticmethod
    def _insertion_seeds(
        visits: list[Visit],
        arrivals: FloatArray,
        departures: FloatArray,
        camp: int,
        i: int,
        k: int,
        asteroid: int,
    ) -> Iterator[tuple[list[Visit], FloatArray, FloatArray]]:
        """Seed epoch vectors for inserting ``asteroid`` after deploy visit ``i`` and after
        collect visit ``k``: (a) the split legs share their gap; (b) both new hops keep 3/4 of
        the split leg's TOF and the camp dwell lends the time (the deploy phase ends later, the
        collect phase starts earlier)."""

        def build(t_deploy: float, shift_deploy: float, t_collect: float, shift_collect: float):
            new_visits = list(visits)
            arr = list(arrivals)
            dep = list(departures)
            # collect side first: its indices are not moved by the deploy insertion
            last = new_visits[k + 1].body == EARTH_ID
            before = new_visits[k]
            new_visits[k] = Visit(
                before.body,
                before.deploy,
                before.collect,
                "collect_hop",
                before.foreign_deploy_epoch,
                before.pinned_arrival,
            )
            new_visits.insert(
                k + 1, Visit(asteroid, False, True, "earth_return" if last else "collect_hop")
            )
            arr.insert(k + 1, t_collect)
            dep.insert(k + 1, t_collect)
            # the collect phase up to visit k (from the camp departure) moves earlier
            for j in range(camp, k + 1):
                if j == camp:
                    dep[j] -= shift_collect
                else:
                    arr[j] -= shift_collect
                    dep[j] -= shift_collect
            # deploy side: the new visit and the deploy phase after it (up to the camp arrival)
            new_visits.insert(i + 1, Visit(asteroid, True, False, "deploy_hop"))
            arr.insert(i + 1, t_deploy)
            dep.insert(i + 1, t_deploy)
            for j in range(i + 2, camp + 2):  # camp index is now camp + 1
                if j == camp + 1:
                    arr[j] += shift_deploy
                else:
                    arr[j] += shift_deploy
                    dep[j] += shift_deploy
            return (
                new_visits,
                np.asarray(arr, dtype=np.float64),
                np.asarray(dep, dtype=np.float64),
            )

        gap_d = arrivals[i + 1] - departures[i]
        gap_c = arrivals[k + 1] - departures[k]
        # the camp dwell is the slack both phases borrow from: never below the one-year stay
        dwell = departures[camp] - arrivals[camp]
        slack = max(0.0, dwell - C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS - 5.0)
        yield build(departures[i] + 0.5 * gap_d, 0.0, departures[k] + 0.5 * gap_c, 0.0)
        for share_d, share_c in ((0.5, 0.5), (1.0, 0.0), (0.0, 1.0)):
            lend_d = min(0.5 * gap_d, share_d * slack)
            lend_c = min(0.5 * gap_c, share_c * slack)
            if lend_d + lend_c <= 1.0:
                continue
            yield build(
                departures[i] + 0.5 * (gap_d + lend_d),
                lend_d,
                departures[k] + 0.5 * (gap_c - lend_c),
                lend_c,
            )


# -- driver ----------------------------------------------------------------------------------


@dataclass(slots=True)
class JointResult:
    before: RefinedRoute
    route: RefinedRoute | None  # best certified improvement; None when nothing beat ``before``
    attempts: list[dict[str, Any]] = field(default_factory=list)
    inserted: int | None = None
    certifications: int = 0
    evaluations: int = 0
    lambert_evaluations: int = 0
    wall_seconds: float = 0.0
    baseline_error_kg: float = math.nan  # |surrogate at the warm start - certified collected|
    stopped: str = ""

    @property
    def before_kg(self) -> float:
        return float(self.before.total_collected_kg)

    @property
    def after_kg(self) -> float:
        return self.before_kg if self.route is None else float(self.route.total_collected_kg)

    @property
    def gain_kg(self) -> float:
        return self.after_kg - self.before_kg

    def summary(self) -> dict[str, Any]:
        return {
            "before_kg": self.before_kg,
            "after_kg": self.after_kg,
            "gain_kg": self.gain_kg,
            "asteroids_before": len(self.before.plan.asteroids),
            "asteroids_after": len(self.before.plan.asteroids)
            if self.route is None
            else len(self.route.plan.asteroids),
            "inserted": self.inserted,
            "certifications": self.certifications,
            "evaluations": self.evaluations,
            "lambert_evaluations": self.lambert_evaluations,
            "baseline_error_kg": self.baseline_error_kg,
            "wall_seconds": self.wall_seconds,
            "stopped": self.stopped,
            "legs_before": _leg_records(self.before),
            "legs_after": None if self.route is None else _leg_records(self.route),
            "attempts": self.attempts,
        }


def _leg_records(route: RefinedRoute) -> list[dict[str, Any]]:
    return [
        {
            "from": leg.planned.from_id,
            "to": leg.planned.to_id,
            "role": leg.planned.role,
            "t0": leg.planned.departure_epoch,
            "tf": leg.planned.arrival_epoch,
            "tof_days": leg.planned.tof_days,
            "propellant_kg": None
            if not (math.isfinite(leg.mass_before) and math.isfinite(leg.mass_after_leg))
            else leg.mass_before - leg.mass_after_leg,
        }
        for leg in route.legs
    ]


def _failing_ratio(refined: RefinedRoute) -> tuple[tuple[int, int], float] | None:
    failing = next((leg for leg in refined.legs if not leg.certified), None)
    if failing is None:
        return None
    p = failing.planned
    if p.delta_v_proxy_km_s <= 0.0 or not math.isfinite(failing.mass_before):
        return None
    ratio = p.delta_v_proxy_km_s / float(
        thrust_authority_km_s(failing.mass_before, p.tof_days, 1.0)
    )
    return (p.from_id, p.to_id), ratio


def optimise_ship(
    route: RefinedRoute,
    catalogue: AsteroidCatalogue,
    retimer: Retimer,
    *,
    weights: dict[int, float] | None = None,
    scvx: ScvxSettings | None = None,
    settings: JointSettings | None = None,
    search_settings: Any = None,
    excluded: set[int] | None = None,
    refine: Callable[[RoutePlan], RefinedRoute] | None = None,
) -> JointResult:
    """Jointly re-optimise every epoch of a certified stand-alone ship (see the module doc).

    ``route`` is the warm start (an archived or freshly refined certified route); ``retimer``
    supplies the calibrated pair-cost surrogate, the TOF/dwell envelopes and the bans;
    ``excluded`` asteroids (other ships') are never inserted; ``refine`` replaces
    :func:`pipeline.refine_route` (tests inject a proxy-trusting stand-in).
    """

    started = time.perf_counter()
    settings = settings or JointSettings()
    deadline = started + settings.time_budget_seconds
    refine = refine or (lambda plan: refine_route(plan, catalogue, scvx=scvx))
    joint = JointItinerary(catalogue, retimer, weights=weights, settings=settings)
    result = JointResult(before=route, route=None)
    if not route.certified:
        result.stopped = "warm start not certified"
        result.wall_seconds = time.perf_counter() - started
        return result
    plan = route.plan
    if plan.foreign_deploy_epochs:
        # a collector of another ship's miners: its collect epochs are tied to that ship
        result.stopped = "cooperative ship (foreign miners)"
        result.wall_seconds = time.perf_counter() - started
        return result
    joint.learn(route)
    retimer.protect_earth_leg(plan)
    visits, arr_list, dep_list = visits_of(plan)
    arr = np.asarray(arr_list, dtype=np.float64)
    dep = np.asarray(dep_list, dtype=np.float64)
    baseline = joint.evaluate(visits, arr, dep)
    incumbent = route
    incumbent_weighted = weighted_mass(route.collected_mass, weights)
    if baseline.feasible:
        result.baseline_error_kg = abs(baseline.collected_kg - route.total_collected_kg)
    else:
        result.baseline_error_kg = math.inf
        result.attempts.append({"stage": "baseline", "failure": baseline.failure})

    def certify(stage: str, ev: Evaluation) -> RefinedRoute | None:
        nonlocal incumbent, incumbent_weighted
        assert ev.plan is not None
        refined = refine(ev.plan)
        result.certifications += 1
        joint.learn(refined)
        record: dict[str, Any] = {
            "stage": stage,
            "surrogate_kg": ev.collected_kg,
            "surrogate_weighted_kg": ev.weighted_kg,
            "surrogate_spare_kg": ev.spare_kg,
            "measured_legs": ev.measured_legs,
            "certified": refined.certified,
            "certified_kg": refined.total_collected_kg,
            "final_mass_kg": refined.final_mass_kg,
            "passes": refined.passes,
            "scvx_seconds": refined.wall_seconds,
            "failures": refined.failures,
            "elapsed_seconds": round(time.perf_counter() - started, 1),
        }
        accepted = None
        if refined.certified:
            gain = weighted_mass(refined.collected_mass, weights) - incumbent_weighted
            if gain > settings.min_gain_kg:
                incumbent = refined
                incumbent_weighted += gain
                accepted = refined
                record["result"] = "accepted"
            else:
                record["result"] = "certified but no gain"
        else:
            failing = _failing_ratio(refined)
            if failing is not None:
                (a_body, b_body), ratio = failing
                retimer.ban(a_body, b_body, ratio)
                record["result"] = f"banned {a_body}->{b_body} at ratio {ratio:.3f}"
            else:
                record["result"] = "not certified (mass budget); pairs recalibrated"
        result.attempts.append(record)
        return accepted

    # -- joint schedule on the fixed visit order ------------------------------------------
    for delta in settings.mesh_days:
        repeats = 0
        while (
            time.perf_counter() < deadline and result.certifications < settings.max_certifications
        ):
            a2, d2, ev, moves = joint.optimise_epochs(
                visits, arr, dep, mesh=(delta,), deadline=deadline
            )
            if moves == 0 or not ev.feasible:
                break
            if ev.weighted_kg <= incumbent_weighted + settings.min_gain_kg:
                # only propellant was freed (or too little ore): keep the epochs as the search
                # base for the finer meshes and the insertion step, nothing to certify
                arr, dep = a2, d2
                break
            accepted = certify(f"mesh {delta:g} d", ev)
            if accepted is not None:
                arr, dep = a2, d2
                repeats = 0
                continue
            repeats += 1
            if repeats >= 2:
                break  # the surrogate keeps proposing what SCvx refuses at this mesh: refine it
        if time.perf_counter() >= deadline or result.certifications >= settings.max_certifications:
            result.stopped = (
                "time budget" if time.perf_counter() >= deadline else "certification budget"
            )
            break

    # -- one more asteroid ----------------------------------------------------------------
    if (
        settings.insert
        and not result.stopped
        and time.perf_counter() < deadline
        and result.certifications < settings.max_certifications
    ):
        from types import SimpleNamespace

        from .clusters import ClusterBands
        from .returnsweep import neighbourhood

        # a wider co-moving neighbourhood than the beam's (the joint schedule can afford a
        # longer detour than a lattice re-timing): ``insert_radius`` band units
        bands = getattr(search_settings, "cluster_bands", None) or ClusterBands.collect_window()
        bands = replace(bands, radius=settings.insert_radius)
        pool = neighbourhood(
            catalogue,
            incumbent.plan,
            SimpleNamespace(cluster_bands=bands),
            count=settings.insert_neighbours,
        )
        banned = set(excluded or ()) | set(incumbent.plan.asteroids)
        candidates = [int(a) for a in pool.tolist() if int(a) not in banned]
        options = joint.insertions(visits, arr, dep, candidates)
        result.attempts.append(
            {
                "stage": "insertion",
                "neighbourhood": int(pool.shape[0]),
                "candidates": len(candidates),
                "feasible_options": len(options),
                "best_surrogate_kg": None if not options else options[0][3].collected_kg,
                "incumbent_weighted_kg": incumbent_weighted,
            }
        )
        tried = 0
        seen: set[int] = set()
        for new_visits, a0, d0, _ev0, asteroid in options:
            if asteroid in seen:
                continue
            if tried >= settings.insert_trials or time.perf_counter() >= deadline:
                break
            if result.certifications >= settings.max_certifications:
                break
            a2, d2, ev, _moves = joint.optimise_epochs(
                new_visits, a0, d0, mesh=settings.insert_mesh_days, deadline=deadline
            )
            if (
                not ev.feasible
                or ev.weighted_kg <= incumbent_weighted + settings.insert_min_gain_kg
            ):
                continue
            seen.add(asteroid)
            tried += 1
            accepted = certify(f"insert {asteroid}", ev)
            if accepted is not None:
                result.inserted = asteroid
                visits, arr, dep = new_visits, a2, d2
                break

    if incumbent is not route:
        result.route = incumbent
    result.evaluations = joint.evaluations
    result.lambert_evaluations = joint.lambert_evaluations
    result.wall_seconds = time.perf_counter() - started
    return result
