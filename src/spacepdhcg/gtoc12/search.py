"""Deterministic beam search for self-cleaning GTOC12 mining-ship routes.

A route is ``Earth -> A1 -> ... -> Ak (deploy at each) -> camp at Ak -> Ak -> ... -> A1 (collect at
each) -> Earth``.  Deploy hops are expanded forwards from a launch-epoch grid with Lambert
rendezvous costs; the collection tour is scheduled *backwards* from the end of the window so every
miner works as long as possible.  Costs are impulsive proxies inflated for finite thrust; the
low-thrust refinement (``pipeline``) replaces them with certified arcs.

Everything is deterministic: candidate order is fixed, ties break on asteroid ID, and the only use
of ``seed`` is to shuffle nothing unless ``randomise`` is requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state, earth_state
from .screening import (
    element_distance_proxy,
    lambert_hops,
    propellant_for_delta_v,
    screen_asteroid_hops,
    screen_earth_to_asteroids,
    thrust_authority_km_s,
)

FloatArray = NDArray[np.float64]
EARTH_ID = 0


@dataclass(frozen=True, slots=True)
class SearchSettings:
    beam_width: int = 24
    max_deploys: int = 4
    min_deploys: int = 1
    launch_epochs: tuple[float, ...] = tuple(C.MISSION_START_MJD + np.arange(0.0, 731.0, 30.0))
    earth_leg_tofs: tuple[float, ...] = tuple(np.arange(300.0, 901.0, 50.0))
    hop_tofs: tuple[float, ...] = (60.0, 90.0, 120.0, 150.0, 180.0, 240.0, 300.0)
    neighbours: int = 40
    earth_leg_inflation: float = 1.6
    hop_inflation: float = 1.25
    duty: float = 0.85
    end_margin_days: float = 2.0
    return_window_days: float = 240.0
    collect_wait_window_days: float = 420.0
    schedule_step_days: float = 15.0
    wait_penalty: float = 0.5  # kg propellant-equivalent per kg of mining mass forgone
    initial_mass: float = C.MAX_INITIAL_MASS_KG
    seed: int = 0
    randomise: bool = False


@dataclass(frozen=True, slots=True)
class PlannedLeg:
    from_id: int  # 0 = Earth
    to_id: int
    departure_epoch: float
    arrival_epoch: float
    delta_v_proxy_km_s: float
    inflation: float
    role: str  # "earth_out" | "deploy_hop" | "collect_hop" | "earth_return" | "camp"

    @property
    def tof_days(self) -> float:
        return self.arrival_epoch - self.departure_epoch


@dataclass(frozen=True, slots=True)
class RoutePlan:
    legs: tuple[PlannedLeg, ...]
    deploy_epochs: dict[int, float]
    collect_epochs: dict[int, float]
    collected_mass: dict[int, float]
    propellant_proxy_kg: float
    final_mass_proxy_kg: float

    @property
    def asteroids(self) -> tuple[int, ...]:
        return tuple(self.deploy_epochs)

    @property
    def total_collected_kg(self) -> float:
        return sum(self.collected_mass.values())

    @property
    def feasible(self) -> bool:
        return self.final_mass_proxy_kg >= C.DRY_MASS_KG + self.total_collected_kg

    def summary(self) -> dict[str, object]:
        return {
            "asteroids": list(self.asteroids),
            "launch_epoch": self.legs[0].departure_epoch,
            "earth_return_epoch": self.legs[-1].arrival_epoch,
            "deploy_epochs": dict(self.deploy_epochs),
            "collect_epochs": dict(self.collect_epochs),
            "collected_mass_kg": dict(self.collected_mass),
            "total_collected_kg": self.total_collected_kg,
            "propellant_proxy_kg": self.propellant_proxy_kg,
            "final_mass_proxy_kg": self.final_mass_proxy_kg,
            "feasible_proxy": self.feasible,
            "legs": [
                {
                    "from": leg.from_id,
                    "to": leg.to_id,
                    "t0": leg.departure_epoch,
                    "tf": leg.arrival_epoch,
                    "dv_proxy_km_s": leg.delta_v_proxy_km_s,
                    "role": leg.role,
                }
                for leg in self.legs
            ],
        }


@dataclass(slots=True)
class _Partial:
    legs: list[PlannedLeg]
    location: int
    epoch: float
    mass: float
    deployed: list[tuple[int, float]]
    score: float = 0.0


@dataclass(slots=True)
class SearchResult:
    best: RoutePlan | None
    candidates: list[RoutePlan]
    expansions: int
    lambert_evaluations: int
    wall_seconds: float
    failures: list[dict[str, object]] = field(default_factory=list)


class RouteSearch:
    def __init__(
        self,
        catalogue: AsteroidCatalogue,
        asteroid_ids: NDArray[np.int64],
        settings: SearchSettings | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.ids = np.asarray(sorted(int(item) for item in asteroid_ids), dtype=np.int64)
        self.settings = settings or SearchSettings()
        self.lambert_evaluations = 0
        self._neighbours: dict[int, NDArray[np.int64]] = {}
        self._hop_cache: dict[tuple[int, float], dict[str, FloatArray]] = {}

    # -- proxies --

    def _propellant(self, mass: float, delta_v: float, inflation: float) -> float:
        return float(propellant_for_delta_v(mass, delta_v * inflation))

    def _feasible(self, mass: float, delta_v: float, tof: float, inflation: float) -> bool:
        return delta_v * inflation <= float(thrust_authority_km_s(mass, tof, self.settings.duty))

    def neighbours(self, asteroid_id: int) -> NDArray[np.int64]:
        if asteroid_id not in self._neighbours:
            proxy = element_distance_proxy(self.catalogue, asteroid_id, self.ids)
            order = np.lexsort((self.ids, proxy))
            self._neighbours[asteroid_id] = self.ids[order[: self.settings.neighbours + 1]]
        return self._neighbours[asteroid_id]

    def hops_from(self, asteroid_id: int, epoch: float) -> dict[str, FloatArray]:
        key = (asteroid_id, round(epoch, 6))
        if key not in self._hop_cache:
            targets = np.asarray(
                [item for item in self.neighbours(asteroid_id) if item != asteroid_id],
                dtype=np.int64,
            )
            result = screen_asteroid_hops(
                self.catalogue, asteroid_id, targets, epoch, np.asarray(self.settings.hop_tofs)
            )
            self.lambert_evaluations += 2 * targets.shape[0] * len(self.settings.hop_tofs)
            self._hop_cache[key] = result
        return self._hop_cache[key]

    # -- search --

    def run(self) -> SearchResult:
        import time

        started = time.perf_counter()
        s = self.settings
        epochs = np.asarray(s.launch_epochs)
        tofs = np.asarray(s.earth_leg_tofs)
        grid = screen_earth_to_asteroids(self.catalogue, self.ids, epochs, tofs)
        self.lambert_evaluations += 2 * grid["total_delta_v"].size
        beam: list[_Partial] = []
        # first deploy: Earth -> A1
        for a_index, asteroid in enumerate(self.ids):
            for e_index, launch in enumerate(epochs):
                for t_index, tof in enumerate(tofs):
                    if not grid["feasible"][a_index, e_index, t_index]:
                        continue
                    dv = float(grid["total_delta_v"][a_index, e_index, t_index])
                    if not self._feasible(s.initial_mass, dv, float(tof), s.earth_leg_inflation):
                        continue
                    propellant = self._propellant(s.initial_mass, dv, s.earth_leg_inflation)
                    arrival = float(launch + tof)
                    leg = PlannedLeg(
                        EARTH_ID,
                        int(asteroid),
                        float(launch),
                        arrival,
                        dv,
                        s.earth_leg_inflation,
                        "earth_out",
                    )
                    mass = s.initial_mass - propellant - C.MINER_MASS_KG
                    beam.append(
                        _Partial([leg], int(asteroid), arrival, mass, [(int(asteroid), arrival)])
                    )
        expansions = 0
        completed: list[RoutePlan] = []
        failures: list[dict[str, object]] = []
        for partial in self._select(beam):
            plan = self._complete(partial)
            if plan is not None:
                completed.append(plan)
        current = self._select(beam)
        for _depth in range(1, s.max_deploys):
            next_beam: list[_Partial] = []
            for partial in current:
                expansions += 1
                hops = self.hops_from(partial.location, partial.epoch)
                visited = {item for item, _ in partial.deployed}
                for t_index, target in enumerate(hops["target_ids"]):
                    target = int(target)
                    if target in visited:
                        continue
                    for f_index, tof in enumerate(hops["tofs_days"]):
                        if not hops["feasible"][t_index, f_index]:
                            continue
                        dv = float(hops["total_delta_v"][t_index, f_index])
                        if not self._feasible(partial.mass, dv, float(tof), s.hop_inflation):
                            continue
                        propellant = self._propellant(partial.mass, dv, s.hop_inflation)
                        arrival = partial.epoch + float(tof)
                        if arrival > C.MISSION_END_MJD - 3.0 * C.YEAR_DAYS:
                            continue
                        leg = PlannedLeg(
                            partial.location,
                            target,
                            partial.epoch,
                            arrival,
                            dv,
                            s.hop_inflation,
                            "deploy_hop",
                        )
                        next_beam.append(
                            _Partial(
                                [*partial.legs, leg],
                                target,
                                arrival,
                                partial.mass - propellant - C.MINER_MASS_KG,
                                [*partial.deployed, (target, arrival)],
                            )
                        )
            current = self._select(next_beam)
            for partial in current:
                if len(partial.deployed) >= s.min_deploys:
                    plan = self._complete(partial)
                    if plan is not None:
                        completed.append(plan)
                    else:
                        failures.append(
                            {
                                "asteroids": [item for item, _ in partial.deployed],
                                "reason": "no feasible collection tour",
                            }
                        )
            if not current:
                break
        completed.sort(
            key=lambda item: (-item.total_collected_kg, item.propellant_proxy_kg, item.asteroids)
        )
        best = next((item for item in completed if item.feasible), None)
        return SearchResult(
            best,
            completed,
            expansions,
            self.lambert_evaluations,
            time.perf_counter() - started,
            failures,
        )

    def _select(self, partials: list[_Partial]) -> list[_Partial]:
        """Stable top-``beam_width`` by heuristic: expected mined mass minus propellant penalty."""

        end = C.MISSION_END_MJD - 2.0 * C.YEAR_DAYS  # rough collection horizon
        for partial in partials:
            mined = sum(
                C.maximum_collected_mass(max(end - deploy_epoch, 0.0))
                for _, deploy_epoch in partial.deployed
            )
            spent = self.settings.initial_mass - partial.mass
            partial.score = mined - 0.15 * spent
        ordered = sorted(
            partials,
            key=lambda item: (-item.score, item.epoch, tuple(a for a, _ in item.deployed)),
        )
        return ordered[: self.settings.beam_width]

    def _return_options(self, asteroid: int, end: float) -> list[tuple[float, float, float]]:
        """Candidate ``(dv, departure, tof)`` Earth returns arriving inside the final window."""

        s = self.settings
        arrivals = np.arange(end - s.return_window_days, end + 1e-9, s.schedule_step_days)
        tofs = np.asarray(s.earth_leg_tofs)
        a_idx, t_idx = np.meshgrid(
            np.arange(arrivals.shape[0]), np.arange(tofs.shape[0]), indexing="ij"
        )
        a_idx, t_idx = a_idx.ravel(), t_idx.ravel()
        departures = arrivals[a_idx] - tofs[t_idx]
        r_s, v_s = asteroid_state(
            self.catalogue, np.full(departures.shape[0], asteroid), departures
        )
        r_e, v_e = earth_state(arrivals[a_idx])
        hop = lambert_hops(
            r_s,
            v_s,
            r_e,
            v_e,
            departures,
            tofs[t_idx],
            arrival_allowance_km_s=C.MAX_VINF_EARTH_KM_S,
        )
        self.lambert_evaluations += 2 * departures.shape[0]
        options = [
            (float(hop.total_delta_v[k]), float(departures[k]), float(tofs[t_idx[k]]))
            for k in range(departures.shape[0])
            if hop.feasible[k] and np.isfinite(hop.total_delta_v[k])
        ]
        options.sort(key=lambda item: (item[0], -item[1]))
        return options

    def _collect_hop_options(
        self, source: int, target: int, latest_arrival: float
    ) -> list[tuple[float, float, float]]:
        """Candidate ``(dv, departure, tof)`` hops ``source -> target`` arriving by the deadline.

        The ship may arrive early and camp at ``target`` until the scheduled collection, and it
        collects at ``source`` when it departs; later departures therefore mine more.
        """

        s = self.settings
        tofs = np.asarray(s.hop_tofs)
        waits = np.arange(0.0, s.collect_wait_window_days + 1e-9, s.schedule_step_days)
        w_idx, t_idx = np.meshgrid(
            np.arange(waits.shape[0]), np.arange(tofs.shape[0]), indexing="ij"
        )
        w_idx, t_idx = w_idx.ravel(), t_idx.ravel()
        arrivals = latest_arrival - waits[w_idx]
        departures = arrivals - tofs[t_idx]
        r_s, v_s = asteroid_state(self.catalogue, np.full(departures.shape[0], source), departures)
        r_t, v_t = asteroid_state(self.catalogue, np.full(departures.shape[0], target), arrivals)
        hop = lambert_hops(r_s, v_s, r_t, v_t, departures, tofs[t_idx])
        self.lambert_evaluations += 2 * departures.shape[0]
        options = [
            (float(hop.total_delta_v[k]), float(departures[k]), float(tofs[t_idx[k]]))
            for k in range(departures.shape[0])
            if hop.feasible[k] and np.isfinite(hop.total_delta_v[k])
        ]
        return options

    def _complete(self, partial: _Partial) -> RoutePlan | None:
        """Schedule the reverse collection tour and Earth return backwards from the window end."""

        s = self.settings
        order = [
            asteroid for asteroid, _ in reversed(partial.deployed)
        ]  # collect last deployed first
        deploy = dict(partial.deployed)
        end = C.MISSION_END_MJD - s.end_margin_days
        first = order[-1]  # the last asteroid collected before returning to Earth
        # Mass proxy after the deploy phase; collected mass is added as the tour is scheduled
        # backwards so the feasibility test sees roughly the right (heavier) ship.
        mass_guess = partial.mass + sum(
            C.maximum_collected_mass(max(end - deploy_epoch, 0.0))
            for _, deploy_epoch in partial.deployed
        )
        best_return = None
        for dv, departure, tof in self._return_options(first, end):
            if self._feasible(mass_guess, dv, tof, s.earth_leg_inflation):
                best_return = (dv, departure, tof)
                break
        if best_return is None:
            return None
        legs_backward: list[PlannedLeg] = [
            PlannedLeg(
                first,
                EARTH_ID,
                best_return[1],
                best_return[1] + best_return[2],
                best_return[0],
                s.earth_leg_inflation,
                "earth_return",
            )
        ]
        collect: dict[int, float] = {first: best_return[1]}
        epoch = best_return[1]  # collection (=departure) epoch at the current asteroid
        location = first
        for previous in reversed(order[:-1]):
            options = self._collect_hop_options(previous, location, epoch)
            best_hop = None
            best_cost = np.inf
            for dv, departure, tof in options:
                if not self._feasible(mass_guess, dv, tof, s.hop_inflation):
                    continue
                # propellant proxy plus the mining mass lost by collecting ``previous`` earlier
                lost = (
                    C.maximum_collected_mass(epoch - departure - tof)
                    if epoch - departure - tof > 0
                    else 0.0
                )
                cost = self._propellant(mass_guess, dv, s.hop_inflation) + s.wait_penalty * lost
                if cost < best_cost - 1e-12 or (
                    abs(cost - best_cost) <= 1e-12
                    and best_hop is not None
                    and departure > best_hop[1]
                ):
                    best_cost = cost
                    best_hop = (dv, departure, tof)
            if best_hop is None:
                return None
            arrival = best_hop[1] + best_hop[2]
            if arrival < epoch:
                legs_backward.append(
                    PlannedLeg(location, location, arrival, epoch, 0.0, 1.0, "camp")
                )
            legs_backward.append(
                PlannedLeg(
                    previous,
                    location,
                    best_hop[1],
                    arrival,
                    best_hop[0],
                    s.hop_inflation,
                    "collect_hop",
                )
            )
            collect[previous] = best_hop[1]
            epoch = best_hop[1]
            location = previous
        # camp at the last deployed asteroid between its deploy and its collection
        camp_start = partial.epoch
        camp_end = epoch
        if camp_end - camp_start < 0.0:
            return None
        for asteroid in order:
            if collect[asteroid] - deploy[asteroid] < C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS:
                return None
        legs = list(partial.legs)
        if camp_end > camp_start:
            legs.append(
                PlannedLeg(
                    partial.location, partial.location, camp_start, camp_end, 0.0, 1.0, "camp"
                )
            )
        legs.extend(reversed(legs_backward))
        # mass proxy forward through the collection tour (heavier ship after each collection)
        mass = partial.mass
        collected: dict[int, float] = {}
        propellant_total = s.initial_mass - partial.mass - C.MINER_MASS_KG * len(deploy)
        for leg in legs[len(partial.legs) :]:
            if leg.role == "camp":
                continue
            if leg.role == "collect_hop" or leg.role == "earth_return":
                # collection happens at departure of the leg
                asteroid = leg.from_id
                gained = C.maximum_collected_mass(collect[asteroid] - deploy[asteroid])
                collected[asteroid] = gained
                mass += gained
            if not self._feasible(mass, leg.delta_v_proxy_km_s, leg.tof_days, leg.inflation):
                return None
            propellant = self._propellant(mass, leg.delta_v_proxy_km_s, leg.inflation)
            propellant_total += propellant
            mass -= propellant
        return RoutePlan(tuple(legs), deploy, collect, collected, propellant_total, mass)
