"""Deterministic beam search for self-cleaning GTOC12 mining-ship routes.

A route is ``Earth -> A1 -> ... -> Ak (deploy at each) -> camp at Ak -> collection tour over
{A1..Ak} -> Earth``.  Deploy hops are expanded forwards from a launch-epoch grid with Lambert
rendezvous costs; the collection tour is scheduled *backwards* from the end of the window so every
miner works as long as possible.  Costs are impulsive proxies inflated for finite thrust; the
low-thrust refinement (``pipeline``) replaces them with certified arcs.

The candidate generator encodes what the archived JPL/Antipodes solutions do (see
``references.py`` and ``docs/GTOC12_TRACK.md``): the next asteroid is picked in *position space
at the departure epoch* — within a few hundredths of an AU in semi-major axis, a few degrees of
inclination, and a few degrees of phase — so that a 100-250 day, sub-revolution hop costs
50-100 kg.  Chains also keep a propellant reserve for the collection tour so the beam does not
fill up with deploy phases that can never be collected.

Everything is deterministic: candidate order is fixed, ties break on asteroid ID, and the only use
of ``seed`` is to shuffle nothing unless ``randomise`` is requested.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state, earth_state
from .proxies import phasing_edelbaum_proxy
from .screening import (
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
    max_deploys: int = 10
    min_deploys: int = 1
    launch_epochs: tuple[float, ...] = tuple(C.MISSION_START_MJD + np.arange(0.0, 731.0, 30.0))
    earth_leg_tofs: tuple[float, ...] = tuple(np.arange(300.0, 901.0, 50.0))
    hop_tofs: tuple[float, ...] = (
        60.0,
        90.0,
        120.0,
        150.0,
        180.0,
        240.0,
        300.0,
        360.0,
        420.0,
        480.0,
    )
    collect_hop_tofs: tuple[float, ...] = (
        90.0,
        120.0,
        150.0,
        180.0,
        240.0,
        300.0,
        360.0,
        420.0,
        480.0,
        600.0,
        720.0,
    )
    deploy_wait_days: tuple[float, ...] = (0.0, 30.0, 60.0, 120.0)
    max_per_first: int = 8  # beam diversity: variants sharing the first asteroid
    neighbours: int = 48
    # position-space candidate metric scales (reference-hop p95 values; see references.py)
    band_a_au: float = 0.04
    band_i_deg: float = 4.5  # relative inclination (vector difference), deg
    band_e: float = 0.06  # eccentricity-vector difference
    band_phase_deg: float = 3.3
    filter_scale: float = 1.5  # element-band filter = filter_scale x reference p95 bands
    # certified Earth legs cost 1.08x (out) / 0.97x (return) their Lambert proxy
    # (results/gtoc12/proxy_validation.json); 1.3 was tried (reduced_v1_search3,
    # full_catalogue_search3-5) and did not beat 1.6 once SCvx-infeasible legs were removed
    earth_leg_inflation: float = 1.6
    # ...but the thrust-authority test keeps the old 1.6 factor: a 5 km/s Earth leg squeezed into
    # 500 days passes the mass budget yet SCvx cannot fly it (reduced_v1_search3)
    earth_leg_authority_inflation: float = 1.6
    hop_inflation: float = (
        1.2  # reference hops: true dV / zero-rev Lambert dV median 1.16, p90 1.34
    )
    duty: float = 0.8  # Earth legs (their 1.6x authority inflation already carries the margin)
    # hops: reference hops use <= 0.79 of full authority with *true* dV (p95); lowering this to
    # 0.75 removed the SCvx-infeasible 120-180 day hops but cost more than it saved
    hop_duty: float = 0.8
    end_margin_days: float = 2.0
    return_window_days: float = 600.0
    collect_wait_window_days: float = 600.0
    max_per_deployed_set: int = 2
    first_level_limit: int = 4000
    earth_block: int = 1500  # asteroids per Earth-leg screening block (bounds memory)
    schedule_step_days: float = 15.0
    wait_penalty: float = 1.0  # kg propellant-equivalent per kg of mining mass forgone
    reserve_fraction: float = 0.9  # collect-phase hop propellant ~ deploy-phase hop propellant
    return_reserve_kg: float = 250.0  # reference Earth returns cost 190-230 kg
    # beam heuristic weights; full-catalogue chains die on the 15-year window with 230-430 kg
    # of propellant unused, but pricing time (0.02-0.05 kg/day) steered the beam into 120-180
    # day hops that SCvx could not fly (full_catalogue_search4/5), so it is off by default
    propellant_weight: float = 0.15
    time_weight: float = 0.0  # kg of heuristic score per day of deploy-phase duration
    initial_mass: float = C.MAX_INITIAL_MASS_KG
    time_budget_seconds: float = float("inf")  # stop expanding (keep completed plans) past this
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
    hop_propellant: float = 0.0
    score: float = 0.0


@dataclass(slots=True)
class SearchResult:
    best: RoutePlan | None
    candidates: list[RoutePlan]
    expansions: int
    lambert_evaluations: int
    wall_seconds: float
    failures: list[dict[str, object]] = field(default_factory=list)
    depth_reached: int = 0
    best_by_depth: dict[int, float] = field(default_factory=dict)


def element_deviations(
    catalogue: AsteroidCatalogue, source: int, pool: NDArray[np.int64]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """|Δa| (AU), |Δe-vector| and relative inclination (deg) between ``source`` and ``pool``.

    Vector forms are used because two orbits with equal e but different perihelion directions
    (or equal i but different nodes) are different ellipses that drift apart within years.
    """

    index = catalogue.index_of(pool)
    s_index = catalogue.index_of(source)
    da = (
        np.abs(catalogue.semi_major_axis_km[index] - catalogue.semi_major_axis_km[s_index])
        / C.AU_KM
    )
    varpi = catalogue.ascending_node_rad + catalogue.argument_of_perihelion_rad
    e_vec = catalogue.eccentricity[:, None] * np.stack([np.cos(varpi), np.sin(varpi)], axis=1)
    de = np.linalg.norm(e_vec[index] - e_vec[s_index], axis=1)
    inc = catalogue.inclination_rad
    node = catalogue.ascending_node_rad
    i_vec = inc[:, None] * np.stack([np.cos(node), np.sin(node)], axis=1)
    di = np.rad2deg(np.linalg.norm(i_vec[index] - i_vec[s_index], axis=1))
    return da, de, di


def positional_candidates(
    catalogue: AsteroidCatalogue,
    source: int,
    pool: NDArray[np.int64],
    epoch: float,
    settings: SearchSettings,
) -> tuple[NDArray[np.int64], FloatArray]:
    """Rank ``pool`` by closeness to ``source`` in (a, e, i, phase) at ``epoch``.

    The metric is the sum of squared deviations scaled by the reference-hop p95 bands, i.e. the
    *cluster-first* generator: it favours asteroids that are physically near the ship at
    departure and on a nearly identical orbit, which is where the 50-100 kg hops live.
    """

    pool = pool[pool != source]
    r_pool, _ = asteroid_state(catalogue, pool, np.full(pool.shape[0], epoch))
    r_source, _ = asteroid_state(catalogue, source, epoch)
    da, de, di = element_deviations(catalogue, source, pool)
    cross_z = r_source[0] * r_pool[:, 1] - r_source[1] * r_pool[:, 0]
    phase = np.rad2deg(np.arctan2(cross_z, r_pool @ r_source))
    metric = (
        (da / settings.band_a_au) ** 2
        + (di / settings.band_i_deg) ** 2
        + (de / settings.band_e) ** 2
        + (phase / settings.band_phase_deg) ** 2
    )
    order = np.lexsort((pool, metric))
    return pool[order], metric[order]


def proxy_candidates(
    catalogue: AsteroidCatalogue,
    source: int,
    pool: NDArray[np.int64],
    epoch: float,
    settings: SearchSettings,
) -> tuple[NDArray[np.int64], FloatArray]:
    """Rank ``pool`` by the Lambert-free phasing/Edelbaum ΔV proxy over the hop TOF grid."""

    pool = pool[pool != source]
    proxy = phasing_edelbaum_proxy(catalogue, source, pool, epoch, np.asarray(settings.hop_tofs))
    order = np.lexsort((pool, proxy["best_delta_v"]))
    return pool[order], proxy["best_delta_v"][order]


class RouteSearch:
    def __init__(
        self,
        catalogue: AsteroidCatalogue,
        asteroid_ids: NDArray[np.int64],
        settings: SearchSettings | None = None,
        excluded: set[int] | frozenset[int] | None = None,
    ) -> None:
        self.catalogue = catalogue
        banned = set(excluded or ())
        self.ids = np.asarray(
            sorted(int(item) for item in asteroid_ids if int(item) not in banned), dtype=np.int64
        )
        self.settings = settings or SearchSettings()
        self.lambert_evaluations = 0
        self._hop_cache: dict[tuple[int, float], dict[str, FloatArray]] = {}
        self._return_cache: dict[int, list[tuple[float, float, float]]] = {}
        self._collect_cache: dict[tuple[int, int, float], list[tuple[float, float, float]]] = {}
        self.last_failure = ""
        self._band_cache: dict[int, NDArray[np.int64]] = {}

    # -- proxies --

    def _propellant(self, mass: float, delta_v: float, inflation: float) -> float:
        return float(propellant_for_delta_v(mass, delta_v * inflation))

    def _feasible(
        self, mass: float, delta_v: float, tof: float, inflation: float, *, hop: bool = False
    ) -> bool:
        duty = self.settings.hop_duty if hop else self.settings.duty
        return delta_v * inflation <= float(thrust_authority_km_s(mass, tof, duty))

    def band_pool(self, asteroid_id: int) -> NDArray[np.int64]:
        """Asteroids on orbits similar enough to ``asteroid_id`` to be collectable years later.

        Reference hops stay within |Δa| 0.04 AU, |Δe| 0.045, |Δi| 3° (p95); the filter uses 1.5x
        those bands and falls back to the nearest ``neighbours`` asteroids by scaled element
        distance when the pool is too sparse (e.g. the reduced instance).
        """

        if asteroid_id in self._band_cache:
            return self._band_cache[asteroid_id]
        s = self.settings
        pool = self.ids[self.ids != asteroid_id]
        da, de, di = element_deviations(self.catalogue, asteroid_id, pool)
        inside = (
            (da <= s.filter_scale * s.band_a_au)
            & (de <= s.filter_scale * s.band_e)
            & (di <= s.filter_scale * s.band_i_deg)
        )
        if inside.sum() >= s.neighbours:
            chosen = pool[inside]
        else:
            metric = (da / s.band_a_au) ** 2 + (de / s.band_e) ** 2 + (di / s.band_i_deg) ** 2
            chosen = pool[np.lexsort((pool, metric))[: s.neighbours]]
        self._band_cache[asteroid_id] = np.sort(chosen)
        return self._band_cache[asteroid_id]

    def candidates(self, asteroid_id: int, epoch: float) -> NDArray[np.int64]:
        """Union of the proxy-ΔV ranking and the positional (cluster) ranking, proxy first."""

        s = self.settings
        pool = self.band_pool(asteroid_id)
        by_proxy, _ = proxy_candidates(self.catalogue, asteroid_id, pool, epoch, s)
        by_position, _ = positional_candidates(self.catalogue, asteroid_id, pool, epoch, s)
        chosen: list[int] = []
        seen: set[int] = set()
        for item in list(by_proxy[: s.neighbours]) + list(by_position[: s.neighbours // 2]):
            if int(item) not in seen:
                seen.add(int(item))
                chosen.append(int(item))
        return np.asarray(chosen, dtype=np.int64)

    def hops_from(self, asteroid_id: int, epoch: float) -> dict[str, FloatArray]:
        key = (asteroid_id, round(epoch, 6))
        if key not in self._hop_cache:
            targets = self.candidates(asteroid_id, epoch)
            result = screen_asteroid_hops(
                self.catalogue, asteroid_id, targets, epoch, np.asarray(self.settings.hop_tofs)
            )
            self.lambert_evaluations += 2 * targets.shape[0] * len(self.settings.hop_tofs)
            self._hop_cache[key] = result
        return self._hop_cache[key]

    def _reserve(self, partial: _Partial) -> float:
        """Propellant the collection tour and Earth return will need (reference-calibrated)."""

        s = self.settings
        return s.reserve_fraction * partial.hop_propellant + s.return_reserve_kg

    # -- search --

    def _first_level(self) -> list[_Partial]:
        """Earth -> A1 candidates, screened block-wise to bound memory at catalogue scale."""

        s = self.settings
        if s.max_deploys < 1 or self.ids.shape[0] == 0:
            return []
        epochs = np.asarray(s.launch_epochs)
        tofs = np.asarray(s.earth_leg_tofs)
        horizon = C.MISSION_END_MJD - 2.0 * C.YEAR_DAYS
        tof_grid = np.broadcast_to(tofs[None, None, :], (1, epochs.shape[0], tofs.shape[0]))
        authority = thrust_authority_km_s(s.initial_mass, tof_grid, s.duty)
        arrival_grid = epochs[None, :, None] + tofs[None, None, :]
        mined_grid = (
            C.MINING_RATE_KG_PER_YEAR * np.maximum(horizon - arrival_grid, 0.0) / C.YEAR_DAYS
        )
        kept: list[tuple[float, int, float, float, float, float]] = []
        for start in range(0, self.ids.shape[0], s.earth_block):
            block = self.ids[start : start + s.earth_block]
            grid = screen_earth_to_asteroids(self.catalogue, block, epochs, tofs)
            self.lambert_evaluations += 2 * grid["total_delta_v"].size
            dv_grid = np.where(grid["feasible"], grid["total_delta_v"], np.inf)
            ok = np.isfinite(dv_grid) & (dv_grid * s.earth_leg_authority_inflation <= authority)
            propellant_grid = propellant_for_delta_v(
                s.initial_mass, dv_grid * s.earth_leg_inflation
            )
            score_grid = np.where(
                ok, mined_grid - s.propellant_weight * (propellant_grid + C.MINER_MASS_KG), -np.inf
            )
            flat = np.argsort(-score_grid.ravel(), kind="stable")[: s.first_level_limit]
            for index in flat:
                a_index, e_index, t_index = np.unravel_index(int(index), score_grid.shape)
                if not ok[a_index, e_index, t_index]:
                    break
                kept.append(
                    (
                        float(score_grid[a_index, e_index, t_index]),
                        int(block[a_index]),
                        float(epochs[e_index]),
                        float(tofs[t_index]),
                        float(dv_grid[a_index, e_index, t_index]),
                        float(propellant_grid[a_index, e_index, t_index]),
                    )
                )
        kept.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        beam: list[_Partial] = []
        for _score, asteroid, launch, tof, dv, propellant in kept[: s.first_level_limit]:
            arrival = launch + tof
            leg = PlannedLeg(
                EARTH_ID, asteroid, launch, arrival, dv, s.earth_leg_inflation, "earth_out"
            )
            mass = s.initial_mass - propellant - C.MINER_MASS_KG
            beam.append(_Partial([leg], asteroid, arrival, mass, [(asteroid, arrival)]))
        return beam

    def _expand(self, partial: _Partial) -> list[_Partial]:
        s = self.settings
        visited = {item for item, _ in partial.deployed}
        children: list[_Partial] = []
        for wait in s.deploy_wait_days:
            departure = partial.epoch + float(wait)
            hops = self.hops_from(partial.location, departure)
            for t_index, target in enumerate(hops["target_ids"]):
                target = int(target)
                if target in visited:
                    continue
                for f_index, tof in enumerate(hops["tofs_days"]):
                    if not hops["feasible"][t_index, f_index]:
                        continue
                    dv = float(hops["total_delta_v"][t_index, f_index])
                    if not self._feasible(partial.mass, dv, float(tof), s.hop_inflation, hop=True):
                        continue
                    propellant = self._propellant(partial.mass, dv, s.hop_inflation)
                    arrival = departure + float(tof)
                    if arrival > C.MISSION_END_MJD - 3.0 * C.YEAR_DAYS:
                        continue
                    legs = list(partial.legs)
                    if wait > 0.0:
                        legs.append(
                            PlannedLeg(
                                partial.location,
                                partial.location,
                                partial.epoch,
                                departure,
                                0.0,
                                1.0,
                                "camp",
                            )
                        )
                    legs.append(
                        PlannedLeg(
                            partial.location,
                            target,
                            departure,
                            arrival,
                            dv,
                            s.hop_inflation,
                            "deploy_hop",
                        )
                    )
                    children.append(
                        _Partial(
                            legs,
                            target,
                            arrival,
                            partial.mass - propellant - C.MINER_MASS_KG,
                            [*partial.deployed, (target, arrival)],
                            partial.hop_propellant + propellant,
                        )
                    )
        return children

    def run(self) -> SearchResult:
        started = time.perf_counter()
        s = self.settings
        beam = self._first_level()
        if not beam:
            return SearchResult(None, [], 0, self.lambert_evaluations, 0.0, [], 0, {})
        expansions = 0
        completed: list[RoutePlan] = []
        failures: list[dict[str, object]] = []
        best_by_depth: dict[int, float] = {}
        current = self._select(beam)
        depth = 1
        for partial in current:
            plan = self._complete(partial)
            if plan is not None:
                completed.append(plan)
                best_by_depth[1] = max(best_by_depth.get(1, 0.0), plan.total_collected_kg)
        for depth in range(2, s.max_deploys + 1):
            if time.perf_counter() - started > s.time_budget_seconds:
                failures.append({"reason": "time budget exhausted", "depth": depth - 1})
                depth -= 1
                break
            next_beam: list[_Partial] = []
            for partial in current:
                expansions += 1
                next_beam.extend(self._expand(partial))
            current = self._select(next_beam)
            if not current:
                depth -= 1
                break
            for partial in current:
                if len(partial.deployed) >= s.min_deploys:
                    plan = self._complete(partial)
                    if plan is not None:
                        completed.append(plan)
                        best_by_depth[depth] = max(
                            best_by_depth.get(depth, 0.0), plan.total_collected_kg
                        )
                    else:
                        failures.append(
                            {
                                "asteroids": [item for item, _ in partial.deployed],
                                "reason": f"no feasible collection tour ({self.last_failure})",
                                "mass_after_deploys_kg": partial.mass,
                                "deploy_end_epoch": partial.epoch,
                            }
                        )
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
            depth,
            best_by_depth,
        )

    def _return_feasible(self, asteroid: int, mass_guess: float) -> bool:
        """Cached test that *some* Earth return from ``asteroid`` fits inside the final window."""

        if asteroid not in self._return_cache:
            end = C.MISSION_END_MJD - self.settings.end_margin_days
            self._return_cache[asteroid] = self._return_options(asteroid, end)
        return any(
            self._feasible(mass_guess, dv, tof, self.settings.earth_leg_authority_inflation)
            for dv, _departure, tof in self._return_cache[asteroid]
        )

    def _select(self, partials: list[_Partial]) -> list[_Partial]:
        """Stable top-``beam_width`` by heuristic, with diversity, reserve and return pruning.

        Score = expected mined mass minus a propellant penalty.  At most
        ``max_per_deployed_set`` variants of one deployed set survive; chains whose mass after the
        deploy phase cannot cover the dry mass plus the collect-phase reserve are dropped, as are
        chains whose first asteroid (the last one collected before the Earth return) has no
        feasible return.
        """

        end = C.MISSION_END_MJD - 2.0 * C.YEAR_DAYS  # rough collection horizon
        for partial in partials:
            mined = sum(
                C.maximum_collected_mass(max(end - deploy_epoch, 0.0))
                for _, deploy_epoch in partial.deployed
            )
            spent = self.settings.initial_mass - partial.mass
            elapsed = partial.epoch - partial.legs[0].departure_epoch
            partial.score = (
                mined
                - self.settings.propellant_weight * spent
                - self.settings.time_weight * elapsed
            )
        ordered = sorted(
            partials,
            key=lambda item: (-item.score, item.epoch, tuple(a for a, _ in item.deployed)),
        )
        selected: list[_Partial] = []
        per_set: dict[tuple[int, ...], int] = {}
        per_first: dict[int, int] = {}
        for partial in ordered:
            if len(selected) >= self.settings.beam_width:
                break
            if partial.mass < C.DRY_MASS_KG + self._reserve(partial):
                continue
            key = tuple(sorted(a for a, _ in partial.deployed))
            if per_set.get(key, 0) >= self.settings.max_per_deployed_set:
                continue
            first = partial.deployed[0][0]
            if per_first.get(first, 0) >= self.settings.max_per_first:
                continue
            guess = partial.mass + sum(
                C.maximum_collected_mass(max(end - d, 0.0)) for _, d in partial.deployed
            )
            if not self._return_feasible(first, guess):
                continue
            per_set[key] = per_set.get(key, 0) + 1
            per_first[first] = per_first.get(first, 0) + 1
            selected.append(partial)
        return selected

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

        key = (source, target, round(latest_arrival, 6))
        if key in self._collect_cache:
            return self._collect_cache[key]
        s = self.settings
        tofs = np.asarray(s.collect_hop_tofs)
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
        self._collect_cache[key] = options
        return options

    def _best_collect_hop(
        self,
        source: int,
        target: int,
        epoch: float,
        mass_guess: float,
        penalty_scale: float = 1.0,
    ) -> tuple[float, tuple[float, float, float] | None]:
        s = self.settings
        best_hop = None
        best_cost = np.inf
        for dv, departure, tof in self._collect_hop_options(source, target, epoch):
            if not self._feasible(mass_guess, dv, tof, s.hop_inflation, hop=True):
                continue
            # propellant proxy plus the mining mass lost by collecting ``source`` earlier (the
            # whole hop duration counts: the miner at ``source`` stops when the ship leaves)
            lost = C.maximum_collected_mass(epoch - departure)
            cost = self._propellant(mass_guess, dv, s.hop_inflation)
            cost += s.wait_penalty * penalty_scale * lost
            if cost < best_cost - 1e-12 or (
                abs(cost - best_cost) <= 1e-12 and best_hop is not None and departure > best_hop[1]
            ):
                best_cost = cost
                best_hop = (dv, departure, tof)
        return best_cost, best_hop

    def _complete(self, partial: _Partial) -> RoutePlan | None:
        """Schedule the collection tour: greedy backward order first, strict reverse as fallback."""

        plans: list[RoutePlan] = []
        reasons: list[str] = []
        for penalty_scale in (1.0, 4.0, 16.0):
            for greedy in (True, False):
                plan = self._schedule(partial, greedy, penalty_scale)
                if plan is not None:
                    plans.append(plan)
                else:
                    reasons.append(
                        f"{'greedy' if greedy else 'reverse'}x{penalty_scale:g}:{self.last_failure}"
                    )
            if plans:
                break
        if not plans:
            self.last_failure = ",".join(reasons)
            return None
        plans.sort(key=lambda item: (-item.total_collected_kg, item.propellant_proxy_kg))
        return plans[0]

    def _schedule(
        self, partial: _Partial, greedy: bool, penalty_scale: float = 1.0
    ) -> RoutePlan | None:
        """Schedule the collection tour and Earth return backwards from the window end.

        The first deployed asteroid is collected last (it has the longest stay and the return
        departs from it); the remaining order is either the strict reverse of deployment or chosen
        greedily backwards by proxy cost, with the camp asteroid (last deployed) forced to be the
        first collected because the ship is already there.
        """

        s = self.settings
        deploy = dict(partial.deployed)
        remaining = [asteroid for asteroid, _ in partial.deployed]
        end = C.MISSION_END_MJD - s.end_margin_days
        first = remaining[0]  # the last asteroid collected before returning to Earth
        camp_asteroid = partial.location
        mass_guess = partial.mass + sum(
            C.maximum_collected_mass(max(end - deploy_epoch, 0.0))
            for _, deploy_epoch in partial.deployed
        )
        best_return = None
        for dv, departure, tof in self._return_options(first, end):
            if self._feasible(mass_guess, dv, tof, s.earth_leg_authority_inflation):
                best_return = (dv, departure, tof)
                break
        if best_return is None:
            self.last_failure = "no_return"
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
        remaining.remove(first)
        while remaining:
            if not greedy:
                # strict reverse of deployment: going backwards in time, the asteroid collected
                # just before ``location`` is the earliest-deployed one still remaining
                choices = [remaining[0]]
            elif len(remaining) == 1:
                choices = list(remaining)
            else:
                choices = [a for a in remaining if a != camp_asteroid] or list(remaining)
            best_choice = None
            for previous in choices:
                cost, hop = self._best_collect_hop(
                    previous, location, epoch, mass_guess, penalty_scale
                )
                if hop is not None and (best_choice is None or cost < best_choice[0] - 1e-12):
                    best_choice = (cost, previous, hop)
            if best_choice is None:
                self.last_failure = "no_collect_hop"
                return None
            _cost, previous, best_hop = best_choice
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
            remaining.remove(previous)
        if location != camp_asteroid:
            self.last_failure = "tour_not_ending_at_camp"
            return None
        # camp at the last deployed asteroid between its deploy and its collection
        camp_start = partial.epoch
        camp_end = epoch
        if camp_end - camp_start < 0.0:
            self.last_failure = "camp_negative"
            return None
        for asteroid in deploy:
            if collect[asteroid] - deploy[asteroid] < C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS:
                self.last_failure = "stay_too_short"
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
            authority_inflation = (
                s.earth_leg_authority_inflation
                if leg.role in ("earth_out", "earth_return")
                else leg.inflation
            )
            is_hop = leg.role in ("deploy_hop", "collect_hop")
            if not self._feasible(
                mass, leg.delta_v_proxy_km_s, leg.tof_days, authority_inflation, hop=is_hop
            ):
                self.last_failure = "leg_authority"
                return None
            propellant = self._propellant(mass, leg.delta_v_proxy_km_s, leg.inflation)
            propellant_total += propellant
            mass -= propellant
        if mass < C.DRY_MASS_KG + sum(collected.values()):
            self.last_failure = "mass_below_dry_plus_collected"
            return None
        return RoutePlan(tuple(legs), deploy, collect, collected, propellant_total, mass)
