"""Cooperative cluster pricing: the master's pricing problem solved per co-moving family.

The archived GTOC12 solutions put every ship inside one tight co-moving family of asteroids
(a 2.72-2.80 AU, hops of 78 kg / 183 days) and let ships collect each other's miners.  Our
earlier greedy fleets could not get there because the Earth-leg gate (Lambert authority ratio
0.5) only admitted legs to the a 2.23-2.43 AU region, where families have 2-43 members.  This
module prices one family at a time:

1. **Earth legs**: the family's members are screened from the launch grid with a permissive
   Lambert limit and the best distinct ``(target, launch)`` legs are *flown by SCvx* before the
   beam sees them (``certify_earth_legs``); the beam is seeded with certified legs only, at
   their measured propellant (``RouteSearch(first_level=...)``).
2. **Ships**: up to ``ships`` itineraries are built one after another inside the family
   (beam -> SCvx -> joint re-timing/extension), sharing a :class:`MinerPool`.  Earlier ships may
   leave miners (orphan credit > 0) and later ships pick them up as foreign collects; the last
   ship never leaves any.
3. **Orphan repair**: a miner nobody in the bundle collects is first offered to every ship as a
   foreign collect; if that does not certify, the deployer drops the visit and is re-timed and
   re-certified, so the emitted bundle has *no orphans*.
4. The result is a :class:`ClusterBundle`: certified ship routes (columns for the master, both as
   one multi-ship bundle and, where self-contained, as single-ship columns) with the cooperative
   statistics the report needs (orphans left, collectors per deployer).

``price_clusters`` runs ``price_cluster`` over many families in forked worker processes with a
bounded worker count; every worker is deterministic given the cluster and the seed, and the
caller consumes results in cluster order.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .clusters import ClusterBands, ComovingClusters
from .cooperative import EPOCH_TOLERANCE_DAYS, FleetColumn, MinerPool
from .data import AsteroidCatalogue
from .earthleg import EarthLegBounds, EarthLegModel, refine_leg_scvx
from .low_thrust import ScvxSettings
from .pipeline import RefinedRoute, refine_route
from .proxies import phasing_edelbaum_proxy
from .retiming import (
    Retimer,
    RetimeSettings,
    Visit,
    build_visits,
    calibrate_from_route,
    extend_plan,
    improve_and_certify,
    orders_of,
    plan_value,
    visits_of,
)
from .screening import propellant_for_delta_v, screen_earth_to_asteroids, thrust_authority_km_s
from .search import EARTH_ID, EarthLeg, PlannedLeg, RoutePlan, RouteSearch, SearchSettings

IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ClusterPricingSettings:
    ships: int = 3  # itineraries priced per family (deployer + collectors)
    # Earth legs: Lambert prescreen limit, then SCvx on the best distinct (target, launch) legs
    earth_authority_ratio: float = 0.95
    earth_inflation: float = 0.9  # reference Earth legs cost 0.83-0.86x their Lambert ΔV
    earth_legs_per_ship: int = 4  # certified legs the beam is seeded with (per ship slot)
    earth_leg_checks: int = 12  # SCvx checks per ship slot before giving up
    # continuous (launch, TOF) refinement of every certified grid leg with SCvx in the loop
    # (earthleg.refine_leg_scvx); ``earth_leg_refinements`` extra SCvx calls per leg
    earth_leg_continuous: bool = True
    earth_leg_refinements: int = 8
    # collector itineraries: later ship slots are seeded to harvest every miner the earlier
    # slots left (all orphans, any position in the tour) instead of only the beam's incidental
    # foreign collects; see ``harvest_orphans``
    collector_harvest: bool = True
    # beam score weight on the collect-time cost of re-flying each deploy pair (SearchSettings
    # .collect_lookahead_weight); 0 = off
    collect_lookahead_weight: float = 0.0
    # exact collect-tour pricing in the beam (SearchSettings.collect_dp): every completed
    # partial also gets the Held-Karp order + timing DP over the pair-cost table at the actual
    # collect epochs; the best of the heuristic tours and the DP tour is kept
    collect_dp: bool = True
    collect_dp_propellant_weight: float = 1.0
    # launch grid spans the mission start plus this window (the references launch over ~3 y)
    launch_window_days: float = 3.0 * C.YEAR_DAYS
    launch_step_days: float = 30.0
    # beam inside the family
    beam_width: int = 24
    max_deploys: int = 10
    neighbours: int = 64
    refine_top: int = 2  # distinct-Earth-leg chains flown per beam run
    search_retries: int = 2  # beam re-runs after SCvx refused every flown chain (legs banned)
    # joint re-timing / extension (SCvx in the loop)
    retime_attempts: int = 4
    retime_rounds: int = 6
    retime_budget_seconds: float = 600.0
    orphan_credit: float = 1.0  # earlier ships may leave miners for the bundle's later ships
    orphan_margin_days: float = 400.0
    # DP hop limit (Lambert ΔV / full authority).  The archived hops fly at up to 0.48 Lambert
    # ratio (0.63 true); SCvx-in-the-loop bans what does not fly, so the re-timer may push
    # harder here than the 0.45 that was calibrated on the inner, eccentric region.
    hop_authority_ratio: float = 0.55
    retime_step_days: float = 15.0
    time_budget_seconds: float = float("inf")  # per cluster
    seed: int = 0
    # beam grids (tests use coarse ones); ``None`` keeps the SearchSettings defaults
    launch_epochs: tuple[float, ...] | None = None
    earth_leg_tofs: tuple[float, ...] | None = None


def cluster_search_settings(settings: ClusterPricingSettings, members: int) -> SearchSettings:
    """Beam settings for a family: reference-calibrated Earth legs, small dense pool."""

    grids: dict[str, Any] = {}
    if settings.launch_epochs is not None:
        grids["launch_epochs"] = settings.launch_epochs
    else:
        grids["launch_epochs"] = tuple(
            C.MISSION_START_MJD
            + np.arange(0.0, settings.launch_window_days + 1e-9, settings.launch_step_days)
        )
    if settings.earth_leg_tofs is not None:
        grids["earth_leg_tofs"] = settings.earth_leg_tofs
    return SearchSettings(
        beam_width=settings.beam_width,
        max_deploys=settings.max_deploys,
        neighbours=min(settings.neighbours, max(members - 1, 1)),
        earth_out_inflation=settings.earth_inflation,
        earth_out_authority_ratio=settings.earth_authority_ratio,
        # the archived returns (443-486 d) cost 0.92x Lambert at ratio 0.40
        earth_return_inflation=1.0,
        earth_return_authority_ratio=0.5,
        return_reserve_kg=200.0,
        seed=settings.seed,
        # With continuously optimised Earth legs the beam is seeded with the certified legs
        # only: the grid neighbours the window would add are shorter legs priced at the long
        # leg's calibration (1.05x Lambert), and those fail SCvx (probe: the beam flew a 450 d
        # neighbour of a certified 610 d leg; with the window off it closed 7 asteroids/414 kg
        # on the certified leg instead of 6/364 kg).
        first_level_window_days=0.0 if settings.earth_leg_continuous else 200.0,
        collect_lookahead_weight=settings.collect_lookahead_weight,
        collect_dp=settings.collect_dp,
        collect_dp_propellant_weight=settings.collect_dp_propellant_weight,
        **grids,
    )


def cluster_retime_settings(settings: ClusterPricingSettings, *, last: bool) -> RetimeSettings:
    return RetimeSettings(
        step_days=settings.retime_step_days,
        orphan_credit=0.0 if last else settings.orphan_credit,
        orphan_margin_days=settings.orphan_margin_days,
        hop_authority_ratio=settings.hop_authority_ratio,
        # the re-timer moves Earth legs too: keep them inside the envelope SCvx flew
        earth_out_authority_ratio=0.85,
    )


def _peak_rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # pragma: no cover - non-POSIX
        return float("nan")


# -- Earth legs ------------------------------------------------------------------------------


def certify_earth_legs(
    catalogue: AsteroidCatalogue,
    targets: IntArray,
    search_settings: SearchSettings,
    *,
    count: int,
    max_checks: int,
    scvx: ScvxSettings | None = None,
    cache: dict[tuple[int, float, float], EarthLeg | None] | None = None,
    weights: dict[int, float] | None = None,
    certify=None,
    continuous: bool = True,
    continuous_evaluations: int = 8,
    scvx_cache: dict[tuple[int, float, float], EarthLeg | None] | None = None,
    optimiser_log: list[dict[str, Any]] | None = None,
) -> tuple[list[EarthLeg], list[dict[str, Any]]]:
    """SCvx-certified Earth legs to ``targets``: the beam's first level for one ship slot.

    Distinct ``(target, launch)`` legs are ranked by the beam's own first-level score (mining
    horizon minus priced propellant at the best admissible TOF) and flown in that order until
    ``count`` certify or ``max_checks`` were tried.  With ``continuous`` (default) every grid
    leg that certifies is then refined by the continuous Earth-leg optimiser
    (``earthleg.refine_leg_scvx``: compass search over launch epoch and TOF inside ±half a grid
    spacing with SCvx as the evaluator, at most ``continuous_evaluations`` extra SCvx calls) and
    the cheapest certified leg of that launch window seeds the beam.  Failed legs are retained
    (returned as the reject log) and cached so no cluster flies the same leg twice.
    """

    s = search_settings
    cache = {} if cache is None else cache
    certify = certify or _certify_single_leg
    targets = np.asarray(sorted(int(a) for a in targets), dtype=np.int64)
    if targets.shape[0] == 0:
        return [], []
    epochs = np.asarray(s.launch_epochs)
    tofs = np.asarray(s.earth_leg_tofs)
    launch_radius = 0.5 * float(np.min(np.diff(epochs))) if epochs.shape[0] > 1 else 15.0
    model = EarthLegModel(
        initial_mass=s.initial_mass,
        inflation=s.earth_out_inflation,
        authority_ratio=s.earth_out_authority_ratio,
        propellant_weight=s.propellant_weight,
    )
    bounds = EarthLegBounds(
        launch_min=float(epochs.min()),
        launch_max=float(epochs.max()),
        tof_min=float(tofs.min()),
        tof_max=float(tofs.max()),
    )
    scvx_cache = {} if scvx_cache is None else scvx_cache
    grid = screen_earth_to_asteroids(catalogue, targets, epochs, tofs)
    dv = np.where(grid["feasible"], grid["total_delta_v"], np.inf)
    authority = s.earth_out_authority_ratio * thrust_authority_km_s(
        s.initial_mass, tofs[None, None, :], 1.0
    )
    ok = np.isfinite(dv) & (dv <= authority)
    propellant = propellant_for_delta_v(s.initial_mass, dv * s.earth_out_inflation)
    horizon = C.MISSION_END_MJD - 2.0 * C.YEAR_DAYS
    arrival = epochs[None, :, None] + tofs[None, None, :]
    mined = C.MINING_RATE_KG_PER_YEAR * np.maximum(horizon - arrival, 0.0) / C.YEAR_DAYS
    weight = np.asarray([1.0 if weights is None else weights.get(int(a), 1.0) for a in targets])
    score = np.where(ok, weight[:, None, None] * mined - s.propellant_weight * propellant, -np.inf)
    # best TOF per (target, launch); then rank the pairs
    best_tof = np.argmax(score, axis=2)
    pair_score = np.take_along_axis(score, best_tof[:, :, None], axis=2)[:, :, 0]
    order = np.argsort(-pair_score.ravel(), kind="stable")
    certified: list[EarthLeg] = []
    rejected: list[dict[str, Any]] = []
    checks = 0
    for flat in order:
        if len(certified) >= count:
            break
        a_index, e_index = np.unravel_index(int(flat), pair_score.shape)
        if not np.isfinite(pair_score[a_index, e_index]):
            break
        t_index = int(best_tof[a_index, e_index])
        target = int(targets[a_index])
        launch = float(epochs[e_index])
        tof = float(tofs[t_index])
        lambert = float(dv[a_index, e_index, t_index])
        grid_key = (target, round(launch, 3), round(tof, 3))
        if grid_key in cache:
            leg = cache[grid_key]
            launch_flown, tof_flown = launch, tof
        elif checks >= max_checks:
            continue  # only cached legs from here on
        else:
            launch_flown, tof_flown = launch, tof
            checks += 1
            leg = certify(catalogue, target, launch_flown, tof_flown, lambert, scvx)
            if leg is not None and continuous and continuous_evaluations > 0:
                refinement = refine_leg_scvx(
                    catalogue,
                    leg,
                    certify,
                    scvx,
                    bounds=bounds,
                    launch_radius_days=launch_radius,
                    max_evaluations=continuous_evaluations,
                    cache=scvx_cache,
                    model=model,
                )
                if optimiser_log is not None:
                    optimiser_log.append(refinement.summary())
                leg = refinement.leg
                launch_flown, tof_flown = leg.launch_epoch, leg.tof_days
            cache[grid_key] = leg
        if leg is None:
            rejected.append(
                {
                    "target": target,
                    "launch_epoch": launch_flown,
                    "tof_days": tof_flown,
                    "lambert_dv_km_s": lambert,
                    "authority_ratio": float(
                        lambert / thrust_authority_km_s(s.initial_mass, tof_flown, 1.0)
                    ),
                    "reason": "earth leg not certified by SCvx",
                }
            )
        else:
            certified.append(leg)
    return certified, rejected


def _certify_single_leg(
    catalogue: AsteroidCatalogue,
    target: int,
    launch: float,
    tof: float,
    lambert_dv: float,
    scvx: ScvxSettings | None,
) -> EarthLeg | None:
    leg = PlannedLeg(EARTH_ID, target, launch, launch + tof, lambert_dv, 1.0, "earth_out")
    plan = RoutePlan((leg,), {target: launch + tof}, {}, {}, 0.0, C.MAX_INITIAL_MASS_KG)
    refined = refine_route(plan, catalogue, scvx=scvx)
    first = refined.legs[0] if refined.legs else None
    if first is None or not first.certified or not np.isfinite(first.mass_after_leg):
        return None
    return EarthLeg(
        target, launch, tof, lambert_dv, C.MAX_INITIAL_MASS_KG - float(first.mass_after_leg)
    )


# -- bundle ----------------------------------------------------------------------------------


@dataclass(slots=True)
class BundleShip:
    slot: int
    route: RefinedRoute  # the ship's emitted (certified) route
    variants: list[RefinedRoute] = field(default_factory=list)  # every certified variant
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClusterBundle:
    label: int
    members: tuple[int, ...]
    ships: list[BundleShip]
    rejected: list[dict[str, Any]] = field(default_factory=list)  # reject-variant log
    earth_legs: dict[str, Any] = field(default_factory=dict)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    harvest: dict[str, Any] = field(default_factory=dict)  # joint collect re-sequencing report
    wall_seconds: float = 0.0
    peak_rss_mb: float = 0.0
    stopped: str = ""

    @property
    def routes(self) -> list[RefinedRoute]:
        return [ship.route for ship in self.ships]

    @property
    def collected_kg(self) -> float:
        return sum(route.total_collected_kg for route in self.routes)

    def pool(self) -> MinerPool:
        pool = MinerPool()
        pool.register_all([(ship.route.plan, ship.slot) for ship in self.ships])
        return pool

    def consistent(self) -> str:
        """Empty when the emitted routes form a valid pool (each miner deployed once, every
        collect matched to a deploy in the bundle); the error text otherwise."""

        try:
            self.pool()
        except ValueError as error:
            return str(error)
        return ""

    def cooperative_statistics(self) -> dict[str, Any]:
        """Orphans left, foreign collects and collectors per deployer for the emitted routes."""

        deployer_of: dict[int, int] = {}
        for ship in self.ships:
            for asteroid in ship.route.plan.deploy_epochs:
                deployer_of[asteroid] = ship.slot
        collectors: dict[int, set[int]] = {ship.slot: set() for ship in self.ships}
        foreign = 0
        for ship in self.ships:
            for asteroid in ship.route.plan.collect_epochs:
                deployer = deployer_of[asteroid]
                if deployer != ship.slot:
                    foreign += 1
                    collectors[deployer].add(ship.slot)
        pool = self.pool()
        return {
            "ships": len(self.ships),
            "asteroids": sum(len(ship.route.plan.deploy_epochs) for ship in self.ships),
            "collected_kg": self.collected_kg,
            "orphans_left": sorted(pool.orphans()),
            "cooperative_collects": foreign,
            "collectors_per_deployer": {
                str(slot): len(items) for slot, items in sorted(collectors.items())
            },
            "deployers_with_collectors": sum(1 for items in collectors.values() if items),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "members": len(self.members),
            "ships": [
                {
                    "slot": ship.slot,
                    "asteroids": list(ship.route.plan.asteroids),
                    "collected_kg": ship.route.total_collected_kg,
                    "final_mass_kg": ship.route.final_mass_kg,
                    "refined_arcs": ship.route.refined_arc_count,
                    "certified_variants": len(ship.variants),
                    "launch_epoch": ship.route.plan.legs[0].departure_epoch,
                    "foreign_collects": sorted(ship.route.plan.foreign_deploy_epochs),
                    "orphaned": list(ship.route.plan.orphaned),
                    **ship.report,
                }
                for ship in self.ships
            ],
            "cooperative": self.cooperative_statistics() if self.ships else None,
            "earth_legs": self.earth_legs,
            "repairs": self.repairs,
            "harvest": self.harvest,
            "rejected": self.rejected,
            "wall_seconds": self.wall_seconds,
            "peak_rss_mb": self.peak_rss_mb,
            "stopped": self.stopped,
        }


def profile_for_orders(
    plan: RoutePlan,
    retimer: Retimer,
    deploy_order: list[int],
    collect_order: list[int],
    foreign: dict[int, float] | None,
) -> list[float]:
    """Per-leg mass guess for a re-ordered plan, taken from the same departures of ``plan``.

    Each new leg departs a body the old plan also departed from (Earth, a deploy visit or a
    collect visit); the old leg's initial mass in the same phase is the guess, falling back to
    the other phase (a new camp asteroid only departed once before).  The DP's mass rounds
    correct the guess.
    """

    old_visits, _, _ = visits_of(plan)
    old_masses = retimer._plan_masses(plan)
    by_phase: dict[tuple[int, bool], float] = {}

    def deploy_phase(visit: Visit) -> bool:
        return visit.body == EARTH_ID or (visit.deploy and not visit.collect)

    for index, mass in enumerate(old_masses):
        visit = old_visits[index]  # the leg departs this visit
        by_phase[(visit.body, deploy_phase(visit))] = mass
    new_visits = build_visits(deploy_order, collect_order, foreign)
    profile: list[float] = []
    for index in range(len(new_visits) - 1):
        visit = new_visits[index]
        mass = by_phase.get((visit.body, deploy_phase(visit)))
        if mass is None:
            mass = by_phase.get((visit.body, not deploy_phase(visit)))
        if mass is None:
            mass = profile[-1] if profile else retimer.search_settings.initial_mass
        profile.append(float(mass))
    return profile


def drop_asteroid(
    plan: RoutePlan,
    asteroid: int,
    retimer: Retimer,
    pinned: dict[int, float] | None = None,
) -> RoutePlan | None:
    """The plan without one of its own deploys (and its collect, if any), re-timed.

    ``pinned`` deploys (miners another ship collects) keep their exact epoch in the re-timing.
    """

    deploy_order, collect_order = orders_of(plan)
    if asteroid not in deploy_order or len(deploy_order) < 2:
        return None
    new_deploy = [a for a in deploy_order if a != asteroid]
    new_collect = [a for a in collect_order if a != asteroid]
    if not new_collect:
        return None
    profile = profile_for_orders(plan, retimer, new_deploy, new_collect, plan.foreign_deploy_epochs)
    result = retimer.retime_order(
        new_deploy,
        new_collect,
        profile,
        before=-np.inf,
        original=plan,
        foreign=plan.foreign_deploy_epochs,
        pinned={a: e for a, e in (pinned or {}).items() if a in new_deploy},
    )
    if result.plan is None or not result.plan.feasible:
        return None
    return result.plan


def refine_candidates(
    candidates: Sequence[RoutePlan],
    certified_keys: set[tuple[int, float, float]],
    settings: ClusterPricingSettings,
) -> list[RoutePlan]:
    """The beam chains worth flying: distinct Earth legs, exactly-certified Earth legs first.

    The beam's top chains are usually near-duplicates on one Earth leg, so refining the top two
    blindly flies the same (possibly refused) leg twice.  One chain per Earth leg is kept, in
    beam order, and chains whose Earth leg is one SCvx already flew go first because they cannot
    fail on leg 0.
    """

    exact: list[RoutePlan] = []
    grid: list[RoutePlan] = []
    seen: set[tuple[int, float, float]] = set()
    for plan in candidates:
        leg = plan.legs[0]
        key = (leg.to_id, leg.departure_epoch, leg.tof_days)
        if key in seen:
            continue
        seen.add(key)
        (exact if key in certified_keys else grid).append(plan)
    return (exact + grid)[: settings.refine_top]


def ban_failed_legs(
    plan: RoutePlan,
    failures: Sequence[dict[str, Any]],
    banned_pairs: set[tuple[int, int]],
    banned_earth: set[tuple[int, float, float]],
) -> None:
    """Record the legs SCvx refused so no later beam in the family rebuilds them."""

    for failure in failures:
        index = failure.get("leg")
        if not isinstance(index, int) or index < 0 or index >= len(plan.legs):
            continue
        leg = plan.legs[index]
        if leg.from_id == EARTH_ID:
            banned_earth.add((leg.to_id, leg.departure_epoch, leg.tof_days))
        elif leg.to_id != EARTH_ID:
            banned_pairs.add((leg.from_id, leg.to_id))


def price_cluster(
    catalogue: AsteroidCatalogue,
    members: IntArray,
    *,
    label: int = 0,
    excluded: set[int] | frozenset[int] | None = None,
    weights: dict[int, float] | None = None,
    settings: ClusterPricingSettings | None = None,
    scvx: ScvxSettings | None = None,
    earth_cache: dict[tuple[int, float, float], EarthLeg | None] | None = None,
    certify_earth=None,
    refine=None,
) -> ClusterBundle:
    """Price one co-moving family: deployer + collector itineraries, all SCvx-certified.

    ``certify_earth`` and ``refine`` are injection points for tests (proxy-trusting stand-ins for
    the SCvx single-leg check and :func:`refine_route`).
    """

    started = time.perf_counter()
    settings = settings or ClusterPricingSettings()
    refine = refine or (lambda plan: refine_route(plan, catalogue, scvx=scvx))
    members = np.asarray(sorted(int(a) for a in members), dtype=np.int64)
    used: set[int] = set(excluded or ())
    pool = MinerPool()
    bundle = ClusterBundle(label, tuple(int(a) for a in members), [])
    earth_cache = {} if earth_cache is None else earth_cache
    earth_scvx_cache: dict[tuple[int, float, float], EarthLeg | None] = {}
    search_settings = cluster_search_settings(settings, members.shape[0])
    earth_report: dict[str, Any] = {
        "checked": 0,
        "certified": 0,
        "rejected": [],
        "continuous": settings.earth_leg_continuous,
        "optimised": [],  # grid leg -> continuous leg records (surrogate propellant before/after)
    }
    retimers: dict[int, Retimer] = {}
    searches: dict[int, RouteSearch] = {}
    # legs SCvx refused anywhere in this family: shared by every ship slot's beam
    banned_pairs: set[tuple[int, int]] = set()
    banned_earth: set[tuple[int, float, float]] = set()

    def remaining_budget() -> float:
        return settings.time_budget_seconds - (time.perf_counter() - started)

    for slot in range(1, settings.ships + 1):
        if remaining_budget() <= 0.0:
            bundle.stopped = f"time budget before ship slot {slot}"
            break
        free = np.asarray([a for a in members if a not in used and a not in pool.touched()])
        if free.shape[0] < 2:
            bundle.stopped = f"family exhausted before ship slot {slot}"
            break
        before = len(earth_cache)
        legs, rejected_legs = certify_earth_legs(
            catalogue,
            free,
            search_settings,
            count=settings.earth_legs_per_ship,
            max_checks=settings.earth_leg_checks,
            scvx=scvx,
            cache=earth_cache,
            weights=weights,
            certify=certify_earth,
            continuous=settings.earth_leg_continuous,
            continuous_evaluations=settings.earth_leg_refinements,
            scvx_cache=earth_scvx_cache,
            optimiser_log=earth_report["optimised"],
        )
        earth_report["checked"] += len(earth_cache) - before
        earth_report["certified"] += len(legs)
        earth_report["rejected"].extend(rejected_legs)
        if not legs:
            bundle.rejected.append({"slot": slot, "reason": "no certified Earth leg"})
            continue
        last = slot == settings.ships
        search = RouteSearch(
            catalogue,
            members,
            search_settings,
            excluded=used | pool.touched(),
            weights=weights,
            seeds=pool.orphans(),
            first_level=legs,
        )
        search.banned_pairs = banned_pairs
        search.banned_earth = banned_earth
        certified_keys = {(leg.target, leg.launch_epoch, leg.tof_days) for leg in legs}
        refined: RefinedRoute | None = None
        variants: list[RefinedRoute] = []
        ship_report: dict[str, Any] = {"search": []}
        # beam -> SCvx; when every flown chain is refused, the refused legs are banned and the
        # beam is re-run without them (bounded by ``search_retries``)
        for attempt in range(settings.search_retries + 1):
            if remaining_budget() <= 0.0:
                break
            result = search.run()
            ship_report["search"].append(
                {
                    "attempt": attempt,
                    "candidates": len(result.candidates),
                    "best_by_depth": result.best_by_depth,
                    "wall_seconds": result.wall_seconds,
                    "failed_chains": len(result.failures),
                    "earth_legs": [leg.target for leg in legs],
                    "banned_pairs": len(banned_pairs),
                    "banned_earth": len(banned_earth),
                    "collect_dp": dict(search.collect_dp_stats),
                    "collect_table_pairs": (
                        search.collect_table.cached_pairs if search.collect_dp_used else 0
                    ),
                }
            )
            if not result.candidates:
                reasons: dict[str, int] = {}
                for failure in result.failures:
                    key = str(failure.get("reason", "?"))
                    reasons[key] = reasons.get(key, 0) + 1
                bundle.rejected.append(
                    {
                        "slot": slot,
                        "attempt": attempt,
                        "reason": "beam found no closing chain",
                        "first_level": result.first_level,
                        "depth_reached": result.depth_reached,
                        "failed_chains": len(result.failures),
                        "failure_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:5]),
                        "earth_legs": [
                            [leg.target, leg.launch_epoch, leg.tof_days, leg.propellant_kg]
                            for leg in legs
                        ],
                    }
                )
                break
            bans_before = len(banned_pairs) + len(banned_earth)
            for rank, plan in enumerate(
                refine_candidates(result.candidates, certified_keys, settings)
            ):
                route = refine(plan)
                if route.certified:
                    refined = route
                    variants.append(route)
                    ship_report["refined_rank"] = rank
                    ship_report["refined_attempt"] = attempt
                    break
                bundle.rejected.append(
                    {
                        "slot": slot,
                        "attempt": attempt,
                        "rank": rank,
                        "reason": "beam chain not certified",
                        "failures": route.failures[:3],
                        "plan_collected_kg": plan.total_collected_kg,
                    }
                )
                ban_failed_legs(plan, route.failures, banned_pairs, banned_earth)
            if refined is not None or len(banned_pairs) + len(banned_earth) == bans_before:
                break  # certified, or nothing new to exclude: a re-run would repeat itself
        if refined is None:
            continue
        retimer = Retimer(
            catalogue,
            search_settings,
            cluster_retime_settings(settings, last=last),
            weights,
        )
        calibrate_from_route(retimer, refined)
        if settings.earth_leg_continuous:
            retimer.protect_earth_leg(refined.plan)
        improvement = improve_and_certify(
            refined.plan,
            search,
            retimer,
            catalogue,
            scvx=scvx,
            max_attempts=settings.retime_attempts,
            max_rounds=settings.retime_rounds,
            time_budget_seconds=min(settings.retime_budget_seconds, max(remaining_budget(), 1.0)),
            pool=pool,
            refine=refine,
            # later slots harvest what the earlier ones left (all orphans, all positions)
            harvest=settings.collector_harvest and bool(pool.orphans()),
        )
        variants.extend(improvement.certified_routes)
        best = refined
        for route in improvement.certified_routes:
            if plan_value(route.plan, retimer) > plan_value(best.plan, retimer) + 1e-9:
                best = route
        ship_report["retiming"] = {
            "before_kg": refined.total_collected_kg,
            "after_kg": best.total_collected_kg,
            "attempts": len(improvement.attempts),
            "certified_variants": len(improvement.certified_routes),
            "wall_seconds": improvement.wall_seconds,
            "bans": {f"{a}->{b}": r for (a, b), r in sorted(retimer.bans.items())},
        }
        for record in improvement.attempts:
            if record.get("refined", {}).get("certified") is False:
                bundle.rejected.append(
                    {
                        "slot": slot,
                        "reason": "re-timed variant not certified",
                        "result": record.get("result"),
                        "failures": record["refined"].get("failures", [])[:2],
                    }
                )
        pool.register(best.plan, slot)
        used |= set(best.plan.asteroids)
        bundle.ships.append(BundleShip(slot, best, variants, ship_report))
        retimers[slot] = retimer
        searches[slot] = search
    bundle.earth_legs = earth_report
    if settings.collector_harvest and len(bundle.ships) >= 2 and remaining_budget() > 0.0:
        pool = _joint_harvest(bundle, retimers, refine, catalogue, search_settings) or pool
    _repair_orphans(bundle, pool, retimers, searches, refine)
    bundle.wall_seconds = time.perf_counter() - started
    bundle.peak_rss_mb = _peak_rss_mb()
    return bundle


def _joint_harvest(
    bundle: ClusterBundle,
    retimers: dict[int, Retimer],
    refine,
    catalogue: AsteroidCatalogue,
    search_settings: SearchSettings,
) -> MinerPool | None:
    """Re-plan the bundle's collect tours jointly over the pooled miners (``harvest`` module).

    Ships whose new tour certifies adopt it; the others keep their original route, and miners a
    kept ship still collects are removed from the new tours (one more re-timing/certification
    round).  The bundle adopts the result only if it collects at least as much as before and
    forms a consistent pool; returns that pool, or ``None`` when the original routes stay.
    """

    from .harvest import harvest_report, joint_collect_orders, retime_harvest

    plans = {ship.slot: ship.route.plan for ship in bundle.ships}
    states, uncollected = joint_collect_orders(catalogue, plans, search_settings)
    report = harvest_report(states, uncollected)
    report["ships_retimed"] = 0
    report["ships_certified"] = 0
    report["rounds"] = 0
    bundle.harvest = report
    # a ship's miners that another ship collects - in the new assignment or in an original route
    # a ship may keep - stay deployed at exactly their current epoch when the ship is re-timed
    collected_elsewhere: dict[int, set[int]] = {ship.slot: set() for ship in bundle.ships}
    for ship in bundle.ships:
        others = [s for s in bundle.ships if s.slot != ship.slot]
        for asteroid in plans[ship.slot].deploy_epochs:
            if any(
                asteroid in states[o.slot].collect_order or asteroid in o.route.plan.collect_epochs
                for o in others
            ):
                collected_elsewhere[ship.slot].add(asteroid)
    new_routes: dict[int, RefinedRoute | None] = {}
    for _round in range(3):
        report["rounds"] += 1
        pending = [
            ship
            for ship in bundle.ships
            if ship.slot not in new_routes
            and states[ship.slot].collect_order != orders_of(ship.route.plan)[1]
        ]
        if not pending:
            break
        for ship in pending:
            state = states[ship.slot]
            pinned = {
                a: e
                for a, e in ship.route.plan.deploy_epochs.items()
                if a in collected_elsewhere[ship.slot]
            }
            plan, dropped, failure = retime_harvest(
                ship.route.plan, state, retimers[ship.slot], pinned=pinned
            )
            state.collect_order = [a for a in state.collect_order if a not in dropped]
            for asteroid in dropped:
                state.foreign.pop(asteroid, None)
            if plan is None:
                new_routes[ship.slot] = None
                bundle.rejected.append(
                    {"slot": ship.slot, "reason": "harvest tour not re-timed", "failure": failure}
                )
                continue
            report["ships_retimed"] += 1
            route = refine(plan)
            if route.certified:
                report["ships_certified"] += 1
                new_routes[ship.slot] = route
            else:
                new_routes[ship.slot] = None
                bundle.rejected.append(
                    {
                        "slot": ship.slot,
                        "reason": "harvest tour not certified",
                        "failures": route.failures[:2],
                        "plan_collected_kg": plan.total_collected_kg,
                    }
                )
        # ships that keep their original route still collect their own miners: take those
        # miners out of the new tours and re-time the affected ships once more
        kept_collects = {
            a
            for ship in bundle.ships
            if new_routes.get(ship.slot, ship) is None
            for a in ship.route.plan.collect_epochs
        }
        for ship in bundle.ships:
            route = new_routes.get(ship.slot)
            if route is None:
                continue
            clash = [a for a in states[ship.slot].collect_order if a in kept_collects]
            if clash:
                states[ship.slot].collect_order = [
                    a for a in states[ship.slot].collect_order if a not in kept_collects
                ]
                for asteroid in clash:
                    states[ship.slot].foreign.pop(asteroid, None)
                del new_routes[ship.slot]  # re-timed in the next round
    for ship in bundle.ships:
        if ship.slot not in new_routes:
            new_routes[ship.slot] = None  # never re-timed (unchanged tour or clash unresolved)
    adopted = {slot: route for slot, route in new_routes.items() if route is not None}
    report["ships_adopted"] = sorted(adopted)
    if not adopted:
        return None
    before_kg = bundle.collected_kg
    original = {ship.slot: ship.route for ship in bundle.ships}
    for ship in bundle.ships:
        if ship.slot in adopted:
            ship.route = adopted[ship.slot]
            ship.variants.append(adopted[ship.slot])
    try:
        pool = bundle.pool()
    except ValueError as error:
        for ship in bundle.ships:
            ship.route = original[ship.slot]
        report["reverted"] = f"inconsistent: {error}"
        return None
    after_kg = bundle.collected_kg
    report["collected_before_kg"] = before_kg
    report["collected_after_kg"] = after_kg
    if after_kg < before_kg - 1e-9:
        for ship in bundle.ships:
            ship.route = original[ship.slot]
        report["reverted"] = "collects less than the self-cleaning routes"
        return None
    for ship in bundle.ships:
        if ship.slot in adopted:
            bundle.repairs.append(
                {
                    "kind": "joint_harvest",
                    "collector": ship.slot,
                    "foreign": sorted(ship.route.plan.foreign_deploy_epochs),
                    "collected_kg": ship.route.total_collected_kg,
                }
            )
    return pool


def _repair_orphans(
    bundle: ClusterBundle,
    pool: MinerPool,
    retimers: dict[int, Retimer],
    searches: dict[int, RouteSearch],
    refine,
) -> None:
    """Leave no orphan: offer each to every ship as a foreign collect, else drop the deploy."""

    for asteroid, deploy_epoch in sorted(pool.orphans().items()):
        deployer_slot = pool.deployed[asteroid][1]
        deployer = next(s for s in bundle.ships if s.slot == deployer_slot)
        if asteroid not in deployer.route.plan.deploy_epochs:
            continue  # already resolved by an earlier repair (the deployer reverted/dropped it)
        fixed = False
        # 1. a collector: any ship (deployer last) inserting the orphan into its collect tour
        order = [s for s in bundle.ships if s.slot != deployer_slot] + [
            s for s in bundle.ships if s.slot == deployer_slot
        ]
        for ship in order:
            retimer = retimers[ship.slot]
            single = MinerPool()
            single.deployed[asteroid] = (deploy_epoch, deployer_slot)
            variants, _failures = extend_plan(
                ship.route.plan,
                searches[ship.slot],
                retimer,
                candidates=0,
                pool=single,
                foreign_candidates=1,
            )
            for variant in variants:
                if variant.plan is None or asteroid not in variant.plan.collect_epochs:
                    continue
                route = refine(variant.plan)
                if route.certified:
                    ship.route = route
                    ship.variants.append(route)
                    fixed = True
                    bundle.repairs.append(
                        {
                            "asteroid": asteroid,
                            "deployer": deployer_slot,
                            "collector": ship.slot,
                            "kind": "foreign_collect",
                            "collected_kg": route.total_collected_kg,
                        }
                    )
                    break
                bundle.rejected.append(
                    {
                        "slot": ship.slot,
                        "reason": "orphan repair variant not certified",
                        "asteroid": asteroid,
                        "failures": route.failures[:2],
                    }
                )
            if fixed:
                break
        if fixed:
            continue
        # 2. the deployer drops the visit (re-timed and re-certified) - or reverts to its best
        #    certified variant without that deploy, whichever collects more.  Both are compared
        #    because a re-timed variant that speculated on orphans can, once they are dropped,
        #    fall below the plain chain it was meant to improve.
        # miners other ships of the bundle already collect from this deployer must stay deployed
        # at exactly their epoch, otherwise the repair strands a collector: the drop re-timing
        # pins them, and a fallback variant must reproduce them (dropping them is a last resort)
        required = {
            a: deployer.route.plan.deploy_epochs[a]
            for s in bundle.ships
            if s.slot != deployer_slot
            for a in s.route.plan.foreign_deploy_epochs
            if a in deployer.route.plan.deploy_epochs
        }
        dropped = drop_asteroid(
            deployer.route.plan, asteroid, retimers[deployer_slot], pinned=required
        )
        route = refine(dropped) if dropped is not None else None
        options: list[tuple[RefinedRoute, str]] = []
        if route is not None and route.certified:
            deployer.variants.append(route)
            options.append((route, "dropped"))
        elif route is not None:
            bundle.rejected.append(
                {
                    "slot": deployer_slot,
                    "reason": "dropped-visit variant not certified",
                    "asteroid": asteroid,
                    "failures": route.failures[:2],
                }
            )
        options.extend(
            (v, "reverted")
            for v in deployer.variants
            if asteroid not in v.plan.deploy_epochs
            and not v.plan.orphaned
            and not v.plan.foreign_deploy_epochs
        )
        keeping = [
            item
            for item in options
            if all(
                a in item[0].plan.deploy_epochs
                and abs(item[0].plan.deploy_epochs[a] - e) <= EPOCH_TOLERANCE_DAYS
                for a, e in required.items()
            )
        ]
        if keeping:
            options = keeping
        if not options:
            bundle.repairs.append(
                {"asteroid": asteroid, "deployer": deployer_slot, "kind": "unrepaired"}
            )
            continue
        deployer.route, kind = max(options, key=lambda item: item[0].total_collected_kg)
        bundle.repairs.append(
            {
                "asteroid": asteroid,
                "deployer": deployer_slot,
                "kind": kind,
                "collected_kg": deployer.route.total_collected_kg,
            }
        )
    make_consistent(bundle)


def make_consistent(bundle: ClusterBundle) -> None:
    """Make the emitted routes mutually consistent, giving up single ships rather than the bundle.

    Dropping a deployer's visit or reverting a deployer to another variant can strand a
    collector's foreign collect (the miner is no longer deployed, or was re-timed to another
    epoch); two ships may also end up deploying or collecting the same asteroid.  Stranded
    collectors revert to their best clean variant (no foreign collects, no orphans) or leave the
    bundle; any remaining pool conflict removes the lightest ship touching the asteroid the pool
    complains about.  Only a bundle that stays inconsistent after that is discarded.
    """

    for _ in range(len(bundle.ships) + 1):
        deployed = {a: e for s in bundle.ships for a, e in s.route.plan.deploy_epochs.items()}
        stranded_ships = [
            (
                ship,
                [
                    a
                    for a, e in ship.route.plan.foreign_deploy_epochs.items()
                    if a not in deployed or abs(deployed[a] - e) > 1e-6
                ],
            )
            for ship in list(bundle.ships)
        ]
        stranded_ships = [(ship, items) for ship, items in stranded_ships if items]
        for ship, stranded in stranded_ships:
            clean = [
                v for v in ship.variants if not v.plan.foreign_deploy_epochs and not v.plan.orphaned
            ]
            if clean:
                ship.route = max(clean, key=lambda v: v.total_collected_kg)
                bundle.repairs.append(
                    {"collector": ship.slot, "kind": "reverted_stranded", "asteroids": stranded}
                )
            else:
                bundle.ships.remove(ship)
                bundle.repairs.append(
                    {"collector": ship.slot, "kind": "removed_stranded", "asteroids": stranded}
                )
        if stranded_ships:
            continue
        problem = bundle.consistent()
        if not problem:
            return
        match = re.search(r"asteroid (\d+)", problem)
        asteroid = int(match.group(1)) if match else None
        involved = [
            s for s in bundle.ships if asteroid is None or asteroid in s.route.plan.asteroids
        ]
        if not involved:
            break
        victim = min(involved, key=lambda s: (s.route.total_collected_kg, s.slot))
        bundle.ships.remove(victim)
        bundle.repairs.append(
            {"collector": victim.slot, "kind": "removed_conflict", "detail": problem}
        )
    # last resort: a bundle the pool rejects is not emitted at all (the master would trip on it)
    problem = bundle.consistent()
    if problem:
        bundle.rejected.append({"reason": "inconsistent bundle", "detail": problem})
        bundle.repairs.append({"kind": "bundle_discarded", "detail": problem})
        bundle.ships.clear()


# -- clusters and parallel pricing -------------------------------------------------------------


def family_clusters(
    catalogue: AsteroidCatalogue,
    ids: IntArray,
    *,
    bands: ClusterBands | None = None,
    min_members: int = 12,
    excluded: set[int] | frozenset[int] | None = None,
) -> list[tuple[int, IntArray]]:
    """Co-moving families of the pool, largest first, as ``(label, member ids)``."""

    banned = set(excluded or ())
    pool = np.asarray([a for a in ids if int(a) not in banned], dtype=np.int64)
    clusters = ComovingClusters(catalogue, pool, bands or ClusterBands())
    sizes = np.bincount(clusters.labels[clusters.labels >= 0])
    order = np.argsort(-sizes, kind="stable")
    return [
        (int(label), clusters.cluster_members(int(label)))
        for label in order
        if sizes[label] >= min_members
    ]


def rank_families(
    catalogue: AsteroidCatalogue,
    families: list[tuple[int, IntArray]],
    settings: SearchSettings | None = None,
    *,
    top: int = 5,
    visit_epochs: Sequence[float] | None = None,
    cheap_hop_kg: float = 75.0,
) -> list[tuple[int, IntArray, dict[str, float]]]:
    """Order families by how cheaply a ship can work them (best first).

    Score = mean of the ``top`` cheapest Lambert Earth legs over the launch grid (kg of
    propellant at 3000 kg) + the family's mean nearest-neighbour hop proxy (kg at 2000 kg),
    the hop proxy averaged over the *visit epochs* (deploy phase at year 3 and collect phase at
    year 10, ``ClusterBands.visit_epochs``) so a family whose members drift apart by the collect
    phase is priced as the expensive family it is.  ``cheap_pair_fraction`` reports the share of
    nearest-neighbour hops at or under ``cheap_hop_kg`` (the references' 75 kg).  Size alone is a
    poor guide: the largest families of the box are often the eccentric or inclined ones whose
    Earth legs cost 600 kg and whose hops cost 1.3 km/s.  Deterministic: ties break on the label.
    """

    settings = settings or SearchSettings()
    epochs = np.asarray(settings.launch_epochs)
    tofs = np.asarray(settings.earth_leg_tofs)
    visit_epochs = tuple(visit_epochs or ClusterBands().phase_epochs)
    ranked: list[tuple[int, IntArray, dict[str, float]]] = []
    for label, members in families:
        members = np.asarray(sorted(int(a) for a in members), dtype=np.int64)
        grid = screen_earth_to_asteroids(catalogue, members, epochs, tofs)
        dv = np.where(grid["feasible"], grid["total_delta_v"], np.inf)
        best_per_asteroid = np.sort(dv.reshape(members.shape[0], -1).min(axis=1))
        best = best_per_asteroid[np.isfinite(best_per_asteroid)][:top]
        if best.size == 0:
            earth_kg = float("inf")
        else:
            earth_kg = float(np.mean(propellant_for_delta_v(settings.initial_mass, best)))
        # internal hops: nearest-neighbour phasing proxy at every visit epoch
        hop_tofs = np.asarray(settings.hop_tofs)
        nearest: list[float] = []
        per_epoch_kg: list[float] = []
        for epoch in visit_epochs:
            at_epoch: list[float] = []
            for source in members.tolist():
                others = members[members != source]
                if others.size == 0:
                    continue
                proxy = phasing_edelbaum_proxy(catalogue, source, others, epoch, hop_tofs)
                at_epoch.extend(np.sort(proxy["best_delta_v"])[: min(3, others.size)].tolist())
            nearest.extend(at_epoch)
            per_epoch_kg.append(
                float(np.mean(propellant_for_delta_v(2000.0, np.asarray(at_epoch))))
                if at_epoch
                else float("inf")
            )
        if nearest:
            hop_kg_all = propellant_for_delta_v(2000.0, np.asarray(nearest))
            hop_kg = float(np.mean(hop_kg_all))
            cheap_fraction = float(np.mean(hop_kg_all <= cheap_hop_kg))
        else:
            hop_kg, cheap_fraction = float("inf"), 0.0
        stats = {
            "members": float(members.shape[0]),
            "earth_leg_kg": earth_kg,
            "hop_kg": hop_kg,
            "hop_kg_by_epoch": per_epoch_kg,
            "cheap_pair_fraction": cheap_fraction,
            "score": earth_kg + 4.0 * hop_kg,  # a chain flies ~4 hops per Earth leg
        }
        ranked.append((int(label), members, stats))
    ranked.sort(key=lambda item: (item[2]["score"], item[0]))
    return ranked


_WORKER: dict[str, Any] = {}


def _price_in_worker(task: tuple[int, list[int]]) -> ClusterBundle:
    label, members = task
    state = _WORKER
    if "catalogue" not in state:  # spawned (not forked) worker: load lazily
        from .data import load_catalogue

        state["catalogue"] = load_catalogue()
    try:
        return price_cluster(
            state["catalogue"],
            np.asarray(members, dtype=np.int64),
            label=label,
            excluded=state.get("excluded"),
            weights=state.get("weights"),
            settings=state.get("settings"),
            scvx=state.get("scvx"),
        )
    except Exception as error:  # one family's crash must not end the campaign
        import traceback

        bundle = ClusterBundle(label, tuple(int(a) for a in members), [])
        bundle.stopped = f"crashed: {error!r}"
        bundle.rejected.append(
            {"reason": "worker exception", "traceback": traceback.format_exc()[-2000:]}
        )
        return bundle


def price_clusters(
    catalogue: AsteroidCatalogue,
    clusters: list[tuple[int, IntArray]],
    *,
    settings: ClusterPricingSettings | None = None,
    scvx: ScvxSettings | None = None,
    weights: dict[int, float] | None = None,
    excluded: set[int] | None = None,
    workers: int = 2,
    on_result=None,
    budget_seconds: float = float("inf"),
):
    """Price many families in forked worker processes (``workers`` at a time).

    Results are delivered to ``on_result(bundle)`` as they complete and returned in cluster
    order.  A worker's memory is bounded by one family's pricing (the per-family search,
    re-timer and SCvx objects are released when the task returns).
    """

    started = time.perf_counter()
    settings = settings or ClusterPricingSettings()
    _WORKER.update(
        catalogue=catalogue, excluded=excluded, weights=weights, settings=settings, scvx=scvx
    )
    tasks = [(int(label), [int(a) for a in members]) for label, members in clusters]
    results: dict[int, ClusterBundle] = {}
    if workers <= 1:
        for task in tasks:
            if time.perf_counter() - started > budget_seconds:
                break
            bundle = _price_in_worker(task)
            results[task[0]] = bundle
            if on_result is not None:
                on_result(bundle)
        return [results[label] for label, _ in tasks if label in results]
    # fork: the workers inherit the loaded catalogue and settings copy-on-write (a spawned
    # worker loads the catalogue itself, see ``_price_in_worker``)
    context = multiprocessing.get_context("fork" if os.name == "posix" else "spawn")
    queue = list(tasks)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as executor:
        # submit as workers free up (not all at once) so the budget stops new families promptly
        running: dict[concurrent.futures.Future, int] = {}
        while queue or running:
            while (
                queue and len(running) < workers and time.perf_counter() - started <= budget_seconds
            ):
                task = queue.pop(0)
                running[executor.submit(_price_in_worker, task)] = task[0]
            if not running:
                break
            done, _ = concurrent.futures.wait(
                running, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in sorted(done, key=lambda item: running[item]):
                label = running.pop(future)
                bundle = future.result()
                results[label] = bundle
                if on_result is not None:
                    on_result(bundle)
            if time.perf_counter() - started > budget_seconds:
                queue.clear()
    return [results[label] for label, _ in tasks if label in results]


def bundle_settings_summary(settings: ClusterPricingSettings) -> dict[str, Any]:
    return asdict(settings)


def bundle_columns(bundle: ClusterBundle, start: int, prefix: str = "f") -> list[FleetColumn]:
    """The master columns of one priced bundle, numbered from ``start``.

    Every emitted ship route is a column; every *stand-alone* certified variant (no orphans, no
    foreign collects) is a column of its own; and when the bundle has several ships they also
    form one multi-ship bundle column so their cooperative collects can be selected together.
    """

    columns: list[FleetColumn] = []
    members: list[FleetColumn] = []
    label = f"{prefix}{bundle.label}"
    for ship in bundle.ships:
        column = FleetColumn.from_plan(
            start + len(columns),
            ship.slot,
            f"{label}_s{ship.slot}",
            ship.route.plan,
            ship.route.collected_mass,
            certified=ship.route.certified,
            route=ship.route,
        )
        columns.append(column)
        members.append(column)
        for index, variant in enumerate(ship.variants):
            standalone = not variant.plan.orphaned and not variant.plan.foreign_deploy_epochs
            if variant is ship.route or not standalone or not variant.certified:
                continue
            columns.append(
                FleetColumn.from_plan(
                    start + len(columns),
                    ship.slot,
                    f"{label}_s{ship.slot}_v{index}",
                    variant.plan,
                    variant.collected_mass,
                    certified=variant.certified,
                    route=variant,
                )
            )
    if len(members) > 1:
        columns.append(FleetColumn.from_bundle(start + len(columns), label, members))
    return columns
