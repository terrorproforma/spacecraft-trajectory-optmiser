"""Joint collect-phase re-sequencing over the pooled miners of a cluster bundle.

Why this exists.  A self-cleaning ship collects the miners it deployed, so its collect hops are
the deploy pairs traversed three years later.  Those pairs were cheap *at deploy time*; the
relative phase drift inside a family (a few degrees per year) makes the same pairs cost 2-3x
more at collection (family 0 probe: 5441 -> 57635 flew at 1.29 km/s on the way out and 3.25 km/s
on the way back, 23907 -> 16356 at 2.29 vs 4.95 km/s) - and traversing them in either direction
does not help.  Over the *pooled* miners of the bundle, however, nearest-neighbour collect chains
at the collect epochs stay at 1.3-2.0 km/s (median 1.6-1.9 km/s vs 2.5-2.7 km/s within one
ship's own set), which is the 66-75 kg hop level the references achieve with cooperative
collection.

What it does.  Given the certified (self-cleaning) routes of a bundle, keep every ship's deploy
chain and re-plan the collect tours jointly: a deterministic multi-ship nearest-neighbour
construction over the pooled miners (each miner collected by exactly one ship, camps first,
Lambert ΔV at the running collect epoch as the cost), then the per-ship DP re-timer
(`Retimer.retime_order`) prices and times each new order with the foreign deploy epochs, and
SCvx certifies it.  Ships whose new tour does not certify keep their original route; miners
that then have two collectors are removed from the new tours and those are re-timed once more;
whatever is left uncollected goes to the bundle's orphan repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state
from .retiming import Retimer, orders_of
from .screening import lambert_hops, propellant_for_delta_v
from .search import RoutePlan, SearchSettings

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HarvestSettings:
    collect_tofs: tuple[float, ...] = (180.0, 240.0, 300.0, 360.0, 420.0, 480.0)
    # a ship stops taking miners when its next arrival would be later than this before the end
    return_reserve_days: float = 900.0 + 60.0
    # a miner deployed later than the ship's arrival + this cannot be collected without a long
    # camp: skip it for this ship at this step
    max_camp_days: float = 240.0
    # collects per ship at most (own + foreign); ``None`` = as many as time allows
    max_collects: int | None = None
    # every ship must collect at least its camp, else it keeps its original tour
    min_collects: int = 2
    # nearest-neighbour cost = propellant of the cheapest Lambert TOF + this x hop days
    time_weight_kg_per_day: float = 0.05


@dataclass(slots=True)
class ShipState:
    slot: int
    deploy_order: list[int]
    location: int
    epoch: float  # earliest departure from ``location``
    mass: float
    collect_order: list[int] = field(default_factory=list)
    foreign: dict[int, float] = field(default_factory=dict)
    active: bool = True
    stop_reason: str = ""


def _cheapest_hop(
    catalogue: AsteroidCatalogue,
    source: int,
    target: int,
    departure: float,
    tofs: FloatArray,
) -> tuple[float, float]:
    """(ΔV km/s, TOF days) of the cheapest zero-revolution Lambert hop over the TOF grid."""

    n = tofs.shape[0]
    r_s, v_s = asteroid_state(catalogue, np.full(n, source), np.full(n, departure))
    r_t, v_t = asteroid_state(catalogue, np.full(n, target), departure + tofs)
    hop = lambert_hops(r_s, v_s, r_t, v_t, np.full(n, departure), tofs)
    dv = np.where(hop.feasible & np.isfinite(hop.total_delta_v), hop.total_delta_v, np.inf)
    k = int(np.argmin(dv))
    return float(dv[k]), float(tofs[k])


def joint_collect_orders(
    catalogue: AsteroidCatalogue,
    plans: dict[int, RoutePlan],
    search_settings: SearchSettings,
    settings: HarvestSettings | None = None,
) -> tuple[dict[int, ShipState], list[int]]:
    """Deterministic multi-ship nearest-neighbour collect tours over the pooled miners.

    Every ship starts at its camp (last deployed asteroid, collected first as usual).  The ship
    with the earliest departure epoch moves next: it takes the unassigned miner with the lowest
    ``propellant(cheapest Lambert TOF) + time_weight x TOF`` from its location, provided the
    miner's minimum stay can be honoured without camping longer than ``max_camp_days`` and the
    arrival leaves the return reserve.  Ties break on asteroid id.  Returns the per-ship states
    and the miners nobody collects (sorted).
    """

    settings = settings or HarvestSettings()
    tofs = np.asarray(settings.collect_tofs, dtype=np.float64)
    end = C.MISSION_END_MJD - search_settings.end_margin_days
    pool: dict[int, tuple[float, int]] = {}
    for slot, plan in sorted(plans.items()):
        for asteroid, epoch in plan.deploy_epochs.items():
            pool[asteroid] = (epoch, slot)
    ships: dict[int, ShipState] = {}
    for slot, plan in sorted(plans.items()):
        deploy_order, _ = orders_of(plan)
        camp = deploy_order[-1]
        # mass after the deploy phase under the plan's own proxies
        mass = search_settings.initial_mass
        for leg in plan.legs:
            if leg.role not in ("earth_out", "deploy_hop"):
                break
            mass -= float(propellant_for_delta_v(mass, leg.delta_v_proxy_km_s * leg.inflation))
            mass -= C.MINER_MASS_KG
        ships[slot] = ShipState(slot, deploy_order, camp, plan.deploy_epochs[camp], mass, [camp])
    assigned = {state.location for state in ships.values()}
    min_stay = C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS
    while True:
        active = [s for s in ships.values() if s.active]
        if not active:
            break
        ship = min(active, key=lambda s: (s.epoch, s.slot))
        if settings.max_collects is not None and len(ship.collect_order) >= settings.max_collects:
            ship.active, ship.stop_reason = False, "max_collects"
            continue
        # the ship collects ``location`` when it leaves: not before the miner's minimum stay
        deploy_epoch = pool[ship.location][0]
        departure = max(ship.epoch, deploy_epoch + min_stay)
        best: tuple[float, int, float, float] | None = None
        for asteroid in sorted(pool):
            if asteroid in assigned:
                continue
            dv, tof = _cheapest_hop(catalogue, ship.location, asteroid, departure, tofs)
            if not np.isfinite(dv):
                continue
            arrival = departure + tof
            if arrival > end - settings.return_reserve_days:
                continue
            ready = pool[asteroid][0] + min_stay  # earliest collection at the target
            if ready - arrival > settings.max_camp_days:
                continue
            cost = float(propellant_for_delta_v(ship.mass, dv * search_settings.hop_inflation))
            cost += settings.time_weight_kg_per_day * tof
            if best is None or cost < best[0] - 1e-12:
                best = (cost, asteroid, dv, tof)
        if best is None:
            ship.active, ship.stop_reason = False, "no_reachable_miner"
            continue
        cost, asteroid, dv, tof = best
        ship.mass -= float(propellant_for_delta_v(ship.mass, dv * search_settings.hop_inflation))
        ship.mass += C.maximum_collected_mass(departure - deploy_epoch)
        ship.epoch = departure + tof
        ship.location = asteroid
        ship.collect_order.append(asteroid)
        if pool[asteroid][1] != ship.slot:
            ship.foreign[asteroid] = pool[asteroid][0]
        assigned.add(asteroid)
        if len(assigned) == len(pool):
            for state in ships.values():
                if state.active:
                    state.active, state.stop_reason = False, "pool_exhausted"
    uncollected = sorted(a for a in pool if a not in assigned)
    return ships, uncollected


def retime_harvest(
    plan: RoutePlan,
    state: ShipState,
    retimer: Retimer,
    *,
    drop_tail: int = 3,
    pinned: dict[int, float] | None = None,
) -> tuple[RoutePlan | None, list[int], str]:
    """DP re-timing of the ship's deploy chain with its new collect order.

    When the order does not close (mass or time), the last collects are dropped one at a time
    (up to ``drop_tail``) - the dropped miners return to the pool as orphans.  ``pinned`` deploys
    (this ship's miners another ship collects) keep their exact epoch.  Returns the plan, the
    dropped miners and the last failure text.
    """

    from .bundles import profile_for_orders

    collect_order = list(state.collect_order)
    dropped: list[int] = []
    failure = ""
    for _ in range(drop_tail + 1):
        foreign = {a: e for a, e in state.foreign.items() if a in collect_order}
        try:
            profile = profile_for_orders(plan, retimer, state.deploy_order, collect_order, foreign)
        except ValueError as error:
            return None, dropped, f"invalid_order: {error}"
        result = retimer.retime_order(
            state.deploy_order,
            collect_order,
            profile,
            before=-np.inf,
            original=plan,
            foreign=foreign,
            pinned=pinned,
        )
        if result.plan is not None and result.plan.feasible:
            return result.plan, dropped, ""
        failure = result.failure or "not_feasible"
        if len(collect_order) <= 1:
            break
        dropped.append(collect_order.pop())
    return None, dropped, failure


def harvest_report(states: dict[int, ShipState], uncollected: list[int]) -> dict[str, Any]:
    return {
        "ships": {
            str(slot): {
                "collects": list(state.collect_order),
                "foreign": sorted(state.foreign),
                "stop": state.stop_reason,
            }
            for slot, state in sorted(states.items())
        },
        "uncollected": list(uncollected),
        "foreign_collects": sum(len(state.foreign) for state in states.values()),
    }
