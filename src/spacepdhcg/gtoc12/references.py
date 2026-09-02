"""Decode GTOC12 solution files into per-ship itineraries and structural statistics.

Used on the archived JPL/Antipodes references to learn what a ~740 kg ship looks like (asteroid
count, deploy/collect phasing, hop durations and propellant, revolutions, cooperation between
ships, launch spread) and to turn those observations into search constraints.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import constants as C
from .data import AsteroidCatalogue
from .ephemeris import earth_state
from .solution import EARTH_ID_EVENT, Solution


@dataclass(slots=True)
class LegInfo:
    ship_id: int
    from_body: int  # 0 Earth, -2/-3/-4 planets, >0 asteroid
    to_body: int
    departure_epoch: float
    arrival_epoch: float
    mass_departure: float
    mass_arrival: float
    burn_days: float
    transfer_angle_rad: float
    revolutions: float
    role: str  # earth_out | deploy_hop | collect_hop | earth_return | flyby_leg | mixed

    @property
    def tof_days(self) -> float:
        return self.arrival_epoch - self.departure_epoch

    @property
    def propellant_kg(self) -> float:
        return self.mass_departure - self.mass_arrival


@dataclass(slots=True)
class ShipItinerary:
    ship_id: int
    launch_epoch: float
    launch_mass: float
    launch_vinf_km_s: float
    final_epoch: float
    final_mass: float
    deploys: list[tuple[int, float]]
    collects: list[tuple[int, float, float]]  # asteroid, epoch, mass
    unloads: list[tuple[float, float]]  # epoch, mass
    legs: list[LegInfo] = field(default_factory=list)
    flybys: int = 0

    @property
    def collected_mass_kg(self) -> float:
        return sum(mass for _, _, mass in self.collects)

    @property
    def unloaded_mass_kg(self) -> float:
        return sum(mass for _, mass in self.unloads)

    @property
    def deploy_phase_days(self) -> float:
        return (max(e for _, e in self.deploys) - self.launch_epoch) if self.deploys else 0.0

    @property
    def collect_phase_days(self) -> float:
        if not self.collects:
            return 0.0
        return self.final_epoch - min(e for _, e, _ in self.collects)


def _angle(position: np.ndarray) -> float:
    return math.atan2(position[1], position[0])


def decode_itineraries(solution: Solution, catalogue: AsteroidCatalogue) -> list[ShipItinerary]:
    """Per-ship itineraries; deploy/collect roles come from the global visit order per asteroid."""

    visits: dict[int, list[tuple[float, int]]] = {}
    for ship in solution.ships:
        for event in ship.events:
            if event.is_asteroid:
                visits.setdefault(event.event_id, []).append((event.epoch, ship.ship_id))
    for items in visits.values():
        items.sort()
    itineraries: list[ShipItinerary] = []
    for ship in solution.ships:
        events = ship.events
        launch = events[0]
        _, v_earth = earth_state(launch.epoch)
        itinerary = ShipItinerary(
            ship.ship_id,
            launch.epoch,
            launch.before.mass,
            float(np.linalg.norm(launch.after.velocity - v_earth)),
            events[-1].epoch,
            events[-1].after.mass,
            [],
            [],
            [],
        )
        burn_days_by_leg: list[float] = []
        pending_burn = 0.0
        item_index = 0
        for item in ship.items:
            if not hasattr(item, "event_id"):
                pending_burn += float(
                    np.sum(np.diff(np.asarray([s.epoch for s in item.samples[1:-1]])))
                )
                continue
            if item_index > 0:
                burn_days_by_leg.append(pending_burn)
            pending_burn = 0.0
            item_index += 1
        for previous, event in itertools.pairwise(events):
            role = "flyby_leg"
            if event.is_asteroid:
                order = [ship_id for _, ship_id in visits[event.event_id]]
                first = visits[event.event_id][0][0] == event.epoch
                role = "deploy_hop" if first else "collect_hop"
                if previous.event_id == EARTH_ID_EVENT:
                    role = "earth_out"
                if first:
                    itinerary.deploys.append((event.event_id, event.epoch))
                else:
                    itinerary.collects.append(
                        (event.event_id, event.epoch, event.after.mass - event.before.mass)
                    )
                del order
            elif event.event_id == C.EVENT_EARTH_FLYBY:
                dropped = previous_mass = event.before.mass - event.after.mass
                if dropped > C.TOLERANCE_MASS_KG:
                    itinerary.unloads.append((event.epoch, dropped))
                    role = "earth_return"
                else:
                    itinerary.flybys += 1
                del previous_mass
            else:
                itinerary.flybys += 1
            r0 = previous.after.position
            r1 = event.before.position
            angle = math.atan2(float(np.linalg.norm(np.cross(r0, r1))), float(np.dot(r0, r1)))
            # revolutions from the mean motion of the two end orbits over the flight time
            radii = 0.5 * (np.linalg.norm(r0) + np.linalg.norm(r1))
            mean_motion = math.sqrt(C.MU_SUN_KM3_S2 / radii**3)
            swept = mean_motion * (event.epoch - previous.epoch) * C.DAY_S
            revolutions = max(0.0, (swept - angle) / (2.0 * math.pi))
            leg_index = len(itinerary.legs)
            itinerary.legs.append(
                LegInfo(
                    ship.ship_id,
                    previous.event_id,
                    event.event_id,
                    previous.epoch,
                    event.epoch,
                    previous.after.mass,
                    event.before.mass,
                    burn_days_by_leg[leg_index] if leg_index < len(burn_days_by_leg) else 0.0,
                    angle,
                    revolutions,
                    role,
                )
            )
        itineraries.append(itinerary)
    return itineraries


def element_spread(catalogue: AsteroidCatalogue, asteroid_ids: list[int]) -> dict[str, float]:
    index = catalogue.index_of(np.asarray(asteroid_ids, dtype=np.int64))
    a_au = catalogue.semi_major_axis_km[index] / C.AU_KM
    return {
        "a_min_au": float(a_au.min()),
        "a_max_au": float(a_au.max()),
        "a_spread_au": float(a_au.max() - a_au.min()),
        "e_max": float(catalogue.eccentricity[index].max()),
        "i_max_deg": float(np.rad2deg(catalogue.inclination_rad[index]).max()),
        "i_spread_deg": float(np.rad2deg(np.ptp(catalogue.inclination_rad[index]))),
        "node_spread_deg": float(
            np.rad2deg(
                np.ptp(
                    np.angle(
                        np.exp(
                            1j
                            * (
                                catalogue.ascending_node_rad[index]
                                - catalogue.ascending_node_rad[index][0]
                            )
                        )
                    )
                )
            )
        ),
    }


def summarise(itineraries: list[ShipItinerary], catalogue: AsteroidCatalogue) -> dict[str, Any]:
    """Fleet-level structural statistics used to set search constraints."""

    hops = [
        leg for it in itineraries for leg in it.legs if leg.role in {"deploy_hop", "collect_hop"}
    ]
    outs = [leg for it in itineraries for leg in it.legs if leg.role == "earth_out"]
    returns = [leg for it in itineraries for leg in it.legs if leg.role == "earth_return"]
    deployed = {a for it in itineraries for a, _ in it.deploys}
    collected = {a for it in itineraries for a, _, _ in it.collects}
    self_cleaning = 0
    cooperative = 0
    for it in itineraries:
        own = {a for a, _ in it.deploys}
        for a, _, _ in it.collects:
            if a in own:
                self_cleaning += 1
            else:
                cooperative += 1

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        arr = np.asarray(values)
        return {
            "min": float(arr.min()),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }

    per_ship_spread = [
        element_spread(catalogue, [a for a, _ in it.deploys]) for it in itineraries if it.deploys
    ]
    return {
        "ships": len(itineraries),
        "asteroids_deployed": len(deployed),
        "asteroids_collected": len(collected),
        "collected_mass_kg": sum(it.collected_mass_kg for it in itineraries),
        "unloaded_mass_kg": sum(it.unloaded_mass_kg for it in itineraries),
        "collections_self_cleaning": self_cleaning,
        "collections_cooperative": cooperative,
        "per_ship_asteroids": stats([float(len(it.deploys)) for it in itineraries]),
        "per_ship_collected_kg": stats([it.collected_mass_kg for it in itineraries]),
        "per_ship_launch_mass_kg": stats([it.launch_mass for it in itineraries]),
        "per_ship_final_mass_kg": stats([it.final_mass for it in itineraries]),
        "launch_epoch_mjd": stats([it.launch_epoch for it in itineraries]),
        "launch_vinf_km_s": stats([it.launch_vinf_km_s for it in itineraries]),
        "deploy_phase_days": stats([it.deploy_phase_days for it in itineraries]),
        "collect_phase_days": stats([it.collect_phase_days for it in itineraries]),
        "flybys": sum(it.flybys for it in itineraries),
        "earth_out_tof_days": stats([leg.tof_days for leg in outs]),
        "earth_out_propellant_kg": stats([leg.propellant_kg for leg in outs]),
        "earth_out_revolutions": stats([leg.revolutions for leg in outs]),
        "hop_tof_days": stats([leg.tof_days for leg in hops]),
        "hop_propellant_kg": stats([leg.propellant_kg for leg in hops]),
        "hop_burn_fraction": stats(
            [leg.burn_days / leg.tof_days for leg in hops if leg.tof_days > 0]
        ),
        "hop_transfer_angle_deg": stats([math.degrees(leg.transfer_angle_rad) for leg in hops]),
        "hop_revolutions": stats([leg.revolutions for leg in hops]),
        "return_tof_days": stats([leg.tof_days for leg in returns]),
        "return_propellant_kg": stats([leg.propellant_kg for leg in returns]),
        "per_ship_a_spread_au": stats([s["a_spread_au"] for s in per_ship_spread]),
        "per_ship_i_spread_deg": stats([s["i_spread_deg"] for s in per_ship_spread]),
        "per_ship_i_max_deg": stats([s["i_max_deg"] for s in per_ship_spread]),
        "per_ship_e_max": stats([s["e_max"] for s in per_ship_spread]),
        "per_ship_node_spread_deg": stats([s["node_spread_deg"] for s in per_ship_spread]),
        "stay_days": stats(
            [
                c_epoch - dict(it.deploys).get(a, c_epoch)
                for it in itineraries
                for a, c_epoch, _ in it.collects
                if a in dict(it.deploys)
            ]
        ),
    }


def decode_file(path: str | Path, catalogue: AsteroidCatalogue) -> dict[str, Any]:
    solution = Solution.read(path)
    itineraries = decode_itineraries(solution, catalogue)
    return {
        "file": str(path),
        "summary": summarise(itineraries, catalogue),
        "ships": [
            {
                **{k: v for k, v in asdict(it).items() if k != "legs"},
                "collected_mass_kg": it.collected_mass_kg,
                "legs": [
                    asdict(leg) | {"tof_days": leg.tof_days, "propellant_kg": leg.propellant_kg}
                    for leg in it.legs
                ],
            }
            for it in itineraries
        ],
    }
