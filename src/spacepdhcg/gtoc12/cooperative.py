"""Cooperative collection and the fleet master for GTOC12.

Miners are shared resources: a ship may collect a miner another ship deployed (as the JPL and
Antipodes solutions do) and may leave its own miners for another ship.  Two objects carry that:

* :class:`MinerPool` - the fleet-level registry of deployed miners (asteroid, deploy epoch,
  deploying ship) and of who collected them, used while ships are built one after another so
  ship ``k`` can price collecting the *orphans* of ships ``1..k-1``.
* :func:`solve_fleet_master` - the G7-style master over certified itinerary columns.  It picks a
  subset of columns maximising the fixed-bonus score ``sum_i B_i M_i`` subject to: every asteroid
  is deployed on at most once and collected at most once; a foreign collect needs the column that
  deploys that miner (same asteroid, same epoch) in the fleet; the ship-count rule
  ``N <= min(100, 2 exp(rho * mean collected mass))``; and an optional ship cap.  Columns of the
  same ship slot exclude each other automatically because they share asteroids.  The search is an
  exact depth-first branch and bound with a node cap so it is deterministic and bounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import constants as C
from .search import RoutePlan

EPOCH_TOLERANCE_DAYS = 1e-6


@dataclass(slots=True)
class MinerPool:
    """Fleet-level registry of deployed and collected miners."""

    deployed: dict[int, tuple[float, int]] = field(default_factory=dict)  # a -> (epoch, ship)
    collected: dict[int, int] = field(default_factory=dict)  # asteroid -> collecting ship

    def register(self, plan: RoutePlan, ship: int) -> None:
        """Record a certified plan's deploys and collects (each asteroid once per role)."""

        # validate everything before mutating so a rejected plan leaves the pool untouched
        for asteroid in plan.deploy_epochs:
            if asteroid in self.deployed:
                raise ValueError(f"asteroid {asteroid} deployed twice")
        for asteroid in plan.collect_epochs:
            if asteroid in self.collected:
                raise ValueError(f"asteroid {asteroid} collected twice")
            if asteroid in plan.deploy_epochs:
                continue
            if asteroid not in self.deployed:
                raise ValueError(f"asteroid {asteroid} collected but never deployed")
            if abs(plan.deploy_epoch_of(asteroid) - self.deployed[asteroid][0]) > (
                EPOCH_TOLERANCE_DAYS
            ):
                raise ValueError(f"asteroid {asteroid} collected against a stale deploy epoch")
        for asteroid, epoch in plan.deploy_epochs.items():
            self.deployed[asteroid] = (float(epoch), ship)
        for asteroid in plan.collect_epochs:
            self.collected[asteroid] = ship

    def orphans(self) -> dict[int, float]:
        """Deployed-but-uncollected miners: asteroid -> deploy epoch (sorted by asteroid)."""

        return {
            asteroid: epoch
            for asteroid, (epoch, _ship) in sorted(self.deployed.items())
            if asteroid not in self.collected
        }

    def touched(self) -> set[int]:
        return set(self.deployed) | set(self.collected)

    def summary(self) -> dict[str, Any]:
        orphans = self.orphans()
        return {
            "deployed": len(self.deployed),
            "collected": len(self.collected),
            "orphans": sorted(orphans),
            "cooperative_collects": sorted(
                a for a, ship in self.collected.items() if self.deployed[a][1] != ship
            ),
        }


def orphan_credit_kg(
    plan: RoutePlan, weights: dict[int, float] | None, credit: float, margin_days: float
) -> float:
    """Fleet value credited for miners the plan leaves for another ship.

    Another ship can at best collect ``k (T_end - margin - t_deploy)`` from each orphan; the
    credit factor discounts that for the collector's extra hop and the risk nobody collects it.
    """

    if credit <= 0.0 or not plan.orphaned:
        return 0.0
    total = 0.0
    for asteroid in plan.orphaned:
        weight = 1.0 if weights is None else weights.get(asteroid, 1.0)
        stay = C.MISSION_END_MJD - margin_days - plan.deploy_epochs[asteroid]
        if stay >= C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS:
            total += weight * C.maximum_collected_mass(stay)
    return credit * total


# -- master --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FleetColumn:
    """One certified ship itinerary offered to the master."""

    identifier: int
    slot: int  # ship slot the column was generated for (report only)
    label: str
    deploys: dict[int, float]  # asteroid -> deploy epoch
    collects: dict[int, float]  # asteroid -> collect epoch
    foreign: dict[int, float]  # asteroid -> deploy epoch required from another column
    collected_mass: dict[int, float]
    certified: bool
    route: Any = None  # RefinedRoute (opaque to the master)

    @classmethod
    def from_plan(
        cls,
        identifier: int,
        slot: int,
        label: str,
        plan: RoutePlan,
        collected_mass: dict[int, float],
        *,
        certified: bool,
        route: Any = None,
    ) -> FleetColumn:
        return cls(
            identifier,
            slot,
            label,
            dict(plan.deploy_epochs),
            dict(plan.collect_epochs),
            dict(plan.foreign_deploy_epochs),
            dict(collected_mass),
            certified,
            route,
        )

    def value(self, weights: dict[int, float] | None) -> float:
        if weights is None:
            return sum(self.collected_mass.values())
        return sum(weights.get(a, 1.0) * m for a, m in self.collected_mass.items())

    @property
    def collected_kg(self) -> float:
        return sum(self.collected_mass.values())

    def summary(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "slot": self.slot,
            "label": self.label,
            "deploys": sorted(self.deploys),
            "collects": sorted(self.collects),
            "foreign": sorted(self.foreign),
            "collected_kg": self.collected_kg,
            "certified": self.certified,
        }


@dataclass(slots=True)
class FleetMasterResult:
    selected: tuple[FleetColumn, ...]
    objective: float
    upper_bound: float
    nodes: int
    exhaustive: bool
    rejected: list[dict[str, Any]] = field(default_factory=list)

    @property
    def collected_kg(self) -> float:
        return sum(column.collected_kg for column in self.selected)

    def summary(self) -> dict[str, Any]:
        mean = self.collected_kg / len(self.selected) if self.selected else 0.0
        return {
            "ships": len(self.selected),
            "objective_kg": self.objective,
            "collected_kg": self.collected_kg,
            "upper_bound_kg": self.upper_bound,
            "gap_kg": self.upper_bound - self.objective,
            "nodes": self.nodes,
            "exhaustive": self.exhaustive,
            "ship_limit": C.maximum_ship_count(mean) if self.selected else None,
            "selected": [column.summary() for column in self.selected],
            "rejected": self.rejected,
        }


def fleet_feasible(columns: tuple[FleetColumn, ...] | list[FleetColumn]) -> str:
    """Empty string when the column set is a valid fleet, else the violated rule."""

    deployed: dict[int, float] = {}
    collected: set[int] = set()
    for column in columns:
        if not column.certified:
            return f"column {column.identifier} is not certified"
        for asteroid, epoch in column.deploys.items():
            if asteroid in deployed:
                return f"asteroid {asteroid} deployed twice"
            deployed[asteroid] = epoch
        for asteroid in column.collects:
            if asteroid in collected:
                return f"asteroid {asteroid} collected twice"
            collected.add(asteroid)
    for column in columns:
        for asteroid, epoch in column.foreign.items():
            if asteroid not in deployed:
                return (
                    f"asteroid {asteroid} collected by column {column.identifier} but not deployed"
                )
            if abs(deployed[asteroid] - epoch) > EPOCH_TOLERANCE_DAYS:
                return f"asteroid {asteroid} deploy epoch differs from the collector's assumption"
    if columns:
        mean = sum(column.collected_kg for column in columns) / len(columns)
        if len(columns) > C.maximum_ship_count(mean) + 1e-9:
            return f"{len(columns)} ships exceed the limit {C.maximum_ship_count(mean):.2f}"
    return ""


def solve_fleet_master(
    columns: list[FleetColumn],
    *,
    weights: dict[int, float] | None = None,
    max_ships: int = C.MAX_SHIPS,
    node_cap: int = 200_000,
) -> FleetMasterResult:
    """Exact branch-and-bound packing master (see module docstring)."""

    certified = sorted(
        (column for column in columns if column.certified),
        key=lambda column: (-column.value(weights), column.identifier),
    )
    rejected = [
        {"identifier": column.identifier, "label": column.label, "reason": "not certified"}
        for column in columns
        if not column.certified
    ]
    deployers: dict[int, list[tuple[float, int]]] = {}
    for column in certified:
        for asteroid, epoch in column.deploys.items():
            deployers.setdefault(asteroid, []).append((epoch, column.identifier))
    # columns whose foreign collects no column can supply can never be selected
    usable: list[FleetColumn] = []
    for column in certified:
        missing = [
            asteroid
            for asteroid, epoch in column.foreign.items()
            if not any(
                abs(e - epoch) <= EPOCH_TOLERANCE_DAYS for e, _ in deployers.get(asteroid, [])
            )
        ]
        if missing:
            rejected.append(
                {
                    "identifier": column.identifier,
                    "label": column.label,
                    "reason": f"no column deploys foreign miners {sorted(missing)}",
                }
            )
        else:
            usable.append(column)
    values = [column.value(weights) for column in usable]
    suffix = [0.0] * (len(usable) + 1)
    for index in range(len(usable) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + values[index]
    best: tuple[FleetColumn, ...] = ()
    best_value = 0.0
    nodes = 0
    exhausted = True

    def compatible(column: FleetColumn, deployed: dict[int, float], collected: set[int]) -> bool:
        return not any(a in deployed for a in column.deploys) and not any(
            a in collected for a in column.collects
        )

    def leaf_ok(selected: tuple[FleetColumn, ...], deployed: dict[int, float]) -> bool:
        for column in selected:
            for asteroid, epoch in column.foreign.items():
                if (
                    asteroid not in deployed
                    or abs(deployed[asteroid] - epoch) > EPOCH_TOLERANCE_DAYS
                ):
                    return False
        mean = sum(column.collected_kg for column in selected) / len(selected)
        return len(selected) <= C.maximum_ship_count(mean) + 1e-9

    def search(
        index: int,
        selected: tuple[FleetColumn, ...],
        value: float,
        deployed: dict[int, float],
        collected: set[int],
    ) -> None:
        nonlocal best, best_value, nodes, exhausted
        nodes += 1
        if nodes > node_cap:
            exhausted = False
            return
        if selected and value > best_value + 1e-9 and leaf_ok(selected, deployed):
            best, best_value = selected, value
        if index == len(usable) or len(selected) >= max_ships:
            return
        if value + suffix[index] <= best_value + 1e-9:
            return  # cannot beat the incumbent even taking every remaining column
        column = usable[index]
        if compatible(column, deployed, collected):
            new_deployed = dict(deployed)
            new_deployed.update(column.deploys)
            search(
                index + 1,
                (*selected, column),
                value + values[index],
                new_deployed,
                collected | set(column.collects),
            )
        search(index + 1, selected, value, deployed, collected)

    search(0, (), 0.0, {}, set())
    chosen = {column.identifier for column in best}
    for column in usable:
        if column.identifier not in chosen:
            rejected.append(
                {
                    "identifier": column.identifier,
                    "label": column.label,
                    "reason": "dominated or incompatible with the incumbent",
                    "value_kg": column.value(weights),
                }
            )
    selected = tuple(sorted(best, key=lambda column: column.identifier))
    return FleetMasterResult(
        selected, best_value, suffix[0], nodes, exhausted, sorted(rejected, key=_rejected_key)
    )


def _rejected_key(item: dict[str, Any]) -> int:
    return int(item["identifier"])
