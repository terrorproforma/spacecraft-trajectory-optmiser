"""Greedy multi-ship fleet construction for GTOC12.

Ships are independent except for asteroid exclusivity (each asteroid may be deployed on once) and
the fleet-size rule ``N <= min(100, 2 exp(rho * M_bar))``.  The fleet is built sequentially:
ship ``k`` searches the pool with the asteroids of ships ``1..k-1`` excluded, its best certified
route is kept, and the routes are assembled into one official-format file (ship IDs 1..N).
The fixed post-competition score is simply the sum of collected masses (all bonus coefficients
are 1.0 for the final leaderboard state; see ``constants.bonus_coefficient``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import constants as C
from .data import AsteroidCatalogue
from .pipeline import RefinedRoute, emit_solution
from .solution import Solution


@dataclass(slots=True)
class FleetPlan:
    routes: list[RefinedRoute] = field(default_factory=list)

    @property
    def collected_kg(self) -> list[float]:
        return [route.total_collected_kg for route in self.routes]

    @property
    def average_collected_kg(self) -> float:
        return sum(self.collected_kg) / len(self.routes) if self.routes else 0.0

    @property
    def ship_limit(self) -> float:
        return C.maximum_ship_count(self.average_collected_kg)

    @property
    def rule_satisfied(self) -> bool:
        return len(self.routes) <= self.ship_limit

    def used_asteroids(self) -> set[int]:
        return {asteroid for route in self.routes for asteroid in route.plan.asteroids}

    def summary(self) -> dict[str, Any]:
        return {
            "ships": len(self.routes),
            "collected_kg_per_ship": self.collected_kg,
            "total_collected_kg": sum(self.collected_kg),
            "average_collected_kg": self.average_collected_kg,
            "ship_limit": self.ship_limit,
            "rule_satisfied": self.rule_satisfied,
            "asteroids": sorted(self.used_asteroids()),
        }


def assemble_fleet(plan: FleetPlan, catalogue: AsteroidCatalogue) -> Solution:
    """One official-format solution with ship IDs 1..N in fleet order."""

    if not plan.rule_satisfied:
        raise ValueError(
            f"fleet of {len(plan.routes)} ships exceeds the limit {plan.ship_limit:.2f} for an "
            f"average collected mass of {plan.average_collected_kg:.1f} kg"
        )
    ships = []
    for index, route in enumerate(plan.routes, start=1):
        ships.extend(emit_solution(route, catalogue, ship_id=index).ships)
    return Solution(ships)
