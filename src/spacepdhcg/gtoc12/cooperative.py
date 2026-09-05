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

import math
import sys
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

    def register_all(self, plans: list[tuple[RoutePlan, int]]) -> None:
        """Register several plans whatever their mutual collection order.

        A repaired or jointly harvested bundle can have ship 1 collecting a miner ship 2 deployed
        *and* ship 2 collecting one of ship 1's (a cycle no slot order resolves), so the plans are
        registered in two phases: every deploy first (each asteroid deployed once), then every
        collect against the complete deploy table (each collected once, deployed by somebody,
        against the deployer's epoch).  A failure leaves the pool untouched.
        """

        deployed: dict[int, tuple[float, int]] = dict(self.deployed)
        for plan, ship in plans:
            for asteroid, epoch in plan.deploy_epochs.items():
                if asteroid in deployed:
                    raise ValueError(f"asteroid {asteroid} deployed twice")
                deployed[asteroid] = (float(epoch), ship)
        collected: dict[int, int] = dict(self.collected)
        for plan, ship in plans:
            for asteroid in plan.collect_epochs:
                if asteroid in collected:
                    raise ValueError(f"asteroid {asteroid} collected twice")
                if asteroid not in deployed:
                    raise ValueError(f"asteroid {asteroid} collected but never deployed")
                if asteroid not in plan.deploy_epochs and (
                    abs(plan.deploy_epoch_of(asteroid) - deployed[asteroid][0])
                    > EPOCH_TOLERANCE_DAYS
                ):
                    raise ValueError(f"asteroid {asteroid} collected against a stale deploy epoch")
                collected[asteroid] = ship
        self.deployed = deployed
        self.collected = collected

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
    """One certified ship itinerary - or a bundle of them - offered to the master.

    A *bundle* column (``members`` non-empty) is a cooperative cluster plan: several ships whose
    foreign collects are satisfied inside the bundle.  The master treats it as one column that
    counts ``len(members)`` ships towards the fleet rule.
    """

    identifier: int
    slot: int  # ship slot the column was generated for (report only)
    label: str
    deploys: dict[int, float]  # asteroid -> deploy epoch
    collects: dict[int, float]  # asteroid -> collect epoch
    foreign: dict[int, float]  # asteroid -> deploy epoch required from another column
    collected_mass: dict[int, float]
    certified: bool
    route: Any = None  # RefinedRoute (opaque to the master)
    members: tuple[FleetColumn, ...] = ()  # bundle members (single-ship columns)

    @property
    def ships(self) -> int:
        return len(self.members) if self.members else 1

    def routes(self) -> list[Any]:
        """The RefinedRoute objects this column puts in the fleet (one per ship)."""

        if self.members:
            return [member.route for member in self.members]
        return [self.route]

    @classmethod
    def from_bundle(
        cls, identifier: int, label: str, members: list[FleetColumn] | tuple[FleetColumn, ...]
    ) -> FleetColumn:
        """One column for a set of ships whose foreign collects are (mostly) mutual."""

        members = tuple(members)
        deploys: dict[int, float] = {}
        collects: dict[int, float] = {}
        collected: dict[int, float] = {}
        for member in members:
            for asteroid, epoch in member.deploys.items():
                if asteroid in deploys:
                    raise ValueError(f"bundle deploys asteroid {asteroid} twice")
                deploys[asteroid] = epoch
            for asteroid, epoch in member.collects.items():
                if asteroid in collects:
                    raise ValueError(f"bundle collects asteroid {asteroid} twice")
                collects[asteroid] = epoch
            collected.update(member.collected_mass)
        foreign = {
            asteroid: epoch
            for member in members
            for asteroid, epoch in member.foreign.items()
            if asteroid not in deploys or abs(deploys[asteroid] - epoch) > EPOCH_TOLERANCE_DAYS
        }
        return cls(
            identifier,
            min(member.slot for member in members) if members else 0,
            label,
            deploys,
            collects,
            foreign,
            collected,
            all(member.certified for member in members) and bool(members),
            None,
            members,
        )

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
            "ships": self.ships,
            "deploys": sorted(self.deploys),
            "collects": sorted(self.collects),
            "foreign": sorted(self.foreign),
            "collected_kg": self.collected_kg,
            "certified": self.certified,
            "members": [member.label for member in self.members],
        }


def ship_count(columns: tuple[FleetColumn, ...] | list[FleetColumn]) -> int:
    return sum(column.ships for column in columns)


@dataclass(slots=True)
class FleetMasterResult:
    selected: tuple[FleetColumn, ...]
    objective: float
    upper_bound: float
    nodes: int
    exhaustive: bool
    rejected: list[dict[str, Any]] = field(default_factory=list)
    greedy_objective: float = 0.0
    lp_bound: float = math.inf  # max_N LP(N) relaxation (inf when not computed)
    lp_relaxations: dict[int, float] = field(default_factory=dict)  # N -> LP(N)
    lp_seconds: float = 0.0
    root_bound: float = math.inf  # min(sum of values, LP bound) before branching
    lp_nodes: int = 0  # LPs solved by the LP branch and bound
    lp_proven: bool = False  # the LP branch and bound closed every fleet size
    lp_sizes_searched: list[int] = field(default_factory=list)

    @property
    def proven(self) -> bool:
        """The incumbent is optimal: exhaustive search, closed LP branch and bound, or it meets
        the LP bound."""

        return self.exhaustive or self.lp_proven or self.objective >= self.lp_bound - 1e-6

    @property
    def collected_kg(self) -> float:
        return sum(column.collected_kg for column in self.selected)

    @property
    def ships(self) -> int:
        return ship_count(self.selected)

    def routes(self) -> list[Any]:
        return [route for column in self.selected for route in column.routes()]

    def cooperative_columns(self) -> dict[str, int]:
        """How much of the incumbent is cooperative: bundle columns, ships with foreign collects
        (collectors), ships whose miners another ship collects (deployers) and foreign collects."""

        bundles = sum(1 for column in self.selected if column.members)
        ships = [
            member
            for column in self.selected
            for member in (column.members if column.members else (column,))
        ]
        collectors = sum(1 for ship in ships if ship.foreign)
        wanted = {a for ship in ships for a in ship.foreign}
        deployers = sum(1 for ship in ships if any(a in wanted for a in ship.deploys))
        return {
            "bundle_columns": bundles,
            "collector_ships": collectors,
            "deployer_ships": deployers,
            "foreign_collects": sum(len(ship.foreign) for ship in ships),
        }

    def summary(self) -> dict[str, Any]:
        mean = self.collected_kg / self.ships if self.selected else 0.0
        return {
            "ships": self.ships,
            "columns": len(self.selected),
            "cooperative": self.cooperative_columns(),
            "objective_kg": self.objective,
            "greedy_objective_kg": self.greedy_objective,
            "collected_kg": self.collected_kg,
            "mean_collected_kg": mean,
            "upper_bound_kg": self.upper_bound,
            "gap_kg": self.upper_bound - self.objective,
            "lp_bound_kg": self.lp_bound,
            "lp_gap_kg": self.lp_bound - self.objective,
            "lp_relaxations_kg": {str(k): v for k, v in sorted(self.lp_relaxations.items())},
            "lp_seconds": self.lp_seconds,
            "lp_nodes": self.lp_nodes,
            "lp_sizes_searched": list(self.lp_sizes_searched),
            "root_bound_kg": self.root_bound,
            "proven_optimal": self.proven,
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
        ships = ship_count(columns)
        mean = sum(column.collected_kg for column in columns) / ships
        if ships > C.maximum_ship_count(mean) + 1e-9:
            return f"{ships} ships exceed the limit {C.maximum_ship_count(mean):.2f}"
    return ""


def _ship_rule_ok(selected: tuple[FleetColumn, ...]) -> bool:
    ships = ship_count(selected)
    mean = sum(column.collected_kg for column in selected) / ships
    return ships <= C.maximum_ship_count(mean) + 1e-9


def _foreign_ok(selected: tuple[FleetColumn, ...], deployed: dict[int, float]) -> bool:
    for column in selected:
        for asteroid, epoch in column.foreign.items():
            if asteroid not in deployed or abs(deployed[asteroid] - epoch) > EPOCH_TOLERANCE_DAYS:
                return False
    return True


def greedy_fleet(usable: list[FleetColumn], max_ships: int) -> tuple[FleetColumn, ...]:
    """Iterated greedy over columns (bundles included) under the mass-average ship rule.

    Columns are taken in value order while they stay compatible; then, because the ship limit
    ``2 exp(rho * mean)`` depends on the selection itself, the lowest value-per-ship columns are
    dropped one at a time until the rule and the foreign-collect closure hold (the fixed-point
    iteration on the mean mass).
    """

    selected: list[FleetColumn] = []
    deployed: dict[int, float] = {}
    collected: set[int] = set()
    for column in usable:
        if ship_count(selected) + column.ships > max_ships:
            continue
        if any(a in deployed for a in column.deploys) or any(
            a in collected for a in column.collects
        ):
            continue
        selected.append(column)
        deployed.update(column.deploys)
        collected |= set(column.collects)
    while selected:
        chosen = tuple(selected)
        deployed = {a: e for column in chosen for a, e in column.deploys.items()}
        if _foreign_ok(chosen, deployed) and _ship_rule_ok(chosen):
            return chosen
        # drop what helps least: a column with an unsatisfied foreign collect first, else the
        # lowest collected mass per ship
        stranded = [
            column
            for column in selected
            if any(
                a not in deployed or abs(deployed[a] - e) > EPOCH_TOLERANCE_DAYS
                for a, e in column.foreign.items()
            )
        ]
        victim = min(
            stranded or selected,
            key=lambda column: (column.collected_kg / column.ships, -column.identifier),
        )
        selected.remove(victim)
    return ()


def ship_rule_bound(
    ships: int,
    mass: float,
    value: float,
    mass_prefix: list[float],
    value_prefix: list[float],
    room: int,
) -> float:
    """Upper bound on the value a partial fleet (``ships`` ships, ``mass`` kg collected, ``value``
    weighted kg) can reach by adding up to ``room`` more ships from the remaining columns.

    ``mass_prefix[k]`` / ``value_prefix[k]`` are the sums of the ``k`` largest remaining per-ship
    masses / values.  Adding ``k`` ships collects at most ``mass_prefix[k]`` kg, so if
    ``ships + k`` breaks ``N <= 2 exp(rho * mean)`` at that mass it breaks it for every real
    k-ship extension; among the feasible ``k`` the value gained is at most ``value_prefix[k]``.
    ``-inf`` when no completion (including stopping here) can satisfy the rule.
    """

    best = -math.inf
    if ships == 0 or ships <= C.maximum_ship_count(mass / ships) + 1e-9:
        best = value
    for k in range(1, min(room, len(mass_prefix) - 1) + 1):
        total = mass + mass_prefix[k]
        if ships + k <= C.maximum_ship_count(total / (ships + k)) + 1e-9:
            best = max(best, value + value_prefix[k])
    return best


def ship_rule_mass_floor(ships: int) -> float:
    """Least total collected mass a fleet of ``ships`` ships needs under the ship rule.

    ``N <= 2 exp(rho * mean)`` with ``mean = mass / N`` is ``mass >= N ln(N/2) / rho`` (0 for
    ``N <= 2``); linear in the columns' masses once ``N`` is fixed, which is what makes the
    per-``N`` LP relaxation below exact for the rule.
    """

    if ships <= 2:
        return 0.0
    return ships * math.log(ships / 2.0) / C.SHIP_COUNT_RHO_PER_KG


@dataclass(slots=True)
class LpBound:
    """LP relaxation of the master, solved once per fleet size ``N`` (see :func:`lp_fleet_bound`).

    ``bound`` is ``max_N LP(N)`` (``-inf`` when no ``N`` is LP-feasible); ``relaxations[N]`` the
    LP optimum per ship count; the dual vectors feed the per-node bound of the branch and bound
    (:meth:`node_bound`): for any ``y >= 0`` (asteroid rows), ``mu`` (ship-count row) and
    ``nu >= 0`` (mass-floor row) weak duality gives the valid upper bound
    ``sum_a y_a + N mu - g(N) nu + sum_c max(0, v_c - A_c y - s_c mu + m_c nu)``.
    """

    bound: float
    relaxations: dict[int, float]
    sizes: Any = None  # np.ndarray of the LP-feasible N (int)
    mu: Any = None  # (n_sizes,)
    nu: Any = None  # (n_sizes,)
    y_total: Any = None  # (n_sizes,) sum of the asteroid duals
    column_dual: Any = None  # (n_columns, n_sizes) A_c y per column
    positive_rc_suffix: Any = None  # (n_columns + 1, n_sizes) suffix sums of max(0, rc)
    floors: Any = None  # (n_sizes,) g(N)
    lp_seconds: float = 0.0

    positive_rc: Any = None  # (n_columns, n_sizes) max(0, rc)

    def node_bound(
        self,
        index: int,
        value: float,
        ships: int,
        mass: float,
        used_dual: Any,
        max_ships: int,
        free: Any = None,
    ) -> float:
        """Upper bound on any completion of a node: ``value`` collected so far by ``ships`` ships
        of ``mass`` kg, with the asteroid duals ``used_dual`` (per N) of the selected columns
        already consumed and the columns from ``index`` on still free.  ``free`` (bool per
        column) restricts the reduced-cost sum to columns compatible with the selection; when
        omitted every remaining column counts (looser, still valid)."""

        import numpy as np

        if self.sizes is None or self.sizes.shape[0] == 0:
            return math.inf
        sizes = self.sizes
        if free is None:
            remaining = self.positive_rc_suffix[index]
        else:
            remaining = (self.positive_rc[index:] * free[index:, None]).sum(axis=0)
        bounds = (
            value
            + (self.y_total - used_dual)
            + (sizes - ships) * self.mu
            - (self.floors - mass) * self.nu
            + remaining
        )
        ok = (sizes >= ships) & (sizes <= max_ships)
        if not np.any(ok):
            return -math.inf
        return float(np.max(np.where(ok, bounds, -math.inf)))


class _LpModel:
    """Sparse LP data of the master: asteroid packing rows, foreign-closure rows, the mass-floor
    row (RHS per fleet size) and the ship-count equality.  ``solve(size, lower, upper)`` returns
    the HiGHS result of ``LP(N)`` with per-column bounds (``None`` when infeasible)."""

    def __init__(
        self,
        usable: list[FleetColumn],
        values: list[float],
        *,
        foreign_rows: bool = True,
    ) -> None:
        import numpy as np
        from scipy.sparse import coo_matrix

        n = len(usable)
        self.n = n
        deploy_rows: dict[int, int] = {}
        collect_rows: dict[int, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        for c, column in enumerate(usable):
            for asteroid in column.deploys:
                rows.append(deploy_rows.setdefault(asteroid, len(deploy_rows)))
                cols.append(c)
        offset = len(deploy_rows)
        for c, column in enumerate(usable):
            for asteroid in column.collects:
                rows.append(offset + collect_rows.setdefault(asteroid, len(collect_rows)))
                cols.append(c)
        self.n_asteroid_rows = offset + len(collect_rows)
        # asteroid -> row index (deploy rows first, then collect rows) for the dual pricing
        self.deploy_rows = dict(deploy_rows)
        self.collect_rows = {a: offset + r for a, r in collect_rows.items()}
        data = [1.0] * len(rows)
        # foreign rows: x_c - sum_{d supplies (a, epoch)} x_d <= 0
        n_rows = self.n_asteroid_rows
        if foreign_rows:
            suppliers: dict[int, list[tuple[float, int]]] = {}
            for d, column in enumerate(usable):
                for asteroid, epoch in column.deploys.items():
                    suppliers.setdefault(asteroid, []).append((epoch, d))
            for c, column in enumerate(usable):
                for asteroid, epoch in column.foreign.items():
                    rows.append(n_rows)
                    cols.append(c)
                    data.append(1.0)
                    for e, d in suppliers.get(asteroid, []):
                        if abs(e - epoch) <= EPOCH_TOLERANCE_DAYS:
                            rows.append(n_rows)
                            cols.append(d)
                            data.append(-1.0)
                    n_rows += 1
        self.masses = np.array([column.collected_kg for column in usable])
        self.ships = np.array([float(column.ships) for column in usable])
        self.values = np.asarray(values, dtype=np.float64)
        # mass floor row (index n_rows): -m.x <= -g(N); RHS set per N
        self.mass_row = n_rows
        n_rows += 1
        rows.extend([self.mass_row] * n)
        cols.extend(range(n))
        data.extend((-self.masses).tolist())
        self.n_rows = n_rows
        self.a_ub = coo_matrix(
            (np.array(data), (np.array(rows), np.array(cols))), shape=(n_rows, n)
        ).tocsr()
        self.a_eq = self.ships[None, :]
        self.total_ships = int(self.ships.sum())
        self.solves = 0

    def sizes(self, max_ships: int) -> list[int]:
        """Fleet sizes whose mass floor the columns can reach at all."""

        total = float(self.masses.sum())
        return [
            size
            for size in range(1, min(max_ships, self.total_ships) + 1)
            if ship_rule_mass_floor(size) <= total + 1e-9
        ]

    def solve(self, size: int, lower: Any = None, upper: Any = None) -> Any:
        import numpy as np
        from scipy.optimize import linprog

        b_ub = np.ones(self.n_rows)
        b_ub[self.n_asteroid_rows : self.mass_row] = 0.0
        b_ub[self.mass_row] = -ship_rule_mass_floor(size)
        lo = np.zeros(self.n) if lower is None else lower
        hi = np.ones(self.n) if upper is None else upper
        self.solves += 1
        result = linprog(
            -self.values,
            A_ub=self.a_ub,
            b_ub=b_ub,
            A_eq=self.a_eq,
            b_eq=np.array([float(size)]),
            bounds=np.column_stack([lo, hi]),
            method="highs",
        )
        return result if result.status == 0 else None


def lp_fleet_bound(
    usable: list[FleetColumn],
    values: list[float],
    *,
    max_ships: int = C.MAX_SHIPS,
    foreign_rows: bool = True,
    model: _LpModel | None = None,
) -> LpBound:
    """LP relaxation of the packing master with the ship rule, per fleet size.

    For each integer ``N`` the LP ``max v.x`` s.t. each asteroid deployed <= 1 and collected
    <= 1, ``x_c <= sum of the columns supplying its foreign miners`` (when ``foreign_rows``),
    ``sum s_c x_c = N``, ``sum m_c x_c >= g(N)`` and ``0 <= x <= 1`` is a relaxation of every
    ``N``-ship fleet; the maximum over ``N`` bounds the master.  Solved with HiGHS via SciPy.
    """

    import time as _time

    import numpy as np

    started = _time.perf_counter()
    n = len(usable)
    empty = LpBound(-math.inf, {})
    if n == 0:
        return empty
    model = model or _LpModel(usable, values, foreign_rows=foreign_rows)
    v, masses, ships = model.values, model.masses, model.ships
    relaxations: dict[int, float] = {}
    duals: list[tuple[int, float, float, np.ndarray]] = []  # (N, mu, nu, y)
    for size in model.sizes(max_ships):
        result = model.solve(size)
        if result is None:
            continue
        relaxations[size] = float(-result.fun)
        # HiGHS marginals of the min problem: <= 0 for A_ub rows (y = -marginal >= 0)
        y = np.maximum(-np.asarray(result.ineqlin.marginals), 0.0)
        mu = float(-result.eqlin.marginals[0])
        nu = float(y[model.mass_row])
        duals.append((size, mu, nu, y[: model.n_asteroid_rows]))
    if not relaxations:
        empty.lp_seconds = _time.perf_counter() - started
        return empty
    sizes = np.array([d[0] for d in duals], dtype=np.int64)
    mu = np.array([d[1] for d in duals])
    nu = np.array([d[2] for d in duals])
    y_matrix = np.stack([d[3] for d in duals], axis=1)  # (n_asteroid_rows, n_sizes)
    a_ast = model.a_ub[: model.n_asteroid_rows]
    column_dual = np.asarray(a_ast.T @ y_matrix)  # (n, n_sizes)
    reduced = (
        v[:, None] - column_dual - ships[:, None] * mu[None, :] + masses[:, None] * nu[None, :]
    )
    positive = np.maximum(reduced, 0.0)
    suffix = np.zeros((n + 1, sizes.shape[0]))
    suffix[:-1] = np.cumsum(positive[::-1], axis=0)[::-1]
    return LpBound(
        max(relaxations.values()),
        relaxations,
        sizes,
        mu,
        nu,
        y_matrix.sum(axis=0),
        column_dual,
        suffix,
        np.array([ship_rule_mass_floor(int(s)) for s in sizes]),
        _time.perf_counter() - started,
        positive,
    )


@dataclass(slots=True)
class AsteroidPrices:
    """Per-asteroid duals of the master LP at one fleet size (see :func:`lp_asteroid_prices`).

    ``prices[a]`` is the marginal value (kg of master objective) of asteroid ``a``'s packing
    rows - its deploy row plus its collect row - i.e. what the master already pays for ``a``
    through the columns that use it.  A new column ``c`` improves the LP only if its reduced
    cost ``v_c - sum_{a in c} prices[a] - mu + m_c nu`` is positive, so a family search that
    subtracts ``prices`` from its chains prices the columns the master would take.
    """

    size: int  # fleet size N the LP was solved at
    prices: dict[int, float]
    mu: float  # ship-count dual
    nu: float  # mass-floor dual (kg of objective per kg of collected mass)
    lp_value: float
    solved_sizes: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        values = sorted(self.prices.values(), reverse=True)
        return {
            "size": self.size,
            "priced_asteroids": len(self.prices),
            "max_kg": values[0] if values else 0.0,
            "sum_kg": float(sum(values)),
            "top": [
                {"asteroid": a, "kg": p}
                for a, p in sorted(self.prices.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            ],
            "mu": self.mu,
            "nu": self.nu,
            "lp_value_kg": self.lp_value,
            "solved_sizes": list(self.solved_sizes),
        }


def usable_columns(
    columns: list[FleetColumn], weights: dict[int, float] | None
) -> tuple[list[FleetColumn], list[dict[str, Any]]]:
    """Certified columns whose foreign collects some column supplies, best value first, and
    the reject log for the rest (shared by the master and the dual pricing)."""

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
    return usable, rejected


def lp_asteroid_prices(
    columns: list[FleetColumn],
    *,
    weights: dict[int, float] | None = None,
    max_ships: int = C.MAX_SHIPS,
    size: int | None = None,
    target_size: int | None = None,
    bound_share: bool = False,
) -> AsteroidPrices | None:
    """Duals of the master LP as per-asteroid prices for the family pricing.

    The LP is the relaxation of :func:`lp_fleet_bound` at one fleet size: ``size`` when given,
    else the largest LP-feasible size not above ``target_size`` (default: the largest feasible
    size at all) - pricing at ``N* + 1`` asks which asteroids stand between the archive and one
    more ship.  Returns ``None`` when no size is LP-feasible.  Prices are non-negative; an
    asteroid appears only when its rows carry a positive dual.  Deterministic for a given
    column list (HiGHS on the same LP).

    With ``bound_share`` the dual of every column's ``x_c <= 1`` bound is shared equally among
    the asteroids the column uses.  A nearly integral LP is degenerate: the rent of a selected
    ship sits on its bound, not on its asteroid rows (over the 915-column archive at N = 21 only
    22 asteroids carried a row dual), so the row duals alone under-price what the fleet already
    holds.  Moving the bound dual of ``c`` onto ``c``'s rows keeps every dual constraint
    satisfied (``c``'s own is unchanged, every other column's reduced cost only falls) and the
    dual objective (``sum y + sum z`` is invariant), so the shared prices are another optimal
    dual solution - one that attributes the rent to the resources a new column would have to
    take away.
    """

    import numpy as np

    usable, _rejected = usable_columns(columns, weights)
    if not usable:
        return None
    values = [column.value(weights) for column in usable]
    model = _LpModel(usable, values)
    sizes = model.sizes(max_ships)
    if size is not None:
        sizes = [s for s in sizes if s == size]
    elif target_size is not None:
        sizes = [s for s in sizes if s <= target_size]
    solved: list[int] = []
    chosen: tuple[int, Any] | None = None
    for candidate in sorted(sizes, reverse=True):
        result = model.solve(candidate)
        solved.append(candidate)
        if result is not None:
            chosen = (candidate, result)
            break
    if chosen is None:
        return None
    n_size, result = chosen
    y = np.maximum(-np.asarray(result.ineqlin.marginals), 0.0)
    mu = float(-result.eqlin.marginals[0])
    nu = float(y[model.mass_row])
    prices: dict[int, float] = {}
    for asteroid, row in model.deploy_rows.items():
        prices[asteroid] = prices.get(asteroid, 0.0) + float(y[row])
    for asteroid, row in model.collect_rows.items():
        prices[asteroid] = prices.get(asteroid, 0.0) + float(y[row])
    if bound_share:
        z = np.maximum(-np.asarray(result.upper.marginals), 0.0)
        for c, column in enumerate(usable):
            if z[c] <= 1e-9:
                continue
            asteroids = sorted(set(column.deploys) | set(column.collects))
            if asteroids:
                share = float(z[c]) / len(asteroids)
                for asteroid in asteroids:
                    prices[asteroid] = prices.get(asteroid, 0.0) + share
    prices = {a: p for a, p in sorted(prices.items()) if p > 1e-9}
    return AsteroidPrices(int(n_size), prices, mu, nu, float(-result.fun), solved)


@dataclass(slots=True)
class LpBranchResult:
    """Outcome of :func:`lp_branch_and_bound`."""

    selection: tuple[int, ...] | None  # column indices of an improving integral fleet
    value: float  # its value (or the incumbent value passed in)
    bound: float  # valid upper bound on every fleet after the search
    nodes: int
    proven: bool  # every fleet size closed: ``value`` is optimal
    sizes_searched: list[int] = field(default_factory=list)


def lp_branch_and_bound(
    model: _LpModel,
    relaxations: dict[int, float],
    incumbent_value: float,
    *,
    node_limit: int = 4000,
    time_limit_seconds: float = math.inf,
    integrality_tolerance: float = 1e-6,
) -> LpBranchResult:
    """LP-based branch and bound over the fleet sizes whose relaxation beats the incumbent.

    Depth-first, branching on the most fractional column (``x = 1`` first); a node is pruned
    when its LP is infeasible or does not beat the incumbent.  Integral LP solutions are
    fleets: the packing rows, the foreign-closure rows and the mass floor at integer ``N`` are
    the master's exact constraints.  Stops early at ``node_limit`` LPs or the time limit, in
    which case ``proven`` is False and ``bound`` is the largest LP of the open sizes.
    """

    import time as _time

    import numpy as np

    started = _time.perf_counter()
    best_value = incumbent_value
    best_selection: tuple[int, ...] | None = None
    nodes = 0
    proven = True
    open_bound = -math.inf
    sizes = sorted(
        (size for size, value in relaxations.items() if value > incumbent_value + 1e-6),
        key=lambda size: -relaxations[size],
    )
    for position, size in enumerate(sizes):
        # (lower bounds, upper bounds, parent LP value) - the parent value bounds the node
        stack: list[tuple[np.ndarray, np.ndarray, float]] = [
            (np.zeros(model.n), np.ones(model.n), relaxations[size])
        ]
        closed = True
        while stack:
            if nodes >= node_limit or _time.perf_counter() - started > time_limit_seconds:
                closed = False
                break
            lower, upper, parent = stack.pop()
            if parent <= best_value + 1e-6:
                continue
            nodes += 1
            result = model.solve(size, lower, upper)
            if result is None:
                continue
            value = float(-result.fun)
            if value <= best_value + 1e-6:
                continue
            x = np.asarray(result.x)
            fractional = np.abs(x - np.rint(x)) > integrality_tolerance
            if not np.any(fractional):
                chosen = tuple(int(i) for i in np.nonzero(np.rint(x) > 0.5)[0])
                best_value, best_selection = value, chosen
                continue
            # branch on the most fractional column: x = 1 explored first (LIFO)
            branch = int(np.argmax(np.where(fractional, -np.abs(x - 0.5), -np.inf)))
            up_lower, up_upper = lower.copy(), upper.copy()
            up_lower[branch] = 1.0
            down_lower, down_upper = lower.copy(), upper.copy()
            down_upper[branch] = 0.0
            stack.append((down_lower, down_upper, value))
            stack.append((up_lower, up_upper, value))
        if not closed:
            proven = False
            open_bound = max([open_bound, *(parent for _, _, parent in stack)])
            # the remaining sizes are not searched either: their root LPs stay open bounds
            for other in sizes[position + 1 :]:
                open_bound = max(open_bound, relaxations[other])
            break
    bound = best_value if proven else max(best_value, open_bound)
    return LpBranchResult(best_selection, best_value, bound, nodes, proven, sizes)


def solve_fleet_master(
    columns: list[FleetColumn],
    *,
    weights: dict[int, float] | None = None,
    max_ships: int = C.MAX_SHIPS,
    node_cap: int = 200_000,
    incumbent: tuple[FleetColumn, ...] | list[FleetColumn] | None = None,
    lp_bound: bool = True,
    lp_node_limit: int = 4000,
) -> FleetMasterResult:
    """Exact branch-and-bound packing master (see module docstring).

    Bundle columns count their member ships towards ``max_ships`` and the ship rule.  The
    search starts from the best of the iterated-greedy fleets (:func:`greedy_fleet` in value and
    in value-per-ship order) and the caller's ``incumbent`` (e.g. the previous master's
    selection, which stays feasible when columns are only added), so when the node cap stops it
    (``exhaustive`` False) the result never regresses.  Nodes are pruned with the ship-rule bound
    (:func:`ship_rule_bound`) on top of the asteroid conflicts.
    """

    # columns whose foreign collects no column can supply can never be selected
    usable, rejected = usable_columns(columns, weights)
    values = [column.value(weights) for column in usable]
    suffix = [0.0] * (len(usable) + 1)
    for index in range(len(usable) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + values[index]
    # per-ship units of the columns from ``index`` on, largest first, with prefix sums: the
    # ingredients of the ship-rule bound at every depth
    mass_prefixes: list[list[float]] = []
    value_prefixes: list[list[float]] = []
    for index in range(len(usable) + 1):
        remaining = usable[index:]
        masses = sorted(
            (c.collected_kg / c.ships for c in remaining for _ in range(c.ships)), reverse=True
        )
        unit_values = sorted(
            (values[index + i] / c.ships for i, c in enumerate(remaining) for _ in range(c.ships)),
            reverse=True,
        )
        mass_prefixes.append(_prefix_sums(masses))
        value_prefixes.append(_prefix_sums(unit_values))

    def total_value(selection) -> float:
        return sum(column.value(weights) for column in selection)

    starts: list[tuple[FleetColumn, ...]] = [
        greedy_fleet(usable, max_ships),
        greedy_fleet(
            sorted(
                usable,
                key=lambda c: (-c.value(weights) / c.ships, -c.value(weights), c.identifier),
            ),
            max_ships,
        ),
    ]
    greedy_value = max(total_value(start) for start in starts)
    if incumbent:
        usable_ids = {column.identifier for column in usable}
        warm = tuple(column for column in incumbent if column.identifier in usable_ids)
        if warm and len(warm) == len(incumbent) and fleet_feasible(warm) == "":
            starts.append(warm)
    best: tuple[FleetColumn, ...] = max(starts, key=total_value)
    best_value = total_value(best)
    nodes = 0
    exhausted = True
    lp_model = _LpModel(usable, values) if lp_bound and usable else None
    relaxation = (
        lp_fleet_bound(usable, values, max_ships=max_ships, model=lp_model)
        if lp_model is not None
        else None
    )
    root_bound = suffix[0]
    if relaxation is not None and math.isfinite(relaxation.bound):
        root_bound = min(root_bound, relaxation.bound)
    import numpy as np

    zero_dual = (
        np.zeros(relaxation.sizes.shape[0])
        if relaxation is not None and relaxation.sizes is not None
        else None
    )
    # pairwise conflicts (shared deploy or collect asteroid): the ``free`` mask of a node is
    # the columns still compatible with its selection, which tightens both the value bound
    # (sum of free values) and the dual bound (positive reduced costs of free columns)
    n_usable = len(usable)
    conflicts = np.zeros((n_usable, n_usable), dtype=bool)
    owners: dict[tuple[str, int], list[int]] = {}
    for c, column in enumerate(usable):
        for asteroid in column.deploys:
            owners.setdefault(("d", asteroid), []).append(c)
        for asteroid in column.collects:
            owners.setdefault(("c", asteroid), []).append(c)
    for members in owners.values():
        if len(members) > 1:
            idx = np.asarray(members)
            conflicts[np.ix_(idx, idx)] = True
    np.fill_diagonal(conflicts, False)
    value_array = np.asarray(values, dtype=np.float64)

    def leaf_ok(selected: tuple[FleetColumn, ...], deployed: dict[int, float]) -> bool:
        return _foreign_ok(selected, deployed) and _ship_rule_ok(selected)

    def search(
        index: int,
        selected: tuple[FleetColumn, ...],
        value: float,
        mass: float,
        deployed: dict[int, float],
        collected: set[int],
        used_dual: Any,
        free: Any,
    ) -> None:
        nonlocal best, best_value, nodes, exhausted
        nodes += 1
        if nodes > node_cap:
            exhausted = False
            return
        if selected and value > best_value + 1e-9 and leaf_ok(selected, deployed):
            best, best_value = selected, value
        ships = ship_count(selected)
        if index == len(usable) or ships >= max_ships:
            return
        if value + suffix[index] <= best_value + 1e-9:
            return  # cannot beat the incumbent even taking every remaining column
        if value + float(value_array[index:] @ free[index:]) <= best_value + 1e-9:
            return  # ... nor taking every remaining *compatible* column
        bound = ship_rule_bound(
            ships,
            mass,
            value,
            mass_prefixes[index],
            value_prefixes[index],
            max_ships - ships,
        )
        if bound <= best_value + 1e-9:
            return  # no rule-feasible completion beats the incumbent
        if zero_dual is not None:
            # LP dual bound: the asteroid capacity the selection consumed, the ships and the
            # mass floor still to fill, and the positive reduced costs of the free columns
            dual_bound = relaxation.node_bound(
                index, value, ships, mass, used_dual, max_ships, free
            )
            if dual_bound <= best_value + 1e-9:
                return
        column = usable[index]
        if ships + column.ships <= max_ships and free[index]:
            new_deployed = dict(deployed)
            new_deployed.update(column.deploys)
            search(
                index + 1,
                (*selected, column),
                value + values[index],
                mass + column.collected_kg,
                new_deployed,
                collected | set(column.collects),
                None if zero_dual is None else used_dual + relaxation.column_dual[index],
                free & ~conflicts[index],
            )
        search(index + 1, selected, value, mass, deployed, collected, used_dual, free)

    # ``search`` recurses once per usable column (the skip branch always walks ``index`` to
    # ``len(usable)``), so an archive-wide master offers more columns than CPython's default
    # 1000-frame limit: 1019 columns (fleet_master_v6) overflowed it after 45 min of
    # re-certification.  Widen it for the search only - two frames per column (WSL fix
    # ba9b764) or 500 frames of room for the callers (H100 fix c4e2c31), whichever is larger -
    # and restore it afterwards.
    required_depth = max(2 * n_usable + 200, n_usable + 500)
    previous_limit = sys.getrecursionlimit()
    if required_depth > previous_limit:
        sys.setrecursionlimit(required_depth)
    try:
        search(0, (), 0.0, 0.0, {}, set(), zero_dual, np.ones(n_usable, dtype=bool))
    finally:
        if required_depth > previous_limit:
            sys.setrecursionlimit(previous_limit)
    # LP-based branch and bound closes (or bounds) what the combinatorial search left open:
    # only the fleet sizes whose relaxation beats the incumbent are branched on
    lp_branch: LpBranchResult | None = None
    proven = exhausted
    if relaxation is not None and lp_model is not None and not exhausted:
        lp_branch = lp_branch_and_bound(
            lp_model, relaxation.relaxations, best_value, node_limit=lp_node_limit
        )
        if lp_branch.selection is not None:
            candidate = tuple(usable[i] for i in lp_branch.selection)
            if fleet_feasible(candidate) == "" and total_value(candidate) > best_value + 1e-9:
                best, best_value = candidate, total_value(candidate)
        proven = lp_branch.proven
        root_bound = (
            min(root_bound, lp_branch.bound)
            if lp_branch.proven
            else min(root_bound, max(lp_branch.bound, best_value))
        )
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
        selected,
        best_value,
        # the LP bound stays valid when the node cap stops the search; a proven incumbent
        # (exhaustive search or closed LP branch and bound) collapses the bound onto it
        best_value if proven else max(root_bound, best_value),
        nodes,
        exhausted,
        sorted(rejected, key=_rejected_key),
        greedy_value,
        lp_bound=relaxation.bound if relaxation is not None else math.inf,
        lp_relaxations=dict(relaxation.relaxations) if relaxation is not None else {},
        lp_seconds=(relaxation.lp_seconds if relaxation is not None else 0.0),
        root_bound=root_bound,
        lp_nodes=lp_branch.nodes if lp_branch is not None else 0,
        lp_proven=proven,
        lp_sizes_searched=list(lp_branch.sizes_searched) if lp_branch is not None else [],
    )


def _rejected_key(item: dict[str, Any]) -> int:
    return int(item["identifier"])


def _prefix_sums(items: list[float]) -> list[float]:
    prefix = [0.0]
    for item in items:
        prefix.append(prefix[-1] + item)
    return prefix
