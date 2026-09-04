"""Archived certified routes as master columns.

Every run writes each emitted ship as ``.../ship_NN/**/route_summary.json``.  Those routes are
already the product of beam search, SCvx and re-timing, so a later fleet master should be able
to select among *all* of them (this run's bundles plus earlier fleets) instead of only the
columns priced in the current campaign.  This module rediscovers such archives, rebuilds their
plans (:func:`plan_from_route_summary`), **re-flies every leg through SCvx** so nothing enters
the master on the strength of an old JSON file, and packs each archived group (a family bundle
directory or a fleet run) into a :class:`ClusterBundle` whose columns the master accepts.

Groups keep their cooperative structure: a ship that collected another ship's miner in the
archive can only be selected together with that deployer (the bundle column), exactly as for
freshly priced families.
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bundles import BundleShip, ClusterBundle, make_consistent
from .data import AsteroidCatalogue
from .low_thrust import ScvxSettings
from .pipeline import RefinedRoute, plan_from_route_summary, refine_route
from .search import RoutePlan

SHIP_DIR = re.compile(r"^ship_(\d+)$")
SKIP_PARTS = {"fleet", "fleets", "viewer"}


@dataclass(slots=True)
class ArchivedShip:
    slot: int
    summaries: list[tuple[Path, dict[str, Any]]]  # certified archives, best first

    @property
    def primary(self) -> dict[str, Any]:
        return self.summaries[0][1]


@dataclass(slots=True)
class ArchivedGroup:
    name: str
    directory: Path
    ships: list[ArchivedShip] = field(default_factory=list)


def discover_archives(sources: list[Path]) -> list[ArchivedGroup]:
    """Find every certified ``route_summary.json`` below ``sources``, grouped by ship directory
    parent (a family directory of a cluster run, or a fleet run root).  Deterministic order."""

    groups: dict[Path, ArchivedGroup] = {}
    for source in sources:
        for path in sorted(Path(source).rglob("route_summary.json")):
            relative = path.relative_to(source).parts
            if SKIP_PARTS.intersection(relative):
                continue
            ship_dir = next((p for p in path.parents if SHIP_DIR.match(p.name)), None)
            if ship_dir is None:
                continue
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not summary.get("certified"):
                continue
            group = groups.setdefault(
                ship_dir.parent,
                ArchivedGroup(
                    str(ship_dir.parent.relative_to(Path(source).parent)), ship_dir.parent
                ),
            )
            slot = int(SHIP_DIR.match(ship_dir.name).group(1))  # type: ignore[union-attr]
            ship = next((s for s in group.ships if s.slot == slot), None)
            if ship is None:
                ship = ArchivedShip(slot, [])
                group.ships.append(ship)
            ship.summaries.append((path, summary))
    ordered = [groups[key] for key in sorted(groups)]
    for group in ordered:
        group.ships.sort(key=lambda s: s.slot)
        for ship in group.ships:
            # the emitted (best) archive first; the others become stand-alone variants
            ship.summaries.sort(key=lambda item: (-float(item[1]["total_collected_kg"]), item[0]))
    return ordered


def group_plans(group: ArchivedGroup) -> dict[int, list[RoutePlan]]:
    """Rebuild the plans of a group; foreign deploy epochs are snapped to the deployer's."""

    first = {
        ship.slot: [plan_from_route_summary(s) for _p, s in ship.summaries] for ship in group.ships
    }
    deployers: dict[int, float] = {}
    for plans in first.values():
        for asteroid, epoch in plans[0].deploy_epochs.items():
            deployers.setdefault(asteroid, epoch)
    return {
        ship.slot: [plan_from_route_summary(s, deployers=deployers) for _p, s in ship.summaries]
        for ship in group.ships
    }


_WORKER: dict[str, Any] = {}


def _recertify_in_worker(task: tuple[str, int, list[dict[str, Any]]]) -> tuple[str, int, list]:
    name, slot, plans = task
    state = _WORKER
    if "catalogue" not in state:
        from .data import load_catalogue

        state["catalogue"] = load_catalogue()
    routes: list[RefinedRoute | None] = []
    for summary in plans:
        try:
            plan = RoutePlan.from_summary(summary)
            routes.append(refine_route(plan, state["catalogue"], scvx=state.get("scvx")))
        except Exception as error:  # one bad archive must not stop the rest
            routes.append(None)
            state.setdefault("errors", []).append(f"{name}/ship_{slot:02d}: {error!r}")
    return name, slot, routes


def recertify_archives(
    catalogue: AsteroidCatalogue,
    groups: list[ArchivedGroup],
    *,
    scvx: ScvxSettings | None = None,
    workers: int = 2,
    first_label: int = 10_000,
    on_progress=None,
) -> list[ClusterBundle]:
    """Re-fly every archived route through SCvx and pack the certified ones into bundles.

    Ships whose primary route no longer certifies are dropped (and logged in the bundle's
    ``rejected`` list); a collector left without its deployer is dropped as well, so every
    returned bundle is pool-consistent.
    """

    started = time.perf_counter()
    _WORKER.update(catalogue=catalogue, scvx=scvx)
    tasks: list[tuple[str, int, list[dict[str, Any]]]] = []
    plans_by_group: dict[str, dict[int, list[RoutePlan]]] = {}
    for group in groups:
        plans = group_plans(group)
        plans_by_group[group.name] = plans
        for slot, variants in plans.items():
            tasks.append((group.name, slot, [plan.summary() for plan in variants]))
    results: dict[tuple[str, int], list[RefinedRoute | None]] = {}

    def record(name: str, slot: int, routes: list) -> None:
        results[(name, slot)] = routes
        if on_progress is not None:
            on_progress(
                {
                    "group": name,
                    "slot": slot,
                    "certified": [r is not None and r.certified for r in routes],
                    "collected_kg": [None if r is None else r.total_collected_kg for r in routes],
                    "elapsed_seconds": time.perf_counter() - started,
                    "done": len(results),
                    "total": len(tasks),
                }
            )

    if workers <= 1:
        for task in tasks:
            record(*_recertify_in_worker(task))
    else:
        context = multiprocessing.get_context("fork" if os.name == "posix" else "spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as executor:
            for name, slot, routes in executor.map(_recertify_in_worker, tasks):
                record(name, slot, routes)

    bundles: list[ClusterBundle] = []
    for index, group in enumerate(groups):
        ships: list[BundleShip] = []
        rejected: list[dict[str, Any]] = []
        for ship in group.ships:
            routes = results.get((group.name, ship.slot), [])
            certified = [r for r in routes if r is not None and r.certified]
            if not routes or routes[0] is None or not routes[0].certified:
                rejected.append(
                    {
                        "reason": "archived primary route failed re-certification",
                        "ship": ship.slot,
                        "path": str(ship.summaries[0][0]),
                        "failures": [] if not routes or routes[0] is None else routes[0].failures,
                    }
                )
                if not certified:
                    continue
            primary = routes[0] if routes and routes[0] in certified else certified[0]
            ships.append(BundleShip(ship.slot, primary, certified, {"archived": group.name}))
        bundle = ClusterBundle(first_label + index, (), ships, rejected=rejected)
        make_consistent(bundle)  # a collector whose deployer failed is dropped, not the bundle
        bundle.members = tuple(sorted({a for s in bundle.ships for a in s.route.plan.asteroids}))
        bundle.wall_seconds = time.perf_counter() - started
        bundles.append(bundle)
    return bundles
