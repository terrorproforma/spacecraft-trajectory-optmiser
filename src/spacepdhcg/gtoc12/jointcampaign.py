"""Archive-wide whole-itinerary joint re-optimisation (``gtoc12 joint-itinerary``).

The ships of a fleet report (``fleet_master_v6``: the 20 ships the master selected, matched
to their archives by asteroid set) come first, then the best remaining stand-alone certified
ships of the archives.  Each is warm-started from its archived certified legs and jointly
re-optimised (:mod:`jointopt`); a ship whose re-optimised route certifies with more collected
mass is archived as ``<output>/<group>/ship_<slot>/route_summary.json`` so the next
``fleet-master`` sees it as a column next to the original.  Other fleet ships' asteroids are
never inserted (the master needs disjoint columns to keep the fleet).
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

from .archive import ArchivedGroup, discover_archives
from .data import AsteroidCatalogue
from .low_thrust import ScvxSettings
from .memory import bound_heap_growth, peak_rss_mb

__all__ = [
    "JointCampaignSettings",
    "JointTask",
    "fleet_asteroid_sets",
    "run_joint_campaign",
    "select_tasks",
]


@dataclass(slots=True)
class JointCampaignSettings:
    workers: int = 3
    top: int | None = None  # stand-alone ships beyond the fleet's (best first); all when None
    min_collected_kg: float = 450.0
    time_budget_seconds: float = 9000.0
    per_ship_seconds: float = 900.0
    fleet_report: str | None = None  # run_report.json whose master.selected ships go first
    mesh_days: tuple[float, ...] = (45.0, 20.0, 8.0, 3.0, 1.0)
    max_certifications: int = 10
    margin_price: float = 0.05
    insert: bool = True
    insert_neighbours: int = 40
    insert_radius: float = 2.5
    insert_trials: int = 3
    neighbourhood: int = 40
    inflation_fit: str | None = "results/gtoc12/hop_inflation_fit.json"
    # Earth-out leg stage (jointopt.JointSettings.earth_leg): earlier chain start bought with
    # Earth-leg propellant, single-leg SCvx measurements, monotone whole-route acceptance
    earth_leg: bool = False
    earth_leg_shifts_days: tuple[float, ...] = (30.0, 60.0, 90.0, 120.0, 150.0)
    earth_leg_certifications: int = 4


@dataclass(slots=True)
class JointTask:
    group: str
    slot: int
    path: Path
    summary: dict[str, Any] = field(repr=False)
    in_fleet: bool = False

    @property
    def collected_kg(self) -> float:
        return float(self.summary["total_collected_kg"])

    @property
    def asteroids(self) -> tuple[int, ...]:
        plan = self.summary.get("plan") or {}
        ids = plan.get("asteroids") or self.summary.get("asteroids") or ()
        return tuple(sorted(int(a) for a in ids))

    @property
    def name(self) -> str:
        return f"{_safe(self.group)}/ship_{self.slot:02d}"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def fleet_asteroid_sets(report: Path | str | None) -> list[tuple[int, ...]]:
    """Asteroid sets (sorted deploys) of the ships a fleet-master report selected."""

    if report is None:
        return []
    data = json.loads(Path(report).read_text(encoding="utf-8"))
    selected = (data.get("master") or {}).get("selected") or []
    sets: list[tuple[int, ...]] = []
    for column in selected:
        deploys = column.get("deploys") or []
        if deploys:
            sets.append(tuple(sorted(int(a) for a in deploys)))
    return sets


def _stand_alone(summary: dict[str, Any]) -> bool:
    """A ship collecting only its own miners and returning to Earth (an uncollected miner of
    its own - an orphan nobody in a fleet of stand-alone ships picks up - is allowed)."""

    plan = summary.get("plan")
    if plan is None:
        # older archives carry only the flown legs: rebuild the plan to classify the ship
        from .pipeline import plan_from_route_summary

        try:
            rebuilt = plan_from_route_summary(summary)
        except (KeyError, ValueError, ZeroDivisionError):
            return False
        return not rebuilt.foreign_deploy_epochs and any(
            leg.role == "earth_return" for leg in rebuilt.legs
        )
    if plan.get("foreign_deploy_epochs"):
        return False
    return any(leg.get("role") == "earth_return" for leg in plan.get("legs", []))


def select_tasks(
    groups: list[ArchivedGroup],
    settings: JointCampaignSettings | None = None,
    fleet_sets: list[tuple[int, ...]] | None = None,
) -> list[JointTask]:
    """Stand-alone certified primaries, one per asteroid set: the fleet's ships first (in the
    report's order), then the rest best first (``top`` of them)."""

    settings = settings or JointCampaignSettings()
    fleet_sets = list(fleet_sets or ())
    fleet_index = {key: k for k, key in enumerate(fleet_sets)}
    tasks: list[JointTask] = []
    for group in groups:
        for ship in group.ships:
            # the master picks among all archived variants of a ship: a fleet ship may be a
            # variant other than the primary (best) archive
            for index, (path, summary) in enumerate(ship.summaries):
                if not _stand_alone(summary):
                    continue
                task = JointTask(group.name, ship.slot, path, summary)
                task.in_fleet = task.asteroids in fleet_index
                if index > 0 and not task.in_fleet:
                    continue
                if not task.in_fleet and task.collected_kg < settings.min_collected_kg:
                    continue
                tasks.append(task)
    # the same asteroid set archived by several runs is optimised once: keep the heaviest
    tasks.sort(key=lambda t: (-t.collected_kg, t.group, t.slot))
    unique: dict[tuple[int, ...], JointTask] = {}
    for task in tasks:
        unique.setdefault(task.asteroids, task)
    fleet = sorted(
        (t for t in unique.values() if t.in_fleet), key=lambda t: fleet_index[t.asteroids]
    )
    others = [t for t in unique.values() if not t.in_fleet]
    others.sort(key=lambda t: (-t.collected_kg, t.group, t.slot))
    if settings.top is not None:
        others = others[: settings.top]
    return [*fleet, *others]


_WORKER: dict[str, Any] = {}


def _optimise_in_worker(
    task: tuple[str, int, str, dict[str, Any], str, list[int]],
) -> dict[str, Any]:
    group, slot, path, summary, output, excluded = task
    state = _WORKER
    if "catalogue" not in state:
        from .data import load_catalogue

        state["catalogue"] = load_catalogue()
    bound_heap_growth()
    settings: JointCampaignSettings = state["settings"]
    from .bundles import ClusterPricingSettings, cluster_retime_settings, cluster_search_settings
    from .jointopt import JointSettings, optimise_ship, route_from_summary
    from .pipeline import write_route_artifacts
    from .retiming import Retimer

    catalogue = state["catalogue"]
    started = time.perf_counter()
    record: dict[str, Any] = {
        "group": group,
        "slot": slot,
        "path": path,
        "archived_kg": float(summary["total_collected_kg"]),
        "asteroids_before": len((summary.get("plan") or {}).get("asteroids", ())),
    }
    try:
        route = route_from_summary(summary)
        pricing = ClusterPricingSettings(collect_dp_inflation_fit=settings.inflation_fit)
        search_settings = cluster_search_settings(pricing, settings.neighbourhood)
        retime_settings = cluster_retime_settings(pricing, last=True)
        retimer = Retimer(catalogue, search_settings, retime_settings, state.get("weights"))
        joint_settings = JointSettings(
            mesh_days=settings.mesh_days,
            max_certifications=settings.max_certifications,
            margin_price=settings.margin_price,
            time_budget_seconds=settings.per_ship_seconds,
            insert=settings.insert,
            insert_neighbours=settings.insert_neighbours,
            insert_radius=settings.insert_radius,
            insert_trials=settings.insert_trials,
            earth_leg=settings.earth_leg,
            earth_leg_shifts_days=tuple(settings.earth_leg_shifts_days),
            earth_leg_certifications=settings.earth_leg_certifications,
        )
        result = optimise_ship(
            route,
            catalogue,
            retimer,
            weights=state.get("weights"),
            scvx=state.get("scvx"),
            settings=joint_settings,
            search_settings=search_settings,
            excluded=set(excluded),
        )
        record.update(result.summary())
        record["status"] = "improved" if result.route is not None else "no gain"
        if result.route is not None:
            directory = Path(output) / _safe(group) / f"ship_{slot:02d}"
            write_route_artifacts(result.route, catalogue, directory)
            record["archived_to"] = str(directory / "route_summary.json")
    except Exception as error:  # one ship's crash must not end the campaign
        import traceback

        record["status"] = f"crashed: {error!r}"
        record["traceback"] = traceback.format_exc()[-2000:]
    record["wall_seconds"] = time.perf_counter() - started
    record["peak_rss_mb"] = peak_rss_mb()
    return record


def run_joint_campaign(
    catalogue: AsteroidCatalogue,
    sources: list[Path],
    output: Path,
    *,
    settings: JointCampaignSettings | None = None,
    scvx: ScvxSettings | None = None,
    weights: dict[int, float] | None = None,
    on_result=None,
) -> dict[str, Any]:
    """Jointly re-optimise the archived ships under ``sources``; per-ship records + totals."""

    settings = settings or JointCampaignSettings()
    started = time.perf_counter()
    groups = discover_archives(sources)
    fleet_sets = fleet_asteroid_sets(settings.fleet_report)
    tasks = select_tasks(groups, settings, fleet_sets)
    fleet_asteroids = {a for key in fleet_sets for a in key}
    output.mkdir(parents=True, exist_ok=True)
    _WORKER.update(catalogue=catalogue, settings=settings, scvx=scvx, weights=weights)
    payload = [
        (
            t.group,
            t.slot,
            str(t.path),
            t.summary,
            str(output),
            sorted(fleet_asteroids - set(t.asteroids)),
        )
        for t in tasks
    ]
    records: list[dict[str, Any]] = []
    in_fleet = {(t.group, t.slot): t.in_fleet for t in tasks}

    def take(record: dict[str, Any]) -> None:
        record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        record["done"] = len(records) + 1
        record["total"] = len(payload)
        record["in_fleet"] = in_fleet.get((record["group"], record["slot"]), False)
        records.append(record)
        if on_result is not None:
            on_result(record)

    if settings.workers <= 1:
        for task in payload:
            if time.perf_counter() - started > settings.time_budget_seconds:
                break
            take(_optimise_in_worker(task))
    else:
        context = multiprocessing.get_context("fork" if os.name == "posix" else "spawn")
        queue = list(payload)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=settings.workers, mp_context=context
        ) as executor:
            running: dict[concurrent.futures.Future, int] = {}
            while queue or running:
                while (
                    queue
                    and len(running) < settings.workers
                    and time.perf_counter() - started <= settings.time_budget_seconds
                ):
                    running[executor.submit(_optimise_in_worker, queue.pop(0))] = len(running)
                if not running:
                    break
                done, _ = concurrent.futures.wait(
                    running, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    running.pop(future)
                    take(future.result())
                if time.perf_counter() - started > settings.time_budget_seconds:
                    queue.clear()
    improved = [r for r in records if r.get("status") == "improved"]
    fleet_records = [r for r in records if r.get("in_fleet")]
    before = [float(r["archived_kg"]) for r in fleet_records]
    after = [float(r.get("after_kg") or r["archived_kg"]) for r in fleet_records]
    return {
        "tasks": len(tasks),
        "fleet_ships": sum(1 for t in tasks if t.in_fleet),
        "attempted": len(records),
        "improved": len(improved),
        "inserted": sum(1 for r in improved if r.get("inserted") is not None),
        "gain_kg_total": round(sum(float(r.get("gain_kg", 0.0)) for r in improved), 2),
        "earth_leg": {
            "ships_with_stage": sum(1 for r in records if r.get("earth_leg")),
            "legs_flown": sum(int((r.get("earth_leg") or {}).get("flown") or 0) for r in records),
            "legs_measured": sum(
                int((r.get("earth_leg") or {}).get("measured") or 0) for r in records
            ),
            "accepted": sum(
                1
                for r in records
                if (r.get("earth_leg") or {}).get("accepted_shift_days") is not None
            ),
            "accepted_shift_days": [
                (r.get("earth_leg") or {}).get("accepted_shift_days")
                for r in records
                if (r.get("earth_leg") or {}).get("accepted_shift_days") is not None
            ],
        },
        "fleet_average_before_kg": (sum(before) / len(before)) if before else None,
        "fleet_average_after_kg": (sum(after) / len(after)) if after else None,
        "records": records,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "worker_peak_rss_mb": max((float(r.get("peak_rss_mb") or 0.0) for r in records), default=0),
        "settings": {
            "workers": settings.workers,
            "top": settings.top,
            "min_collected_kg": settings.min_collected_kg,
            "time_budget_seconds": settings.time_budget_seconds,
            "per_ship_seconds": settings.per_ship_seconds,
            "fleet_report": settings.fleet_report,
            "mesh_days": list(settings.mesh_days),
            "max_certifications": settings.max_certifications,
            "margin_price": settings.margin_price,
            "insert": settings.insert,
            "insert_neighbours": settings.insert_neighbours,
            "insert_radius": settings.insert_radius,
            "insert_trials": settings.insert_trials,
            "earth_leg": settings.earth_leg,
            "earth_leg_shifts_days": list(settings.earth_leg_shifts_days),
            "earth_leg_certifications": settings.earth_leg_certifications,
        },
    }
