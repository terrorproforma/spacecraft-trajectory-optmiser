"""Archive-wide Earth-return re-timing (``gtoc12 retime-returns``).

Every certified stand-alone ship in the archives (no foreign miners, no orphans) is re-flown
through SCvx, its Earth return swept on the lattice (:mod:`returnsweep`) and the route re-timed
against the measured return costs.  Ships whose re-timed route certifies with more collected
mass are archived as ``<output>/<group>/ship_<slot>/route_summary.json`` so the next
``fleet-master`` picks them up as columns next to the originals (the master keeps whichever
the LP prefers).  Best ships first, so a time budget spends its SCvx on the ships that decide
the fleet average.
"""

from __future__ import annotations

import concurrent.futures
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

__all__ = ["ReturnCampaignSettings", "ReturnTask", "run_return_campaign", "select_tasks"]


@dataclass(slots=True)
class ReturnCampaignSettings:
    workers: int = 3
    top: int | None = None  # ships to re-time (best first); all when None
    min_collected_kg: float = 450.0  # below this a ship cannot reach the fleet average anyway
    time_budget_seconds: float = 6000.0
    per_ship_seconds: float = 900.0
    back_steps: int = 6
    forward_steps: int = 6
    max_attempts: int = 2
    neighbourhood: int = 40
    inflation_fit: str | None = "results/gtoc12/hop_inflation_fit.json"


@dataclass(slots=True)
class ReturnTask:
    group: str
    slot: int
    path: Path
    summary: dict[str, Any] = field(repr=False)

    @property
    def collected_kg(self) -> float:
        return float(self.summary["total_collected_kg"])

    @property
    def asteroids(self) -> tuple[int, ...]:
        plan = self.summary.get("plan") or {}
        return tuple(sorted(int(a) for a in plan.get("asteroids", ())))

    @property
    def name(self) -> str:
        return f"{_safe(self.group)}/ship_{self.slot:02d}"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def select_tasks(
    groups: list[ArchivedGroup], settings: ReturnCampaignSettings | None = None
) -> list[ReturnTask]:
    """Stand-alone certified primaries, best first, one per asteroid set (the same route
    archived by several runs is re-timed once)."""

    settings = settings or ReturnCampaignSettings()
    tasks: list[ReturnTask] = []
    for group in groups:
        for ship in group.ships:
            path, summary = ship.summaries[0]
            plan = summary.get("plan") or {}
            if plan.get("foreign_deploy_epochs") or plan.get("orphaned"):
                continue
            if not any(leg.get("role") == "earth_return" for leg in plan.get("legs", [])):
                continue
            if float(summary["total_collected_kg"]) < settings.min_collected_kg:
                continue
            tasks.append(ReturnTask(group.name, ship.slot, path, summary))
    tasks.sort(key=lambda t: (-t.collected_kg, t.group, t.slot))
    seen: set[tuple[int, ...]] = set()
    unique: list[ReturnTask] = []
    for task in tasks:
        key = task.asteroids
        if key in seen:
            continue
        seen.add(key)
        unique.append(task)
    if settings.top is not None:
        unique = unique[: settings.top]
    return unique


_WORKER: dict[str, Any] = {}


def _retime_in_worker(task: tuple[str, int, str, dict[str, Any], str]) -> dict[str, Any]:
    group, slot, path, summary, output = task
    state = _WORKER
    if "catalogue" not in state:
        from .data import load_catalogue

        state["catalogue"] = load_catalogue()
    bound_heap_growth()
    settings: ReturnCampaignSettings = state["settings"]
    from .bundles import ClusterPricingSettings, cluster_retime_settings, cluster_search_settings
    from .pipeline import plan_from_route_summary, refine_route, write_route_artifacts
    from .returnsweep import retime_return

    catalogue = state["catalogue"]
    started = time.perf_counter()
    record: dict[str, Any] = {
        "group": group,
        "slot": slot,
        "path": path,
        "archived_kg": float(summary["total_collected_kg"]),
    }
    try:
        plan = plan_from_route_summary(summary)
        route = refine_route(plan, catalogue, scvx=state.get("scvx"))
        record["recertified"] = route.certified
        record["recertify_seconds"] = time.perf_counter() - started
        if not route.certified:
            record["status"] = "archived route failed re-certification"
            record["failures"] = route.failures
            return record
        pricing = ClusterPricingSettings(collect_dp_inflation_fit=settings.inflation_fit)
        search_settings = cluster_search_settings(pricing, settings.neighbourhood)
        retime_settings = cluster_retime_settings(pricing, last=True)
        result = retime_return(
            route,
            catalogue,
            search_settings=search_settings,
            retime_settings=retime_settings,
            weights=state.get("weights"),
            scvx=state.get("scvx"),
            back_steps=settings.back_steps,
            forward_steps=settings.forward_steps,
            max_attempts=settings.max_attempts,
            time_budget_seconds=max(
                settings.per_ship_seconds - (time.perf_counter() - started), 60.0
            ),
        )
        record.update(result.summary())
        record["status"] = "improved" if result.route is not None else "no gain"
        if result.route is not None:
            directory = Path(output) / _safe(group) / f"ship_{slot:02d}"
            write_route_artifacts(result.route, catalogue, directory)
            record["archived_to"] = str(directory / "route_summary.json")
            record["legs"] = [
                {
                    "role": leg.planned.role,
                    "tof_days": leg.planned.tof_days,
                    "propellant_kg": None if leg.solution is None else leg.solution.propellant_kg,
                }
                for leg in result.route.legs
            ]
    except Exception as error:  # one ship's crash must not end the campaign
        import traceback

        record["status"] = f"crashed: {error!r}"
        record["traceback"] = traceback.format_exc()[-2000:]
    record["wall_seconds"] = time.perf_counter() - started
    record["peak_rss_mb"] = peak_rss_mb()
    return record


def run_return_campaign(
    catalogue: AsteroidCatalogue,
    sources: list[Path],
    output: Path,
    *,
    settings: ReturnCampaignSettings | None = None,
    scvx: ScvxSettings | None = None,
    weights: dict[int, float] | None = None,
    on_result=None,
) -> dict[str, Any]:
    """Re-time the returns of the archived ships under ``sources``; per-ship records + totals."""

    settings = settings or ReturnCampaignSettings()
    started = time.perf_counter()
    groups = discover_archives(sources)
    tasks = select_tasks(groups, settings)
    output.mkdir(parents=True, exist_ok=True)
    _WORKER.update(catalogue=catalogue, settings=settings, scvx=scvx, weights=weights)
    payload = [(t.group, t.slot, str(t.path), t.summary, str(output)) for t in tasks]
    records: list[dict[str, Any]] = []

    def take(record: dict[str, Any]) -> None:
        record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        record["done"] = len(records) + 1
        record["total"] = len(payload)
        records.append(record)
        if on_result is not None:
            on_result(record)

    if settings.workers <= 1:
        for task in payload:
            if time.perf_counter() - started > settings.time_budget_seconds:
                break
            take(_retime_in_worker(task))
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
                    running[executor.submit(_retime_in_worker, queue.pop(0))] = len(running)
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
    return {
        "tasks": len(tasks),
        "attempted": len(records),
        "improved": len(improved),
        "gain_kg_total": round(sum(float(r.get("gain_kg", 0.0)) for r in improved), 2),
        "return_before_kg": [r.get("return_before_kg") for r in records],
        "return_after_kg": [r.get("return_after_kg") for r in records],
        "records": records,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "worker_peak_rss_mb": max((float(r.get("peak_rss_mb") or 0.0) for r in records), default=0),
        "settings": {
            "workers": settings.workers,
            "top": settings.top,
            "min_collected_kg": settings.min_collected_kg,
            "time_budget_seconds": settings.time_budget_seconds,
            "per_ship_seconds": settings.per_ship_seconds,
            "back_steps": settings.back_steps,
            "forward_steps": settings.forward_steps,
            "max_attempts": settings.max_attempts,
        },
    }
