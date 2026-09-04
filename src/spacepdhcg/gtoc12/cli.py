"""``spacepdhcg gtoc12`` command group."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=float)


def _commit(repository: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def cmd_fetch(args: argparse.Namespace) -> int:
    from .data import REPOSITORY_ROOT

    script = REPOSITORY_ROOT / "scripts" / "gtoc12" / "fetch_gtoc12_data.py"
    command = [sys.executable, str(script)]
    if args.data_dir:
        command += ["--data-dir", str(args.data_dir)]
    return subprocess.run(command, check=False).returncode


def cmd_verify(args: argparse.Namespace) -> int:
    from .data import load_bonus_table, load_catalogue
    from .official import official_verifier_available, run_official_verifier
    from .verifier import Gtoc12Verifier

    catalogue = load_catalogue()
    bonus = None
    try:
        bonus = load_bonus_table()
    except Exception:
        pass
    started = time.perf_counter()
    report = Gtoc12Verifier(catalogue, bonus=bonus, rtol=args.rtol).verify_file(args.solution)
    output: dict[str, Any] = {
        "independent": report.summary(),
        "independent_seconds": time.perf_counter() - started,
    }
    output["independent"]["scored_masses"] = report.scored_masses
    if args.official:
        if not official_verifier_available():
            output["official"] = {"error": "official verifier binary or catalogue not available"}
        else:
            official = run_official_verifier(args.solution)
            output["official"] = official.summary()
            output["official"]["score_data"] = official.score_data
            output["agreement"] = {
                "ships": official.ships == report.ship_count,
                "mined_asteroids": official.mined_asteroids == report.mined_asteroid_count,
                "asteroid_set": set(official.score_data) == set(report.scored_masses),
                "max_mass_difference_kg": max(
                    (
                        abs(official.score_data[k] - report.scored_masses[k])
                        for k in report.scored_masses
                        if k in official.score_data
                    ),
                    default=None,
                ),
            }
    print(_json(output))
    return 0 if report.ok else 1


def cmd_reduced_instance(args: argparse.Namespace) -> int:
    from .data import load_catalogue
    from .reduced_instance import build_reduced_instance

    instance = build_reduced_instance(load_catalogue(), args.rule)
    summary = instance.summary()
    if args.list_ids:
        summary["asteroid_ids"] = instance.asteroid_ids.tolist()
    print(_json(summary))
    return 0


def _peak_rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # pragma: no cover - non-POSIX
        return float("nan")


def _process_tree_pss_mb() -> float:
    """Proportional set size (MB) of this process and its children, from ``/proc``.

    Forked workers share the catalogue and the imported libraries copy-on-write, so the sum
    of their RSS over-counts the memory the campaign really holds; PSS splits every shared
    page between its owners and sums to the true total.  ``nan`` where ``/proc`` is absent.
    """

    try:
        me = os.getpid()
        pids = [me]
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/status", encoding="utf-8") as handle:
                    for line in handle:
                        if line.startswith("PPid:"):
                            if int(line.split()[1]) == me:
                                pids.append(int(entry))
                            break
            except OSError:
                continue
        total_kb = 0
        for pid in pids:
            try:
                with open(f"/proc/{pid}/smaps_rollup", encoding="utf-8") as handle:
                    for line in handle:
                        if line.startswith("Pss:"):
                            total_kb += int(line.split()[1])
                            break
            except OSError:
                continue
        return total_kb / 1024.0
    except Exception:  # pragma: no cover - non-Linux
        return float("nan")


class _MemorySampler:
    """Background thread recording the peak process-tree PSS every ``interval`` seconds."""

    def __init__(self, interval: float = 15.0) -> None:
        import threading

        self.interval = interval
        self.peak_mb = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="memory-sampler", daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = _process_tree_pss_mb()
            if value == value:  # not nan
                self.peak_mb = max(self.peak_mb, value)
                self.samples += 1
            self._stop.wait(self.interval)

    def start(self) -> _MemorySampler:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


def catalogue_pool(catalogue, args: argparse.Namespace):
    """Full-catalogue pool: the reference a/e/i box, so screening memory stays bounded."""

    import numpy as np

    from . import constants as C

    a_au = catalogue.semi_major_axis_km / C.AU_KM
    mask = (
        (a_au >= args.pool_a_min)
        & (a_au <= args.pool_a_max)
        & (catalogue.eccentricity <= args.pool_e_max)
        & (np.rad2deg(catalogue.inclination_rad) <= args.pool_i_max)
    )
    return catalogue.ids[mask]


def cmd_run(args: argparse.Namespace) -> int:
    """Search -> refine -> emit -> verify (official + independent) -> viewer export.

    With ``--ships N`` the same loop builds a greedy fleet: each ship searches with the previous
    ships' asteroids excluded, and the certified routes are assembled into one fleet file that is
    verified as a whole.
    """

    from .cooperative import FleetColumn, MinerPool, solve_fleet_master
    from .data import REPOSITORY_ROOT, load_bonus_table, load_catalogue
    from .fleet import FleetPlan, assemble_fleet
    from .low_thrust import ScvxSettings
    from .official import official_verifier_available, run_official_verifier
    from .pipeline import refine_route, write_route_artifacts
    from .reduced_instance import build_reduced_instance
    from .retiming import Retimer, improve_and_certify
    from .search import RouteSearch, SearchSettings
    from .solution import Solution
    from .verifier import Gtoc12Verifier
    from .viewer_export import write_viewer_dataset

    started = time.perf_counter()
    catalogue = load_catalogue()
    bonus_table = _optional_bonus(load_bonus_table)
    weights: dict[int, float] | None = None
    if bonus_table is not None and not args.no_bonus_weights:
        weights = {
            int(asteroid): float(bonus_table.coefficient[asteroid - 1])
            for asteroid in catalogue.ids
        }
    if args.full_catalogue:
        ids = catalogue_pool(catalogue, args)
        instance_summary = {
            "instance_id": "gtoc12-full-catalogue",
            "catalogue_asteroids": int(catalogue.ids.shape[0]),
            "pool_filter": {
                "a_au": [args.pool_a_min, args.pool_a_max],
                "e_max": args.pool_e_max,
                "i_max_deg": args.pool_i_max,
            },
            "pool_asteroids": int(ids.shape[0]),
        }
    else:
        instance = build_reduced_instance(catalogue, args.rule)
        ids = instance.asteroid_ids
        instance_summary = instance.summary()
    settings = SearchSettings(
        beam_width=args.beam_width,
        max_deploys=args.max_deploys,
        seed=args.seed,
        neighbours=args.neighbours,
        time_budget_seconds=args.search_budget_seconds,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "run_id": args.run_id,
        "commit": _commit(REPOSITORY_ROOT),
        "instance": instance_summary,
        "settings": {
            "beam_width": args.beam_width,
            "max_deploys": args.max_deploys,
            "neighbours": args.neighbours,
            "seed": args.seed,
            "ships": args.ships,
            "refine_top": args.refine_top,
            "search_budget_seconds": args.search_budget_seconds,
            "budget_seconds": args.budget_seconds,
            "retime": not args.no_retime,
            "retime_attempts": args.retime_attempts,
            "retime_budget_seconds": args.retime_budget_seconds,
            "bonus_weights": weights is not None,
            "cooperative": not args.no_cooperative,
        },
        "cpu_only": True,
        "gpu_used": False,
        "ships": [],
        "timeline": [],  # (elapsed seconds, ships, fleet collected kg) after each ship
    }
    scvx = ScvxSettings(max_iterations=args.scvx_iterations, node_days=args.node_days)
    fleet = FleetPlan()  # greedy incumbent (one certified route per ship slot)
    pool = MinerPool()  # shared miners: later ships may collect earlier ships' orphans
    columns: list[FleetColumn] = []  # every certified itinerary, for the master
    verifier = Gtoc12Verifier(catalogue, bonus=bonus_table)

    def add_column(slot: int, label: str, refined) -> None:
        columns.append(
            FleetColumn.from_plan(
                len(columns),
                slot,
                label,
                refined.plan,
                refined.collected_mass,
                certified=refined.certified,
                route=refined,
            )
        )

    def run_master():
        master = solve_fleet_master(columns, weights=weights, max_ships=args.ships)
        report["master"] = master.summary()
        report["master"]["greedy_collected_kg"] = sum(fleet.collected_kg)
        report["master"]["greedy_ships"] = len(fleet.routes)
        report["master"]["columns"] = len(columns)
        report["miner_pool"] = pool.summary()
        return master

    def checkpoint() -> None:
        report["fleet"] = fleet.summary()
        report["wall_seconds_total"] = time.perf_counter() - started
        report["peak_rss_mb"] = _peak_rss_mb()
        (output_dir / "run_report.json").write_text(_json(report) + "\n", encoding="utf-8")

    def verify(solution_path: Path, histories: dict | None = None) -> dict[str, Any]:
        checker = (
            verifier
            if histories is None
            else Gtoc12Verifier(
                catalogue, bonus=_optional_bonus(load_bonus_table), history=histories
            )
        )
        independent = checker.verify_file(solution_path)
        entry: dict[str, Any] = {"independent": independent.summary()}
        entry["independent"]["scored_masses"] = independent.scored_masses
        if official_verifier_available():
            official = run_official_verifier(solution_path)
            entry["official"] = official.summary()
            entry["official"]["score_data"] = official.score_data
        score = entry.get("official", {}).get("total_mass_kg")
        entry["score_kg"] = independent.total_mass_kg if score is None else score
        return entry

    for ship_index in range(1, args.ships + 1):
        elapsed = time.perf_counter() - started
        if elapsed > args.budget_seconds:
            report["stopped"] = f"wall-clock budget reached before ship {ship_index}"
            break
        ship_dir = output_dir / f"ship_{ship_index:02d}"
        ship_dir.mkdir(parents=True, exist_ok=True)
        search = RouteSearch(
            catalogue,
            ids,
            settings,
            excluded=fleet.used_asteroids() | pool.touched(),
            weights=weights,
            seeds=None if args.no_cooperative else pool.orphans(),
        )
        result = search.run()
        ship_report: dict[str, Any] = {
            "ship": ship_index,
            "excluded_asteroids": sorted(fleet.used_asteroids()),
            "search": {
                "expansions": result.expansions,
                "lambert_evaluations": result.lambert_evaluations,
                "wall_seconds": result.wall_seconds,
                "candidates": len(result.candidates),
                "failures": len(result.failures),
                "depth_reached": result.depth_reached,
                "best_by_depth": result.best_by_depth,
                "top_candidates": [item.summary() for item in result.candidates[: args.refine_top]],
                "failed_chains": result.failures[:200],
            },
            "peak_rss_mb": _peak_rss_mb(),
        }
        (ship_dir / "search.json").write_text(_json(ship_report) + "\n", encoding="utf-8")
        print(
            _json(
                {
                    "ship": ship_index,
                    "candidates": len(result.candidates),
                    "best_by_depth": result.best_by_depth,
                    "search_wall_seconds": round(result.wall_seconds, 1),
                    "peak_rss_mb": round(_peak_rss_mb(), 1),
                }
            ),
            flush=True,
        )
        report["ships"].append(ship_report)
        if args.search_only or not result.candidates:
            ship_report["status"] = "search_only" if result.candidates else "no_candidates"
            if not result.candidates:
                break
            continue
        best_entry = None
        refinements = []
        failed_legs: set[tuple[int, int, float]] = set()
        attempts = 0
        for rank, plan in enumerate(result.candidates):
            if attempts >= args.refine_top:
                break
            # retain failed chains: never re-refine a plan that reuses a leg SCvx already
            # proved infeasible (same bodies and departure epoch)
            if any(
                (leg.from_id, leg.to_id, round(leg.departure_epoch, 3)) in failed_legs
                for leg in plan.legs
            ):
                refinements.append({"rank": rank, "skipped": "contains a failed leg"})
                continue
            attempts += 1
            refined = refine_route(plan, catalogue, scvx=scvx)
            for leg in refined.legs:
                if not leg.certified:
                    failed_legs.add(
                        (
                            leg.planned.from_id,
                            leg.planned.to_id,
                            round(leg.planned.departure_epoch, 3),
                        )
                    )
            entry: dict[str, Any] = {
                "rank": rank,
                "plan": plan.summary(),
                "refined": refined.summary(),
            }
            if refined.certified:
                directory = ship_dir / f"candidate_{rank:02d}"
                artifacts = write_route_artifacts(refined, catalogue, directory)
                solution_path = Path(artifacts["solution"])
                histories: dict = {}
                entry.update(verify(solution_path, histories))
                viewer = write_viewer_dataset(
                    directory / "viewer",
                    Solution.read(solution_path),
                    histories,
                    catalogue,
                    run_id=f"{args.run_id}_s{ship_index:02d}_c{rank:02d}",
                    commit=report["commit"],
                    verification=entry["independent"],
                    solution_path=solution_path,
                    instance_id=str(instance_summary["instance_id"]),
                )
                entry["viewer_manifest"] = viewer
                entry["artifacts"] = {
                    "solution": str(solution_path),
                    "viewer": str(directory / "viewer" / "trajectories.json"),
                }
                if best_entry is None or entry["score_kg"] > best_entry[0]["score_kg"]:
                    best_entry = (entry, refined)
                add_column(ship_index, f"s{ship_index:02d}_c{rank:02d}", refined)
            refinements.append(entry)
            (ship_dir / "refinements.json").write_text(_json(refinements) + "\n", encoding="utf-8")
            print(
                _json(
                    {
                        "ship": ship_index,
                        "rank": rank,
                        "certified": refined.certified,
                        "score_kg": entry.get("score_kg"),
                        "refined_arcs": refined.refined_arc_count,
                    }
                ),
                flush=True,
            )
            if best_entry is not None and args.stop_at_first_certified:
                break
        ship_report["refinements"] = refinements
        if best_entry is None:
            ship_report["status"] = "no_certified_route"
            checkpoint()
            break
        if not args.no_retime:
            # joint re-timing + chain extension with SCvx in the loop; the previously
            # certified route is kept unless the re-timed one certifies with a higher score
            retimer = Retimer(catalogue, settings, weights=weights)
            improvement = improve_and_certify(
                best_entry[1].plan,
                search,
                retimer,
                catalogue,
                scvx=scvx,
                max_attempts=args.retime_attempts,
                time_budget_seconds=args.retime_budget_seconds,
                pool=None if args.no_cooperative else pool,
            )
            for index, variant in enumerate(improvement.certified_routes):
                add_column(ship_index, f"s{ship_index:02d}_retimed{index}", variant)
            improvement_summary = improvement.summary()
            improvement_summary["before"] = {
                "score_kg": best_entry[0]["score_kg"],
                "asteroids": len(best_entry[1].plan.asteroids),
                "final_mass_kg": best_entry[1].final_mass_kg,
            }
            if improvement.route is not None:
                directory = ship_dir / "retimed"
                artifacts = write_route_artifacts(improvement.route, catalogue, directory)
                solution_path = Path(artifacts["solution"])
                entry = {"rank": "retimed", "plan": improvement.route.plan.summary()}
                entry["refined"] = improvement.route.summary()
                entry.update(verify(solution_path))
                entry["artifacts"] = {"solution": str(solution_path)}
                improvement_summary["after"] = {
                    "score_kg": entry["score_kg"],
                    "asteroids": len(improvement.route.plan.asteroids),
                    "final_mass_kg": improvement.route.final_mass_kg,
                }
                if entry["score_kg"] > best_entry[0]["score_kg"]:
                    best_entry = (entry, improvement.route)
            ship_report["retiming"] = improvement_summary
            (ship_dir / "retiming.json").write_text(
                _json(improvement_summary) + "\n", encoding="utf-8"
            )
            print(
                _json(
                    {
                        "ship": ship_index,
                        "retiming": {
                            "before": improvement_summary["before"],
                            "after": improvement_summary.get("after"),
                            "attempts": len(improvement.attempts),
                            "wall_seconds": round(improvement.wall_seconds, 1),
                        },
                    }
                ),
                flush=True,
            )
        ship_report["best"] = best_entry[0]
        ship_report["status"] = "scored"
        fleet.routes.append(best_entry[1])
        pool.register(best_entry[1].plan, ship_index)
        master = run_master()
        report["timeline"].append(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "ships": len(fleet.routes),
                "fleet_collected_kg": sum(fleet.collected_kg),
                "master_ships": len(master.selected),
                "master_collected_kg": master.collected_kg,
                "master_objective_kg": master.objective,
                "peak_rss_mb": _peak_rss_mb(),
            }
        )
        checkpoint()
    report["fleet"] = fleet.summary()
    if fleet.routes:
        # the master picks the fleet from every certified column (the greedy fleet is one of
        # its feasible subsets, so it never scores lower)
        master = run_master()
        selected = FleetPlan(master.routes())
        report["fleet"] = selected.summary()
        report["fleet"]["greedy"] = fleet.summary()
        fleet_dir = output_dir / "fleet"
        fleet_dir.mkdir(parents=True, exist_ok=True)
        (fleet_dir / "master.json").write_text(
            _json({"master": report["master"], "columns": [c.summary() for c in columns]}) + "\n",
            encoding="utf-8",
        )
        fleet_solution = assemble_fleet(selected, catalogue)
        fleet_path = fleet_dir / "Result.txt"
        fleet_solution.write(fleet_path)
        histories = {}
        fleet_entry = verify(fleet_path, histories)
        fleet_entry["artifacts"] = {"solution": str(fleet_path)}
        viewer = write_viewer_dataset(
            fleet_dir / "viewer",
            Solution.read(fleet_path),
            histories,
            catalogue,
            run_id=f"{args.run_id}_fleet",
            commit=report["commit"],
            verification=fleet_entry["independent"],
            solution_path=fleet_path,
            instance_id=str(instance_summary["instance_id"]),
        )
        fleet_entry["viewer_manifest"] = viewer
        fleet_entry["artifacts"]["viewer"] = str(fleet_dir / "viewer" / "trajectories.json")
        report["best"] = fleet_entry
        report["status"] = "scored"
    else:
        report["best"] = None
        report["status"] = "no_certified_route" if not args.search_only else "search_only"
    report["wall_seconds_total"] = time.perf_counter() - started
    report["peak_rss_mb"] = _peak_rss_mb()
    (output_dir / "run_report.json").write_text(_json(report) + "\n", encoding="utf-8")
    best = report["best"]
    print(
        _json(
            {
                "run_id": args.run_id,
                "status": report["status"],
                "fleet": report["fleet"],
                "score_kg": None if best is None else best["score_kg"],
                "official": None if best is None else best.get("official"),
                "wall_seconds_total": report["wall_seconds_total"],
                "peak_rss_mb": report["peak_rss_mb"],
            }
        )
    )
    return 0


def _optional_bonus(loader):
    try:
        return loader()
    except Exception:
        return None


BUDGET_MARKS_MINUTES = (30, 60, 120, 240)


def cmd_cluster_fleet(args: argparse.Namespace) -> int:
    """Cooperative cluster pricing -> bundle master -> verified fleet, with checkpoints.

    Co-moving families of the pool are priced in parallel worker processes (deployer +
    collector itineraries, all SCvx-certified; see ``bundles.py``).  Every finished family adds
    its bundle and single-ship columns to the master; whenever the master's fleet verifies with
    a higher score than the incumbent it is retained under ``fleets/`` (the intermediate verified
    fleets the budget report refers to).  The final fleet is written to ``fleet/Result.txt``.
    """

    from .bundles import (
        ClusterPricingSettings,
        bundle_columns,
        bundle_settings_summary,
        cluster_search_settings,
        family_clusters,
        price_clusters,
        rank_families,
    )
    from .clusters import ClusterBands
    from .cooperative import FleetColumn, solve_fleet_master
    from .data import REPOSITORY_ROOT, load_bonus_table, load_catalogue
    from .fleet import FleetPlan, assemble_fleet
    from .low_thrust import ScvxSettings
    from .official import official_verifier_available, run_official_verifier
    from .pipeline import write_route_artifacts
    from .solution import Solution
    from .verifier import Gtoc12Verifier
    from .viewer_export import write_viewer_dataset

    started = time.perf_counter()
    catalogue = load_catalogue()
    bonus_table = _optional_bonus(load_bonus_table)
    weights: dict[int, float] | None = None
    if bonus_table is not None and not args.no_bonus_weights:
        weights = {
            int(asteroid): float(bonus_table.coefficient[asteroid - 1])
            for asteroid in catalogue.ids
        }
    ids = catalogue_pool(catalogue, args)
    if args.collect_epoch_families:
        bands = ClusterBands.collect_window(
            radius=args.cluster_radius, phase_deg=args.cluster_phase_deg
        )
    else:
        bands = ClusterBands(
            radius=args.cluster_radius,
            phase_deg=args.cluster_phase_deg,
            visit_epochs=None if args.static_families else ClusterBands().visit_epochs,
        )
    clusters = family_clusters(catalogue, ids, bands=bands, min_members=args.min_members)
    # cheapest families first (Earth access + internal hops over the visit epochs), not largest
    ranked = rank_families(
        catalogue,
        clusters,
        cluster_search_settings(ClusterPricingSettings(), 2),
        visit_epochs=bands.phase_epochs,
    )
    if args.families:
        wanted = {int(item) for item in args.families.split(",") if item.strip()}
        ranked = [item for item in ranked if int(item[0]) in wanted]
    ranked = ranked[args.skip_clusters : args.skip_clusters + args.max_clusters]
    clusters = [(label, members) for label, members, _stats in ranked]
    family_stats = {label: stats for label, _members, stats in ranked}
    settings = ClusterPricingSettings(
        ships=args.ships_per_cluster,
        beam_width=args.beam_width,
        max_deploys=args.max_deploys,
        refine_top=args.refine_top,
        retime_attempts=args.retime_attempts,
        retime_budget_seconds=args.retime_budget_seconds,
        orphan_credit=args.orphan_credit,
        hop_authority_ratio=args.hop_authority_ratio,
        time_budget_seconds=args.cluster_budget_seconds,
        seed=args.seed,
        earth_leg_continuous=args.earth_leg_refinements > 0,
        earth_leg_refinements=args.earth_leg_refinements,
        collector_harvest=args.collector_harvest,
        collect_lookahead_weight=args.collect_lookahead,
        collect_dp=not args.no_collect_dp,
        collect_dp_propellant_weight=args.collect_dp_weight,
        collect_dp_step_days=args.collect_dp_step_days,
        collect_dp_inflation_fit=args.collect_dp_inflation_fit or "",
        earth_prescreen_ratio=args.earth_prescreen_ratio,
    )
    scvx = ScvxSettings(max_iterations=args.scvx_iterations, node_days=args.node_days)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    verifier = Gtoc12Verifier(catalogue, bonus=bonus_table)
    report: dict[str, Any] = {
        "run_id": args.run_id,
        "commit": _commit(REPOSITORY_ROOT),
        "instance": {
            "instance_id": "gtoc12-full-catalogue",
            "pool_asteroids": int(ids.shape[0]),
            "pool_filter": {
                "a_au": [args.pool_a_min, args.pool_a_max],
                "e_max": args.pool_e_max,
                "i_max_deg": args.pool_i_max,
            },
            "cluster_bands": {
                "radius": bands.radius,
                "phase_deg": bands.phase_deg,
                "visit_epochs": list(bands.phase_epochs),
                "phase_weights": list(bands.epoch_weights),
                "phasing_aware": bands.visit_epochs is not None,
                "collect_epoch_families": bool(args.collect_epoch_families),
            },
            "families_priced": [
                {"label": int(label), "members": int(members.shape[0]), **family_stats[label]}
                for label, members in clusters
            ],
        },
        "settings": {
            **bundle_settings_summary(settings),
            "workers": args.workers,
            "max_ships": args.max_ships,
            "budget_seconds": args.budget_seconds,
            "scvx_iterations": args.scvx_iterations,
            "node_days": args.node_days,
            "bonus_weights": weights is not None,
        },
        "cpu_only": True,
        "gpu_used": False,
        "bundles": [],
        "timeline": [],
        "fleets": [],  # every verified incumbent (elapsed, ships, score, path)
    }
    columns: list[FleetColumn] = []
    incumbent: dict[str, Any] | None = None
    worker_rss: list[float] = []

    def verify(solution_path: Path, histories: dict | None = None) -> dict[str, Any]:
        checker = (
            verifier
            if histories is None
            else Gtoc12Verifier(catalogue, bonus=bonus_table, history=histories)
        )
        independent = checker.verify_file(solution_path)
        entry: dict[str, Any] = {"independent": independent.summary()}
        if official_verifier_available():
            official = run_official_verifier(solution_path)
            entry["official"] = official.summary()
        score = entry.get("official", {}).get("total_mass_kg")
        entry["score_kg"] = independent.total_mass_kg if score is None else score
        entry["ok"] = bool(independent.ok) and entry.get("official", {}).get("ok", True)
        return entry

    memory = _MemorySampler().start()

    def checkpoint() -> None:
        report["wall_seconds_total"] = time.perf_counter() - started
        report["peak_rss_mb"] = _peak_rss_mb()
        report["worker_peak_rss_mb"] = max(worker_rss) if worker_rss else None
        report["memory_bound_mb"] = _peak_rss_mb() + args.workers * (
            max(worker_rss) if worker_rss else 0.0
        )
        # measured total: peak PSS of the main process + live workers, sampled every 15 s
        report["memory_total_pss_peak_mb"] = memory.peak_mb
        report["memory_samples"] = memory.samples
        (output_dir / "run_report.json").write_text(_json(report) + "\n", encoding="utf-8")

    def add_bundle_columns(bundle) -> None:
        columns.extend(bundle_columns(bundle, len(columns)))

    previous: list[FleetColumn] = []

    def run_master():
        # the previous selection stays feasible (columns are only added): warm start so the
        # node-capped search never regresses below the last incumbent
        master = solve_fleet_master(
            columns,
            weights=weights,
            max_ships=args.max_ships,
            node_cap=args.node_cap,
            incumbent=tuple(previous),
        )
        previous[:] = list(master.selected)
        report["master"] = master.summary()
        report["master"]["columns"] = len(columns)
        return master

    def try_fleet(master, final: bool) -> dict[str, Any] | None:
        nonlocal incumbent
        if not master.selected:
            return None
        plan = FleetPlan(master.routes())
        if final:
            directory = output_dir / "fleet"
        else:
            name = f"fleet_{len(report['fleets']):03d}_{len(plan.routes):02d}ships"
            directory = output_dir / "fleets" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "Result.txt"
        assemble_fleet(plan, catalogue).write(path)
        histories: dict = {}
        entry = verify(path, histories if final else None)
        entry["fleet"] = plan.summary()
        entry["artifacts"] = {"solution": str(path)}
        entry["elapsed_seconds"] = time.perf_counter() - started
        entry["master"] = {
            k: v for k, v in report["master"].items() if k not in ("selected", "rejected")
        }
        (directory / "fleet.json").write_text(_json(entry) + "\n", encoding="utf-8")
        if final:
            viewer = write_viewer_dataset(
                directory / "viewer",
                Solution.read(path),
                histories,
                catalogue,
                run_id=f"{args.run_id}_fleet",
                commit=report["commit"],
                verification=entry["independent"],
                solution_path=path,
            )
            entry["viewer_manifest"] = viewer
        if entry["ok"] and (incumbent is None or entry["score_kg"] > incumbent["score_kg"] + 1e-9):
            incumbent = entry
            report["fleets"].append(
                {
                    "elapsed_seconds": entry["elapsed_seconds"],
                    "ships": entry["fleet"]["ships"],
                    "asteroids": len(entry["fleet"]["asteroids"]),
                    "score_kg": entry["score_kg"],
                    "average_collected_kg": entry["fleet"]["average_collected_kg"],
                    "path": str(path),
                }
            )
        elif not entry["ok"]:
            report.setdefault("failed_fleets", []).append(
                {"path": str(path), "independent": entry["independent"]}
            )
        return entry

    def on_result(bundle) -> None:
        problem = bundle.consistent()
        if problem:  # never let one family's inconsistency end the campaign
            bundle.rejected.append({"reason": "inconsistent bundle at master", "detail": problem})
            bundle.ships.clear()
        summary = bundle.summary()
        worker_rss.append(bundle.peak_rss_mb)
        cluster_dir = output_dir / "clusters" / f"family_{bundle.label:04d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for ship in bundle.ships:
            write_route_artifacts(ship.route, catalogue, cluster_dir / f"ship_{ship.slot:02d}")
        (cluster_dir / "bundle.json").write_text(_json(summary) + "\n", encoding="utf-8")
        report["bundles"].append(
            {k: v for k, v in summary.items() if k not in ("rejected", "earth_legs", "repairs")}
            | {
                "rejected": len(summary["rejected"]),
                "earth_legs_checked": summary["earth_legs"].get("checked"),
                "earth_legs_certified": summary["earth_legs"].get("certified"),
                "repairs": len(summary["repairs"]),
            }
        )
        add_bundle_columns(bundle)
        master = run_master()
        entry = try_fleet(master, final=False)
        report["timeline"].append(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "families_priced": len(report["bundles"]),
                "columns": len(columns),
                "master_ships": master.ships,
                "master_collected_kg": master.collected_kg,
                "master_objective_kg": master.objective,
                "master_exhaustive": master.exhaustive,
                "verified_score_kg": None if entry is None else entry["score_kg"],
                "verified_ok": None if entry is None else entry["ok"],
                "incumbent_score_kg": None if incumbent is None else incumbent["score_kg"],
                "peak_rss_mb": _peak_rss_mb(),
                "worker_peak_rss_mb": bundle.peak_rss_mb,
            }
        )
        checkpoint()
        coop = summary["cooperative"] or {}
        print(
            _json(
                {
                    "family": bundle.label,
                    "members": len(bundle.members),
                    "ships": len(bundle.ships),
                    "collected_kg": [round(s.route.total_collected_kg, 1) for s in bundle.ships],
                    "cooperative_collects": coop.get("cooperative_collects"),
                    "orphans_left": coop.get("orphans_left"),
                    "wall_seconds": round(bundle.wall_seconds),
                    "elapsed_minutes": round((time.perf_counter() - started) / 60.0, 1),
                    "master": {
                        "ships": master.ships,
                        "collected_kg": round(master.collected_kg, 1),
                        "exhaustive": master.exhaustive,
                    },
                    "incumbent_kg": None if incumbent is None else round(incumbent["score_kg"], 1),
                }
            ),
            flush=True,
        )

    price_clusters(
        catalogue,
        clusters,
        settings=settings,
        scvx=scvx,
        weights=weights,
        workers=args.workers,
        on_result=on_result,
        budget_seconds=args.budget_seconds,
    )
    if columns:
        master = run_master()
        final = try_fleet(master, final=True)
        report["best"] = incumbent if final is None or not final["ok"] else final
        report["final_fleet"] = final
        report["status"] = "scored" if incumbent is not None else "no_verified_fleet"
    else:
        report["best"] = None
        report["status"] = "no_certified_route"
    report["budget_marks"] = {
        f"{minutes}_min": max(
            (f for f in report["fleets"] if f["elapsed_seconds"] <= minutes * 60.0),
            key=lambda f: f["score_kg"],
            default=None,
        )
        for minutes in BUDGET_MARKS_MINUTES
    }
    memory.stop()
    checkpoint()
    best = report["best"]
    print(
        _json(
            {
                "run_id": args.run_id,
                "status": report["status"],
                "score_kg": None if best is None else best["score_kg"],
                "ships": None if best is None else best["fleet"]["ships"],
                "asteroids": None if best is None else len(best["fleet"]["asteroids"]),
                "average_collected_kg": None
                if best is None
                else best["fleet"]["average_collected_kg"],
                "budget_marks": {
                    k: None if v is None else round(v["score_kg"], 1)
                    for k, v in report["budget_marks"].items()
                },
                "wall_seconds_total": report["wall_seconds_total"],
                "peak_rss_mb": report["peak_rss_mb"],
                "memory_bound_mb": report["memory_bound_mb"],
                "memory_total_pss_peak_mb": report["memory_total_pss_peak_mb"],
            }
        )
    )
    return 0


def cmd_fleet_master(args: argparse.Namespace) -> int:
    """Master over archived certified routes (this and earlier runs) -> verified fleet.

    Every ``route_summary.json`` below the ``--source`` directories is rebuilt into its plan,
    re-flown through SCvx in worker processes, packed into bundle columns (archived families
    keep their cooperative structure) and handed to the fleet master.  The selected fleet is
    verified independently and officially before it is reported.
    """

    from .archive import discover_archives, recertify_archives
    from .bundles import bundle_columns
    from .cooperative import FleetColumn, solve_fleet_master
    from .data import REPOSITORY_ROOT, load_bonus_table, load_catalogue
    from .fleet import FleetPlan, assemble_fleet
    from .low_thrust import ScvxSettings
    from .official import official_verifier_available, run_official_verifier
    from .pipeline import write_route_artifacts
    from .solution import Solution
    from .verifier import Gtoc12Verifier
    from .viewer_export import write_viewer_dataset

    started = time.perf_counter()
    catalogue = load_catalogue()
    bonus_table = _optional_bonus(load_bonus_table)
    weights: dict[int, float] | None = None
    if bonus_table is not None and not args.no_bonus_weights:
        weights = {
            int(asteroid): float(bonus_table.coefficient[asteroid - 1])
            for asteroid in catalogue.ids
        }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = discover_archives([Path(s) for s in args.source])
    report: dict[str, Any] = {
        "run_id": args.run_id,
        "commit": _commit(REPOSITORY_ROOT),
        "sources": [str(s) for s in args.source],
        "groups": [
            {
                "name": g.name,
                "ships": len(g.ships),
                "archives": sum(len(s.summaries) for s in g.ships),
            }
            for g in groups
        ],
        "settings": {
            "workers": args.workers,
            "max_ships": args.max_ships,
            "node_cap": args.node_cap,
            "scvx_iterations": args.scvx_iterations,
            "node_days": args.node_days,
            "bonus_weights": weights is not None,
        },
        "cpu_only": True,
        "gpu_used": False,
        "recertification": [],
        "bundles": [],
    }
    print(
        _json(
            {
                "groups": len(groups),
                "ships": sum(len(g.ships) for g in groups),
                "archives": sum(len(s.summaries) for g in groups for s in g.ships),
            }
        ),
        flush=True,
    )

    def on_progress(entry: dict[str, Any]) -> None:
        report["recertification"].append(entry)
        print(_json(entry), flush=True)

    bundles = recertify_archives(
        catalogue,
        groups,
        scvx=ScvxSettings(max_iterations=args.scvx_iterations, node_days=args.node_days),
        workers=args.workers,
        on_progress=on_progress,
    )
    columns: list[FleetColumn] = []
    for bundle in bundles:
        summary = bundle.summary()
        report["bundles"].append(summary)
        for ship in bundle.ships:
            write_route_artifacts(
                ship.route,
                catalogue,
                output_dir / "columns" / summary_dir(bundle) / f"ship_{ship.slot:02d}",
            )
        columns.extend(bundle_columns(bundle, len(columns), prefix="a"))
    report["columns"] = len(columns)
    report["recertified_routes"] = sum(len(s.variants) for b in bundles for s in b.ships)
    report["recertification_wall_seconds"] = time.perf_counter() - started
    if not columns:
        report["status"] = "no_certified_route"
        (output_dir / "run_report.json").write_text(_json(report) + "\n", encoding="utf-8")
        print(_json({"run_id": args.run_id, "status": report["status"]}))
        return 1
    master = solve_fleet_master(
        columns,
        weights=weights,
        max_ships=args.max_ships,
        node_cap=args.node_cap,
        lp_node_limit=args.lp_node_limit,
    )
    report["master"] = master.summary() | {"columns": len(columns)}
    report["master_wall_seconds"] = (
        time.perf_counter() - started - report["recertification_wall_seconds"]
    )
    plan = FleetPlan(master.routes())
    directory = output_dir / "fleet"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "Result.txt"
    assemble_fleet(plan, catalogue).write(path)
    histories: dict = {}
    independent = Gtoc12Verifier(catalogue, bonus=bonus_table, history=histories).verify_file(path)
    entry: dict[str, Any] = {"independent": independent.summary()}
    if official_verifier_available():
        entry["official"] = run_official_verifier(path).summary()
    score = entry.get("official", {}).get("total_mass_kg")
    entry["score_kg"] = independent.total_mass_kg if score is None else score
    entry["ok"] = bool(independent.ok) and entry.get("official", {}).get("ok", True)
    entry["fleet"] = plan.summary()
    entry["artifacts"] = {"solution": str(path)}
    entry["master"] = {
        k: v for k, v in report["master"].items() if k not in ("selected", "rejected")
    }
    if entry["ok"]:
        entry["viewer_manifest"] = write_viewer_dataset(
            directory / "viewer",
            Solution.read(path),
            histories,
            catalogue,
            run_id=f"{args.run_id}_fleet",
            commit=report["commit"],
            verification=entry["independent"],
            solution_path=path,
        )
    (directory / "fleet.json").write_text(_json(entry) + "\n", encoding="utf-8")
    report["best"] = entry if entry["ok"] else None
    report["final_fleet"] = entry
    report["status"] = "scored" if entry["ok"] else "fleet_failed_verification"
    report["wall_seconds_total"] = time.perf_counter() - started
    report["peak_rss_mb"] = _peak_rss_mb()
    (output_dir / "run_report.json").write_text(_json(report) + "\n", encoding="utf-8")
    print(
        _json(
            {
                "run_id": args.run_id,
                "status": report["status"],
                "score_kg": entry["score_kg"],
                "ships": entry["fleet"]["ships"],
                "asteroids": len(entry["fleet"]["asteroids"]),
                "average_collected_kg": entry["fleet"]["average_collected_kg"],
                "columns": len(columns),
                "master_exhaustive": master.exhaustive,
                "wall_seconds_total": report["wall_seconds_total"],
                "peak_rss_mb": report["peak_rss_mb"],
            }
        )
    )
    return 0 if entry["ok"] else 1


def summary_dir(bundle) -> str:
    """Directory name of an archived bundle's re-certified columns."""

    archived = next((s.report.get("archived") for s in bundle.ships if s.report), None)
    if archived:
        return str(archived).replace("/", "__").replace("\\", "__")
    return f"bundle_{bundle.label}"


def cmd_export_viewer(args: argparse.Namespace) -> int:
    from .data import REPOSITORY_ROOT, load_bonus_table, load_catalogue
    from .solution import Solution
    from .verifier import Gtoc12Verifier
    from .viewer_export import write_viewer_dataset

    catalogue = load_catalogue()
    histories: dict = {}
    report = Gtoc12Verifier(
        catalogue, bonus=_optional_bonus(load_bonus_table), history=histories
    ).verify_file(args.solution)
    manifest = write_viewer_dataset(
        Path(args.output),
        Solution.read(args.solution),
        histories,
        catalogue,
        run_id=args.run_id,
        commit=_commit(REPOSITORY_ROOT),
        verification=report.summary(),
        solution_path=Path(args.solution),
    )
    print(_json({"verification": report.summary(), "manifest": manifest}))
    return 0 if report.ok else 1


def cmd_leg_stats(args: argparse.Namespace) -> int:
    from .data import load_catalogue
    from .legstats import compare, format_table

    solutions: dict[str, str] = {}
    for item in args.solution:
        name, _, path = item.partition("=")
        if not path:
            name, path = Path(item).parent.name or Path(item).stem, item
        solutions[name] = path
    comparison = compare(solutions, load_catalogue(), cheap_hop_kg=args.cheap_hop_kg)
    print(format_table(comparison))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(_json(comparison) + "\n", encoding="utf-8")
    return 0


def cmd_hop_calibration(args: argparse.Namespace) -> int:
    from .data import REPOSITORY_ROOT, load_catalogue
    from .hopcalib import certified_hops, fit_inflation

    catalogue = load_catalogue()
    train = certified_hops(catalogue, [Path(p) for p in args.train])
    holdout = certified_hops(catalogue, [Path(p) for p in args.holdout]) if args.holdout else None
    fit = fit_inflation(train, holdout, quantile=args.quantile)
    summary = fit.summary()
    summary["train_sources"] = [str(p) for p in args.train]
    summary["holdout_sources"] = [str(p) for p in args.holdout]
    summary["train_hops"] = len(train)
    summary["holdout_hops"] = 0 if holdout is None else len(holdout)
    summary["commit"] = _commit(REPOSITORY_ROOT)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(_json(summary) + "\n", encoding="utf-8")
    names = ["1", "r", "TOF/yr", "|Δa|/0.1AU", "|Δλ|/π"]
    terms = " ".join(f"{c:+.3f}·{n}" for c, n in zip(fit.coefficients, names, strict=True))
    print(f"inflation = {terms}  (quantile {fit.quantile}, {len(train)} hops)")
    for key in ("train", "holdout", "holdout_propellant_error_kg"):
        stats = fit.residuals.get(key)
        if isinstance(stats, dict) and stats.get("n"):
            print(
                f"{key:>28s}: n={stats['n']} rms={stats['rms']:.3f} median={stats['median']:+.3f}"
                f" p10={stats['p10']:+.3f} p90={stats['p90']:+.3f}"
                f" under-priced={stats['share_under_priced']:.2f}"
            )
    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("gtoc12", help="GTOC12 Sustainable Asteroid Mining replay track")
    commands = parser.add_subparsers(dest="gtoc12_command", required=True)

    fetch = commands.add_parser("fetch", help="download and checksum the pinned official data")
    fetch.add_argument("--data-dir", type=Path, default=None)
    fetch.set_defaults(function=cmd_fetch)

    verify = commands.add_parser("verify", help="independently verify and score a solution file")
    verify.add_argument("solution")
    verify.add_argument(
        "--official", action="store_true", help="also run the organisers' binary and compare"
    )
    verify.add_argument("--rtol", type=float, default=1e-12)
    verify.set_defaults(function=cmd_verify)

    reduced = commands.add_parser(
        "reduced-instance", help="print the preregistered reduced instance"
    )
    reduced.add_argument("--rule", type=Path, default=None)
    reduced.add_argument("--list-ids", action="store_true")
    reduced.set_defaults(function=cmd_reduced_instance)

    run = commands.add_parser("run", help="search, refine, emit, verify and export a route")
    run.add_argument("--run-id", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--rule", type=Path, default=None)
    run.add_argument("--beam-width", type=int, default=24)
    run.add_argument("--max-deploys", type=int, default=10)
    run.add_argument("--neighbours", type=int, default=48)
    run.add_argument("--refine-top", type=int, default=3)
    run.add_argument("--scvx-iterations", type=int, default=40)
    run.add_argument("--node-days", type=float, default=2.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--search-only", action="store_true")
    run.add_argument(
        "--full-catalogue", action="store_true", help="screen the full 60,000-asteroid catalogue"
    )
    run.add_argument("--ships", type=int, default=1, help="greedy fleet size")
    run.add_argument("--search-budget-seconds", type=float, default=float("inf"))
    run.add_argument("--stop-at-first-certified", action="store_true")
    run.add_argument("--pool-a-min", type=float, default=2.2)
    run.add_argument("--pool-a-max", type=float, default=3.0)
    run.add_argument("--pool-e-max", type=float, default=0.15)
    run.add_argument("--pool-i-max", type=float, default=8.0)
    run.add_argument(
        "--budget-seconds",
        type=float,
        default=float("inf"),
        help="declared wall-clock budget: no new ship is started after it (partial results kept)",
    )
    run.add_argument("--no-retime", action="store_true", help="skip joint re-timing/extension")
    run.add_argument("--retime-attempts", type=int, default=4)
    run.add_argument("--retime-budget-seconds", type=float, default=900.0)
    run.add_argument(
        "--no-bonus-weights",
        action="store_true",
        help="score plain collected mass instead of the fixed-bonus weighted mass",
    )
    run.add_argument(
        "--no-cooperative",
        action="store_true",
        help="self-cleaning ships only (no shared miner pool / orphan collection)",
    )
    run.set_defaults(function=cmd_run)

    cluster = commands.add_parser(
        "cluster-fleet",
        help="cooperative cluster pricing in parallel workers -> bundle master -> verified fleet",
    )
    cluster.add_argument("--run-id", required=True)
    cluster.add_argument("--output", required=True)
    cluster.add_argument("--workers", type=int, default=2, help="pricing worker processes")
    cluster.add_argument("--ships-per-cluster", type=int, default=3)
    cluster.add_argument(
        "--max-clusters", type=int, default=60, help="families priced (cheapest first)"
    )
    cluster.add_argument("--skip-clusters", type=int, default=0)
    cluster.add_argument(
        "--families",
        default="",
        help="comma-separated family labels to price (default: all, cheapest first)",
    )
    cluster.add_argument("--min-members", type=int, default=12)
    cluster.add_argument("--cluster-radius", type=float, default=1.5)
    cluster.add_argument("--cluster-phase-deg", type=float, default=8.0)
    cluster.add_argument("--max-ships", type=int, default=100)
    cluster.add_argument("--node-cap", type=int, default=200_000, help="master node cap")
    cluster.add_argument("--beam-width", type=int, default=24)
    cluster.add_argument("--max-deploys", type=int, default=10)
    cluster.add_argument("--refine-top", type=int, default=2)
    cluster.add_argument("--retime-attempts", type=int, default=4)
    cluster.add_argument("--retime-budget-seconds", type=float, default=600.0)
    cluster.add_argument("--orphan-credit", type=float, default=1.0)
    cluster.add_argument("--hop-authority-ratio", type=float, default=0.55)
    cluster.add_argument("--cluster-budget-seconds", type=float, default=1800.0)
    cluster.add_argument("--scvx-iterations", type=int, default=40)
    cluster.add_argument("--node-days", type=float, default=2.0)
    cluster.add_argument("--seed", type=int, default=0)
    cluster.add_argument("--pool-a-min", type=float, default=2.2)
    cluster.add_argument("--pool-a-max", type=float, default=3.0)
    cluster.add_argument("--pool-e-max", type=float, default=0.15)
    cluster.add_argument("--pool-i-max", type=float, default=8.0)
    cluster.add_argument(
        "--budget-seconds",
        type=float,
        default=float("inf"),
        help="declared wall-clock budget: no new family is started after it",
    )
    cluster.add_argument("--no-bonus-weights", action="store_true")
    cluster.add_argument(
        "--static-families",
        action="store_true",
        help="single-epoch (static) family membership instead of the phasing-aware default",
    )
    cluster.add_argument(
        "--earth-leg-refinements",
        type=int,
        default=8,
        help="SCvx evaluations of the continuous Earth-leg optimiser per certified leg (0: off)",
    )
    cluster.add_argument("--collector-harvest", action="store_true")
    cluster.add_argument(
        "--collect-lookahead",
        type=float,
        default=0.0,
        help="beam score weight on the collect-time cost of re-flying each deploy pair (0: off)",
    )
    cluster.add_argument(
        "--collect-epoch-families",
        action="store_true",
        help="cluster families on collect-window (years 8.5-13.5) phase co-motion",
    )
    cluster.add_argument(
        "--no-collect-dp",
        action="store_true",
        help="disable the exact collect-tour order/timing DP in the beam",
    )
    cluster.add_argument(
        "--collect-dp-weight",
        type=float,
        default=1.0,
        help="collect DP objective: kg of value per kg of propellant (default 1.0)",
    )
    cluster.add_argument(
        "--collect-dp-step-days",
        type=float,
        default=15.0,
        help="departure lattice of the collect DP in days (default 15)",
    )
    cluster.add_argument(
        "--collect-dp-inflation-fit",
        default="",
        help="hop-calibration fit JSON pricing the DP pair table (default: flat inflation)",
    )
    cluster.add_argument(
        "--earth-prescreen-ratio",
        type=float,
        default=0.7,
        help="Earth legs above this Lambert authority ratio are flown last (default 0.7)",
    )
    cluster.set_defaults(function=cmd_cluster_fleet)

    calib = commands.add_parser(
        "hop-calibration",
        help="fit the low-thrust hop inflation model on archived SCvx-certified hops",
    )
    calib.add_argument(
        "--train",
        action="append",
        required=True,
        help="run directory whose route_summary.json legs train the fit (repeatable)",
    )
    calib.add_argument(
        "--holdout",
        action="append",
        default=[],
        help="run directory whose legs report out-of-sample residuals (repeatable)",
    )
    calib.add_argument("--quantile", type=float, default=0.65)
    calib.add_argument("--output", required=True, help="fit JSON path")
    calib.set_defaults(function=cmd_hop_calibration)

    master = commands.add_parser(
        "fleet-master",
        help="re-certify archived routes (this and earlier runs) and solve the fleet master",
    )
    master.add_argument("--run-id", required=True)
    master.add_argument("--output", required=True)
    master.add_argument(
        "--source",
        action="append",
        required=True,
        help="run directory holding ship_NN/**/route_summary.json archives (repeatable)",
    )
    master.add_argument("--workers", type=int, default=2)
    master.add_argument("--max-ships", type=int, default=100)
    master.add_argument(
        "--node-cap",
        type=int,
        default=2_000_000,
        help="branch-and-bound node cap (about 10 s per million nodes)",
    )
    master.add_argument(
        "--lp-node-limit",
        type=int,
        default=20_000,
        help="LPs solved by the LP branch and bound that closes or bounds the master",
    )
    master.add_argument("--scvx-iterations", type=int, default=40)
    master.add_argument("--node-days", type=float, default=2.0)
    master.add_argument("--no-bonus-weights", action="store_true")
    master.set_defaults(function=cmd_fleet_master)

    export = commands.add_parser("export-viewer", help="propagate a solution and write viewer data")
    export.add_argument("solution")
    export.add_argument("--output", required=True)
    export.add_argument("--run-id", default="manual")
    export.set_defaults(function=cmd_export_viewer)

    legs = commands.add_parser(
        "leg-stats",
        help="per-role leg cost distributions (Earth legs, hops, returns) of solution files",
    )
    legs.add_argument(
        "--solution",
        action="append",
        required=True,
        help="name=path of a solution file (repeatable; the references are decoded the same way)",
    )
    legs.add_argument("--cheap-hop-kg", type=float, default=75.0)
    legs.add_argument("--output", default="", help="optional JSON output path")
    legs.set_defaults(function=cmd_leg_stats)
