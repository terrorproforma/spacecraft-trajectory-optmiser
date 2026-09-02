"""``spacepdhcg gtoc12`` command group."""

from __future__ import annotations

import argparse
import json
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

    from .data import REPOSITORY_ROOT, load_bonus_table, load_catalogue
    from .fleet import FleetPlan, assemble_fleet
    from .low_thrust import ScvxSettings
    from .official import official_verifier_available, run_official_verifier
    from .pipeline import refine_route, write_route_artifacts
    from .reduced_instance import build_reduced_instance
    from .search import RouteSearch, SearchSettings
    from .solution import Solution
    from .verifier import Gtoc12Verifier
    from .viewer_export import write_viewer_dataset

    started = time.perf_counter()
    catalogue = load_catalogue()
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
        },
        "cpu_only": True,
        "gpu_used": False,
        "ships": [],
    }
    scvx = ScvxSettings(max_iterations=args.scvx_iterations, node_days=args.node_days)
    fleet = FleetPlan()
    verifier = Gtoc12Verifier(catalogue, bonus=_optional_bonus(load_bonus_table))

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
        ship_dir = output_dir / f"ship_{ship_index:02d}"
        ship_dir.mkdir(parents=True, exist_ok=True)
        search = RouteSearch(catalogue, ids, settings, excluded=fleet.used_asteroids())
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
                )
                entry["viewer_manifest"] = viewer
                entry["artifacts"] = {
                    "solution": str(solution_path),
                    "viewer": str(directory / "viewer" / "trajectories.json"),
                }
                if best_entry is None or entry["score_kg"] > best_entry[0]["score_kg"]:
                    best_entry = (entry, refined)
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
            break
        ship_report["best"] = best_entry[0]
        ship_report["status"] = "scored"
        fleet.routes.append(best_entry[1])
    report["fleet"] = fleet.summary()
    if fleet.routes:
        fleet_dir = output_dir / "fleet"
        fleet_dir.mkdir(parents=True, exist_ok=True)
        fleet_solution = assemble_fleet(fleet, catalogue)
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
    run.set_defaults(function=cmd_run)

    export = commands.add_parser("export-viewer", help="propagate a solution and write viewer data")
    export.add_argument("solution")
    export.add_argument("--output", required=True)
    export.add_argument("--run-id", default="manual")
    export.set_defaults(function=cmd_export_viewer)
