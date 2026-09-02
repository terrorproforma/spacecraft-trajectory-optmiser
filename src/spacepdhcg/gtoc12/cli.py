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


def cmd_run(args: argparse.Namespace) -> int:
    """Search -> refine -> emit -> verify (official + independent) -> viewer export."""

    from .data import REPOSITORY_ROOT, load_bonus_table, load_catalogue
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
    instance = build_reduced_instance(catalogue, args.rule)
    ids = instance.asteroid_ids if not args.full_catalogue else catalogue.ids
    settings = SearchSettings(
        beam_width=args.beam_width,
        max_deploys=args.max_deploys,
        seed=args.seed,
        neighbours=args.neighbours,
    )
    search = RouteSearch(catalogue, ids, settings)
    result = search.run()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "run_id": args.run_id,
        "commit": _commit(REPOSITORY_ROOT),
        "instance": instance.summary()
        if not args.full_catalogue
        else {"instance_id": "gtoc12-full-catalogue", "asteroids": int(catalogue.ids.shape[0])},
        "search": {
            "settings": {
                "beam_width": args.beam_width,
                "max_deploys": args.max_deploys,
                "neighbours": args.neighbours,
                "seed": args.seed,
            },
            "expansions": result.expansions,
            "lambert_evaluations": result.lambert_evaluations,
            "wall_seconds": result.wall_seconds,
            "candidates": len(result.candidates),
            "failures": len(result.failures),
            "top_candidates": [item.summary() for item in result.candidates[: args.refine_top]],
        },
        "cpu_only": True,
        "gpu_used": False,
    }
    (output_dir / "search.json").write_text(_json(report) + "\n", encoding="utf-8")
    if args.search_only or not result.candidates:
        report["status"] = "search_only" if result.candidates else "no_candidates"
        print(_json({key: report[key] for key in ("run_id", "search", "status")}))
        return 0
    best_report = None
    refinements = []
    scvx = ScvxSettings(max_iterations=args.scvx_iterations, node_days=args.node_days)
    for rank, plan in enumerate(result.candidates[: args.refine_top]):
        refined = refine_route(plan, catalogue, scvx=scvx)
        entry: dict[str, Any] = {"rank": rank, "plan": plan.summary(), "refined": refined.summary()}
        if refined.certified:
            directory = output_dir / f"candidate_{rank:02d}"
            artifacts = write_route_artifacts(refined, catalogue, directory)
            solution_path = Path(artifacts["solution"])
            histories: dict = {}
            independent = Gtoc12Verifier(
                catalogue, bonus=_optional_bonus(load_bonus_table), history=histories
            ).verify_file(solution_path)
            entry["independent"] = independent.summary()
            entry["independent"]["scored_masses"] = independent.scored_masses
            if official_verifier_available():
                official = run_official_verifier(solution_path)
                entry["official"] = official.summary()
                entry["official"]["score_data"] = official.score_data
            viewer = write_viewer_dataset(
                directory / "viewer",
                Solution.read(solution_path),
                histories,
                catalogue,
                run_id=f"{args.run_id}_c{rank:02d}",
                commit=report["commit"],
                verification=independent.summary(),
                solution_path=solution_path,
            )
            entry["viewer_manifest"] = viewer
            entry["artifacts"] = {
                "solution": str(solution_path),
                "viewer": str(directory / "viewer" / "trajectories.json"),
            }
            score = entry.get("official", {}).get("total_mass_kg")
            if score is None:
                score = independent.total_mass_kg
            entry["score_kg"] = score
            if best_report is None or score > best_report["score_kg"]:
                best_report = entry
        refinements.append(entry)
        (output_dir / "refinements.json").write_text(_json(refinements) + "\n", encoding="utf-8")
    report["refinements"] = refinements
    report["best"] = best_report
    report["wall_seconds_total"] = time.perf_counter() - started
    report["status"] = "scored" if best_report else "no_certified_route"
    (output_dir / "run_report.json").write_text(_json(report) + "\n", encoding="utf-8")
    print(
        _json(
            {
                "run_id": args.run_id,
                "status": report["status"],
                "best": None
                if best_report is None
                else {
                    k: best_report[k]
                    for k in ("rank", "score_kg", "official", "artifacts")
                    if k in best_report
                },
                "wall_seconds_total": report["wall_seconds_total"],
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
    run.add_argument("--max-deploys", type=int, default=4)
    run.add_argument("--neighbours", type=int, default=40)
    run.add_argument("--refine-top", type=int, default=3)
    run.add_argument("--scvx-iterations", type=int, default=40)
    run.add_argument("--node-days", type=float, default=2.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--search-only", action="store_true")
    run.add_argument(
        "--full-catalogue", action="store_true", help="screen the full 60,000-asteroid catalogue"
    )
    run.set_defaults(function=cmd_run)

    export = commands.add_parser("export-viewer", help="propagate a solution and write viewer data")
    export.add_argument("solution")
    export.add_argument("--output", required=True)
    export.add_argument("--run-id", default="manual")
    export.set_defaults(function=cmd_export_viewer)
