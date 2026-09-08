"""CPU-only final checks and one-time ready manifest for the prepared kit."""

import ctypes
import datetime
import subprocess
import sys
from unittest.mock import patch

import common


def main():
    root = common.ROOT
    if any((root / name).exists() for name in ("ready-manifest.json", "launch.json", "output")):
        raise FileExistsError("Ready or launched kit must remain frozen")
    evidence = root / "validation"
    evidence.mkdir(exist_ok=False)
    profile, env = common.environment()
    common.runtime_check(env)
    authored = sorted(str(p) for p in root.glob("*.py") if p.name != "fixed_refine.py")
    commands = {
        "pytest": [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(root / "test_experiment.py"),
        ],
        "ruff-check": [sys.executable, "-B", "-m", "ruff", "check", *authored],
        "ruff-format": [sys.executable, "-B", "-m", "ruff", "format", "--check", *authored],
    }
    results = {}
    for name, command in commands.items():
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**env, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        (evidence / (name + ".log")).write_text(result.stdout + result.stderr)
        results[name] = {"command": command, "returncode": result.returncode}
        if result.returncode:
            common.write(evidence / "failed.json", results)
            raise RuntimeError(name + " failed; preserve this validation attempt")
    with patch.object(ctypes, "CDLL", side_effect=AssertionError("CPU-only validation")):
        common.activate()
        from domain import bounds

        counts = bounds()
    controls = common.read(root / "incumbent-controls.json")
    if (
        controls["fresh_control_refinement_required"]
        or controls["critical_source_differences"]
        or not all(controls["exact_ship_bytes_equal"].values())
    ):
        raise ValueError("Archived controls do not qualify")
    core = profile["native_libraries"]["SPACEPDHCG_GTOC12_CUDA_LIBRARY"]["path"]
    symbols = subprocess.run(
        ["nm", "-D", "--defined-only", core],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    search = [
        line for line in symbols.splitlines() if "spacepdhcg_gtoc12_joint_search_host" in line
    ]
    if len(search) != 1:
        raise ValueError("Actual pinned local library lacks unique native search export")
    source = common.read(root / "source-sha256.json")
    inputs = common.read(root / "inputs-sha256.json")
    assert len(source) == 190 and len(inputs) == 33
    inventory = common.read(root / "incumbent-inventory-01.json")["routes"]
    selected = [next(row for row in inventory if row["ship"] == ship) for ship in common.SHIPS]
    common.write(
        root / "plan.json",
        {
            "status": "prepared_no_GPU_execution",
            "hypothesis": "Coupled itinerary timing with increased proxy fuel-margin valuation",
            "source_commit": common.read(root / "preparation.json")["source_commit"],
            "incumbent": {
                "ships": 23,
                "raw_kg": 14051.854893908598,
                "weighted_fixed_bonus_kg": 12810.135953048577,
                "raw_ship_count_floor_kg": common.RAW_FLOOR,
            },
            "selected_ships": selected,
            "margin_prices": common.PRICES,
            "mesh_days": common.MESH,
            "max_moves_per_level": common.MAX_MOVES,
            "per_search_soft_deadline_seconds": 30,
            "bounds": counts,
            "maximum_selected_final_candidates": 3,
            "planned_new_native_leg_cap": 51,
            "archived_control_reuse": (
                "Exact original ship bytes, same core/QOCO and critical source"
            ),
            "fresh_control": (
                "Both complete fleet checkers before GPU work; zero repeated full routes"
            ),
            "comparison": "Same four seeds/caps per price; actual work may differ",
            "acceptance": {
                "objective": "Verified fixed-bonus weighted cargo gain",
                "raw_fleet_ship_rule": True,
                "both_complete_fleet_checkers": True,
                "exact_selected_cargo_and_epochs_during_refinement": True,
                "unchanged_physics_tolerances": True,
            },
            "requested_initial_wall_seconds": 600,
            "outer_timeout_seconds": 630,
            "termination_grace_seconds": 10,
            "launch": "launch.py --execute --wall-seconds 600",
            "resource_scope": (
                "One local RTX5090, serial graph searches/refinements; CPU fleet checks"
            ),
            "limitations": [
                "Proxy fuel is an estimate, not a physical certificate",
                "Only final search outcomes are retained by existing graph API",
                "Existing local active-loop memcheck CUDA999 is unresolved",
                "No new performance or score measurements; no guaranteed gain",
            ],
            "rejected_initial_preparation": (
                "seed-audit-01.json retains four infeasible substitution starts"
            ),
        },
    )
    report = {
        "status": "ready_for_root_review_no_GPU_executed",
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "commands": results,
        "GPU_calls": 0,
        "native_loading_prohibited": True,
        "source_commit": common.read(root / "preparation.json")["source_commit"],
        "final_source_count": len(source),
        "final_input_count": len(inputs),
        "original_preparation_input_count": 27,
        "runtime_libraries_rehashed": profile["native_libraries"],
        "native_search_symbol": search,
        "archived_controls_qualified": True,
        "bounds": counts,
        "planned_probe_cap": 3,
        "planned_new_native_leg_cap": 51,
        "requested_initial_wall_seconds": 600,
        "supervisor_timeout_seconds": 630,
        "termination_grace_seconds": 10,
        "test_scope": "CPU behavior, real harmless child timeout, no CUDA execution",
    }
    common.write(evidence / "report.json", report)
    ignored = {"__pycache__", ".pytest_cache", ".ruff_cache"}
    files = {
        str(p.relative_to(root)): common.sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and not ignored.intersection(p.relative_to(root).parts)
    }
    manifest = {
        "source_commit": report["source_commit"],
        "files": files,
        "file_count": len(files),
        "total_bytes": sum((root / name).stat().st_size for name in files),
        "prepared_only": True,
        "GPU_calls": 0,
    }
    common.write(root / "ready-manifest.json", manifest)
    common.ready_check()
    print(
        {
            "ready_sha256": common.sha(root / "ready-manifest.json"),
            "driver_sha256": common.sha(root / "run.py"),
            "supervisor_sha256": common.sha(root / "launch.py"),
            "file_count": len(files),
            "total_bytes": manifest["total_bytes"],
        }
    )


if __name__ == "__main__":
    main()
