"""CPU-only validation and immutable ready index; refuses to overwrite evidence."""

import ctypes
import datetime
import json
import os
import subprocess
import sys
from unittest.mock import patch

import support


def main():
    root = support.ROOT
    evidence = root / "validation"
    evidence.mkdir(exist_ok=False)
    report = {"GPU_calls": 0, "utc": datetime.datetime.now(datetime.UTC).isoformat(), "checks": []}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    authored = [
        root / name
        for name in (
            "prepare.py",
            "support.py",
            "prefix.py",
            "screen.py",
            "run.py",
            "launch.py",
            "test_rescue.py",
            "validate.py",
        )
    ]
    commands = [
        ("pytest", [sys.executable, "-B", "-m", "pytest", "-q", str(root / "test_rescue.py")]),
        ("ruff-check", [sys.executable, "-m", "ruff", "check", *map(str, authored)]),
        ("ruff-format", [sys.executable, "-m", "ruff", "format", "--check", *map(str, authored)]),
    ]
    for name, command in commands:
        completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
        log = evidence / (name + ".log")
        log.write_text(completed.stdout + completed.stderr)
        report["checks"].append(
            {
                "name": name,
                "command": command,
                "returncode": completed.returncode,
                "log_sha256": support.sha(log),
            }
        )
        print(name, completed.returncode, completed.stdout.strip(), flush=True)
        if completed.returncode:
            support.write(evidence / "report.json", report)
            return completed.returncode
    with patch.object(
        ctypes, "CDLL", side_effect=AssertionError("CPU audit forbids native loading")
    ):
        support.activate()
        profile, runtime = support.environment({})
        support.validate_runtime(runtime)
        os.environ["SPACEPDHCG_GTOC12_DATA"] = profile["data"]
        from prefix import assert_prefix_emission, certify_wait, load_prefix, p
        from screen import coarse_windows

        from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
        from spacepdhcg.gtoc12.solution import format_solution

        catalogue, bonus = load_catalogue(), load_bonus_table()
        prefixes = {name: load_prefix(name, catalogue) for name in ("control", "probe")}
        original = load_prefix("control", catalogue, include_archived_return=True)
        summary = support.read(root / "inputs/control/refinement.json")
        route = p.RefinedRoute(
            original.plan,
            original.legs,
            original.cargo,
            summary["final_mass_kg"],
            True,
            True,
            1,
            0,
            {},
            [],
        )
        solution = p.emit_solution(route, catalogue)
        emitted = format_solution(solution).encode()
        if emitted != (root / "inputs/control/Result.txt").read_bytes():
            raise AssertionError("original full-control Result changed")
        report.update(
            source_commit=support.read(root / "preparation.json")["source_commit"],
            source_files=len(support.read(root / "source-sha256.json")),
            input_files=len(support.read(root / "inputs-sha256.json")),
            native_libraries=profile["native_libraries"],
            catalogue_sha256=catalogue.source_sha256,
            bonus_sha256=bonus.source_sha256,
            original_control_Result_byte_identical=True,
            original_control_Result_sha256=support.sha(root / "inputs/control/Result.txt"),
            control_prefix_emission=assert_prefix_emission(prefixes["control"], solution),
            prefix_audits={
                name: prefix.audit | {"fingerprint": prefix.fingerprint()}
                for name, prefix in prefixes.items()
            },
            positive_waits=[
                certify_wait(prefixes["probe"], catalogue, 69218 + wait)
                for wait in (1, 30, 79, 300, 588)
            ],
            coarse_rows=len(coarse_windows()),
            maximum_fine_rows=2592,
            maximum_geometry_pairs=9614,
            maximum_direction_requests=19228,
            maximum_native_returns=5,
            maximum_candidate_returns=4,
            whole_route_reruns=0,
            soft_wall_seconds=1800,
            control_failure_stops_candidates=True,
            ready=True,
        )
    support.write(evidence / "report.json", report)
    if any((root / name).exists() for name in ("ready-manifest.json", "launch.json", "output")):
        raise FileExistsError("ready/launch/output must be absent before freezing")
    files = {
        path.relative_to(root).as_posix(): support.sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not set(path.parts) & {"__pycache__", ".pytest_cache", ".ruff_cache"}
    }
    ready = {
        "ready": True,
        "GPU_calls": 0,
        "source_commit": report["source_commit"],
        "files": files,
        "file_count": len(files),
        "indexed_bytes": sum((root / name).stat().st_size for name in files),
        "validation_report_sha256": support.sha(evidence / "report.json"),
        "maximum_native_returns": 5,
        "maximum_candidate_returns": 4,
        "maximum_geometry_pairs": 9614,
        "whole_route_reruns": 0,
        "scope": "CPU-qualified preparation; no fresh GPU convergence or score claim",
    }
    support.write(root / "ready-manifest.json", ready)
    support.validate_ready()
    print(
        json.dumps(
            {key: ready[key] for key in ("file_count", "indexed_bytes", "GPU_calls")}
            | {"ready_manifest_sha256": support.sha(root / "ready-manifest.json")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
