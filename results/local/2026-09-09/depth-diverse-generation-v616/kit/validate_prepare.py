"""CPU-only final evidence and immutable manifest. Never launches the generation."""

import ctypes
import dataclasses
import datetime
import os
import re
import subprocess
import sys
from unittest.mock import patch

import common


def main():
    root = common.ROOT
    validation = root / "validation-final-02"
    validation.mkdir(exist_ok=False)
    authored = sorted(str(path.relative_to(root)) for path in root.glob("*.py"))
    commands = [
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "test_generation.py",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        [sys.executable, "-B", "-m", "ruff", "check", *authored],
        [sys.executable, "-B", "-m", "ruff", "format", "--check", *authored],
    ]
    evidence = {
        "GPU_calls": 0,
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "commands": [],
    }
    for i, command in enumerate(commands):
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=90)
        log = validation / f"{i + 1:02d}.log"
        log.write_text(result.stdout + result.stderr)
        evidence["commands"].append(
            {
                "command": command,
                "returncode": result.returncode,
                "log": str(log.relative_to(root)),
                "sha256": common.sha(log),
            }
        )
        common.write(validation / "report.json", evidence)
        if result.returncode:
            raise RuntimeError(
                "CPU validation failed; retain evidence and review before refreezing"
            )
    common.activate()
    profile, env = common.environment({})
    common.runtime_check(env)
    os.environ["SPACEPDHCG_GTOC12_DATA"] = profile["data"]
    with patch.object(
        ctypes, "CDLL", side_effect=AssertionError("No native loading in preparation")
    ):
        from domain import LIMITS, load_inputs, settings

        _catalogue, _weights, seed, allowed, excluded, baseline = load_inputs()
        modules = [
            "gpu_lambert.py",
            "gpu_neighbours.py",
            "gpu_collection.py",
            "gpu_collect_dp.py",
            "gpu_collect_tables.py",
            "gpu_options.py",
        ]
        names = set()
        for name in modules:
            names.update(
                re.findall(
                    r"spacepdhcg_[A-Za-z0-9_]+",
                    (root / "source/src/spacepdhcg/gtoc12" / name).read_text(),
                )
            )
        library = profile["native_libraries"]["SPACEPDHCG_GTOC12_CUDA_LIBRARY"]["path"]
        nm = subprocess.run(
            ["nm", "-D", "--defined-only", library],
            check=True,
            text=True,
            capture_output=True,
            timeout=15,
        )
        exported = {line.split()[-1] for line in nm.stdout.splitlines() if line.split()}
        missing = sorted(names - exported)
        evidence["native_symbol_inventory"] = {
            "scope": "Export presence only, no library load or GPU execution",
            "requested": sorted(names),
            "missing": missing,
            "library_sha256": common.sha(library),
        }
        if missing:
            common.write(validation / "report.json", evidence)
            raise ValueError("Native exports missing: " + str(missing))
        common.write(
            root / "plan.json",
            {
                "status": "prepared_not_executed",
                "limits": LIMITS,
                "settings": dataclasses.asdict(settings()),
                "seed": dataclasses.asdict(seed),
                "family_count": len(allowed),
                "excluded_count": len(excluded),
                "baseline": baseline,
                "source_commit": "f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7",
                "no_production_source_change": True,
                "no_depth_diverse_policy_implemented": True,
                "matched_refinement_comparison_prepared": False,
                "reason": (
                    "No adequate saved generated pool; bounded fresh generation is prerequisite"
                ),
                "driver_sha256": common.sha(root / "run.py"),
                "supervisor_sha256": common.sha(root / "launch.py"),
            },
        )
    evidence.update(
        passed=True,
        GPU_calls=0,
        finished_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        authored_file_hashes={name: common.sha(root / name) for name in authored},
    )
    common.write(validation / "report.json", evidence)
    files = {
        str(path.relative_to(root)): common.sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not any(
            part in ("__pycache__", ".pytest_cache", ".ruff_cache")
            for part in path.relative_to(root).parts
        )
        and path.name != "ready-manifest.json"
    }
    common.write(
        root / "ready-manifest.json",
        {
            "status": "prepared_not_executed",
            "files": files,
            "file_count": len(files),
            "bytes": sum((root / name).stat().st_size for name in files),
            "source_files": len(common.read(root / "source-sha256.json")),
            "input_files": len(common.read(root / "inputs-sha256.json")),
            "GPU_calls_during_preparation": 0,
            "full_route_refinements": 0,
            "single_gpu_launch_authority": "Root review required; no launch by preparation agent",
        },
    )
    print(
        {
            "ready_sha256": common.sha(root / "ready-manifest.json"),
            "files": len(files),
            "driver_sha256": common.sha(root / "run.py"),
            "GPU_calls": 0,
        }
    )


if __name__ == "__main__":
    main()
