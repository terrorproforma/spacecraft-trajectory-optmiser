"""Prepare-only construction checks and retained linter output."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent


def main():
    output = KIT / (sys.argv[1] if len(sys.argv) > 1 else "validation-01")
    output.mkdir(exist_ok=False)
    files = [p.name for p in KIT.glob("*.py") if p.name != "control_comparator.py"]
    commands = [
        [sys.executable, "-m", "ruff", "format", *files],
        [sys.executable, "-m", "ruff", "check", "--fix", *files],
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "test_construction.py",
        ],
        [sys.executable, "-m", "ruff", "check", *files],
        [sys.executable, "-m", "ruff", "format", "--check", *files],
    ]
    report = {"GPU_calls": 0, "CPU_finish_calls": 0, "commands": []}
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    for number, command in enumerate(commands):
        result = subprocess.run(
            command,
            cwd=KIT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
        (output / f"{number:02}.log").write_text(result.stdout)
        report["commands"].append({"command": command, "exit_code": result.returncode})
        print(result.stdout, flush=True)
        if result.returncode:
            break
    report["passed"] = len(report["commands"]) == len(commands) and all(
        x["exit_code"] == 0 for x in report["commands"]
    )
    report["source_sha256"] = {
        name: hashlib.sha256((KIT / name).read_bytes()).hexdigest() for name in files
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["passed"]


if __name__ == "__main__":
    main()
