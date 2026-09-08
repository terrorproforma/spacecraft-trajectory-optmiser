"""Validate only adapter construction/comparison; no GPU or CPU cost-model calls."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent


def main():
    destination = KIT / (sys.argv[1] if len(sys.argv) > 1 else "runner-validation-01")
    destination.mkdir(exist_ok=False)
    files = ["gpu_batch.py", "gpu_runner.py", "test_gpu_runner_contract.py", "validate_runner.py"]
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
            "test_gpu_runner_contract.py",
        ],
        [sys.executable, "-m", "ruff", "check", *files],
        [sys.executable, "-m", "ruff", "format", "--check", *files],
    ]
    report = {"cost_model_calls": 0, "GPU_calls": 0, "commands": []}
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    for number, command in enumerate(commands):
        done = subprocess.run(
            command,
            cwd=KIT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        (destination / f"{number:02}.log").write_text(done.stdout)
        report["commands"].append({"command": command, "exit_code": done.returncode})
        print(done.stdout, flush=True)
        if done.returncode:
            break
    report["passed"] = len(report["commands"]) == len(commands) and all(
        row["exit_code"] == 0 for row in report["commands"]
    )
    report["files_sha256"] = {
        name: hashlib.sha256((KIT / name).read_bytes()).hexdigest() for name in files
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["passed"]


if __name__ == "__main__":
    main()
