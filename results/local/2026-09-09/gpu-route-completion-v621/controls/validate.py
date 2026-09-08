"""CPU-only validation with complete command logs and source hashes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = KIT / (sys.argv[1] if len(sys.argv) > 1 else "validation-02")
    destination.mkdir(exist_ok=False)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1"}
    scripts = sorted(p.name for p in KIT.glob("*.py"))
    commands = [
        [sys.executable, "-m", "ruff", "format", *scripts],
        [sys.executable, "-m", "ruff", "check", "--fix", *scripts],
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "test_fixture_evidence.py",
        ],
        [sys.executable, "-m", "ruff", "check", *scripts],
        [sys.executable, "-m", "ruff", "format", "--check", *scripts],
    ]
    results = []
    for number, command in enumerate(commands):
        done = subprocess.run(
            command,
            cwd=KIT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
        (destination / f"{number:02}.log").write_text(done.stdout)
        results.append(
            {"command": command, "exit_code": done.returncode, "log": f"{number:02}.log"}
        )
        print(done.stdout, flush=True)
        if done.returncode:
            break
    report = {
        "python": sys.version,
        "commands": results,
        "all_passed": len(results) == len(commands)
        and all(row["exit_code"] == 0 for row in results),
        "scripts_sha256": {p.name: sha(p) for p in sorted(KIT.glob("*.py"))},
        "fixtures_sha256": sha(KIT / "fixtures.json"),
        "GPU_calls": 0,
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["all_passed"]


if __name__ == "__main__":
    main()
