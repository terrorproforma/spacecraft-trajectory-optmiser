"""Retain CPU-only preparation test and lint output, including failed attempts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]


def main():
    root = KIT / "validation"
    root.mkdir(exist_ok=True)
    folder = root / f"attempt-{len(list(root.glob('attempt-*'))) + 1:02}"
    folder.mkdir(exist_ok=False)
    files = [
        str(KIT / name)
        for name in (
            "prepare.py",
            "common.py",
            "run.py",
            "launch.py",
            "test_preparation.py",
            "validate.py",
            "freeze.py",
        )
    ]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1"}
    commands = [
        ("ruff", [sys.executable, "-B", "-m", "ruff", "check", *files]),
        ("format", [sys.executable, "-B", "-m", "ruff", "format", "--check", *files]),
        (
            "pytest",
            [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                str(KIT / "test_preparation.py"),
                "-q",
                "-p",
                "no:cacheprovider",
                "--junitxml=" + str(folder / "pytest.xml"),
            ],
        ),
    ]
    rows = []
    for name, command in commands:
        p = subprocess.run(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        (folder / f"{name}.log").write_text(p.stdout)
        rows.append({"name": name, "command": command, "exit_code": p.returncode})
    (folder / "report.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(
        json.dumps({"folder": str(folder), "results": [(x["name"], x["exit_code"]) for x in rows]})
    )
    raise SystemExit(int(any(x["exit_code"] for x in rows)))


if __name__ == "__main__":
    main()
