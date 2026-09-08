"""Lint and index the finished preparation; never execute the benchmark."""

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
    assert not (KIT / "ready.json").exists()
    assert not (KIT / "launch-marker.json").exists() and not (KIT / "output").exists()
    files = sorted(p.name for p in KIT.glob("*.py") if p.name != "control_comparator.py")
    logs = KIT / "validation-final-02"
    logs.mkdir(exist_ok=False)
    commands = [
        [sys.executable, "-m", "ruff", "format", *files],
        [sys.executable, "-m", "ruff", "check", *files],
        [sys.executable, "-m", "ruff", "format", "--check", *files],
    ]
    report = {"GPU_calls": 0, "CPU_finish_calls": 0, "commands": []}
    for i, command in enumerate(commands):
        done = subprocess.run(
            command,
            cwd=KIT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        (logs / f"{i:02}.log").write_text(done.stdout)
        report["commands"].append({"command": command, "exit_code": done.returncode})
        print(done.stdout, flush=True)
        if done.returncode:
            break
    report["passed"] = len(report["commands"]) == len(commands) and all(
        row["exit_code"] == 0 for row in report["commands"]
    )
    report["files_sha256"] = {name: sha(KIT / name) for name in files}
    (logs / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["passed"]
    assert json.loads((KIT / "validation-03/report.json").read_text())["passed"]
    records = {p.relative_to(KIT).as_posix(): sha(p) for p in sorted(KIT.rglob("*")) if p.is_file()}
    index = {
        "schema": "completion-benchmark-v621",
        "GPU_executed": False,
        "files": records,
        "file_count": len(records),
        "bytes": sum((KIT / name).stat().st_size for name in records),
        "budget": {
            "GPU_calls": 60,
            "candidate_evaluations_per_backend": 14200,
            "maximum_worker_seconds": 180,
            "retries": 0,
        },
    }
    (KIT / "ready.json").write_text(json.dumps(index, indent=2) + "\n")
    for name, digest in records.items():
        assert sha(KIT / name) == digest
    print(
        json.dumps(
            {
                "ready_sha256": sha(KIT / "ready.json"),
                "file_count": len(records),
                "bytes": index["bytes"],
                "launcher_sha256": sha(KIT / "launch.py"),
                "worker_sha256": sha(KIT / "run.py"),
                "common_sha256": sha(KIT / "common.py"),
            }
        )
    )


if __name__ == "__main__":
    main()
