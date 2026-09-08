"""Run and preserve the bounded CPU-only preparation checks, never CUDA."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parents[2]
environment = dict(os.environ)
environment.update(
    CUDA_VISIBLE_DEVICES="",
    SPACEPDHCG_GTOC12_GPU_TESTS="0",
    SPACEPDHCG_TEST_GTOC12_JOINT_BATCH="0",
    SPACEPDHCG_GTOC12_DATA="/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    PYTHONHASHSEED="0",
)
files = [
    root / name
    for name in (
        "run.py",
        "prepare.py",
        "attach_audit.py",
        "cpu_audit.py",
        "launch.py",
        "test_driver.py",
        "validate.py",
        "finalize.py",
    )
]
commands = [
    (
        "pytest",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(root / "test_driver.py"),
            f"--junitxml={root / 'pytest.xml'}",
        ],
    ),
    ("cpu-audit", [sys.executable, str(root / "cpu_audit.py")]),
    ("ruff-check", [sys.executable, "-m", "ruff", "check", *map(str, files)]),
    ("ruff-format", [sys.executable, "-m", "ruff", "format", "--check", *map(str, files)]),
    ("launch-recipe", [sys.executable, str(root / "launch.py")]),
]
results = []
for name, command in commands:
    completed = subprocess.run(command, cwd=repo, env=environment, text=True, capture_output=True)
    (root / f"{name}.log").write_text(completed.stdout + completed.stderr)
    results.append({"name": name, "command": command, "returncode": completed.returncode})
    print(json.dumps(results[-1]), flush=True)
    if completed.returncode:
        print(completed.stdout + completed.stderr)
        break
record = {
    "kind": "CPU_only_no_GPU_execution",
    "passed": len(results) == len(commands) and all(r["returncode"] == 0 for r in results),
    "checks": results,
    "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
}
(root / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
if not record["passed"]:
    raise SystemExit(1)
