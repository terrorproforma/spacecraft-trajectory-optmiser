"""Save focused CPU test/lint evidence; never invoke a native solver."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]


def main():
    base = KIT / "validation"
    base.mkdir(exist_ok=True)
    attempt = base / f"attempt-{len(list(base.iterdir())) + 1:02}"
    attempt.mkdir(exist_ok=False)
    files = [
        "src/spacepdhcg/gtoc12/refinement_admission.py",
        "tests/test_gtoc12_refinement_admission.py",
        "build/performance/refinement-admission-v622/prepare.py",
        "build/performance/refinement-admission-v622/replay.py",
        "build/performance/refinement-admission-v622/validate.py",
    ]
    tests = ["tests/test_gtoc12_refinement_admission.py"]
    for source, test in (
        ("src/spacepdhcg/gtoc12/fixed_refinement.py", "tests/test_gtoc12_fixed_refinement.py"),
    ):
        if (ROOT / source).exists():
            files += [source, test]
            tests.append(test)
    if (ROOT / "tests/test_gtoc12_completion_capture.py").exists():
        files += [
            "src/spacepdhcg/gtoc12/completion_capture.py",
            "tests/test_gtoc12_completion_capture.py",
        ]
        tests.append("tests/test_gtoc12_completion_capture.py")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1")
    pytest_args = [
        *tests,
        "-q",
        "-p",
        "no:cacheprovider",
        "--junitxml=" + str(attempt / "pytest.xml"),
    ]
    script = (
        "import sys; sys.meta_path=[f for f in sys.meta_path if "
        "f.__class__.__module__!='_editable_skbc_spacepdhcg']; "
        f"sys.path.insert(0,{str(ROOT / 'src')!r}); import pytest; "
        f"raise SystemExit(pytest.main({pytest_args!r}))"
    )
    commands = [
        ("ruff-check", [sys.executable, "-B", "-m", "ruff", "check", *files]),
        ("ruff-format", [sys.executable, "-B", "-m", "ruff", "format", "--check", *files]),
        ("pytest", [sys.executable, "-B", "-c", script]),
    ]
    report = []
    for name, command in commands:
        result = subprocess.run(
            command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (attempt / f"{name}.log").write_text(result.stdout)
        report.append({"name": name, "command": command, "returncode": result.returncode})
    (attempt / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"path": str(attempt), "commands": [(x["name"], x["returncode"]) for x in report]}
        )
    )
    raise SystemExit(int(any(x["returncode"] for x in report)))


if __name__ == "__main__":
    main()
