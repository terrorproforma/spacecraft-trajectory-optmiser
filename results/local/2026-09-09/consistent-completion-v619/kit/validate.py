"""CPU-only pinned regression evidence; rejects native library loading."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from prepare import KIT, ROOT, read, sha, write


def tests():
    source = KIT / "source/candidate-final"
    sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"]
    sys.path.insert(0, str(source / "src"))
    os.environ["SPACEPDHCG_GTOC12_DATA"] = read(KIT / "inputs/v616-profile.json")["data"]
    with patch.object(ctypes, "CDLL", side_effect=AssertionError("Validation forbids native loading")):
        import pytest
        import spacepdhcg.gtoc12.search as search

        assert Path(search.__file__).resolve() == source / "src/spacepdhcg/gtoc12/search.py"
        return pytest.main([str(KIT / "test_completion_costs_final.py"),
                            str(KIT / "validation/tests/test_gtoc12_collectdp.py"),
                            str(KIT / "validation/tests/test_gtoc12_returns.py"),
                            "-q", "-p", "no:cacheprovider"])


def main():
    if "--tests" in sys.argv:
        raise SystemExit(tests())
    dest = KIT / "validation"
    dest.mkdir(exist_ok=True)
    (dest / "tests").mkdir(exist_ok=True)
    for name in ("test_gtoc12_collectdp.py", "test_gtoc12_returns.py"):
        payload = subprocess.check_output(["git", "show", f"f8b2ac7a:tests/{name}"], cwd=ROOT)
        with (dest / "tests" / name).open("xb") as stream:
            stream.write(payload)
    commands = [
        ("pytest", [sys.executable, "-B", str(__file__), "--tests"]),
        ("ruff", [sys.executable, "-B", "-m", "ruff", "check", "--no-cache",
                   str(ROOT / "src/spacepdhcg/gtoc12/search.py"),
                   str(ROOT / "tests/test_gtoc12_completion_costs.py")]),
        ("format", [sys.executable, "-B", "-m", "ruff", "format", "--check", "--no-cache",
                     str(ROOT / "src/spacepdhcg/gtoc12/search.py"),
                     str(ROOT / "tests/test_gtoc12_completion_costs.py")]),
    ]
    checks = []
    for label, cmd in commands:
        completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=240)
        (dest / f"{label}.log").write_text(completed.stdout + completed.stderr)
        checks.append({"label": label, "command": cmd, "exit_code": completed.returncode,
                       "log_sha256": sha(dest / f"{label}.log")})
        print(label, completed.returncode, completed.stdout[-3000:], completed.stderr[-2000:])
    write(dest / "report.json", {
        "checks": checks,
        "passed": all(check["exit_code"] == 0 for check in checks),
        "GPU_calls": 0, "native_loading_forbidden": True,
        "final_search_sha256": sha(KIT / "source/candidate-final/src/spacepdhcg/gtoc12/search.py"),
        "test_files_sha256": {
            str(p.relative_to(KIT)): sha(p) for p in [KIT / "test_completion_costs_final.py", *sorted((dest / "tests").iterdir())]
        },
    })
    assert all(check["exit_code"] == 0 for check in checks)


if __name__ == "__main__":
    main()
