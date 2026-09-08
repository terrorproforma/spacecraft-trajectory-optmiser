"""Final CPU regression harness with the committed host-table fixture correction."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from prepare import KIT, ROOT, read, sha, write

DEST = KIT / "validation-final-02"


def tests():
    source = KIT / "source/candidate-final"
    sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"]
    sys.path.insert(0, str(source / "src"))
    os.environ["SPACEPDHCG_GTOC12_DATA"] = read(KIT / "inputs/v616-profile.json")["data"]
    original_cdll = ctypes.CDLL

    def guarded_cdll(name, *args, **kwargs):
        if name != "libc.so.6":
            raise AssertionError(f"CPU validation forbids native solver/GPU loading: {name}")
        return original_cdll(name, *args, **kwargs)

    with patch.object(ctypes, "CDLL", side_effect=guarded_cdll):
        import pytest
        import spacepdhcg.gtoc12.search as search

        assert Path(search.__file__).resolve() == source / "src/spacepdhcg/gtoc12/search.py"
        return pytest.main([str(KIT / "test_completion_costs_final.py"),
                            str(DEST / "tests/test_gtoc12_collectdp.py"),
                            str(DEST / "tests/test_gtoc12_returns.py"),
                            "-q", "-p", "no:cacheprovider"])


def main():
    if "--tests" in sys.argv:
        raise SystemExit(tests())
    DEST.mkdir()
    (DEST / "tests").mkdir()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name in ("test_gtoc12_collectdp.py", "test_gtoc12_returns.py"):
        payload = subprocess.check_output(["git", "show", f"{head}:tests/{name}"], cwd=ROOT)
        (DEST / "tests" / name).write_bytes(payload)
    delta = subprocess.check_output(["git", "diff", "f8b2ac7a", head, "--", "tests/test_gtoc12_collectdp.py", "tests/test_gtoc12_returns.py"], cwd=ROOT)
    (DEST / "committed-fixture.diff").write_bytes(delta)
    commands = [
        ("pytest", [sys.executable, "-B", str(__file__), "--tests"]),
        ("ruff", [sys.executable, "-B", "-m", "ruff", "check", "--no-cache",
                   str(ROOT / "src/spacepdhcg/gtoc12/search.py"), str(ROOT / "tests/test_gtoc12_completion_costs.py")]),
        ("format", [sys.executable, "-B", "-m", "ruff", "format", "--check", "--no-cache",
                     str(ROOT / "src/spacepdhcg/gtoc12/search.py"), str(ROOT / "tests/test_gtoc12_completion_costs.py")]),
    ]
    checks = []
    for label, cmd in commands:
        completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=240)
        (DEST / f"{label}.log").write_text(completed.stdout + completed.stderr)
        checks.append({"label": label, "command": cmd, "exit_code": completed.returncode,
                       "log_sha256": sha(DEST / f"{label}.log")})
        print(label, completed.returncode, completed.stdout[-4000:], completed.stderr[-2000:])
    write(DEST / "report.json", {
        "checks": checks, "passed": all(check["exit_code"] == 0 for check in checks),
        "GPU_calls": 0, "native_loading_allowlist": ["libc.so.6"],
        "GPU_and_solver_libraries_forbidden": True,
        "production_source_unchanged_from_first_regression_pass": True,
        "test_fixture_commit": head, "test_fixture_diff_sha256": sha(DEST / "committed-fixture.diff"),
        "prior_validation_preserved": "validation/report.json",
        "final_search_sha256": sha(KIT / "source/candidate-final/src/spacepdhcg/gtoc12/search.py"),
        "test_files_sha256": {str(p.relative_to(KIT)): sha(p) for p in [KIT / "test_completion_costs_final.py", *sorted((DEST / "tests").iterdir())]},
    })
    assert all(check["exit_code"] == 0 for check in checks)


if __name__ == "__main__":
    main()
