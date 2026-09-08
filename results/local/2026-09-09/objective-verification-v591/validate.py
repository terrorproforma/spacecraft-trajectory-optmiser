"""CPU-only objective regression evidence; execute from the repository root."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import time

ROOT = Path.cwd().resolve()
OUT = ROOT / "build/performance/objective-verification-v591"
TESTS = [
    "tests/test_gtoc12_run_objective.py",
    "tests/test_gtoc12_fleet_master_objective.py",
    "tests/test_gtoc12_cluster_objective.py",
    "tests/test_gtoc12_run_final_verification.py",
    "tests/test_gtoc12_run_refinement.py",
    "tests/test_gtoc12_certified_objective.py",
    "tests/test_gtoc12_gpu_cli.py",
]
SOURCES = [
    "src/spacepdhcg/gtoc12/cli.py",
    "src/spacepdhcg/gtoc12/cooperative.py",
    "src/spacepdhcg/gtoc12/constants.py",
    "src/spacepdhcg/gtoc12/verifier.py",
    "src/spacepdhcg/gtoc12/official.py",
    "src/spacepdhcg/gtoc12/fleet.py",
    "src/spacepdhcg/gtoc12/pipeline.py",
    "src/spacepdhcg/gtoc12/solution.py",
    *TESTS,
    "tests/test_gtoc12_verifier.py",
    "build/performance/objective-verification-v591/validate.py",
]


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def command(name, arguments):
    started = time.perf_counter()
    completed = subprocess.run(arguments, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    (OUT / f"{name}.txt").write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="", flush=True)
    return {"command": arguments, "return_code": completed.returncode,
            "wall_seconds": time.perf_counter() - started}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    test_code = (
        "import sys; from pathlib import Path; sys.path.insert(0, str(Path('src').resolve())); "
        "from spacepdhcg.gtoc12 import cli; "
        "assert Path(cli.__file__).resolve() == Path('src/spacepdhcg/gtoc12/cli.py').resolve(); "
        "print('CLI source:', cli.__file__); import pytest; "
        f"raise SystemExit(pytest.main({['-q', *TESTS]!r}))"
    )
    checks = {"pytest": command("pytest", [sys.executable, "-c", test_code])}
    checks["ruff"] = command("ruff", [str(Path(sys.executable).with_name("ruff")), "check",
                                      "src/spacepdhcg/gtoc12/cli.py", *TESTS])
    sys.path.insert(0, str(ROOT / "src"))
    from spacepdhcg.gtoc12.data import official_verifier_binary
    from spacepdhcg.gtoc12.official import run_official_verifier
    from spacepdhcg.gtoc12.solution import Solution
    from spacepdhcg.gtoc12.verifier import Gtoc12Verifier

    fixtures = runpy.run_path(str(ROOT / "tests/test_gtoc12_verifier.py"))
    binary = official_verifier_binary()
    probes = []
    for name, mass_drop in (("ship_rule_only", 0.0), ("ship_rule_and_mass_error", 1.0)):
        ships = [fixtures["_coasting_earth_ship"]().ships[0] for _ in range(2)]
        ships.append(fixtures["_coasting_earth_ship"](mass_drop=mass_drop).ships[0])
        for ship_id, ship in enumerate(ships, start=1):
            ship.ship_id = ship_id
        path = OUT / f"{name}.txt"
        solution = Solution(ships)
        solution.write(path)
        independent = Gtoc12Verifier(fixtures["_empty_catalogue"]()).verify(solution)
        official = run_official_verifier(path)
        probes.append({"name": name, "solution": str(path.relative_to(ROOT)),
                       "independent": independent.summary(),
                       "independent_codes": [v.code for v in independent.violations],
                       "official": official.summary(), "official_stdout": official.stdout,
                       "official_stderr": official.stderr})
    (OUT / "official_probes.json").write_text(json.dumps({
        "purpose": "Synthetic classifier evidence; these are not accepted mission fleets.",
        "official_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "probes": probes,
    }, indent=2) + "\n", encoding="utf-8")
    after = hashes()
    assert set(probes[0]["independent_codes"]) == {"Error301"}
    assert {"Error203", "Error301"} <= set(probes[1]["independent_codes"])
    assert probes[0]["official"]["return_code"] == 3
    summary = {"started_utc": start, "python": sys.version, "platform": platform.platform(),
               "gpu_work_launched": False, "checks": checks,
               "sources_unchanged_during_validation": before == after,
               "source_sha256": after}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evidence": str(OUT), "sources_unchanged": before == after,
                      "check_returns": {key: value["return_code"] for key, value in checks.items()}}))
    return 0 if before == after and all(check["return_code"] == 0 for check in checks.values()) else 1


if __name__ == "__main__":
    os.environ.setdefault("SPACEPDHCG_GTOC12_DATA",
                          "/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data")
    raise SystemExit(main())
