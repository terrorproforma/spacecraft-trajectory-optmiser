"""Record CPU validation for archive column persistence; run from the repository root."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path.cwd().resolve()
OUT = ROOT / "build/performance/archive-variant-persistence-20260909"
TESTS = [
    "tests/test_gtoc12_archive_variant_artifacts.py",
    "tests/test_gtoc12_archive.py",
    "tests/test_gtoc12_run_objective.py",
    "tests/test_gtoc12_fleet_master_objective.py",
    "tests/test_gtoc12_cluster_objective.py",
    "tests/test_gtoc12_run_final_verification.py",
    "tests/test_gtoc12_run_refinement.py",
    "tests/test_gtoc12_certified_objective.py",
    "tests/test_gtoc12_gpu_cli.py",
]
CHANGED = ["src/spacepdhcg/gtoc12/cli.py", "src/spacepdhcg/gtoc12/archive.py", TESTS[0]]
SOURCES = [*CHANGED[:2], "src/spacepdhcg/gtoc12/bundles.py",
           "src/spacepdhcg/gtoc12/pipeline.py", "src/spacepdhcg/gtoc12/search.py",
           *TESTS, "build/performance/archive-variant-persistence-20260909/validate.py"]


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0, str(Path('src').resolve())); "
        "from spacepdhcg.gtoc12 import cli; "
        "assert Path(cli.__file__).resolve() == Path('src/spacepdhcg/gtoc12/cli.py').resolve(); "
        "print('CLI source:', cli.__file__); import pytest; "
        f"raise SystemExit(pytest.main({['-q', *TESTS]!r}))"
    )
    commands = {
        "pytest": [sys.executable, "-c", code],
        "ruff": [str(Path(sys.executable).with_name("ruff")), "check", *CHANGED],
    }
    results = {}
    for name, arguments in commands.items():
        start = time.perf_counter()
        completed = subprocess.run(arguments, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, check=False)
        (OUT / f"{name}.txt").write_text(completed.stdout, encoding="utf-8")
        print(completed.stdout, end="", flush=True)
        results[name] = {"command": arguments, "return_code": completed.returncode,
                         "wall_seconds": time.perf_counter() - start}
    after = hashes()
    summary = {"checks": results, "source_sha256": after, "gpu_work_launched": False,
               "sources_unchanged_during_validation": before == after,
               "scope": "Archive persistence and command behavior; no new physical mission result."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources_unchanged": before == after,
                      "changed_file_sha256": {name: after[name] for name in CHANGED}}))
    return 0 if before == after and all(item["return_code"] == 0 for item in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
