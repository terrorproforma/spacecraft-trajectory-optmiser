"""Re-run saved-vector and coefficient audits in a new materialized CPU-only tree."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from portable import materialize, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verification = verify(args.package)
    out = materialize(args.package, args.out.resolve())
    perf = out / "build/performance"
    logs = out / "cpu-recheck"
    logs.mkdir()
    commands = [
        ("native", perf / "completion-integration-review-v621/review_native_readback.py", []),
        ("adapter", perf / "completion-integration-review-v621/review_adapter_readback.py", []),
        ("mass", perf / "mass-structure-review-v621/audit_mass.py", []),
        ("metric", perf / "mass-diagonal-metric-v621/audit_metric.py", []),
        ("benchmark", perf / "completion-benchmark-review-v621/audit_benchmark.py", []),
    ]
    report = {"passed": False, "GPU_calls": 0, "native_library_loads": 0,
              "package_verification": verification, "commands": []}
    for label, script, extra in commands:
        start = time.monotonic()
        command = [sys.executable, "-B", str(script), *extra]
        done = subprocess.run(command, cwd=out, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        log = logs / (label + ".log")
        log.write_bytes(done.stdout)
        report["commands"].append({"name": label, "command": command, "returncode": done.returncode,
                                  "seconds": time.monotonic()-start,
                                  "log_sha256": hashlib.sha256(done.stdout).hexdigest()})
        (logs / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        assert done.returncode == 0, f"{label}: see {log}"
    for relative in ("reviews/mass-design/findings.json", "reviews/mass-metric/findings.json",
                     "reviews/orchestration/adapter-readback-findings.json"):
        directory = {"reviews/mass-design": "mass-structure-review-v621",
                     "reviews/mass-metric": "mass-diagonal-metric-v621",
                     "reviews/orchestration": "completion-integration-review-v621"}[str(Path(relative).parent).replace("\\", "/")]
        assert (args.package / relative).read_bytes() == (perf / directory / Path(relative).name).read_bytes(), relative
    report["passed"] = True
    report["byte_identical_regenerated_reports"] = ["mass", "metric", "adapter"]
    (logs / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
