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
        ("metric", perf / "mass-diagonal-metric-v621/audit_metric.py", []),
        ("mass", perf / "mass-structure-review-v621/audit_mass.py", []),
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
        # Original evidence keeps its Windows-authored CRLF bytes. A Linux rerun
        # writes LF: compare JSON values, never rewrite the pinned input first.
        expected = json.loads((args.package / relative).read_text(encoding="utf-8"))
        actual = json.loads((perf / directory / Path(relative).name).read_text(encoding="utf-8"))
        if "input_sha256" in expected:
            # Relative path labels are Windows-authored in the original audit.
            # Hash values and every numerical/result field remain exact.
            for value in (expected, actual):
                value["input_sha256"] = {name.replace("\\", "/"): digest for name, digest in value["input_sha256"].items()}
        assert expected == actual, relative
    report["passed"] = True
    report["JSON_value_identical_regenerated_reports"] = ["mass", "metric", "adapter"]
    report["serialization_note"] = "Original bytes remain untouched; generated JSON is compared by value with input-hash path labels normalized from Windows separators."
    (logs / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
