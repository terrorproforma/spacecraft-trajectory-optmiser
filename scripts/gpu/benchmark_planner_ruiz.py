"""Sweep native GPU Ruiz passes against a fixed, previously qualified objective.

This is an exploratory sweep: failures are retained and disqualify that setting.
Every sample uses a fresh process and unchanged problem tolerances. GPU jobs are
serialized with the same advisory lock as the other local benchmark runners.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "core", "qoco", "cudss", "problem", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reference-objective", type=float, required=True)
    parser.add_argument("--passes", type=int, nargs="+", default=[0, 1, 2, 4, 8, 12])
    parser.add_argument("--warm-starts", nargs="+", choices=["none", "primal"])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if (
        not math.isfinite(args.reference_objective)
        or args.warmups < 0 or args.repeats < 1
        or not math.isfinite(args.timeout) or args.timeout <= 0
        or len(set(args.passes)) != len(args.passes)
        or any(p < 0 or p > 100 for p in args.passes)
    ):
        parser.error("invalid objective, passes, repeat count or timeout")
    problem = json.loads(args.problem.read_text())
    if "units" in problem or problem["solver"]["backend"] != "pure_qoco":
        parser.error("requires a canonical pure_qoco problem")
    warm_starts = args.warm_starts or [problem["solver"].get("warm_start_mode", "primal")]
    if len(set(warm_starts)) != len(warm_starts):
        parser.error("warm-start modes must be distinct")
    variants = [(p, w) for p in args.passes for w in warm_starts]
    lock = (Path.home() / ".spacepdhcg-gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw = args.output.with_suffix(".samples")
    raw.mkdir(parents=True, exist_ok=False)
    paths = [args.executable, args.core, args.qoco, args.cudss, args.problem]
    report = {
        "reference_objective": args.reference_objective,
        "objective_absolute_tolerance": 1e-8,
        "problem": problem,
        "passes": args.passes,
        "warm_starts": warm_starts,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "timeout_seconds": args.timeout,
        "timing_boundary": "fresh native process including independent replay",
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "samples": [],
    }
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPACEPDHCG_TEST_")}
    env.pop("SPACEPDHCG_QOCO_VERBOSE", None)
    env["SPACEPDHCG_QOCO_LIBRARY"] = str(args.qoco.resolve())
    env["LD_LIBRARY_PATH"] = ":".join([
        str(args.core.resolve().parent), str(args.cudss.resolve().parent),
        "/usr/local/cuda-12.8/lib64",
    ])
    inputs = {}
    for passes, warm in variants:
        document = copy.deepcopy(problem)
        document["solver"]["qoco_ruiz_iterations"] = passes
        document["solver"]["warm_start_mode"] = warm
        inputs[passes, warm] = raw / f"input-{passes}-{warm}.json"
        inputs[passes, warm].write_text(json.dumps(document, indent=2) + "\n")
    for repeat in range(args.warmups + args.repeats):
        offset = repeat % len(variants)
        order = variants[offset:] + variants[:offset]
        for passes, warm in order:
            output = raw / f"ruiz-{passes}-{warm}-{repeat}.json"
            command = [
                str(args.executable), str(inputs[passes, warm]), "--quiet", "--output", str(output)
            ]
            started = time.perf_counter()
            try:
                run = subprocess.run(command, env=env, capture_output=True, text=True,
                                     timeout=args.timeout, check=False)
                rc, stdout, stderr = run.returncode, run.stdout, run.stderr
            except subprocess.TimeoutExpired as error:
                rc = None
                stdout = (error.stdout or b"").decode(errors="replace")
                stderr = (error.stderr or b"").decode(errors="replace")
            elapsed = time.perf_counter() - started
            result = json.loads(output.read_text()) if output.exists() else {}
            certificate = result.get("certificate", {})
            backend = result.get("backend", {})
            objective = result.get("summary", {}).get("objective")
            objective = objective if isinstance(objective, (int, float)) else math.nan
            gates = {
                "process_success": rc == 0,
                "certified": certificate.get("certified") is True,
                "unchanged_tolerance": certificate.get("tolerance")
                == problem["solver"]["tolerance"],
                "continuous_time": certificate.get("continuous_time_within_tolerance") is True,
                "physics_gates": bool(certificate.get("gates")) and all(
                    g["passed"] and math.isfinite(g["value"])
                    for g in certificate.get("gates", {}).values()
                ),
                "gpu_backend": backend.get("hidden_cpu_fallback") is False,
                "requested_passes": backend.get("requested_qoco_ruiz_iterations") == passes,
                "applied_passes": backend.get("qoco_ruiz_iterations") == passes,
                "warm_start_mode": backend.get("warm_start_mode") == warm,
                "fixed_objective": math.isfinite(objective)
                and abs(objective - args.reference_objective) <= 1e-8,
            }
            record = {
                "passes": passes, "warm_start_mode": warm,
                "repeat": repeat, "warmup": repeat < args.warmups,
                "passed": all(gates.values()), "qualification": gates,
                "returncode": rc, "process_seconds": elapsed,
                "command": command, "stdout": stdout, "stderr": stderr,
                "result_path": str(output),
                "result_sha256": hashlib.sha256(output.read_bytes()).hexdigest()
                if output.exists() else None,
                **{k: result.get(k) for k in (
                    "status", "summary", "timings", "certificate", "backend", "independent_replay"
                )},
            }
            report["samples"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: record[k] for k in (
                "passes", "warm_start_mode", "repeat", "passed", "qualification",
                "summary", "timings"
            )}), flush=True)
    report["settings"] = {}
    for passes, warm in variants:
        all_samples = [s for s in report["samples"]
                       if s["passes"] == passes and s["warm_start_mode"] == warm]
        measured = [s for s in all_samples if not s["warmup"]]
        qualified = all(s["passed"] for s in all_samples)
        key = str(passes) if len(warm_starts) == 1 else f"{passes}:{warm}"
        report["settings"][key] = {
            "all_qualified": qualified,
            "failures_including_warmups": sum(not s["passed"] for s in all_samples),
            # No success-only latency summary for disqualified settings.
            "median_scvx_seconds": statistics.median(
                s["timings"]["scvx_total_seconds"] for s in measured
            ) if qualified else None,
            "median_inner_iterations": statistics.median(
                s["summary"]["inner_iterations"] for s in measured
            ) if qualified else None,
            "p95_scvx_seconds_nearest_rank": sorted(
                s["timings"]["scvx_total_seconds"] for s in measured
            )[math.ceil(0.95 * len(measured)) - 1] if qualified else None,
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["settings"], indent=2))


if __name__ == "__main__":
    main()
