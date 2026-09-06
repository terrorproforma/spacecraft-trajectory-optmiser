"""Compare native GPU backends on a canonical planner document at equal accuracy.

Use `spacepdhcg validate` to normalize user units before this benchmark. Each
sample includes the native planner's independent replay. Full result documents
are retained beside the report; compact records include their SHA-256 hashes.
Optional per-variant cuDSS paths isolate and fingerprint runtime comparisons.
"""

from __future__ import annotations

import argparse
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
    for name in ("executable", "problem", "baseline", "optimized", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--baseline-core", type=Path)
    parser.add_argument("--optimized-core", type=Path)
    parser.add_argument("--baseline-cudss", type=Path)
    parser.add_argument("--optimized-cudss", type=Path)
    parser.add_argument("--reference-objective", type=float)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats positive")
    if args.reference_objective is not None and not math.isfinite(args.reference_objective):
        parser.error("reference objective must be finite")
    problem = json.loads(args.problem.read_text())
    if "units" in problem:
        parser.error("normalize the problem with spacepdhcg validate first")
    lock = (Path.home() / ".spacepdhcg-gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw_dir = args.output.with_suffix(".samples")
    raw_dir.mkdir(parents=True, exist_ok=False)
    paths = [args.executable, args.problem, args.baseline, args.optimized]
    default_core = args.executable.parent.parent / "cuda/libspacepdhcg_cuda.so"
    cores = {
        "baseline": args.baseline_core or default_core,
        "optimized": args.optimized_core or default_core,
    }
    paths.extend(cores.values())
    runtimes = {"baseline": args.baseline_cudss, "optimized": args.optimized_cudss}
    for runtime in runtimes.values():
        if runtime is not None:
            if (runtime.parent / "libcudss.so").resolve() != runtime.resolve():
                parser.error("cuDSS directory must expose libcudss.so for the selected runtime")
            paths.append(runtime)
    report = {
        "problem": problem,
        "timing_boundary": "fresh native process including replay; unit parsing excluded",
        "objective_comparison_absolute_tolerance": 1e-8,
        "reference_objective": args.reference_objective,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "core_libraries": {k: str(v.resolve()) for k, v in cores.items()},
        "cudss_libraries": {k: str(v.resolve()) if v else None for k, v in runtimes.items()},
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "samples": [],
    }
    objective = args.reference_objective
    for repeat in range(args.warmups + args.repeats):
        variants = [("baseline", args.baseline), ("optimized", args.optimized)]
        if repeat % 2:
            variants.reverse()
        for variant, backend in variants:
            env = {k: v for k, v in os.environ.items() if not k.startswith("SPACEPDHCG_TEST_")}
            env["SPACEPDHCG_QOCO_LIBRARY"] = str(backend.resolve())
            runtime = runtimes[variant]
            runtime_path = str(runtime.parent.resolve()) + ":" if runtime else ""
            env["LD_LIBRARY_PATH"] = (
                runtime_path + str(cores[variant].resolve().parent)
                + ":" + env.get("LD_LIBRARY_PATH", "")
            )
            output = raw_dir / f"{variant}-{repeat}.json"
            started = time.perf_counter()
            run = subprocess.run(
                [str(args.executable), str(args.problem), "--quiet", "--output", str(output)],
                env=env, capture_output=True, text=True, timeout=120, check=False,
            )
            elapsed = time.perf_counter() - started
            result = json.loads(output.read_text()) if output.exists() else {}
            certificate = result.get("certificate", {})
            summary = result.get("summary", {})
            current_objective = summary.get("objective", math.nan)
            if objective is None and math.isfinite(current_objective):
                objective = current_objective
            passed = (
                run.returncode == 0
                and certificate.get("certified", False)
                and certificate.get("tolerance") == problem["solver"]["tolerance"]
                and certificate.get("continuous_time_within_tolerance", False)
                and bool(certificate.get("gates"))
                and all(
                    g["passed"] and math.isfinite(g["value"])
                    for g in certificate.get("gates", {}).values()
                )
                and result.get("backend", {}).get("hidden_cpu_fallback") is False
                and math.isfinite(current_objective)
                and abs(current_objective - objective) <= 1e-8
            )
            record = {
                "variant": variant, "repeat": repeat, "warmup": repeat < args.warmups,
                "passed": bool(passed), "returncode": run.returncode,
                "process_seconds": elapsed, "stdout": run.stdout, "stderr": run.stderr,
                "result_path": str(output),
                "result_sha256": hashlib.sha256(output.read_bytes()).hexdigest()
                if output.exists() else None,
                **{k: result.get(k) for k in (
                    "status", "summary", "timings", "certificate", "backend", "independent_replay"
                )},
            }
            report["samples"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            if not passed:
                raise RuntimeError(f"{variant} failed equal-accuracy qualification: {args.output}")
            print(
                f"{variant} repeat={repeat} scvx={result['timings']['scvx_total_seconds']:.6f}s "
                f"inner={summary['inner_iterations']} "
                f"accepted={summary['accepted_steps']} certified",
                flush=True,
            )
    medians = {}
    for variant in ("baseline", "optimized"):
        samples = [s for s in report["samples"] if s["variant"] == variant and not s["warmup"]]
        medians[variant] = {
            "process_seconds": statistics.median(s["process_seconds"] for s in samples),
            **{k: statistics.median(s["timings"][k] for s in samples)
               for k in ("scvx_total_seconds", "qoco_solve_seconds", "plan_wall_seconds")},
            "inner_iterations": statistics.median(
                s["summary"]["inner_iterations"] for s in samples
            ),
        }
    report["medians"] = medians
    report["median_speedup"] = {
        k: medians["baseline"][k] / medians["optimized"][k]
        for k in (
            "process_seconds", "scvx_total_seconds", "qoco_solve_seconds", "plan_wall_seconds"
        )
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"medians": medians, "speedup": report["median_speedup"]}, indent=2))


if __name__ == "__main__":
    main()
