"""Alternate GPU IPM backend builds on the same independently qualified landing case.

Run under WSL with the CUDA/cuDSS runtime directories in LD_LIBRARY_PATH. Each
sample starts a new process. Iteration counts are recorded, not assumed equal:
GPU factorization/reductions can change the convergence path at roundoff level.
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
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--baseline-core", type=Path)
    parser.add_argument("--optimized-core", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups nonnegative")
    lock = (Path.home() / ".spacepdhcg-gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = [
        args.executable,
        args.baseline,
        args.optimized,
        args.executable.parent.parent / "cuda/libspacepdhcg_cuda.so",
    ]
    cores = {
        "baseline": args.baseline_core or paths[-1],
        "optimized": args.optimized_core or paths[-1],
    }
    paths.extend(cores.values())
    document = {
        "fixture": "P1-C-pd3, 20 intervals, two accepted outer steps, tolerance 1e-8",
        "warmups_per_variant": args.warmups,
        "measured_samples_per_variant": args.repeats,
        "core_libraries": {k: str(v.resolve()) for k, v in cores.items()},
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    objective = None
    for repeat in range(args.warmups + args.repeats):
        variants = [("baseline", args.baseline), ("optimized", args.optimized)]
        if repeat % 2:
            variants.reverse()
        for variant, library in variants:
            env = {k: v for k, v in os.environ.items() if not k.startswith("SPACEPDHCG_TEST_")}
            env["SPACEPDHCG_QOCO_LIBRARY"] = str(library.resolve())
            env["LD_LIBRARY_PATH"] = (
                str(cores[variant].resolve().parent) + ":" + env.get("LD_LIBRARY_PATH", "")
            )
            start = time.perf_counter()
            run = subprocess.run(
                [str(args.executable), "--p1c-qoco-repeatability", "1"],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            elapsed = time.perf_counter() - start
            records = [
                json.loads(line) for line in run.stdout.splitlines() if line.startswith('{"case":')
            ]
            samples = [r for r in records if r["case"] == "g4_sample"]
            iterations = [r for r in records if r["case"] == "g4_iteration"]
            passed = run.returncode == 0 and len(samples) == 1 and len(iterations) == 2
            if passed:
                sample = samples[0]
                passed = (
                    sample["qualified"]
                    and sample["hidden_cpu_fallback"] == 0
                    and all(
                        math.isfinite(sample[k])
                        for k in ("canonical_residual", "objective", "terminal", "dynamics", "path")
                    )
                    and sample["canonical_residual"] <= 1e-8
                    and sample["terminal"] <= 1e-8
                    and sample["dynamics"] <= 1e-8
                    and sample["path"] <= 1e-8
                    and all(
                        r["accepted"] == 1
                        and r["forcing_satisfied"] == 1
                        and math.isfinite(r["natural"])
                        and r["natural"] <= 1e-8
                        for r in iterations
                    )
                )
                objective = sample["objective"] if objective is None else objective
                passed = passed and abs(sample["objective"] - objective) <= 1e-8
            record = {
                "variant": variant,
                "repeat": repeat,
                "warmup": repeat < args.warmups,
                "process_seconds": elapsed,
                "passed": bool(passed),
                "records": records,
                "stdout": run.stdout,
                "stderr": run.stderr,
                "returncode": run.returncode,
            }
            document["samples"].append(record)
            args.output.write_text(json.dumps(document, indent=2) + "\n")
            if not passed:
                raise RuntimeError(
                    f"{variant} failed qualification; raw evidence saved to {args.output}"
                )
            print(
                f"{variant} repeat={repeat} scvx={sample['scvx_seconds']:.6f}s "
                f"inner={sample['inner_iterations']} qualified",
                flush=True,
            )
    medians = {}
    for variant in ("baseline", "optimized"):
        rows = [r for r in document["samples"] if r["variant"] == variant and not r["warmup"]]
        medians[variant] = {
            "process_seconds": statistics.median(r["process_seconds"] for r in rows),
            **{
                k: statistics.median(
                    next(s for s in r["records"] if s["case"] == "g4_sample")[k] for r in rows
                )
                for k in ("scvx_seconds", "qoco_solve_seconds", "inner_iterations")
            },
        }
    document["medians"] = medians
    document["median_speedup"] = {
        k: medians["baseline"][k] / medians["optimized"][k]
        for k in ("process_seconds", "scvx_seconds", "qoco_solve_seconds")
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"medians": medians, "speedup": document["median_speedup"]}, indent=2))


if __name__ == "__main__":
    main()
