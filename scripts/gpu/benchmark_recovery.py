"""Matched local native recovery benchmarks with independent trajectory qualification."""

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
    parser.add_argument("--baseline-library", type=Path, required=True)
    parser.add_argument("--optimized-library", type=Path)
    parser.add_argument("--family", choices=("pd3", "pd6", "low-thrust"), default="pd3")
    parser.add_argument("--intervals", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.intervals < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error("intervals/repeats must be positive and warmups nonnegative")
    optimized = (
        args.optimized_library or args.executable.parent.parent / "cuda/libspacepdhcg_cuda.so"
    )
    lock = (Path.home() / ".spacepdhcg-gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_paths = [
        Path("cpp/cuda/include/spacepdhcg/cuda/persistent_pdhcg_c_api.h"),
        Path("cpp/cuda/src/persistent_pdhcg.cu"),
        Path("cpp/cuda/src/cooperative_pdhg.cuh"),
        Path("cpp/cuda/tests/device_scvx_integration_test.cu"),
    ]
    provenance = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [args.executable, args.baseline_library, optimized, *source_paths]
        },
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,temperature.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "family": args.family,
        "intervals": args.intervals,
        "warmups": args.warmups,
        "repeats": args.repeats,
    }
    samples: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_objective = None
    baseline_inner_qualified = False
    for repeat in range(args.warmups + args.repeats):
        variants = [("baseline", args.baseline_library), ("optimized", optimized)]
        if repeat % 2:
            variants.reverse()
        for variant, library in variants:
            env = os.environ.copy()
            for key in list(env):
                if key.startswith("SPACEPDHCG_TEST_"):
                    del env[key]
            env["LD_LIBRARY_PATH"] = (
                str(library.resolve().parent) + ":" + env.get("LD_LIBRARY_PATH", "")
            )
            started = time.perf_counter()
            result = subprocess.run(
                [str(args.executable), "--production-outer", args.family, str(args.intervals)],
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            elapsed = time.perf_counter() - started
            if result.returncode:
                raise RuntimeError(
                    f"{variant} failed: {result.returncode}\n{result.stdout}\n{result.stderr}"
                )
            records = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.startswith('{"case":"production_outer"')
            ]
            if len(records) != 1:
                raise RuntimeError(f"Expected one production result: {result.stdout}")
            record = records[0]
            profiles = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.startswith('{"case":"recovery_profile"')
            ]
            for metric in ("canonical", "dynamics", "path", "terminal", "virtual"):
                if not math.isfinite(record[metric]) or record[metric] > 1e-6:
                    raise RuntimeError(f"Unqualified {metric}: {record}")
            for metric in ("cpu_gpu_trajectory", "cpu_gpu_replay"):
                if not math.isfinite(record[metric]) or record[metric] > 1e-9:
                    raise RuntimeError(f"Trajectory parity changed: {record}")
            if not math.isfinite(record["objective"]):
                raise RuntimeError(f"Nonfinite objective: {record}")
            inner_qualified = math.isfinite(record["achieved"]) and (
                record["achieved"] <= record["requested"]
            )
            if variant == "baseline":
                baseline_inner_qualified |= inner_qualified
            elif baseline_inner_qualified and not inner_qualified:
                raise RuntimeError(f"Inner solver lost baseline qualification: {record}")
            if expected_objective is None:
                expected_objective = record["objective"]
            if abs(record["objective"] - expected_objective) > 1e-8 * max(
                1.0, abs(expected_objective)
            ):
                raise RuntimeError(f"Objective changed: {record}")
            samples.append(
                {
                    "variant": variant,
                    "repeat": repeat,
                    "warmup": repeat < args.warmups,
                    "process_seconds": elapsed,
                    "record": record,
                    "recovery_profile": profiles[-1] if profiles else None,
                }
            )
            args.output.write_text(
                json.dumps({"provenance": provenance, "samples": samples}, indent=2) + "\n"
            )
            print(
                f"{variant} repeat={repeat} scvx={record['t_scvx']:.6f}s "
                f"recovery={record['recovery_seconds']:.6f}s "
                f"steps={record['inner_iterations']}+{record['recovery_iterations']}",
                flush=True,
            )
    medians = {}
    if baseline_inner_qualified:
        for sample in samples:
            record = sample["record"]
            if sample["variant"] == "optimized" and (
                not math.isfinite(record["achieved"]) or record["achieved"] > record["requested"]
            ):
                raise RuntimeError(f"Inner solver lost baseline qualification: {record}")
    for variant in ("baseline", "optimized"):
        selected = [s for s in samples if s["variant"] == variant and not s["warmup"]]
        medians[variant] = {
            key: statistics.median(s["record"][key] for s in selected)
            for key in (
                "t_scvx",
                "t_cqp",
                "recovery_seconds",
                "inner_iterations",
                "recovery_iterations",
            )
        }
        medians[variant]["process_seconds"] = statistics.median(
            s["process_seconds"] for s in selected
        )
    args.output.write_text(
        json.dumps({"provenance": provenance, "medians": medians, "samples": samples}, indent=2)
        + "\n"
    )
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    main()
