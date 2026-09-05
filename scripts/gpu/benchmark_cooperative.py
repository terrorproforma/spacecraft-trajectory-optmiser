"""Matched local CUDA hill climb; each sample must retain HCW qualification.

Run under WSL/Linux with an idle GPU and a freshly built integration executable.
The baseline library is an immutable copy built before the optimization.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--baseline-library", type=Path)
    parser.add_argument("--optimized-library", type=Path)
    parser.add_argument("--blocks", default="0,1,2,4,8,16,32,64,128,170")
    parser.add_argument("--intervals", type=int, default=2000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups nonnegative")
    optimized_library = (
        args.optimized_library or args.executable.parent.parent / "cuda" / "libspacepdhcg_cuda.so"
    ).resolve()
    lock = (Path.home() / ".spacepdhcg-gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configurations = [
        "auto" if value == "auto" else str(int(value)) for value in args.blocks.split(",")
    ]
    if args.baseline_library:
        configurations.insert(0, "baseline")
    samples: list[dict] = []
    provenance = {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        "optimized_library_sha256": hashlib.sha256(optimized_library.read_bytes()).hexdigest(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path("cpp/cuda/src/persistent_pdhcg.cu"),
                Path("cpp/cuda/src/cooperative_pdhg.cuh"),
                Path("cpp/cuda/src/device_scvx.cu"),
                Path("cpp/cuda/internal/numeric_fingerprint.cuh"),
                Path("cpp/cuda/internal/scvx_metrics.cuh"),
                Path("cpp/cuda/internal/hcw_replay.cuh"),
            )
        },
        "baseline_sha256": hashlib.sha256(args.baseline_library.read_bytes()).hexdigest()
        if args.baseline_library
        else None,
        "gpu": subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,pstate,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "intervals": args.intervals,
        "warmups": args.warmups,
        "repeats": args.repeats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.warmups + args.repeats):
        # Alternate ordering to reduce systematic thermal/order bias.
        order = configurations if repeat % 2 == 0 else list(reversed(configurations))
        for configuration in order:
            env = os.environ.copy()
            env.pop("SPACEPDHCG_TEST_GRID_BLOCKS", None)
            env.pop("SPACEPDHCG_TEST_SCALING_PARITY", None)
            env.pop("SPACEPDHCG_TEST_FINGERPRINT_PARITY", None)
            if configuration == "baseline":
                env["LD_LIBRARY_PATH"] = (
                    str(args.baseline_library.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
                )
            else:
                env["LD_LIBRARY_PATH"] = (
                    str(optimized_library.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
                )
                if configuration != "auto":
                    env["SPACEPDHCG_TEST_GRID_BLOCKS"] = configuration
            started = time.perf_counter()
            run = subprocess.run(
                [str(args.executable), "--h1-hcw", str(args.intervals), "1"],
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            if run.returncode:
                raise RuntimeError(
                    f"{configuration} failed ({run.returncode}): {run.stderr}\n{run.stdout}"
                )
            wall = time.perf_counter() - started
            records = [
                json.loads(line)
                for line in run.stdout.splitlines()
                if line.startswith('{"case":"h1_hcw"')
            ]
            if len(records) != 1:
                raise RuntimeError(f"missing/ambiguous H1 output: {run.stdout}")
            record = records[0]
            for metric in ("canonical_residual", "nonlinear_residual", "cpu_gpu_trajectory"):
                if record[metric] != 0:
                    raise RuntimeError(f"HCW qualification changed: {record}")
            if (
                record["inner_iterations"],
                record["outer_iterations"],
                record["accepted_steps"],
            ) != (1, 1, 0):
                raise RuntimeError(f"HCW work count changed: {record}")
            samples.append(
                {
                    "configuration": configuration,
                    "repeat": repeat,
                    "warmup": repeat < args.warmups,
                    "process_seconds": wall,
                    "record": record,
                }
            )
            args.output.write_text(
                json.dumps({"provenance": provenance, "samples": samples}, indent=2) + "\n"
            )
            print(
                f"{configuration:>8} repeat={repeat} cqp={record['cqp_total_seconds']:.6f}s "
                f"scvx={record['scvx_total_seconds']:.6f}s",
                flush=True,
            )
    medians = {}
    for configuration in configurations:
        records = [
            sample["record"]
            for sample in samples
            if sample["configuration"] == configuration and not sample["warmup"]
        ]
        medians[configuration] = {
            metric: statistics.median(record[metric] for record in records)
            for metric in (
                "update_seconds",
                "scaling_seconds",
                "solve_seconds",
                "cqp_total_seconds",
                "scvx_total_seconds",
            )
        }
    args.output.write_text(
        json.dumps({"provenance": provenance, "medians": medians, "samples": samples}, indent=2)
        + "\n"
    )
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    main()
