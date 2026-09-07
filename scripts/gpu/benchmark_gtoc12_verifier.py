"""Compare a batched GPU propagation of a real solution with CPU certificates.

Run with PYTHONPATH=src and SPACEPDHCG_GTOC12_CUDA_LIBRARY set. Times include
host uploads/final downloads, exclude file parsing and report comparisons.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12.gpu_verifier import GpuVerifier, pack_solution
from spacepdhcg.gtoc12.solution import Solution
from spacepdhcg.gtoc12.verifier import propagate_burn, propagate_coast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu-legs", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or args.cpu_legs < 1:
        parser.error("output must be new; repeats and cpu-legs must be positive")
    start = time.perf_counter()
    legs, arcs, samples, metadata = pack_solution(Solution.read(args.solution))
    packing_s = time.perf_counter() - start
    selected = np.unique(np.linspace(0, len(legs) - 1, min(args.cpu_legs, len(legs)), dtype=int))
    with GpuVerifier(len(legs), len(arcs), len(samples)) as gpu:
        start = time.perf_counter()
        out = gpu.propagate(legs, arcs, samples)
        cold_s = time.perf_counter() - start
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            out = gpu.propagate(legs, arcs, samples)
            times.append(time.perf_counter() - start)
        subset_times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            gpu.propagate(legs[selected], arcs, samples)
            subset_times.append(time.perf_counter() - start)
    report = dict(
        schema_version=1,
        solution_sha256=hashlib.sha256(args.solution.read_bytes()).hexdigest(),
        runtime_sha256=hashlib.sha256(
            Path(os.environ["SPACEPDHCG_GTOC12_CUDA_LIBRARY"]).read_bytes()
        ).hexdigest(),
        legs=len(legs),
        arcs=len(arcs),
        samples=len(samples),
        packing_seconds=packing_s,
        gpu_cold_seconds=cold_s,
        gpu_batch_seconds=times,
        gpu_subset_seconds=subset_times,
        completed=int(np.count_nonzero(out["status"] == 0)),
        status=out["status"].tolist(),
        accepted_steps=out["accepted_steps"].tolist(),
        rejected_steps=out["rejected_steps"].tolist(),
        comparisons=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Persist GPU outcomes before running the slower independent comparisons.
    args.output.write_text(json.dumps(report, indent=2))
    for index in selected:
        sid, previous, target, burns = metadata[index]
        initial = previous.after
        r, v, m, epoch = (
            initial.position.copy(),
            initial.velocity.copy(),
            initial.mass,
            initial.epoch,
        )
        radius = float(np.linalg.norm(r))
        start = time.perf_counter()
        for arc in sorted(burns, key=lambda arc: arc.start):
            if arc.start > epoch:
                r, v, minimum = propagate_coast(epoch, r, v, m, arc.start)
                radius = min(radius, minimum)
                epoch = arc.start
            r, v, m, minimum = propagate_burn(epoch, r, v, m, arc)
            radius = min(radius, minimum)
            epoch = arc.end
        if target.epoch > epoch:
            r, v, minimum = propagate_coast(epoch, r, v, m, target.epoch)
            radius = min(radius, minimum)
        cpu_s = time.perf_counter() - start
        row = out[index]
        comparison = dict(
            index=int(index),
            ship=sid,
            cpu_seconds=cpu_s,
            status=int(row["status"]),
            position_difference_km=float(np.linalg.norm(row["final_state"][:3] - r)),
            velocity_difference_km_s=float(np.linalg.norm(row["final_state"][3:6] - v)),
            mass_difference_kg=float(abs(row["final_state"][6] - m)),
            radius_difference_km=float(abs(row["minimum_radius_km"] - radius)),
            gpu_target_position_km=float(
                np.linalg.norm(row["final_state"][:3] - target.before.position)
            ),
            gpu_target_velocity_km_s=float(
                np.linalg.norm(row["final_state"][3:6] - target.before.velocity)
            ),
            gpu_target_mass_kg=float(abs(row["final_state"][6] - target.before.mass)),
        )
        report["comparisons"].append(comparison)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(comparison), flush=True)
    report["cpu_subset_seconds"] = sum(row["cpu_seconds"] for row in report["comparisons"])
    report["matched_subset_speedup"] = report["cpu_subset_seconds"] / float(np.median(subset_times))
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "legs",
                    "completed",
                    "gpu_batch_seconds",
                    "cpu_subset_seconds",
                    "matched_subset_speedup",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
