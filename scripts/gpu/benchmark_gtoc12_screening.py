"""Matched Earth-to-catalogue screening, including both branches and cost selection."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.ephemeris import asteroid_state, earth_state, propagate_kepler
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.screening import lambert_hops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.count <= 60000 or args.repeats < 1:
        parser.error("new output, count 1..60000, and positive repeats required")
    catalogue = load_catalogue()
    ids = np.linspace(1, 60000, args.count, dtype=int)
    epochs = C.MISSION_START_MJD + np.linspace(0, 730, args.count)
    tofs = 300 + (np.arange(args.count) * 71 % 601)
    r1, v1 = earth_state(epochs)
    r2, v2 = asteroid_state(catalogue, ids, epochs + tofs)
    report = dict(
        count=args.count,
        branch_solves_per_batch=2 * args.count,
        catalogue_sha256=catalogue.source_sha256,
        input_sha256=hashlib.sha256(
            b"".join(a.tobytes() for a in [r1, v1, r2, v2, epochs, tofs])
        ).hexdigest(),
        runtime_sha256=hashlib.sha256(
            Path(os.environ["SPACEPDHCG_GTOC12_CUDA_LIBRARY"]).read_bytes()
        ).hexdigest(),
        runs={},
    )
    outputs = {}
    for backend in ["numpy", "cuda"]:
        with using_lambert_backend(backend) as gpu:
            times = []
            for _ in range(args.repeats + 1):
                start = time.perf_counter()
                output = lambert_hops(r1, v1, r2, v2, epochs, tofs, departure_allowance_km_s=6)
                times.append(time.perf_counter() - start)
            outputs[backend] = output
            report["runs"][backend] = dict(
                cold_seconds=times[0],
                warm_seconds=times[1:],
                median_seconds=statistics.median(times[1:]),
                transfers_per_second=args.count / statistics.median(times[1:]),
                feasible=int(output.feasible.sum()),
                submitted_branches=gpu.evaluations if gpu else None,
            )
    cpu, gpu = outputs["numpy"], outputs["cuda"]
    np.testing.assert_array_equal(cpu.feasible, gpu.feasible)
    for name in ["departure_delta_v", "arrival_delta_v", "departure_velocity", "arrival_velocity"]:
        np.testing.assert_allclose(getattr(cpu, name), getattr(gpu, name), rtol=1e-9, atol=1e-8)
    cpu_cost = cpu.departure_delta_v + cpu.arrival_delta_v
    gpu_cost = gpu.departure_delta_v + gpu.arrival_delta_v
    np.testing.assert_array_equal(np.argsort(cpu_cost)[:100], np.argsort(gpu_cost)[:100])
    valid = gpu.feasible
    r, v = propagate_kepler(r1[valid], gpu.departure_velocity[valid], tofs[valid] * C.DAY_S)
    position = float(np.max(np.linalg.norm(r - r2[valid], axis=1)))
    velocity = float(np.max(np.linalg.norm(v - gpu.arrival_velocity[valid], axis=1)))
    cr, _cv = propagate_kepler(r1[valid], cpu.departure_velocity[valid], tofs[valid] * C.DAY_S)
    worst = int(np.flatnonzero(valid)[np.argmax(np.linalg.norm(r - r2[valid], axis=1))])
    report.update(
        complete=bool(position < 0.01 and velocity < 1e-8),
        top100_identical=True,
        kepler_position_error_km=position,
        kepler_velocity_error_km_s=velocity,
        cpu_kepler_position_error_km=float(np.max(np.linalg.norm(cr - r2[valid], axis=1))),
        worst_case=dict(
            index=worst,
            asteroid_id=int(ids[worst]),
            epoch=float(epochs[worst]),
            tof_days=float(tofs[worst]),
            r1=r1[worst].tolist(),
            r2=r2[worst].tolist(),
            cpu_v1=cpu.departure_velocity[worst].tolist(),
            gpu_v1=gpu.departure_velocity[worst].tolist(),
        ),
        maximum_cost_difference_km_s=float(np.max(abs(cpu_cost[valid] - gpu_cost[valid]))),
        speedup=report["runs"]["numpy"]["median_seconds"]
        / report["runs"]["cuda"]["median_seconds"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    assert report["complete"], (position, velocity)


if __name__ == "__main__":
    main()
