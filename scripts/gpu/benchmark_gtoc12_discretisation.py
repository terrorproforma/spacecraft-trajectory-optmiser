"""Matched CPU/CUDA GTOC12 interval and full-leg timings at fixed accuracy.

Run serialized with other GPU work. This measures the transitional host bridge,
including transfers, and a complete SCvx leg including native workspace lifetime.
The remaining CPU assembly/Clarabel/seed/certificate are not GPU-native.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import statistics
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12.gpu_discretisation import GpuDiscretisation
from spacepdhcg.gtoc12.low_thrust import ScvxSettings, _Discretisation, certify_leg, solve_leg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("repeats must be positive and output must be a new file")
    os.environ["SPACEPDHCG_GTOC12_CUDA_LIBRARY"] = str(args.library.resolve(strict=True))
    # Use the exact regression inputs and independently checked synthetic leg.
    fixtures = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "tests/test_gtoc12_gpu_discretisation.py")
    )
    report = {
        "scope": "Alternating calls in one process, 2 warmups + measured repeats. "
        "Interval timings include host outputs/transfers, exclude workspace construction. "
        "Full-leg wall time includes native workspace lifetime, excludes independent certificate. "
        "CPU sparse assembly/Clarabel/seed and independent verifier remain.",
        "repeats": args.repeats,
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "source_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in [
                "src/spacepdhcg/gtoc12/low_thrust.py",
                "src/spacepdhcg/gtoc12/gpu_discretisation.py",
                "tests/test_gtoc12_gpu_discretisation.py",
            ]
        },
        "phases": [],
        "legs": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for n in [3, 17, 75, 257, 2001]:
        for hold in ["zoh", "lagrange"]:
            model, times, states, controls = fixtures["fixture"](n, hold)
            cpu = _Discretisation(model, times, 8, hold)
            started = time.perf_counter()
            with closing(GpuDiscretisation(model, times, 8, hold)) as gpu:
                setup = time.perf_counter() - started
                for phase in ["linearise", "propagate"]:
                    expected = getattr(cpu, phase)(states, controls)
                    actual = getattr(gpu, phase)(states, controls)
                    if phase == "propagate":
                        expected, actual = (expected,), (actual,)
                    for a, b in zip(actual, expected, strict=True):
                        np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)
                    rows = []
                    for repeat in range(args.repeats + 2):
                        for backend in ["numpy", "cuda"] if repeat % 2 == 0 else ["cuda", "numpy"]:
                            function = getattr(cpu if backend == "numpy" else gpu, phase)
                            started = time.perf_counter()
                            function(states, controls)
                            rows.append(
                                dict(
                                    backend=backend,
                                    warmup=repeat < 2,
                                    repeat=repeat,
                                    seconds=time.perf_counter() - started,
                                )
                            )
                    medians = {
                        backend: statistics.median(
                            row["seconds"]
                            for row in rows
                            if row["backend"] == backend and not row["warmup"]
                        )
                        for backend in ["numpy", "cuda"]
                    }
                    report["phases"].append(
                        dict(
                            intervals=n,
                            hold=hold,
                            phase=phase,
                            gpu_setup_seconds=setup,
                            samples=rows,
                            medians=medians,
                            speedup=medians["numpy"] / medians["cuda"],
                        )
                    )
                    save()
                    print(n, hold, phase, medians, flush=True)
    boundary = fixtures["synthetic_boundary"]()
    for repeat in range(args.repeats + 2):
        for backend in ["numpy", "cuda"] if repeat % 2 == 0 else ["cuda", "numpy"]:
            settings = ScvxSettings(
                max_iterations=30, time_limit_s=300.0, discretisation_backend=backend
            )
            started = time.perf_counter()
            solution = solve_leg(boundary, settings)
            elapsed = time.perf_counter() - started
            certificate = certify_leg(solution)
            passed = (
                solution.status == "converged"
                and certificate.within_tolerance
                and abs(solution.final_mass_kg - 2445.3111007852112) <= 1e-5
            )
            report["legs"].append(
                dict(
                    backend=backend,
                    repeat=repeat,
                    warmup=repeat < 2,
                    seconds=elapsed,
                    passed=passed,
                    status=solution.status,
                    iterations=solution.iterations,
                    accepted=solution.accepted_iterations,
                    final_mass_kg=solution.final_mass_kg,
                    certificate=asdict(certificate),
                )
            )
            save()
            assert passed, report["legs"][-1]
    report["leg_medians"] = {
        backend: statistics.median(
            row["seconds"]
            for row in report["legs"]
            if row["backend"] == backend and not row["warmup"]
        )
        for backend in ["numpy", "cuda"]
    }
    save()
    print("full leg", report["leg_medians"], flush=True)


if __name__ == "__main__":
    main()
