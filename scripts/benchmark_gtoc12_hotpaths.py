"""Matched-input CPU timings and numerical comparisons against a git revision.

Run with PYTHONPATH=src python scripts/benchmark_gtoc12_hotpaths.py --baseline-ref HEAD.
The baseline is loaded from git without changing the checkout. No external data is required.
"""

from __future__ import annotations

import argparse
import cProfile
import dataclasses
import io
import json
import platform
import pstats
import statistics
import subprocess
import sys
import time
import types
from functools import partial
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12 import lambert, low_thrust
from spacepdhcg.gtoc12.ephemeris import elements_to_state


def baseline_module(name, revision):
    path = f"src/spacepdhcg/gtoc12/{name}.py"
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)
    module = types.ModuleType(f"spacepdhcg.gtoc12._baseline_{name}")
    module.__package__ = "spacepdhcg.gtoc12"
    sys.modules[module.__name__] = module
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def timed(function, repeats):
    function()  # warm imports and numerical-library initialization
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return {"median_s": statistics.median(samples), "samples_s": samples}, result


def circular(radius, phase, inclination=0.0):
    r, v = elements_to_state(
        np.array([radius * C.AU_KM]),
        np.array([0.0]),
        np.array([inclination]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([phase]),
    )
    return r[0], v[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--scan-samples", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=Path("artifacts/performance/gtoc12.json"))
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.scan_samples < 1:
        parser.error("repeats and scan-samples must be positive")
    old_lambert = baseline_module("lambert", args.baseline_ref)
    old_lt = baseline_module("low_thrust", args.baseline_ref)
    report = {
        "baseline_ref": args.baseline_ref,
        "baseline_sha": subprocess.check_output(
            ["git", "rev-parse", args.baseline_ref], text=True
        ).strip(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "repeats": args.repeats,
        "scan_samples": args.scan_samples,
        "benchmarks": {},
    }

    def compare(name, old, new, check):
        before, expected = timed(old, args.repeats)
        after, actual = timed(new, args.repeats)
        quality = check(expected, actual)
        report["benchmarks"][name] = {
            "baseline": before,
            "current": after,
            "speedup": before["median_s"] / after["median_s"],
            "quality": quality,
        }
        print(name, json.dumps(report["benchmarks"][name]), flush=True)

    rng = np.random.default_rng(20260905)
    for count in (1, 64, 256, 1024):
        r1 = np.stack([circular(1.0, p)[0] for p in rng.uniform(0, 2 * np.pi, count)])
        r2 = np.stack([circular(2.7, p, 0.05)[0] for p in rng.uniform(0, 2 * np.pi, count)])
        tof = rng.uniform(150, 900, count) * C.DAY_S
        long_way = rng.random(count) < 0.5

        def check_lambert(expected, actual):
            for field in dataclasses.fields(expected):
                a, b = getattr(expected, field.name), getattr(actual, field.name)
                # Compare IEEE bits too (including signed zero and NaN payloads).
                if a.dtype == np.float64:
                    a, b = a.view(np.uint64), b.view(np.uint64)
                np.testing.assert_array_equal(a, b)
            return {"all_outputs_bitwise_equal": True, "feasible": int(actual.feasible.sum())}

        compare(
            f"lambert_{count}",
            partial(
                old_lambert.lambert_batch,
                r1,
                r2,
                tof,
                long_way=long_way,
                scan_samples=args.scan_samples,
            ),
            partial(
                lambert.lambert_batch,
                r1,
                r2,
                tof,
                long_way=long_way,
                scan_samples=args.scan_samples,
            ),
            check_lambert,
        )

    r0, v0 = circular(2.70, 0.0)
    motion = np.sqrt(C.MU_SUN_KM3_S2 / (2.73 * C.AU_KM) ** 3)
    rf, vf = circular(2.73, motion * 150 * C.DAY_S, 0.002)
    boundary = low_thrust.LegBoundary(65000.0, r0, v0, 65150.0, rf, vf, 2500.0)
    for hold in ("zoh", "lagrange"):
        times = np.linspace(0, 150 * C.DAY_S / low_thrust.TU_S, 76)
        states, controls = low_thrust._ballistic_reference(boundary, times, 2500.0)
        controls[:, :3] = rng.uniform(-0.2, 0.2, (76, 3))
        controls[:, 3] = np.linalg.norm(controls[:, :3], axis=1)
        old_disc = old_lt._Discretisation(old_lt._Model(2500), times, 8, hold)
        new_disc = low_thrust._Discretisation(low_thrust._Model(2500), times, 8, hold)

        def check_linearisation(expected, actual):
            errors = [float(np.max(np.abs(a - b))) for a, b in zip(expected, actual, strict=True)]
            for a, b in zip(expected, actual, strict=True):
                np.testing.assert_allclose(a, b, rtol=2e-14, atol=2e-15)
            return {"max_abs_error_A_B_c_state": errors}

        compare(
            f"linearise_{hold}",
            partial(old_disc.linearise, states, controls),
            partial(new_disc.linearise, states, controls),
            check_linearisation,
        )

    settings = low_thrust.ScvxSettings(max_iterations=30, time_limit_s=300)

    def check_solve(expected, actual):
        assert actual.status == expected.status == "converged"
        certificate = low_thrust.certify_leg(actual)
        assert certificate.within_tolerance
        np.testing.assert_allclose(
            actual.states_scaled, expected.states_scaled, rtol=1e-7, atol=1e-8
        )
        assert abs(actual.final_mass_kg - expected.final_mass_kg) < 1e-4
        return {
            "status": actual.status,
            "iterations_before": expected.iterations,
            "iterations_after": actual.iterations,
            "final_mass_difference_kg": actual.final_mass_kg - expected.final_mass_kg,
            "certificate": dataclasses.asdict(certificate),
        }

    # Baseline low_thrust imports Lambert at call time: bind the baseline for the whole solve.
    def old_solve():
        current = lambert.lambert_batch
        lambert.lambert_batch = old_lambert.lambert_batch
        try:
            return old_lt.solve_leg(boundary, settings)
        finally:
            lambert.lambert_batch = current

    compare(
        "solve_leg_150_days",
        old_solve,
        lambda: low_thrust.solve_leg(boundary, settings),
        check_solve,
    )
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(low_thrust.solve_leg, boundary, settings)
        output = io.StringIO()
        pstats.Stats(profiler, stream=output).sort_stats("cumtime").print_stats(30)
        report["solve_profile"] = output.getvalue()
        print(output.getvalue())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
