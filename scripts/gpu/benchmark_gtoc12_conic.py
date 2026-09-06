"""Compare complete GTOC12 coefficient paths and full legs at fixed physics accuracy.

Run under the shared GPU lock. Interleaved samples include bridge transfers and
CPU solver costs; no claim of a complete GPU-native refiner or kernel-only time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import statistics
import time
from contextlib import ExitStack, closing
from dataclasses import asdict
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12.gpu_conic import GpuConvexProblem
from spacepdhcg.gtoc12.gpu_discretisation import GpuDiscretisation
from spacepdhcg.gtoc12.low_thrust import (
    ScvxSettings,
    _ConvexProblem,
    _Discretisation,
    certify_leg,
    solve_leg,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("positive repeats and a new output path required")
    os.environ["SPACEPDHCG_GTOC12_CUDA_LIBRARY"] = str(args.library.resolve(strict=True))
    fixture = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "tests/test_gtoc12_gpu_discretisation.py")
    )
    report = dict(
        scope="One process, one BLAS thread, 2 warmups + measured repeats. "
        "Three backends rotate order. "
        "Coefficient path includes linearisation, assembly and host outputs/transfers; excludes "
        "workspace construction. Full-leg time includes construction/destruction, CPU solver and "
        "outer loop, excludes independent certification. Clarabel/seed/outer decisions remain CPU.",
        runtime_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        sources={
            p: hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in (
                "src/spacepdhcg/gtoc12/gpu_conic.py",
                "src/spacepdhcg/gtoc12/low_thrust.py",
                "src/spacepdhcg/gtoc12/gpu_discretisation.py",
                "tests/test_gtoc12_gpu_discretisation.py",
                "scripts/gpu/benchmark_gtoc12_conic.py",
            )
        },
        phases=[],
        legs=[],
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    names = ["numpy", "cuda_dynamics", "cuda_assembly"]
    for intervals in [3, 75, 257, 2001]:
        for hold in ["zoh", "lagrange"]:
            model, times, states, controls = fixture["fixture"](intervals, hold)
            boundary = dict(
                r0=states[0, :3], v0=states[0, 3:6], rf=states[-1, :3], vf=states[-1, 3:6]
            )
            weights = np.r_[np.diff(times), 0.0] * model.lam
            parameters = (0.1, 0.3, 13.0, 0.3, 0.05, 0.02, 0.001)
            reference = _ConvexProblem(len(times), False, False)
            with ExitStack() as resources:
                cpu = _Discretisation(model, times, 8, hold)
                dynamics = resources.enter_context(
                    closing(GpuDiscretisation(model, times, 8, hold))
                )
                started = time.perf_counter()
                conic = resources.enter_context(
                    closing(GpuConvexProblem(model, times, hold, False, False, boundary, weights))
                )
                setup = time.perf_counter() - started

                def run(
                    backend, conic=conic, states=states, controls=controls, parameters=parameters,
                    cpu=cpu, dynamics=dynamics, reference=reference, boundary=boundary,
                    weights=weights, hold=hold,
                ):
                    if backend == "cuda_assembly":
                        return conic.build_linearised(states, controls, 8, *parameters)
                    disc = cpu if backend == "numpy" else dynamics
                    phi, psi, c, _ = disc.linearise(states, controls)
                    return reference.build(
                        phi,
                        psi,
                        c,
                        disc.stencils,
                        states,
                        controls,
                        boundary,
                        *parameters[:6],
                        weights,
                        parameters[6],
                        zero_last_control=hold == "zoh",
                    )

                expected, actual = run("numpy"), run("cuda_assembly")
                for i in [0, 4]:
                    np.testing.assert_allclose(
                        (actual[i] - expected[i]).data, 0.0, atol=2e-12, rtol=0.0
                    )
                for i in [1, 2]:
                    np.testing.assert_allclose(actual[i], expected[i], atol=2e-12, rtol=2e-12)
                assert [str(c) for c in expected[3]] == [str(c) for c in actual[3]]
                rows = []
                for repeat in range(args.repeats + 2):
                    order = names[repeat % 3 :] + names[: repeat % 3]
                    for backend in order:
                        started = time.perf_counter()
                        run(backend)
                        rows.append(
                            dict(
                                backend=backend,
                                repeat=repeat,
                                warmup=repeat < 2,
                                seconds=time.perf_counter() - started,
                            )
                        )
                medians = {
                    name: statistics.median(
                        r["seconds"] for r in rows if r["backend"] == name and not r["warmup"]
                    )
                    for name in names
                }
                report["phases"].append(
                    dict(
                        intervals=intervals,
                        hold=hold,
                        setup_seconds=setup,
                        samples=rows,
                        medians=medians,
                    )
                )
                save()
                print(intervals, hold, medians, flush=True)
    for repeat in range(args.repeats + 2):
        for backend in names[repeat % 3 :] + names[: repeat % 3]:
            settings = ScvxSettings(
                max_iterations=30,
                time_limit_s=300.0,
                discretisation_backend="numpy" if backend == "numpy" else "cuda",
                assembly_backend="cuda" if backend == "cuda_assembly" else "numpy",
            )
            started = time.perf_counter()
            solution = solve_leg(fixture["synthetic_boundary"](), settings)
            seconds = time.perf_counter() - started
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
                    seconds=seconds,
                    status=solution.status,
                    iterations=solution.iterations,
                    accepted=solution.accepted_iterations,
                    mass=solution.final_mass_kg,
                    certificate=asdict(certificate),
                    passed=passed,
                )
            )
            save()
            assert passed, report["legs"][-1]
    report["leg_medians"] = {
        name: statistics.median(
            r["seconds"] for r in report["legs"] if r["backend"] == name and not r["warmup"]
        )
        for name in names
    }
    save()
    print("full_leg", report["leg_medians"], flush=True)


if __name__ == "__main__":
    main()
