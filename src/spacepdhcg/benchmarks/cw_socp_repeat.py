"""Repeated fixed-pattern CW rendezvous SOCP benchmark."""

from __future__ import annotations

import argparse
import json
from statistics import fmean, median
from time import perf_counter
from typing import Any

import numpy as np

from spacepdhcg.backends import PersistentClarabel
from spacepdhcg.models import CWRendezvousConfig, CWRendezvousProblem, ThrustConstraint


def run_benchmark(
    *,
    repeats: int = 20,
    intervals: int = 40,
    seed: int = 11,
    tolerance: float = 1.0e-8,
    update_magnitude: float | None = None,
) -> dict[str, Any]:
    """Run repeated SOCP numerical updates against one Clarabel workspace."""

    if repeats < 2:
        raise ValueError("repeats must be at least two")

    rng = np.random.default_rng(seed)
    problem = CWRendezvousProblem(
        CWRendezvousConfig(
            intervals=intervals,
            max_component_acceleration=5.0e-2,
            thrust_constraint=ThrustConstraint.SECOND_ORDER_CONE,
        )
    )

    initial_states: list[np.ndarray] = []
    target_states: list[np.ndarray] = []
    if update_magnitude is not None and (
        not np.isfinite(update_magnitude) or update_magnitude < 0.0
    ):
        raise ValueError("update_magnitude must be finite and non-negative")
    base_initial = (
        None
        if update_magnitude is None
        else np.concatenate((rng.uniform(-100.0, 100.0, 3), rng.uniform(-0.05, 0.05, 3)))
    )
    base_target = (
        None
        if update_magnitude is None
        else np.concatenate((rng.uniform(-5.0, 5.0, 3), np.zeros(3)))
    )
    for _ in range(repeats):
        if update_magnitude is None:
            initial = np.concatenate((rng.uniform(-100.0, 100.0, 3), rng.uniform(-0.05, 0.05, 3)))
            target = np.concatenate((rng.uniform(-5.0, 5.0, 3), np.zeros(3)))
        else:
            assert base_initial is not None and base_target is not None
            initial_delta = rng.normal(size=6)
            target_delta = rng.normal(size=6)
            initial_delta /= max(float(np.linalg.norm(initial_delta)), 1.0)
            target_delta /= max(float(np.linalg.norm(target_delta)), 1.0)
            initial = base_initial + update_magnitude * initial_delta
            target = base_target + update_magnitude * target_delta
        initial_states.append(initial)
        target_states.append(target)

    backend = PersistentClarabel(
        problem.canonical(initial_states[0], target_states[0]),
        tolerance=tolerance,
    )
    update_times: list[float] = []
    solve_times: list[float] = []
    iterations_taken: list[int] = []
    terminal_errors: list[float] = []
    dynamics_defects: list[float] = []
    control_violations: list[float] = []

    for initial, target in zip(initial_states, target_states, strict=True):
        start = perf_counter()
        backend.update(problem.values(initial, target))
        update_times.append(perf_counter() - start)
        solution = backend.solve()
        if not solution.solved:
            raise RuntimeError(f"Clarabel failed with status {solution.status!r}")
        diagnostics = problem.diagnostics(solution.primal, initial, target)
        if not diagnostics.feasible(max(1.0e-5, 10.0 * tolerance)):
            raise RuntimeError(f"trajectory failed independent checks: {diagnostics}")

        solve_times.append(solution.solve_seconds)
        iterations_taken.append(solution.iterations)
        terminal_errors.append(diagnostics.terminal_error_inf)
        dynamics_defects.append(diagnostics.dynamics_defect_inf)
        control_violations.append(diagnostics.control_violation_inf)

    return {
        "backend": "Clarabel persistent CPU conic reference",
        "repeats": repeats,
        "intervals": intervals,
        "variables": problem.layout.n_variables,
        "scalar_constraints": problem.layout.n_constraints,
        "affine_cone_rows": problem.layout.n_affine_constraints,
        "soc_blocks": len(problem.structure.affine_cones),
        "setup_seconds": backend.setup_seconds,
        "mean_update_seconds": fmean(update_times),
        "median_update_seconds": median(update_times),
        "mean_solve_seconds": fmean(solve_times),
        "median_solve_seconds": median(solve_times),
        "mean_iterations": fmean(iterations_taken),
        "maximum_terminal_error": max(terminal_errors),
        "maximum_dynamics_defect": max(dynamics_defects),
        "maximum_control_violation": max(control_violations),
        "workspace_updates": backend.update_count,
        "seed": seed,
        "tolerance": tolerance,
        "update_magnitude": update_magnitude,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--intervals", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    arguments = parser.parse_args()
    result = run_benchmark(
        repeats=arguments.repeats,
        intervals=arguments.intervals,
        seed=arguments.seed,
        tolerance=arguments.tolerance,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
