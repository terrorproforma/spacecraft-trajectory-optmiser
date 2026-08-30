"""Repeated fixed-pattern CW rendezvous benchmark."""

from __future__ import annotations

import argparse
import json
from statistics import fmean, median
from time import perf_counter
from typing import Any

import numpy as np

from spacepdhcg.backends import PersistentOSQP
from spacepdhcg.models import CWRendezvousConfig, CWRendezvousProblem


def run_benchmark(
    *,
    repeats: int = 20,
    intervals: int = 40,
    seed: int = 7,
    tolerance: float = 1.0e-7,
) -> dict[str, Any]:
    """Run repeated numerical updates against one allocated OSQP workspace."""

    if repeats < 2:
        raise ValueError("repeats must be at least two")

    rng = np.random.default_rng(seed)
    config = CWRendezvousConfig(
        intervals=intervals,
        max_component_acceleration=5.0e-2,
    )
    problem = CWRendezvousProblem(config)

    initial_states: list[np.ndarray] = []
    target_states: list[np.ndarray] = []
    for _ in range(repeats):
        initial = np.concatenate(
            (rng.uniform(-100.0, 100.0, 3), rng.uniform(-0.05, 0.05, 3))
        )
        target = np.concatenate((rng.uniform(-5.0, 5.0, 3), np.zeros(3)))
        initial_states.append(initial)
        target_states.append(target)

    backend = PersistentOSQP(problem.canonical(initial_states[0], target_states[0]))
    update_times: list[float] = []
    solve_times: list[float] = []
    iterations: list[int] = []
    terminal_errors: list[float] = []
    dynamics_defects: list[float] = []
    previous = None

    for initial, target in zip(initial_states, target_states, strict=True):
        values = problem.values(initial, target)
        update_start = perf_counter()
        backend.update(values)
        update_times.append(perf_counter() - update_start)

        if previous is not None:
            backend.warm_start(previous.primal, previous.dual)

        solution = backend.solve(tolerance=tolerance)
        if not solution.solved:
            raise RuntimeError(f"OSQP failed with status {solution.status!r}")
        diagnostics = problem.diagnostics(solution.primal, initial, target)
        if not diagnostics.feasible(max(1.0e-5, 10.0 * tolerance)):
            raise RuntimeError(f"trajectory failed independent checks: {diagnostics}")

        solve_times.append(solution.solve_seconds)
        iterations.append(solution.iterations)
        terminal_errors.append(diagnostics.terminal_error_inf)
        dynamics_defects.append(diagnostics.dynamics_defect_inf)
        previous = solution

    return {
        "backend": "OSQP persistent CPU reference",
        "repeats": repeats,
        "intervals": intervals,
        "variables": problem.layout.n_variables,
        "constraints": problem.layout.n_constraints,
        "setup_seconds": backend.setup_seconds,
        "mean_update_seconds": fmean(update_times),
        "median_update_seconds": median(update_times),
        "mean_solve_seconds": fmean(solve_times),
        "median_solve_seconds": median(solve_times),
        "mean_iterations": fmean(iterations),
        "maximum_terminal_error": max(terminal_errors),
        "maximum_dynamics_defect": max(dynamics_defects),
        "workspace_updates": backend.update_count,
        "explicit_warm_starts": backend.warm_start_count,
        "seed": seed,
        "tolerance": tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--intervals", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
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
