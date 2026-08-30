"""Run the nonlinear 3-DoF powered-descent SCvx reference benchmark."""

from __future__ import annotations

import argparse
import json

import numpy as np

from spacepdhcg.models import PoweredDescent3DOFModel
from spacepdhcg.scvx import (
    ForcingRuleConfig,
    PoweredDescentOuterConfig,
    PoweredDescentSCvxSolver,
    TrustRegionConfig,
)
from spacepdhcg.transcription import (
    PoweredDescent3DOFSubproblem,
    PoweredDescentSCvxConfig,
)


def run(
    *,
    intervals: int,
    step_seconds: float,
    max_iterations: int,
    tolerance: float,
) -> dict[str, object]:
    model = PoweredDescent3DOFModel()
    initial = np.asarray([20.0, -10.0, 120.0, 0.0, 0.0, -7.0, 2_000.0])
    target_position = np.zeros(3)
    target_velocity = np.zeros(3)
    subproblem = PoweredDescent3DOFSubproblem(
        model,
        PoweredDescentSCvxConfig(
            intervals=intervals,
            step_seconds=step_seconds,
            trust_radius=1.0,
        ),
    )
    solver = PoweredDescentSCvxSolver(
        subproblem,
        outer_config=PoweredDescentOuterConfig(
            max_iterations=max_iterations,
            convergence_tolerance=tolerance,
        ),
        forcing_config=ForcingRuleConfig(),
        trust_config=TrustRegionConfig(initial_radius=1.0),
    )
    result = solver.solve(initial, target_position, target_velocity)
    return {
        "benchmark": "nonlinear 3-DoF powered-descent SCvx CPU reference",
        "status": result.status,
        "converged": result.converged,
        "intervals": intervals,
        "step_seconds": step_seconds,
        "outer_iterations": result.outer_iterations,
        "accepted_iterations": result.accepted_iterations,
        "final_merit": result.merit,
        "final_outer_residual": result.residual.maximum,
        "final_dynamics_residual": result.residual.dynamics,
        "final_path_residual": result.residual.path,
        "final_terminal_residual": result.residual.terminal,
        "path_violation": result.path_diagnostics.maximum_violation,
        "final_mass": float(result.states[-1, 6]),
        "normalised_mean_thrust": float(
            np.mean(result.controls[:, 3]) / model.config.maximum_thrust
        ),
        "total_setup_seconds": result.total_setup_seconds,
        "total_solve_seconds": result.total_solve_seconds,
        "iterations": [record.as_dict() for record in result.iterations],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=int, default=10)
    parser.add_argument("--step-seconds", type=float, default=2.0)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=2.0e-3)
    arguments = parser.parse_args()
    payload = run(
        intervals=arguments.intervals,
        step_seconds=arguments.step_seconds,
        max_iterations=arguments.max_iterations,
        tolerance=arguments.tolerance,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
