"""Run the nonlinear 3-DoF powered-descent SCvx reference benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from spacepdhcg.backends import PersistentClarabel
from spacepdhcg.cqp import residual_qualified
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
    initial_dispersion_scale: float = 0.0,
    final_polish: bool = False,
) -> dict[str, object]:
    model = PoweredDescent3DOFModel()
    initial = np.asarray([20.0, -10.0, 120.0, 0.0, 0.0, -7.0, 2_000.0])
    if not np.isfinite(initial_dispersion_scale) or initial_dispersion_scale < 0.0:
        raise ValueError("initial_dispersion_scale must be finite and non-negative")
    initial += initial_dispersion_scale * np.asarray([20.0, -10.0, 30.0, 1.0, -0.5, -1.0, 0.0])
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
    polish_payload: dict[str, object] | None = None
    if final_polish:
        problem = subproblem.canonical(
            result.states,
            result.controls,
            initial,
            target_position,
            target_velocity,
        )
        polish_backend = PersistentClarabel(
            problem,
            tolerance=min(tolerance, 1.0e-8),
            iteration_limit=2_000,
            verbose=False,
        )
        polish = polish_backend.solve()
        audit = polish_backend.independent_residuals(polish.primal)
        diagnostics = subproblem.diagnostics(polish.primal, problem.values)
        polish_payload = {
            "status": polish.status,
            "iterations": polish.iterations,
            "objective": polish.objective,
            "primal_residual": polish.primal_residual,
            "dual_residual": polish.dual_residual,
            "independent_primal_residual": audit.primal,
            "independent_dual_residual": audit.dual,
            "independent_natural_residual": audit.natural,
            "independent_cone_residual": audit.cone,
            "independent_complementarity": audit.complementarity,
            "residual_qualified": residual_qualified(
                polish,
                tolerance=max(min(tolerance, 1.0e-8), 2.0e-8),
            ),
            "maximum_convex_violation": diagnostics.convex_violation_inf,
            "solve_seconds": polish.solve_seconds,
            "setup_seconds": polish_backend.setup_seconds,
        }
    accepted_optimizer = next(
        (record.as_dict() for record in reversed(result.iterations) if record.accepted),
        None,
    )
    return {
        "benchmark": "nonlinear 3-DoF powered-descent SCvx CPU reference",
        "status": result.status,
        "converged": result.converged,
        "intervals": intervals,
        "step_seconds": step_seconds,
        "initial_dispersion_scale": initial_dispersion_scale,
        "final_polish": final_polish,
        "polish": polish_payload,
        "accepted_optimizer": accepted_optimizer,
        "outer_iterations": result.outer_iterations,
        "accepted_iterations": result.accepted_iterations,
        "final_merit": result.merit,
        "final_outer_residual": result.residual.maximum,
        "final_dynamics_residual": result.residual.dynamics,
        "final_path_residual": result.residual.path,
        "final_terminal_residual": result.residual.terminal,
        "path_violation": result.path_diagnostics.maximum_violation,
        "maximum_virtual_control": (
            result.iterations[-1].convex_diagnostics.virtual_control_inf
            if result.iterations
            else 0.0
        ),
        "final_mass": float(result.states[-1, 6]),
        "normalised_mean_thrust": float(
            np.mean(result.controls[:, 3]) / model.config.maximum_thrust
        ),
        "total_setup_seconds": result.total_setup_seconds,
        "total_solve_seconds": result.total_solve_seconds,
        "iterations": [record.as_dict() for record in result.iterations],
    }


def json_safe(value: Any) -> Any:
    """Convert diagnostics to standards-compliant JSON without losing semantics."""

    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=int, default=10)
    parser.add_argument("--step-seconds", type=float, default=2.0)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=2.0e-3)
    parser.add_argument("--initial-dispersion-scale", type=float, default=0.0)
    parser.add_argument("--final-polish", action="store_true")
    arguments = parser.parse_args()
    payload = run(
        intervals=arguments.intervals,
        step_seconds=arguments.step_seconds,
        max_iterations=arguments.max_iterations,
        tolerance=arguments.tolerance,
        initial_dispersion_scale=arguments.initial_dispersion_scale,
        final_polish=arguments.final_polish,
    )
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
