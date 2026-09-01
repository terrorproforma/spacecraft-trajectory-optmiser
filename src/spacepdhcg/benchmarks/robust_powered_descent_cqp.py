"""Solve one monolithic robust powered-descent CQP with shared controls."""

from __future__ import annotations

import argparse
import json

import numpy as np

from spacepdhcg.backends import PersistentClarabel
from spacepdhcg.cqp import residual_qualified
from spacepdhcg.distributed import ScenarioCQPBundle, ScenarioTree
from spacepdhcg.models import (
    PoweredDescent3DOFConfig,
    PoweredDescent3DOFModel,
)
from spacepdhcg.scvx import make_dynamics_consistent_reference
from spacepdhcg.transcription import (
    PoweredDescent3DOFSubproblem,
    PoweredDescentSCvxConfig,
)


def run(
    *,
    scenarios: int,
    intervals: int,
    gravity_spread: float,
    tolerance: float,
    common_prefix_fraction: float = 1.0,
) -> dict[str, object]:
    if scenarios <= 0:
        raise ValueError("scenarios must be positive")
    if not 0.0 <= gravity_spread < 0.5:
        raise ValueError("gravity_spread must lie in [0, 0.5)")
    if not 0.0 <= common_prefix_fraction <= 1.0:
        raise ValueError("common_prefix_fraction must lie in [0, 1]")

    step_seconds = 2.0
    initial = np.asarray([8.0, -4.0, 60.0, 0.0, 0.0, -4.5, 2_000.0])
    target_position = np.zeros(3)
    target_velocity = np.zeros(3)
    nominal_model = PoweredDescent3DOFModel()
    _, shared_reference_controls = make_dynamics_consistent_reference(
        nominal_model,
        initial,
        target_position,
        target_velocity,
        intervals=intervals,
        step_seconds=step_seconds,
    )

    deltas = (
        np.zeros(1) if scenarios == 1 else np.linspace(-gravity_spread, gravity_spread, scenarios)
    )
    subproblems = []
    models = []
    local_values = []
    for delta in deltas:
        gravity = nominal_model.config.gravity_vector.copy()
        gravity[2] *= 1.0 + delta
        model = PoweredDescent3DOFModel(
            PoweredDescent3DOFConfig(gravity=tuple(float(value) for value in gravity))
        )
        subproblem = PoweredDescent3DOFSubproblem(
            model,
            PoweredDescentSCvxConfig(
                intervals=intervals,
                step_seconds=step_seconds,
                trust_radius=2.0,
            ),
        )
        reference_states = model.rollout(
            initial,
            shared_reference_controls,
            step_seconds,
        )
        models.append(model)
        subproblems.append(subproblem)
        local_values.append(
            subproblem.values(
                reference_states,
                shared_reference_controls,
                initial,
                target_position,
                target_velocity,
                trust_radius=2.0,
            )
        )

    first = subproblems[0]
    common_prefix = round(intervals * common_prefix_fraction)
    tree = ScenarioTree.common_open_loop(
        scenarios,
        intervals,
        common_prefix=common_prefix,
    )
    bundle = ScenarioCQPBundle(
        tree,
        first.structure,
        state_dimension=7,
        control_dimension=4,
        local_auxiliary_dimension=(
            first.layout.n_variables - first.layout.state_count - first.layout.control_count
        ),
    )
    problem = bundle.problem(local_values)
    solver = PersistentClarabel(
        problem,
        tolerance=tolerance,
        iteration_limit=2_000,
        verbose=False,
    )
    solution = solver.solve()
    audit = solver.independent_residuals(solution.primal)
    if not residual_qualified(solution, tolerance=max(tolerance, 2.0e-8)):
        raise RuntimeError(
            "robust CQP failed residual qualification with "
            f"status {solution.status}, primal={solution.primal_residual}, "
            f"dual={solution.dual_residual}"
        )

    decoded = bundle.decode_primal(solution.primal)
    diagnostics = [
        subproblem.diagnostics(local, values)
        for subproblem, local, values in zip(
            subproblems,
            decoded.local,
            local_values,
            strict=True,
        )
    ]
    objectives = bundle.local_objectives(decoded.local, local_values)
    controls = [
        subproblem.decode(local)[1]
        for subproblem, local in zip(subproblems, decoded.local, strict=True)
    ]
    maximum_nonlinear_dynamics_defect = 0.0
    maximum_nonlinear_path_violation = 0.0
    maximum_nonlinear_terminal_error = 0.0
    for model, subproblem, local in zip(models, subproblems, decoded.local, strict=True):
        states, scenario_controls, _, _ = subproblem.decode(local)
        rollout = model.rollout(initial, scenario_controls, step_seconds)
        maximum_nonlinear_dynamics_defect = max(
            maximum_nonlinear_dynamics_defect,
            float(np.max(np.abs(states - rollout), initial=0.0)),
        )
        maximum_nonlinear_path_violation = max(
            maximum_nonlinear_path_violation,
            model.path_diagnostics(rollout, scenario_controls).maximum_violation,
        )
        maximum_nonlinear_terminal_error = max(
            maximum_nonlinear_terminal_error,
            float(np.max(np.abs(rollout[-1, :3] - target_position), initial=0.0)),
            float(np.max(np.abs(rollout[-1, 3:6] - target_velocity), initial=0.0)),
        )
    maximum_pairwise_control_difference = 0.0
    for controls_a in controls:
        for controls_b in controls:
            maximum_pairwise_control_difference = max(
                maximum_pairwise_control_difference,
                float(np.max(np.abs(controls_a - controls_b))),
            )

    return {
        "benchmark": "uncertain robust 3-DoF powered-descent CQP",
        "status": solution.status,
        "residual_qualified": True,
        "scenarios": scenarios,
        "intervals": intervals,
        "gravity_spread": gravity_spread,
        "common_prefix_fraction": common_prefix_fraction,
        "common_prefix_intervals": common_prefix,
        "variables": problem.structure.n_variables,
        "scalar_rows": problem.structure.n_constraints,
        "affine_rows": problem.structure.n_affine_constraints,
        "nonanticipativity_rows": bundle.nonanticipativity_rows,
        "objective": solution.objective,
        "expected_objective_recomputed": bundle.expected_objective(
            decoded.local,
            local_values,
        ),
        "local_objectives": [float(value) for value in objectives],
        "local_objective_min": float(np.min(objectives)),
        "local_objective_max": float(np.max(objectives)),
        "nonanticipativity_violation": bundle.maximum_nonanticipativity_violation(solution.primal),
        "maximum_pairwise_control_difference": maximum_pairwise_control_difference,
        "maximum_scalar_violation": max(
            diagnostic.scalar_violation_inf for diagnostic in diagnostics
        ),
        "maximum_cone_violation": max(diagnostic.cone_violation_inf for diagnostic in diagnostics),
        "maximum_linearised_dynamics_defect": max(
            diagnostic.linearised_dynamics_defect_inf for diagnostic in diagnostics
        ),
        "maximum_virtual_control": max(
            diagnostic.virtual_control_inf for diagnostic in diagnostics
        ),
        "maximum_nonlinear_dynamics_defect": maximum_nonlinear_dynamics_defect,
        "maximum_nonlinear_path_violation": maximum_nonlinear_path_violation,
        "maximum_nonlinear_terminal_error": maximum_nonlinear_terminal_error,
        "primal_residual": solution.primal_residual,
        "dual_residual": solution.dual_residual,
        "independent_primal_residual": audit.primal,
        "independent_dual_residual": audit.dual,
        "independent_natural_residual": audit.natural,
        "independent_cone_residual": audit.cone,
        "independent_complementarity": audit.complementarity,
        "iterations": solution.iterations,
        "setup_seconds": solver.setup_seconds,
        "solve_seconds": solution.solve_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=4)
    parser.add_argument("--intervals", type=int, default=6)
    parser.add_argument("--gravity-spread", type=float, default=0.02)
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
    parser.add_argument("--common-prefix-fraction", type=float, default=1.0)
    arguments = parser.parse_args()
    payload = run(
        scenarios=arguments.scenarios,
        intervals=arguments.intervals,
        gravity_spread=arguments.gravity_spread,
        tolerance=arguments.tolerance,
        common_prefix_fraction=arguments.common_prefix_fraction,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
