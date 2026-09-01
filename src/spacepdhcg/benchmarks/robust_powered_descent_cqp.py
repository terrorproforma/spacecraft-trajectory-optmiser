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
) -> dict[str, object]:
    if scenarios <= 0:
        raise ValueError("scenarios must be positive")
    if not 0.0 <= gravity_spread < 0.5:
        raise ValueError("gravity_spread must lie in [0, 0.5)")

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
    tree = ScenarioTree.common_open_loop(scenarios, intervals)
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
        "variables": problem.structure.n_variables,
        "scalar_rows": problem.structure.n_constraints,
        "affine_rows": problem.structure.n_affine_constraints,
        "nonanticipativity_rows": bundle.nonanticipativity_rows,
        "objective": solution.objective,
        "expected_objective_recomputed": bundle.expected_objective(
            decoded.local,
            local_values,
        ),
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
        "primal_residual": solution.primal_residual,
        "dual_residual": solution.dual_residual,
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
    arguments = parser.parse_args()
    payload = run(
        scenarios=arguments.scenarios,
        intervals=arguments.intervals,
        gravity_spread=arguments.gravity_spread,
        tolerance=arguments.tolerance,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
