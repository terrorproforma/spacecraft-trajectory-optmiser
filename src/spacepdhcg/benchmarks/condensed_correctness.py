"""Diagnose condensed scenario-CQP equivalence independently of a solver status flag."""

from __future__ import annotations

import argparse
import json

import numpy as np

from spacepdhcg.backends import PersistentClarabel
from spacepdhcg.cqp import CanonicalCQP
from spacepdhcg.distributed import CondensedScenarioCQPBundle, ScenarioTree
from spacepdhcg.models import PoweredDescent3DOFModel
from spacepdhcg.scvx import make_dynamics_consistent_reference
from spacepdhcg.transcription import (
    PoweredDescent3DOFSubproblem,
    PoweredDescentSCvxConfig,
)


def _objective(problem: CanonicalCQP, primal: np.ndarray) -> float:
    quadratic = problem.structure.quadratic.matrix(problem.values.quadratic)
    return float(
        0.5 * primal @ (quadratic @ primal)
        + problem.values.linear @ primal
    )


def _maximum_scalar_violation(problem: CanonicalCQP, primal: np.ndarray) -> float:
    activity = problem.structure.constraint.matrix(problem.values.constraint) @ primal
    violation = np.maximum(
        np.maximum(problem.values.lower - activity, 0.0),
        np.maximum(activity - problem.values.upper, 0.0),
    )
    return float(np.max(violation, initial=0.0))


def _maximum_variable_violation(problem: CanonicalCQP, primal: np.ndarray) -> float:
    violation = np.maximum(
        np.maximum(problem.values.variable_lower - primal, 0.0),
        np.maximum(primal - problem.values.variable_upper, 0.0),
    )
    return float(np.max(violation, initial=0.0))


def _maximum_soc_violation(problem: CanonicalCQP, primal: np.ndarray) -> float:
    if problem.structure.affine_cone is None:
        return 0.0
    activity = (
        problem.structure.affine_cone.matrix(problem.values.affine_cone) @ primal
        + problem.values.affine_offset
    )
    maximum = 0.0
    for cone in problem.structure.affine_cones:
        segment = activity[cone.start : cone.stop]
        maximum = max(
            maximum,
            float(max(np.linalg.norm(segment[:-1]) - segment[-1], 0.0)),
        )
    return maximum


def run(*, intervals: int, scenarios: int, tolerance: float) -> dict[str, object]:
    model = PoweredDescent3DOFModel()
    subproblem = PoweredDescent3DOFSubproblem(
        model,
        PoweredDescentSCvxConfig(
            intervals=intervals,
            step_seconds=2.0,
            trust_radius=1.0,
        ),
    )
    initial = np.asarray([5.0, -2.0, 50.0, 0.0, 0.0, -4.0, 2_000.0])
    target_position = np.zeros(3)
    target_velocity = np.zeros(3)
    states, controls = make_dynamics_consistent_reference(
        model,
        initial,
        target_position,
        target_velocity,
        intervals=intervals,
        step_seconds=2.0,
    )
    local_values = subproblem.values(
        states,
        controls,
        initial,
        target_position,
        target_velocity,
    )
    local_problem = CanonicalCQP(subproblem.structure, local_values)
    local_solver = PersistentClarabel(
        local_problem,
        tolerance=tolerance,
        iteration_limit=2_000,
    )
    local_solution = local_solver.solve()

    tree = ScenarioTree.common_open_loop(scenarios, intervals)
    bundle = CondensedScenarioCQPBundle(
        tree,
        subproblem.structure,
        state_dimension=7,
        control_dimension=4,
        local_auxiliary_dimension=(
            subproblem.layout.n_variables
            - subproblem.layout.state_count
            - subproblem.layout.control_count
        ),
    )
    global_problem = bundle.problem([local_values] * scenarios)
    repeated = np.zeros(global_problem.structure.n_variables)
    assigned = np.zeros(global_problem.structure.n_variables, dtype=bool)
    for scenario in range(scenarios):
        mapping = bundle.local_to_global(scenario)
        for local_index, global_index in enumerate(mapping):
            value = local_solution.primal[local_index]
            if assigned[global_index] and not np.isclose(
                repeated[global_index],
                value,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise RuntimeError("identical local solutions disagree on a shared variable")
            repeated[global_index] = value
            assigned[global_index] = True
    if not np.all(assigned):
        raise RuntimeError("condensed primal contains unassigned variables")

    global_solver = PersistentClarabel(
        global_problem,
        tolerance=tolerance,
        iteration_limit=2_000,
    )
    global_solution = global_solver.solve()
    repeated_decoded = bundle.decode_primal(repeated)

    return {
        "benchmark": "condensed identical-scenario correctness diagnostic",
        "intervals": intervals,
        "scenarios": scenarios,
        "local_status": local_solution.status,
        "global_status": global_solution.status,
        "local_objective": local_solution.objective,
        "repeated_local_expected_objective": bundle.expected_objective(
            repeated_decoded.local,
            [local_values] * scenarios,
        ),
        "repeated_local_global_objective": _objective(global_problem, repeated),
        "global_solver_objective": global_solution.objective,
        "global_solver_minus_repeated": (
            global_solution.objective - _objective(global_problem, repeated)
        ),
        "repeated_scalar_violation": _maximum_scalar_violation(
            global_problem,
            repeated,
        ),
        "repeated_variable_violation": _maximum_variable_violation(
            global_problem,
            repeated,
        ),
        "repeated_soc_violation": _maximum_soc_violation(
            global_problem,
            repeated,
        ),
        "global_primal_residual": global_solution.primal_residual,
        "global_dual_residual": global_solution.dual_residual,
        "global_iterations": global_solution.iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=int, default=5)
    parser.add_argument("--scenarios", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                intervals=arguments.intervals,
                scenarios=arguments.scenarios,
                tolerance=arguments.tolerance,
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
