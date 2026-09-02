import numpy as np
import pytest

from spacepdhcg.backends import PersistentClarabel
from spacepdhcg.cqp import CanonicalCQP, residual_qualified
from spacepdhcg.distributed import (
    CondensedScenarioCQPBundle,
    ScenarioCQPBundle,
    ScenarioTree,
    encode_condensed_primal,
)
from spacepdhcg.models import PoweredDescent3DOFModel
from spacepdhcg.scvx import make_dynamics_consistent_reference
from spacepdhcg.transcription import (
    PoweredDescent3DOFSubproblem,
    PoweredDescentSCvxConfig,
)


def _local_problem(intervals: int = 5):
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
    values = subproblem.values(
        states,
        controls,
        initial,
        target_position,
        target_velocity,
    )
    return subproblem, values


def _local_auxiliary(subproblem) -> int:
    return (
        subproblem.layout.n_variables
        - subproblem.layout.state_count
        - subproblem.layout.control_count
    )


def _bundle(subproblem, scenario_count: int = 3) -> ScenarioCQPBundle:
    tree = ScenarioTree.common_open_loop(
        scenario_count,
        subproblem.layout.intervals,
    )
    return ScenarioCQPBundle(
        tree,
        subproblem.structure,
        state_dimension=7,
        control_dimension=4,
        local_auxiliary_dimension=_local_auxiliary(subproblem),
    )


def _condensed_bundle(
    subproblem,
    scenario_count: int = 3,
) -> CondensedScenarioCQPBundle:
    tree = ScenarioTree.common_open_loop(
        scenario_count,
        subproblem.layout.intervals,
    )
    return CondensedScenarioCQPBundle(
        tree,
        subproblem.structure,
        state_dimension=7,
        control_dimension=4,
        local_auxiliary_dimension=_local_auxiliary(subproblem),
    )


def _local_solution(subproblem, local_values):
    solution = PersistentClarabel(
        CanonicalCQP(subproblem.structure, local_values),
        tolerance=1.0e-8,
        iteration_limit=1_000,
    ).solve()
    assert residual_qualified(solution, tolerance=1.0e-8)
    return solution


def _canonical_objective(problem, primal) -> float:
    quadratic = problem.structure.quadratic.matrix(problem.values.quadratic)
    return float(0.5 * primal @ (quadratic @ primal) + problem.values.linear @ primal)


def test_condensed_identical_bundle_contains_repeated_local_optimum() -> None:
    subproblem, local_values = _local_problem()
    local_solution = _local_solution(subproblem, local_values)
    bundle = _condensed_bundle(subproblem)
    problem = bundle.problem([local_values] * bundle.scenario_count)
    repeated = encode_condensed_primal(
        bundle,
        [local_solution.primal] * bundle.scenario_count,
    )

    decoded = bundle.decode_primal(repeated)
    assert bundle.maximum_nonanticipativity_violation(repeated) < 1.0e-12
    for local in decoded.local:
        np.testing.assert_allclose(local, local_solution.primal, atol=0.0, rtol=0.0)
    expected = bundle.expected_objective(
        decoded.local,
        [local_values] * bundle.scenario_count,
    )
    assert abs(expected - local_solution.objective) < 1.0e-8
    assert abs(_canonical_objective(problem, repeated) - expected) < 1.0e-8

    scalar = problem.structure.constraint.matrix(problem.values.constraint) @ repeated
    scalar_violation = np.maximum(
        np.maximum(problem.values.lower - scalar, 0.0),
        np.maximum(scalar - problem.values.upper, 0.0),
    )
    variable_violation = np.maximum(
        np.maximum(problem.values.variable_lower - repeated, 0.0),
        np.maximum(repeated - problem.values.variable_upper, 0.0),
    )
    assert np.max(scalar_violation, initial=0.0) < 2.0e-8
    assert np.max(variable_violation, initial=0.0) < 2.0e-8

    assert problem.structure.affine_cone is not None
    affine = (
        problem.structure.affine_cone.matrix(problem.values.affine_cone) @ repeated
        + problem.values.affine_offset
    )
    for cone in problem.structure.affine_cones:
        segment = affine[cone.start : cone.stop]
        assert np.linalg.norm(segment[:-1]) <= segment[-1] + 2.0e-8


def test_known_incumbent_prevents_false_solver_qualification() -> None:
    subproblem, local_values = _local_problem()
    local_solution = _local_solution(subproblem, local_values)
    bundle = _condensed_bundle(subproblem)
    problem = bundle.problem([local_values] * bundle.scenario_count)
    repeated = encode_condensed_primal(
        bundle,
        [local_solution.primal] * bundle.scenario_count,
    )
    incumbent_objective = _canonical_objective(problem, repeated)
    solution = PersistentClarabel(
        problem,
        tolerance=1.0e-8,
        iteration_limit=1_000,
    ).solve()

    qualified_without_incumbent = residual_qualified(
        solution,
        tolerance=2.0e-8,
    )
    qualified_with_incumbent = residual_qualified(
        solution,
        tolerance=2.0e-8,
        objective_upper_bound=incumbent_objective,
        objective_tolerance=2.0e-6,
    )
    expected_qualification = bool(
        qualified_without_incumbent and solution.objective <= incumbent_objective + 2.0e-6
    )
    assert qualified_with_incumbent is expected_qualification

    decoded = bundle.decode_primal(solution.primal)
    expected = bundle.expected_objective(
        decoded.local,
        [local_values] * bundle.scenario_count,
    )
    assert abs(solution.objective - expected) < 1.0e-6


def test_condensed_encoder_rejects_inconsistent_shared_controls() -> None:
    subproblem, local_values = _local_problem(intervals=4)
    local_solution = _local_solution(subproblem, local_values)
    bundle = _condensed_bundle(subproblem, scenario_count=2)
    local_primals = [local_solution.primal.copy(), local_solution.primal.copy()]
    shared = subproblem.layout.control_slice(0)
    local_primals[1][shared.start] += 1.0e-3

    with pytest.raises(ValueError, match="shared information-node control"):
        encode_condensed_primal(bundle, local_primals)


def test_consensus_row_bundle_contains_the_repeated_local_optimum() -> None:
    subproblem, local_values = _local_problem()
    local_solution = _local_solution(subproblem, local_values)
    bundle = _bundle(subproblem)
    candidate = np.zeros(bundle.structure.n_variables)
    for scenario in range(bundle.scenario_count):
        candidate[bundle.layout.scenario_slice(scenario)] = local_solution.primal
    for block in bundle.layout.consensus_blocks:
        candidate[block.variable_slice] = local_solution.primal[
            subproblem.layout.control_slice(block.node.stage)
        ]

    assert bundle.maximum_nonanticipativity_violation(candidate) == 0.0
    decoded_candidate = bundle.decode_primal(candidate)
    for local in decoded_candidate.local:
        np.testing.assert_allclose(local, local_solution.primal, atol=0.0, rtol=0.0)
    assert (
        abs(
            bundle.expected_objective(
                decoded_candidate.local,
                [local_values] * bundle.scenario_count,
            )
            - local_solution.objective
        )
        < 1.0e-8
    )

    global_solution = PersistentClarabel(
        bundle.problem([local_values] * bundle.scenario_count),
        tolerance=1.0e-8,
        iteration_limit=1_000,
    ).solve()
    assert residual_qualified(global_solution, tolerance=2.0e-8)
    decoded = bundle.decode_primal(global_solution.primal)
    expected = bundle.expected_objective(
        decoded.local,
        [local_values] * bundle.scenario_count,
    )
    assert abs(global_solution.objective - expected) < 1.0e-6


def test_bundle_decodes_dual_blocks_and_consensus_controls() -> None:
    subproblem, local_values = _local_problem(intervals=4)
    bundle = _bundle(subproblem, scenario_count=2)
    solution = PersistentClarabel(
        bundle.problem([local_values, local_values]),
        tolerance=1.0e-8,
        iteration_limit=1_000,
    ).solve()

    assert residual_qualified(solution, tolerance=2.0e-8)
    primal = bundle.decode_primal(solution.primal)
    dual = bundle.decode_dual(solution.dual)
    assert len(primal.local) == 2
    assert len(primal.consensus) == subproblem.layout.intervals
    assert len(dual.local_scalar) == 2
    assert len(dual.local_affine) == 2
    assert dual.nonanticipativity.shape == (bundle.nonanticipativity_rows,)

    for block, consensus in zip(
        bundle.layout.consensus_blocks,
        primal.consensus,
        strict=True,
    ):
        for scenario in block.node.scenario_indices:
            local_control = primal.local[scenario][
                subproblem.layout.control_slice(block.node.stage)
            ]
            np.testing.assert_allclose(
                local_control,
                consensus,
                atol=1.0e-7,
                rtol=0.0,
            )


@pytest.mark.parametrize(
    ("measure", "alpha"),
    (("worst", None), ("cvar", 0.9)),
)
def test_sparse_risk_epigraph_matches_identical_scenario_costs(measure, alpha) -> None:
    subproblem, local_values = _local_problem(intervals=4)
    bundle = _bundle(subproblem, scenario_count=2)
    problem, layout = bundle.risk_problem(
        [local_values, local_values],
        measure,
        alpha=alpha,
    )
    solution = PersistentClarabel(
        problem,
        tolerance=1.0e-8,
        iteration_limit=2_000,
    ).solve()

    assert residual_qualified(solution, tolerance=2.0e-8)
    decoded = bundle.decode_primal(solution.primal[: layout.base_variables])
    local_objectives = bundle.local_objectives(
        decoded.local,
        [local_values, local_values],
    )
    represented = solution.primal[layout.scenario_costs]
    assert np.max(local_objectives - represented) <= 2.0e-5
    assert abs(solution.objective - np.max(local_objectives)) <= 2.0e-4


def test_numerical_updates_preserve_global_sparse_structure() -> None:
    subproblem, local_values = _local_problem(intervals=4)
    bundle = _bundle(subproblem, scenario_count=2)
    condensed = _condensed_bundle(subproblem, scenario_count=2)
    perturbed = local_values.copy()
    perturbed.linear += np.linspace(0.0, 1.0e-4, perturbed.linear.size)

    first = bundle.values([local_values, local_values])
    second = bundle.values([local_values, perturbed])
    first_condensed = condensed.values([local_values, local_values])
    second_condensed = condensed.values([local_values, perturbed])

    assert first.quadratic.shape == second.quadratic.shape
    assert first.constraint.shape == second.constraint.shape
    assert first.affine_cone.shape == second.affine_cone.shape
    assert not np.array_equal(first.linear, second.linear)
    first_matrix = bundle.structure.constraint.matrix(first.constraint)
    second_matrix = bundle.structure.constraint.matrix(second.constraint)
    np.testing.assert_array_equal(
        first_matrix[-bundle.nonanticipativity_rows :].toarray(),
        second_matrix[-bundle.nonanticipativity_rows :].toarray(),
    )
    assert first_condensed.quadratic.shape == second_condensed.quadratic.shape
    assert first_condensed.constraint.shape == second_condensed.constraint.shape
    assert not np.array_equal(first_condensed.linear, second_condensed.linear)
