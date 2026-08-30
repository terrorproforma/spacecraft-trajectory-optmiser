import numpy as np

from spacepdhcg.backends import PersistentClarabel
from spacepdhcg.cqp import CanonicalCQP, residual_qualified
from spacepdhcg.distributed import (
    CondensedScenarioCQPBundle,
    ScenarioCQPBundle,
    ScenarioTree,
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


def test_condensed_identical_bundle_matches_single_scenario_objective() -> None:
    subproblem, local_values = _local_problem()
    local_solution = _local_solution(subproblem, local_values)
    bundle = _condensed_bundle(subproblem)
    global_solution = PersistentClarabel(
        bundle.problem([local_values] * bundle.scenario_count),
        tolerance=1.0e-8,
        iteration_limit=1_000,
    ).solve()

    assert residual_qualified(global_solution, tolerance=2.0e-8)
    decoded = bundle.decode_primal(global_solution.primal)
    assert bundle.maximum_nonanticipativity_violation(global_solution.primal) < 1.0e-12
    for local in decoded.local[1:]:
        np.testing.assert_allclose(local, decoded.local[0], atol=2.0e-6, rtol=0.0)
    expected = bundle.expected_objective(decoded.local, [local_values] * 3)
    assert abs(global_solution.objective - expected) < 1.0e-6
    assert abs(global_solution.objective - local_solution.objective) < 2.0e-6


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
