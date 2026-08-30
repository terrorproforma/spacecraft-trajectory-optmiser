import numpy as np

from spacepdhcg.backends import PersistentClarabel
from spacepdhcg.cqp import CanonicalCQP
from spacepdhcg.distributed import ScenarioCQPBundle, ScenarioTree
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


def _bundle(subproblem, scenario_count: int = 3) -> ScenarioCQPBundle:
    tree = ScenarioTree.common_open_loop(
        scenario_count,
        subproblem.layout.intervals,
    )
    local_auxiliary = (
        subproblem.layout.n_variables
        - subproblem.layout.state_count
        - subproblem.layout.control_count
    )
    return ScenarioCQPBundle(
        tree,
        subproblem.structure,
        state_dimension=7,
        control_dimension=4,
        local_auxiliary_dimension=local_auxiliary,
    )


def test_identical_scenario_bundle_matches_single_scenario_objective() -> None:
    subproblem, local_values = _local_problem()
    local_solution = PersistentClarabel(
        CanonicalCQP(subproblem.structure, local_values),
        tolerance=1.0e-8,
        iteration_limit=1_000,
    ).solve()
    assert local_solution.solved

    bundle = _bundle(subproblem)
    global_problem = bundle.problem([local_values] * bundle.scenario_count)
    global_solution = PersistentClarabel(
        global_problem,
        tolerance=1.0e-8,
        iteration_limit=1_000,
    ).solve()

    assert global_solution.solved
    decoded = bundle.decode_primal(global_solution.primal)
    assert bundle.maximum_nonanticipativity_violation(global_solution.primal) < 1.0e-7
    for local in decoded.local[1:]:
        np.testing.assert_allclose(local, decoded.local[0], atol=2.0e-6, rtol=0.0)
    expected = bundle.expected_objective(decoded.local, [local_values] * 3)
    assert abs(global_solution.objective - expected) < 1.0e-6
    assert abs(global_solution.objective - local_solution.objective) < 2.0e-6


def test_bundle_decodes_dual_blocks_and_consensus_controls() -> None:
    subproblem, local_values = _local_problem(intervals=4)
    bundle = _bundle(subproblem, scenario_count=2)
    solution = PersistentClarabel(
        bundle.problem([local_values, local_values]),
        tolerance=1.0e-8,
        iteration_limit=1_000,
    ).solve()

    assert solution.solved
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
    perturbed = local_values.copy()
    perturbed.linear += np.linspace(0.0, 1.0e-4, perturbed.linear.size)

    first = bundle.values([local_values, local_values])
    second = bundle.values([local_values, perturbed])

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
