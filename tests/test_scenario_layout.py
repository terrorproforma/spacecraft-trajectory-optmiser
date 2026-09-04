import numpy as np

from spacepdhcg.distributed import (
    BlockArrowLayout,
    LogicalGPUGrid,
    Scenario,
    ScenarioTree,
    partition_scenarios,
)


def test_common_open_loop_prefix_produces_exact_information_groups() -> None:
    tree = ScenarioTree.common_open_loop(4, 5, common_prefix=2)

    assert tree.control_groups(0) == ((0, 1, 2, 3),)
    assert tree.control_groups(1) == ((0, 1, 2, 3),)
    assert tree.control_groups(2) == ((0,), (1,), (2,), (3,))
    assert len(tree.shared_nodes) == 2
    assert tree.anchor_edges() == (
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
        (1, 0, 1),
        (1, 0, 2),
        (1, 0, 3),
    )


def test_general_information_histories_branch_by_exact_prefix() -> None:
    scenarios = (
        Scenario("a", 0.25, ("root", "left", "left/low")),
        Scenario("b", 0.25, ("root", "left", "left/high")),
        Scenario("c", 0.25, ("root", "right", "right/low")),
        Scenario("d", 0.25, ("root", "right", "right/high")),
    )
    tree = ScenarioTree(scenarios)

    assert tree.control_groups(0) == ((0, 1, 2, 3),)
    assert tree.control_groups(1) == ((0, 1), (2, 3))
    assert tree.control_groups(2) == ((0,), (1,), (2,), (3,))


def test_block_arrow_operator_enforces_local_controls_against_consensus() -> None:
    tree = ScenarioTree.common_open_loop(3, 4, common_prefix=3)
    layout = BlockArrowLayout(tree, state_dimension=2, control_dimension=2)
    operator = layout.nonanticipativity_operator()

    assert operator.shape == (3 * 3 * 2, layout.total_variables)
    assert operator.nnz == 2 * operator.shape[0]

    vector = np.zeros(layout.total_variables)
    for block in layout.consensus_blocks:
        consensus = np.array([block.node.stage + 1.0, -block.node.stage - 0.5])
        vector[block.variable_slice] = consensus
        for scenario_index in block.node.scenario_indices:
            vector[layout.control_slice(scenario_index, block.node.stage)] = consensus
    assert layout.nonanticipativity_violation(vector) == 0.0

    vector[layout.control_slice(2, 1).start] += 0.125
    assert layout.nonanticipativity_violation(vector) == 0.125


def test_consensus_payload_does_not_grow_with_scenario_count() -> None:
    small = BlockArrowLayout(
        ScenarioTree.common_open_loop(2, 6),
        state_dimension=7,
        control_dimension=4,
    )
    large = BlockArrowLayout(
        ScenarioTree.common_open_loop(32, 6),
        state_dimension=7,
        control_dimension=4,
    )

    assert small.consensus_dimension == large.consensus_dimension == 24
    small_profile = small.communication_profile(4)
    large_profile = large.communication_profile(4)
    assert small_profile.payload_bytes == large_profile.payload_bytes
    assert large.nonanticipativity_rows == 16 * small.nonanticipativity_rows


def test_scenario_partition_is_deterministic_and_balances_heavy_blocks() -> None:
    weights = [10.0, 9.0, 8.0, 7.0, 3.0, 2.0, 1.0]
    first = partition_scenarios(weights, 3)
    second = partition_scenarios(weights, 3)

    assert first == second
    assigned = sorted(index for assignment in first.assignments for index in assignment)
    assert assigned == list(range(len(weights)))
    assert first.imbalance <= 1.15
    assert all(first.owner(index) >= 0 for index in range(len(weights)))


def test_logical_gpu_grid_uses_scenario_major_rank_order() -> None:
    grid = LogicalGPUGrid(scenario_partitions=3, time_partitions=2)

    assert grid.device_count == 6
    assert grid.rank(0, 0) == 0
    assert grid.rank(1, 0) == 2
    assert grid.rank(2, 1) == 5


def test_probabilities_must_sum_to_one() -> None:
    with np.testing.assert_raises(ValueError):
        ScenarioTree(
            (
                Scenario("a", 0.4, ("root",)),
                Scenario("b", 0.4, ("root",)),
            )
        )
