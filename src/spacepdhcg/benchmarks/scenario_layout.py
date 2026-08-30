"""Inspect scenario-aware block-arrow structure and communication scaling."""

from __future__ import annotations

import argparse
import json

import numpy as np

from spacepdhcg.distributed import (
    BlockArrowLayout,
    LogicalGPUGrid,
    ScenarioTree,
    partition_scenarios,
)


def run(
    *,
    scenarios: int,
    intervals: int,
    common_prefix: int,
    scenario_partitions: int,
    time_partitions: int,
) -> dict[str, object]:
    tree = ScenarioTree.common_open_loop(
        scenarios,
        intervals,
        common_prefix=common_prefix,
    )
    layout = BlockArrowLayout(
        tree,
        state_dimension=7,
        control_dimension=4,
        local_auxiliary_dimension=7 * intervals,
    )
    grid = LogicalGPUGrid(scenario_partitions, time_partitions)
    scenario_work = np.asarray(
        [
            layout.local_variables_per_scenario * (1.0 + 0.05 * (index % 5))
            for index in range(scenarios)
        ]
    )
    partition = partition_scenarios(scenario_work, scenario_partitions)
    communication = layout.communication_profile(
        grid.device_count,
        collective_count=2,
    )
    return {
        "benchmark": "scenario-aware block-arrow layout",
        "grid": {
            "scenario_partitions": scenario_partitions,
            "time_partitions": time_partitions,
            "devices": grid.device_count,
        },
        "tree": {
            "scenarios": scenarios,
            "intervals": intervals,
            "common_prefix": common_prefix,
            "information_nodes": len(tree.nodes),
            "shared_nodes": len(tree.shared_nodes),
        },
        "layout": layout.statistics(),
        "partition": {
            "assignments": partition.assignments,
            "loads": partition.loads,
            "imbalance": partition.imbalance,
        },
        "communication": {
            "shared_dimension": communication.shared_dimension,
            "payload_bytes": communication.payload_bytes,
            "bytes_per_device": communication.bytes_per_device,
            "aggregate_bytes": communication.aggregate_bytes,
            "collective_count": communication.collective_count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=16)
    parser.add_argument("--intervals", type=int, default=20)
    parser.add_argument("--common-prefix", type=int, default=20)
    parser.add_argument("--scenario-partitions", type=int, default=4)
    parser.add_argument("--time-partitions", type=int, default=1)
    arguments = parser.parse_args()
    payload = run(
        scenarios=arguments.scenarios,
        intervals=arguments.intervals,
        common_prefix=arguments.common_prefix,
        scenario_partitions=arguments.scenario_partitions,
        time_partitions=arguments.time_partitions,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
