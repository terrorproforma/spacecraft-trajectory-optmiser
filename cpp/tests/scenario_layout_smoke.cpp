#include "spacepdhcg/distributed/scenario_layout.hpp"

#include <cmath>
#include <vector>

int main() {
    using spacepdhcg::distributed::BlockArrowLayout;
    using spacepdhcg::distributed::LogicalGpuGrid;
    using spacepdhcg::distributed::ScenarioTree;
    using spacepdhcg::distributed::partition_scenarios;

    const auto tree = ScenarioTree::common_open_loop(4U, 5U, 3U);
    if (tree.scenario_count() != 4U || tree.horizon() != 5U
        || tree.shared_nodes().size() != 3U) {
        return 1;
    }

    const BlockArrowLayout layout(tree, 7U, 4U, 12U);
    if (layout.consensus_dimension() != 12U || layout.nonanticipativity_rows() != 48U) {
        return 2;
    }
    const auto triplets = layout.nonanticipativity_triplets();
    if (triplets.size() != 96U) {
        return 3;
    }
    const auto profile = layout.communication_profile(2U);
    if (profile.payload_bytes != 96U || std::abs(profile.bytes_per_device - 96.0) > 1.0e-12
        || std::abs(profile.aggregate_bytes - 192.0) > 1.0e-12) {
        return 4;
    }

    const auto partition = partition_scenarios(std::vector<double>{5.0, 4.0, 3.0, 2.0}, 2U);
    if (partition.assignments.size() != 2U || std::abs(partition.maximum_load() - 7.0) > 1.0e-12
        || std::abs(partition.imbalance() - 1.0) > 1.0e-12) {
        return 5;
    }

    const LogicalGpuGrid grid{2U, 2U};
    if (grid.device_count() != 4U || grid.rank(1U, 1U) != 3U) {
        return 6;
    }
    return 0;
}
