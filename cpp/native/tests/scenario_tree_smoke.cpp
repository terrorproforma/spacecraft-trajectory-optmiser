#include "spacepdhcg/native/scenario_tree.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace native = spacepdhcg::native;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_tree_and_operator() {
    const auto tree = native::ScenarioTree::common_open_loop(4, 5, 2);
    require(tree.scenario_count() == 4, "scenario-tree count is wrong");
    require(tree.horizon() == 5, "scenario-tree horizon is wrong");
    require(tree.nodes_at_stage(0).size() == 1, "root information node is wrong");
    require(tree.nodes_at_stage(1).size() == 1, "shared prefix node is wrong");
    require(tree.nodes_at_stage(2).size() == 4, "recourse split is wrong");
    require(tree.shared_nodes().size() == 2, "shared-node count is wrong");

    native::BlockArrowLayout layout(tree, 7, 4, 14);
    require(layout.state_variables_per_scenario() == 42,
            "scenario state dimension is wrong");
    require(layout.control_variables_per_scenario() == 20,
            "scenario control dimension is wrong");
    require(layout.local_variables_per_scenario() == 76,
            "scenario local dimension is wrong");
    require(layout.consensus_dimension() == 8,
            "scenario consensus dimension is wrong");
    require(layout.nonanticipativity_rows() == 32,
            "non-anticipativity row count is wrong");

    const auto operator_matrix = layout.nonanticipativity_operator();
    require(operator_matrix.rows == 32,
            "non-anticipativity matrix row count is wrong");
    require(operator_matrix.columns == static_cast<spacepdhcg::Index>(layout.total_variables()),
            "non-anticipativity matrix column count is wrong");
    require(operator_matrix.nonzeros() == 64,
            "non-anticipativity matrix sparsity is wrong");

    std::vector<double> decision(layout.total_variables(), 0.0);
    for (const auto& block : layout.consensus_blocks()) {
        for (std::size_t component = 0; component < layout.control_dimension(); ++component) {
            const double value = 10.0 * static_cast<double>(block.node.stage + 1U) +
                                 static_cast<double>(component);
            decision[block.variables.offset + component] = value;
            for (std::size_t scenario : block.node.scenarios) {
                const auto local = layout.control_slice(scenario, block.node.stage);
                decision[local.offset + component] = value;
            }
        }
    }
    require(layout.nonanticipativity_violation(decision) < 1.0e-15,
            "consistent shared controls violate non-anticipativity");

    const auto first_shared = layout.consensus_blocks().front();
    const auto first_local = layout.control_slice(
        first_shared.node.scenarios.front(),
        first_shared.node.stage
    );
    decision[first_local.offset] += 0.25;
    require(layout.nonanticipativity_violation(decision) > 0.249,
            "non-anticipativity violation was not detected");
}

}  // namespace

int main() {
    test_tree_and_operator();
    return 0;
}
