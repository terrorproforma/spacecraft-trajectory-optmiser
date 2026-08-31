#include "spacepdhcg/orbitweaver/column_generation.hpp"
#include "spacepdhcg/orbitweaver/restricted_master_lp.hpp"
#include "spacepdhcg/orbitweaver/route_master.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

int main() {
    using spacepdhcg::orbitweaver::ColumnGenerationConfig;
    using spacepdhcg::orbitweaver::DenseRestrictedMasterLP;
    using spacepdhcg::orbitweaver::RouteColumn;
    using spacepdhcg::orbitweaver::RouteMasterDualPrices;
    using spacepdhcg::orbitweaver::route_reduced_cost;
    using spacepdhcg::orbitweaver::run_column_generation;

    const RouteColumn target_zero{
        0U,
        0U,
        {0U},
        5.0,
        4.0,
        1.0,
    };
    const RouteColumn target_one{
        1U,
        1U,
        {1U},
        5.0,
        4.0,
        1.0,
    };
    const RouteColumn combined{
        2U,
        0U,
        {0U, 1U},
        6.0,
        5.0,
        1.5,
    };

    const DenseRestrictedMasterLP master{2U, 2U};
    const auto initial = master({target_zero, target_one});
    if (!initial.feasible || std::abs(initial.objective - 10.0) > 1.0e-9
        || std::abs(initial.lower_bound - 10.0) > 1.0e-9) {
        return 1;
    }
    const auto combined_reduced_cost = route_reduced_cost(combined, initial.duals);
    if (!(combined_reduced_cost < -1.0e-8)) {
        return 2;
    }

    std::size_t pricing_calls{0U};
    const auto pricing = [&pricing_calls, combined](const RouteMasterDualPrices&) {
        ++pricing_calls;
        return std::vector<RouteColumn>{combined};
    };
    ColumnGenerationConfig config{};
    config.maximum_iterations = 5U;
    config.maximum_columns_per_iteration = 2U;
    const auto result = run_column_generation(
        2U,
        2U,
        {target_zero, target_one},
        master.callback(),
        pricing,
        config
    );
    if (!result.master_feasible || !result.converged
        || result.iteration_limit_reached || result.history.size() != 2U
        || pricing_calls != 2U) {
        return 3;
    }
    if (!result.incumbent.feasible || std::abs(result.incumbent.cost - 6.0) > 1.0e-9
        || result.incumbent.selected_column_indices.size() != 1U) {
        return 4;
    }
    if (result.history.front().columns_added != 1U
        || result.history.back().columns_added != 0U
        || result.columns.size() != 3U) {
        return 5;
    }
    const auto final_master = master(result.columns);
    if (!final_master.feasible || std::abs(final_master.objective - 6.0) > 1.0e-9
        || route_reduced_cost(combined, final_master.duals) < -1.0e-8) {
        return 6;
    }
    return 0;
}
