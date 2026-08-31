#include "spacepdhcg/orbitweaver/column_generation.hpp"

#include <cmath>
#include <vector>

int main() {
    using spacepdhcg::orbitweaver::ColumnGenerationConfig;
    using spacepdhcg::orbitweaver::RestrictedMasterResult;
    using spacepdhcg::orbitweaver::RouteColumn;
    using spacepdhcg::orbitweaver::RouteMasterDualPrices;
    using spacepdhcg::orbitweaver::run_column_generation;

    const std::vector<RouteColumn> initial{
        RouteColumn{10U, 0U, {0U, 1U}, 10.0, 8.0, 3.0},
        RouteColumn{20U, 1U, {2U, 3U}, 10.0, 8.0, 3.0},
    };
    const auto master = [](const std::vector<RouteColumn>& columns) {
        if (columns.size() < 4U) {
            return RestrictedMasterResult{
                true,
                12.0,
                12.0,
                RouteMasterDualPrices{{3.0, 3.0, 3.0, 3.0}, {0.0, 0.0}},
                4U,
                1.0e-3,
            };
        }
        return RestrictedMasterResult{
            true,
            8.0,
            8.0,
            RouteMasterDualPrices{{2.0, 2.0, 2.0, 2.0}, {0.0, 0.0}},
            2U,
            5.0e-4,
        };
    };
    const auto pricing = [](const RouteMasterDualPrices&) {
        return std::vector<RouteColumn>{
            RouteColumn{11U, 0U, {0U, 2U}, 4.0, 3.5, 1.5},
            RouteColumn{12U, 0U, {0U, 2U}, 5.5, 5.0, 2.0},
            RouteColumn{21U, 1U, {1U, 3U}, 4.0, 3.5, 1.0},
        };
    };

    ColumnGenerationConfig config{};
    config.maximum_iterations = 8U;
    config.maximum_columns_per_iteration = 8U;
    config.reduced_cost_tolerance = 1.0e-10;
    const auto result = run_column_generation(
        2U,
        4U,
        initial,
        master,
        pricing,
        config
    );

    if (!result.master_feasible || !result.converged || result.iteration_limit_reached) {
        return 1;
    }
    if (!result.incumbent.feasible || std::abs(result.incumbent.cost - 8.0) > 1.0e-12) {
        return 2;
    }
    if (result.columns.size() != 4U || result.history.size() != 2U) {
        return 3;
    }
    if (result.history[0].columns_added != 2U
        || result.history[0].negative_reduced_cost_columns != 3U) {
        return 4;
    }
    if (std::abs(result.history[0].best_reduced_cost + 2.0) > 1.0e-12) {
        return 5;
    }
    if (result.history[1].columns_added != 0U
        || result.history[1].negative_reduced_cost_columns != 0U) {
        return 6;
    }
    if (std::abs(result.master_lower_bound - 8.0) > 1.0e-12) {
        return 7;
    }
    return 0;
}
