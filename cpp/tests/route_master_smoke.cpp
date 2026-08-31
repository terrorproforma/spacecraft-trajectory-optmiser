#include "spacepdhcg/orbitweaver/route_master.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

int main() {
    using spacepdhcg::orbitweaver::RouteColumn;
    using spacepdhcg::orbitweaver::RouteColumnPool;
    using spacepdhcg::orbitweaver::RouteMasterDualPrices;
    using spacepdhcg::orbitweaver::negative_reduced_cost_columns;
    using spacepdhcg::orbitweaver::solve_exact_route_master;

    RouteColumnPool pool(2U, 4U);
    if (!pool.add(RouteColumn{10U, 0U, {0U, 1U}, 5.0, 4.0, 2.0})) {
        return 1;
    }
    if (!pool.add(RouteColumn{11U, 0U, {0U, 2U}, 4.0, 3.5, 1.5})) {
        return 2;
    }
    if (!pool.add(RouteColumn{20U, 1U, {2U, 3U}, 5.0, 4.0, 2.5})) {
        return 3;
    }
    if (!pool.add(RouteColumn{21U, 1U, {1U, 3U}, 4.0, 3.5, 1.0})) {
        return 4;
    }
    if (pool.add(RouteColumn{12U, 0U, {0U, 2U}, 6.0, 5.0, 2.0})) {
        return 5;
    }

    const auto solution = solve_exact_route_master(
        pool.spacecraft_count(),
        pool.target_count(),
        pool.columns()
    );
    if (!solution.feasible || std::abs(solution.cost - 8.0) > 1.0e-12) {
        return 6;
    }
    if (solution.selected_column_indices.size() != 2U) {
        return 7;
    }
    std::vector<std::size_t> identifiers{};
    for (const auto index : solution.selected_column_indices) {
        identifiers.push_back(pool.columns().at(index).identifier);
    }
    std::sort(identifiers.begin(), identifiers.end());
    if (identifiers != std::vector<std::size_t>{11U, 21U}) {
        return 8;
    }

    const RouteMasterDualPrices duals{
        {3.0, 3.0, 3.0, 3.0},
        {0.0, 0.0},
    };
    const std::vector<RouteColumn> candidates{
        RouteColumn{30U, 0U, {0U, 3U}, 3.0, 2.5, 1.0},
        RouteColumn{31U, 1U, {1U, 2U}, 7.0, 6.0, 2.0},
    };
    const auto priced = negative_reduced_cost_columns(candidates, duals);
    if (priced.size() != 1U || priced.front().column.identifier != 30U) {
        return 9;
    }
    if (std::abs(priced.front().reduced_cost + 3.0) > 1.0e-12) {
        return 10;
    }
    if (!pool.add(priced.front().column)) {
        return 11;
    }
    return 0;
}
