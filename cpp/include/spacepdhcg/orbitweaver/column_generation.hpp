#pragma once

#include "spacepdhcg/orbitweaver/route_master.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

struct RestrictedMasterResult {
    bool feasible{false};
    double objective{std::numeric_limits<double>::infinity()};
    double lower_bound{std::numeric_limits<double>::infinity()};
    RouteMasterDualPrices duals{};
    std::size_t iterations{0U};
    double solve_seconds{0.0};

    void validate(std::size_t target_count, std::size_t spacecraft_count) const {
        if (!feasible) {
            return;
        }
        if (!std::isfinite(objective) || !std::isfinite(lower_bound)
            || lower_bound > objective) {
            throw std::runtime_error("restricted master returned invalid objective bounds");
        }
        if (!std::isfinite(solve_seconds) || solve_seconds < 0.0) {
            throw std::runtime_error("restricted master returned invalid solve time");
        }
        duals.validate(target_count, spacecraft_count);
    }
};

struct ColumnGenerationConfig {
    std::size_t maximum_iterations{100U};
    std::size_t maximum_columns_per_iteration{64U};
    double reduced_cost_tolerance{1.0e-8};
    double absolute_gap_tolerance{1.0e-8};
    double relative_gap_tolerance{1.0e-6};

    void validate() const {
        if (maximum_iterations == 0U || maximum_columns_per_iteration == 0U) {
            throw std::invalid_argument("column-generation iteration limits must be positive");
        }
        for (const auto value : {
                 reduced_cost_tolerance,
                 absolute_gap_tolerance,
                 relative_gap_tolerance,
             }) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(
                    "column-generation tolerances must be finite and non-negative"
                );
            }
        }
    }
};

struct ColumnGenerationIteration {
    std::size_t iteration{0U};
    std::size_t columns_before{0U};
    std::size_t candidate_columns{0U};
    std::size_t negative_reduced_cost_columns{0U};
    std::size_t columns_added{0U};
    double master_objective{std::numeric_limits<double>::infinity()};
    double master_lower_bound{std::numeric_limits<double>::infinity()};
    double best_reduced_cost{std::numeric_limits<double>::infinity()};
    double incumbent_cost{std::numeric_limits<double>::infinity()};
    double optimality_gap{std::numeric_limits<double>::infinity()};
    std::size_t master_iterations{0U};
    double master_solve_seconds{0.0};
};

struct ColumnGenerationResult {
    bool master_feasible{false};
    bool converged{false};
    bool iteration_limit_reached{false};
    RouteMasterSolution incumbent{};
    double master_lower_bound{std::numeric_limits<double>::infinity()};
    std::vector<RouteColumn> columns{};
    std::vector<ColumnGenerationIteration> history{};
};

using RestrictedMasterSolver = std::function<RestrictedMasterResult(
    const std::vector<RouteColumn>& columns
)>;
using RoutePricingOracle = std::function<std::vector<RouteColumn>(
    const RouteMasterDualPrices& duals
)>;

namespace column_generation_detail {

inline double gap(double incumbent, double lower_bound) noexcept {
    if (!std::isfinite(incumbent) || !std::isfinite(lower_bound)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max(0.0, incumbent - lower_bound);
}

inline double gap_tolerance(double incumbent, const ColumnGenerationConfig& config) noexcept {
    return config.absolute_gap_tolerance
           + config.relative_gap_tolerance * std::max(1.0, std::abs(incumbent));
}

}  // namespace column_generation_detail

/// Deterministic column-generation control loop with an exact small-instance incumbent.
///
/// The restricted-master LP solver and mission pricing oracle are injected so this controller
/// can drive a CPU LP reference now and a GPU/batched trajectory oracle later without changing
/// convergence accounting. Candidate columns are independently re-priced and de-duplicated by
/// the native route pool before they can enter the master.
inline ColumnGenerationResult run_column_generation(
    std::size_t spacecraft_count,
    std::size_t target_count,
    std::vector<RouteColumn> initial_columns,
    RestrictedMasterSolver master_solver,
    RoutePricingOracle pricing_oracle,
    ColumnGenerationConfig config = ColumnGenerationConfig{},
    std::vector<std::size_t> required_targets = {}
) {
    config.validate();
    if (!master_solver || !pricing_oracle) {
        throw std::invalid_argument("column generation requires master and pricing callbacks");
    }

    RouteColumnPool pool(spacecraft_count, target_count);
    for (auto& column : initial_columns) {
        static_cast<void>(pool.add(std::move(column)));
    }
    if (pool.columns().empty()) {
        throw std::invalid_argument("column generation requires at least one initial column");
    }

    ColumnGenerationResult result{};
    for (std::size_t iteration = 0; iteration < config.maximum_iterations; ++iteration) {
        const auto master = master_solver(pool.columns());
        master.validate(target_count, spacecraft_count);
        ColumnGenerationIteration record{};
        record.iteration = iteration;
        record.columns_before = pool.columns().size();
        record.master_iterations = master.iterations;
        record.master_solve_seconds = master.solve_seconds;
        record.master_objective = master.objective;
        record.master_lower_bound = master.lower_bound;
        if (!master.feasible) {
            result.history.push_back(std::move(record));
            result.columns = pool.columns();
            return result;
        }
        result.master_feasible = true;
        result.master_lower_bound = master.lower_bound;

        result.incumbent = solve_exact_route_master(
            spacecraft_count,
            target_count,
            pool.columns(),
            required_targets
        );
        record.incumbent_cost = result.incumbent.cost;
        record.optimality_gap = column_generation_detail::gap(
            result.incumbent.cost,
            master.lower_bound
        );

        auto candidates = pricing_oracle(master.duals);
        record.candidate_columns = candidates.size();
        auto priced = negative_reduced_cost_columns(
            candidates,
            master.duals,
            config.reduced_cost_tolerance
        );
        record.negative_reduced_cost_columns = priced.size();
        if (!priced.empty()) {
            record.best_reduced_cost = priced.front().reduced_cost;
        }

        const auto addition_limit = std::min(
            config.maximum_columns_per_iteration,
            priced.size()
        );
        for (std::size_t index = 0; index < addition_limit; ++index) {
            if (pool.add(std::move(priced[index].column))) {
                ++record.columns_added;
            }
        }
        result.history.push_back(record);

        const auto gap_closed = result.incumbent.feasible
                                && record.optimality_gap
                                       <= column_generation_detail::gap_tolerance(
                                           result.incumbent.cost,
                                           config
                                       );
        if (record.columns_added == 0U) {
            result.converged = true;
            if (!gap_closed && result.incumbent.feasible) {
                // Pricing closure certifies the restricted-master relaxation; the exact
                // incumbent gap is retained rather than being silently labelled zero.
                result.converged = record.negative_reduced_cost_columns == 0U;
            }
            result.columns = pool.columns();
            return result;
        }
    }

    result.iteration_limit_reached = true;
    result.columns = pool.columns();
    result.incumbent = solve_exact_route_master(
        spacecraft_count,
        target_count,
        result.columns,
        required_targets
    );
    return result;
}

}  // namespace spacepdhcg::orbitweaver
