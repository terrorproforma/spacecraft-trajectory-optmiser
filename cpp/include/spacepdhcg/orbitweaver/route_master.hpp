#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

struct RouteColumn {
    std::size_t identifier{0U};
    std::size_t spacecraft{0U};
    std::vector<std::size_t> targets{};
    double cost{0.0};
    double lower_bound{0.0};
    double propellant{0.0};

    void validate(std::size_t spacecraft_count, std::size_t target_count) const {
        if (spacecraft >= spacecraft_count) {
            throw std::invalid_argument("route column spacecraft index is outside the fleet");
        }
        for (const auto value : {cost, lower_bound, propellant}) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument("route column metrics must be finite and non-negative");
            }
        }
        if (lower_bound > cost) {
            throw std::invalid_argument("route column lower bound may not exceed its cost");
        }
        std::size_t previous{0U};
        for (std::size_t index = 0; index < targets.size(); ++index) {
            if (targets[index] >= target_count) {
                throw std::invalid_argument("route column target is outside the master problem");
            }
            if (index > 0U && targets[index] <= previous) {
                throw std::invalid_argument("route column targets must be sorted and unique");
            }
            previous = targets[index];
        }
    }
};

struct RouteMasterSolution {
    bool feasible{false};
    std::vector<std::size_t> selected_column_indices{};
    double cost{std::numeric_limits<double>::infinity()};
    double lower_bound{std::numeric_limits<double>::infinity()};
    double propellant{std::numeric_limits<double>::infinity()};
    std::size_t explored_nodes{0U};
};

struct RouteMasterDualPrices {
    std::vector<double> target{};
    std::vector<double> spacecraft{};

    void validate(std::size_t target_count, std::size_t spacecraft_count) const {
        if (target.size() != target_count || spacecraft.size() != spacecraft_count) {
            throw std::invalid_argument("route-master dual prices have the wrong dimensions");
        }
        for (const auto price : target) {
            if (!std::isfinite(price)) {
                throw std::invalid_argument("target dual prices must be finite");
            }
        }
        for (const auto price : spacecraft) {
            if (!std::isfinite(price)) {
                throw std::invalid_argument("spacecraft dual prices must be finite");
            }
        }
    }
};

struct PricedRouteColumn {
    RouteColumn column{};
    double reduced_cost{0.0};
};

class RouteColumnPool {
  public:
    RouteColumnPool(std::size_t spacecraft_count, std::size_t target_count)
        : spacecraft_count_(spacecraft_count), target_count_(target_count) {
        if (spacecraft_count_ == 0U || target_count_ == 0U) {
            throw std::invalid_argument("route column pool dimensions must be positive");
        }
    }

    [[nodiscard]] std::size_t spacecraft_count() const noexcept {
        return spacecraft_count_;
    }
    [[nodiscard]] std::size_t target_count() const noexcept { return target_count_; }
    [[nodiscard]] const std::vector<RouteColumn>& columns() const noexcept { return columns_; }

    /// Add a route unless an equal-coverage route for the same spacecraft dominates it.
    /// Returns true when the retained pool changes.
    bool add(RouteColumn column, double tolerance = 1.0e-12) {
        column.validate(spacecraft_count_, target_count_);
        if (!std::isfinite(tolerance) || tolerance < 0.0) {
            throw std::invalid_argument("column dominance tolerance must be finite and non-negative");
        }
        const auto key = std::pair{column.spacecraft, target_mask(column.targets)};
        const auto iterator = lookup_.find(key);
        if (iterator == lookup_.end()) {
            lookup_.emplace(key, columns_.size());
            columns_.push_back(std::move(column));
            return true;
        }
        auto& incumbent = columns_[iterator->second];
        const auto candidate_key = std::tie(
            column.cost,
            column.lower_bound,
            column.propellant,
            column.identifier
        );
        const auto incumbent_key = std::tie(
            incumbent.cost,
            incumbent.lower_bound,
            incumbent.propellant,
            incumbent.identifier
        );
        if (column.cost + tolerance < incumbent.cost
            || (std::abs(column.cost - incumbent.cost) <= tolerance
                && candidate_key < incumbent_key)) {
            incumbent = std::move(column);
            return true;
        }
        return false;
    }

  private:
    std::size_t spacecraft_count_{0U};
    std::size_t target_count_{0U};
    std::vector<RouteColumn> columns_{};
    std::map<std::pair<std::size_t, std::uint64_t>, std::size_t> lookup_{};

    [[nodiscard]] std::uint64_t target_mask(
        const std::vector<std::size_t>& targets
    ) const {
        if (target_count_ > 64U) {
            throw std::invalid_argument("native exact route pools support at most 64 targets");
        }
        std::uint64_t mask{0U};
        for (const auto target : targets) {
            mask |= std::uint64_t{1U} << target;
        }
        return mask;
    }
};

inline double route_reduced_cost(
    const RouteColumn& column,
    const RouteMasterDualPrices& duals
) {
    column.validate(duals.spacecraft.size(), duals.target.size());
    double reduced = column.cost - duals.spacecraft[column.spacecraft];
    for (const auto target : column.targets) {
        reduced -= duals.target[target];
    }
    return reduced;
}

inline std::vector<PricedRouteColumn> negative_reduced_cost_columns(
    const std::vector<RouteColumn>& candidates,
    const RouteMasterDualPrices& duals,
    double tolerance = 1.0e-9
) {
    duals.validate(duals.target.size(), duals.spacecraft.size());
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::invalid_argument("pricing tolerance must be finite and non-negative");
    }
    std::vector<PricedRouteColumn> result{};
    for (const auto& column : candidates) {
        const auto reduced = route_reduced_cost(column, duals);
        if (reduced < -tolerance) {
            result.push_back(PricedRouteColumn{column, reduced});
        }
    }
    std::sort(
        result.begin(),
        result.end(),
        [](const auto& left, const auto& right) {
            return std::tie(
                       left.reduced_cost,
                       left.column.cost,
                       left.column.spacecraft,
                       left.column.identifier
                   )
                   < std::tie(
                       right.reduced_cost,
                       right.column.cost,
                       right.column.spacecraft,
                       right.column.identifier
                   );
        }
    );
    return result;
}

namespace route_master_detail {

inline std::uint64_t mask(const std::vector<std::size_t>& targets) {
    std::uint64_t result{0U};
    for (const auto target : targets) {
        result |= std::uint64_t{1U} << target;
    }
    return result;
}

struct SearchState {
    const std::vector<RouteColumn>* columns{nullptr};
    std::vector<std::uint64_t> masks{};
    std::vector<std::vector<std::size_t>> target_columns{};
    std::uint64_t required_mask{0U};
    RouteMasterSolution best{};
};

inline std::optional<std::size_t> branching_target(
    const SearchState& state,
    std::uint64_t covered,
    std::uint64_t used_spacecraft
) {
    std::optional<std::size_t> selected{};
    std::size_t selected_count = std::numeric_limits<std::size_t>::max();
    for (std::size_t target = 0; target < state.target_columns.size(); ++target) {
        const auto bit = std::uint64_t{1U} << target;
        if ((state.required_mask & bit) == 0U || (covered & bit) != 0U) {
            continue;
        }
        std::size_t count{0U};
        for (const auto column_index : state.target_columns[target]) {
            const auto& column = state.columns->at(column_index);
            const auto spacecraft_bit = std::uint64_t{1U} << column.spacecraft;
            if ((used_spacecraft & spacecraft_bit) == 0U
                && (state.masks[column_index] & covered) == 0U) {
                ++count;
            }
        }
        if (count < selected_count) {
            selected = target;
            selected_count = count;
        }
    }
    return selected;
}

inline void search(
    SearchState& state,
    std::uint64_t covered,
    std::uint64_t used_spacecraft,
    double cost,
    double lower_bound,
    double propellant,
    std::vector<std::size_t>& selected
) {
    ++state.best.explored_nodes;
    if (cost >= state.best.cost) {
        return;
    }
    if ((covered & state.required_mask) == state.required_mask) {
        state.best.feasible = true;
        state.best.cost = cost;
        state.best.lower_bound = lower_bound;
        state.best.propellant = propellant;
        state.best.selected_column_indices = selected;
        return;
    }
    const auto target = branching_target(state, covered, used_spacecraft);
    if (!target.has_value()) {
        return;
    }
    auto candidates = state.target_columns[*target];
    std::sort(
        candidates.begin(),
        candidates.end(),
        [&state](std::size_t left, std::size_t right) {
            const auto& left_column = state.columns->at(left);
            const auto& right_column = state.columns->at(right);
            return std::tie(
                       left_column.cost,
                       left_column.lower_bound,
                       left_column.spacecraft,
                       left_column.identifier
                   )
                   < std::tie(
                       right_column.cost,
                       right_column.lower_bound,
                       right_column.spacecraft,
                       right_column.identifier
                   );
        }
    );
    for (const auto column_index : candidates) {
        const auto& column = state.columns->at(column_index);
        const auto spacecraft_bit = std::uint64_t{1U} << column.spacecraft;
        const auto column_mask = state.masks[column_index];
        if ((used_spacecraft & spacecraft_bit) != 0U || (covered & column_mask) != 0U) {
            continue;
        }
        selected.push_back(column_index);
        search(
            state,
            covered | column_mask,
            used_spacecraft | spacecraft_bit,
            cost + column.cost,
            lower_bound + column.lower_bound,
            propellant + column.propellant,
            selected
        );
        selected.pop_back();
    }
}

}  // namespace route_master_detail

/// Exact small-instance set-partitioning truth model for multi-spacecraft route columns.
///
/// Every required target is covered exactly once and each spacecraft contributes at most one
/// route. The exponential search is intentionally bounded to 64 targets and 64 spacecraft; it
/// validates future MILP/column-generation masters rather than replacing them at large scale.
inline RouteMasterSolution solve_exact_route_master(
    std::size_t spacecraft_count,
    std::size_t target_count,
    const std::vector<RouteColumn>& columns,
    std::vector<std::size_t> required_targets = {}
) {
    if (spacecraft_count == 0U || target_count == 0U) {
        throw std::invalid_argument("route master dimensions must be positive");
    }
    if (spacecraft_count > 64U || target_count > 64U) {
        throw std::invalid_argument("exact native route master supports at most 64 axes");
    }
    if (required_targets.empty()) {
        required_targets.resize(target_count);
        for (std::size_t target = 0; target < target_count; ++target) {
            required_targets[target] = target;
        }
    }
    std::sort(required_targets.begin(), required_targets.end());
    if (std::adjacent_find(required_targets.begin(), required_targets.end())
        != required_targets.end()) {
        throw std::invalid_argument("required route-master targets must be unique");
    }

    route_master_detail::SearchState state{};
    state.columns = &columns;
    state.masks.reserve(columns.size());
    state.target_columns.resize(target_count);
    for (std::size_t column_index = 0; column_index < columns.size(); ++column_index) {
        columns[column_index].validate(spacecraft_count, target_count);
        const auto column_mask = route_master_detail::mask(columns[column_index].targets);
        state.masks.push_back(column_mask);
        for (const auto target : columns[column_index].targets) {
            state.target_columns[target].push_back(column_index);
        }
    }
    for (const auto target : required_targets) {
        if (target >= target_count) {
            throw std::invalid_argument("required target is outside the route master");
        }
        state.required_mask |= std::uint64_t{1U} << target;
    }
    std::vector<std::size_t> selected{};
    route_master_detail::search(state, 0U, 0U, 0.0, 0.0, 0.0, selected);
    return state.best;
}

}  // namespace spacepdhcg::orbitweaver
