#pragma once

#include "spacepdhcg/orbitweaver/time_expanded_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

struct DynamicDiscretisationConfig {
    std::size_t maximum_iterations{16U};
    std::size_t maximum_epochs{512U};
    std::size_t maximum_new_epochs_per_iteration{16U};
    double absolute_gap_tolerance{1.0e-4};
    double relative_gap_tolerance{1.0e-4};
    double minimum_interval{1.0e-6};

    void validate() const {
        if (maximum_iterations == 0U || maximum_epochs < 2U
            || maximum_new_epochs_per_iteration == 0U) {
            throw std::invalid_argument("dynamic discretisation iteration limits must be positive");
        }
        for (const auto value : {
                 absolute_gap_tolerance,
                 relative_gap_tolerance,
                 minimum_interval,
             }) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(
                    "dynamic discretisation tolerances must be finite and non-negative"
                );
            }
        }
        if (minimum_interval <= 0.0) {
            throw std::invalid_argument("dynamic discretisation minimum interval must be positive");
        }
    }
};

struct DynamicDiscretisationIteration {
    std::size_t iteration{0U};
    std::size_t epoch_count{0U};
    bool route_found{false};
    double nominal_cost{std::numeric_limits<double>::infinity()};
    double lower_bound{std::numeric_limits<double>::infinity()};
    double optimality_gap{std::numeric_limits<double>::infinity()};
    std::vector<double> inserted_epochs{};
};

struct DynamicDiscretisationResult {
    std::vector<double> epochs{};
    std::optional<ScheduledRoute> route{};
    std::vector<DynamicDiscretisationIteration> history{};
    bool converged{false};
};

namespace dynamic_discretisation_detail {

inline void validate_epochs(const std::vector<double>& epochs) {
    if (epochs.size() < 2U) {
        throw std::invalid_argument("dynamic discretisation requires at least two epochs");
    }
    for (std::size_t index = 0; index < epochs.size(); ++index) {
        if (!std::isfinite(epochs[index])) {
            throw std::invalid_argument("dynamic discretisation epochs must be finite");
        }
        if (index > 0U && epochs[index] <= epochs[index - 1U]) {
            throw std::invalid_argument("dynamic discretisation epochs must be strictly increasing");
        }
    }
}

inline std::size_t epoch_index(const std::vector<double>& epochs, double epoch) {
    const auto iterator = std::lower_bound(epochs.begin(), epochs.end(), epoch);
    if (iterator == epochs.end() || *iterator != epoch) {
        throw std::invalid_argument("route start epoch must be present in the discretisation");
    }
    return static_cast<std::size_t>(std::distance(epochs.begin(), iterator));
}

inline double route_gap(const ScheduledRoute& route) noexcept {
    return std::max(0.0, route.nominal_cost - route.lower_bound);
}

inline double gap_tolerance(
    const ScheduledRoute& route,
    const DynamicDiscretisationConfig& config
) noexcept {
    return config.absolute_gap_tolerance
           + config.relative_gap_tolerance * std::max(1.0, std::abs(route.nominal_cost));
}

inline void propose_adjacent_midpoints(
    const std::vector<double>& epochs,
    std::size_t epoch,
    double score,
    const DynamicDiscretisationConfig& config,
    std::map<double, double>& candidates
) {
    const auto add_interval = [&](double lower, double upper) {
        if (upper - lower < 2.0 * config.minimum_interval) {
            return;
        }
        const auto midpoint = std::midpoint(lower, upper);
        const auto [iterator, inserted] = candidates.emplace(midpoint, score);
        if (!inserted) {
            iterator->second = std::max(iterator->second, score);
        }
    };
    if (epoch > 0U) {
        add_interval(epochs[epoch - 1U], epochs[epoch]);
    }
    if (epoch + 1U < epochs.size()) {
        add_interval(epochs[epoch], epochs[epoch + 1U]);
    }
}

inline std::vector<double> select_candidates(
    const std::map<double, double>& candidates,
    const DynamicDiscretisationConfig& config,
    std::size_t current_epoch_count
) {
    std::vector<std::pair<double, double>> ordered{};
    ordered.reserve(candidates.size());
    for (const auto& [epoch, score] : candidates) {
        ordered.emplace_back(score, epoch);
    }
    std::sort(
        ordered.begin(),
        ordered.end(),
        [](const auto& left, const auto& right) {
            if (left.first != right.first) {
                return left.first > right.first;
            }
            return left.second < right.second;
        }
    );
    const auto capacity = config.maximum_epochs > current_epoch_count
                              ? config.maximum_epochs - current_epoch_count
                              : 0U;
    const auto count = std::min(
        {ordered.size(), config.maximum_new_epochs_per_iteration, capacity}
    );
    std::vector<double> selected{};
    selected.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        selected.push_back(ordered[index].second);
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

inline std::vector<double> largest_interval_midpoints(
    const std::vector<double>& epochs,
    const DynamicDiscretisationConfig& config
) {
    std::vector<std::pair<double, double>> intervals{};
    intervals.reserve(epochs.size() - 1U);
    for (std::size_t index = 0; index + 1U < epochs.size(); ++index) {
        const auto width = epochs[index + 1U] - epochs[index];
        if (width >= 2.0 * config.minimum_interval) {
            intervals.emplace_back(width, std::midpoint(epochs[index], epochs[index + 1U]));
        }
    }
    std::sort(
        intervals.begin(),
        intervals.end(),
        [](const auto& left, const auto& right) {
            if (left.first != right.first) {
                return left.first > right.first;
            }
            return left.second < right.second;
        }
    );
    const auto capacity = config.maximum_epochs > epochs.size()
                              ? config.maximum_epochs - epochs.size()
                              : 0U;
    const auto count = std::min(
        {intervals.size(), config.maximum_new_epochs_per_iteration, capacity}
    );
    std::vector<double> selected{};
    selected.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        selected.push_back(intervals[index].second);
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

inline void insert_epochs(std::vector<double>& epochs, const std::vector<double>& additions) {
    epochs.insert(epochs.end(), additions.begin(), additions.end());
    std::sort(epochs.begin(), epochs.end());
    epochs.erase(std::unique(epochs.begin(), epochs.end()), epochs.end());
}

}  // namespace dynamic_discretisation_detail

/// Route-driven dynamic discretisation discovery for a moving-target graph.
///
/// The scheduled oracle must provide valid lower bounds. Selected-route arcs with a material
/// nominal/lower-bound gap trigger midpoint insertion around their departure and arrival nodes.
/// When no route exists, the widest temporal intervals are bisected to recover connectivity.
template <typename ScheduledOracle>
DynamicDiscretisationResult discover_time_discretisation(
    std::size_t target_count,
    std::vector<double> initial_epochs,
    std::size_t start_target,
    double start_epoch,
    std::size_t required_legs,
    ScheduledOracle&& oracle,
    DynamicDiscretisationConfig config = {}
) {
    config.validate();
    dynamic_discretisation_detail::validate_epochs(initial_epochs);
    if (initial_epochs.size() > config.maximum_epochs) {
        throw std::invalid_argument("initial epoch count exceeds the configured maximum");
    }

    DynamicDiscretisationResult result{};
    result.epochs = std::move(initial_epochs);
    for (std::size_t iteration = 0; iteration < config.maximum_iterations; ++iteration) {
        const auto start_index = dynamic_discretisation_detail::epoch_index(
            result.epochs,
            start_epoch
        );
        const auto graph = build_time_expanded_graph(
            target_count,
            result.epochs,
            oracle
        );
        auto route = minimum_cost_elementary_route(
            graph,
            start_target,
            start_index,
            required_legs
        );
        DynamicDiscretisationIteration record{};
        record.iteration = iteration;
        record.epoch_count = result.epochs.size();
        record.route_found = route.has_value();

        std::vector<double> additions{};
        if (!route.has_value()) {
            additions = dynamic_discretisation_detail::largest_interval_midpoints(
                result.epochs,
                config
            );
        } else {
            record.nominal_cost = route->nominal_cost;
            record.lower_bound = route->lower_bound;
            record.optimality_gap = dynamic_discretisation_detail::route_gap(*route);
            result.route = route;
            if (record.optimality_gap
                <= dynamic_discretisation_detail::gap_tolerance(*route, config)) {
                result.converged = true;
                result.history.push_back(std::move(record));
                return result;
            }

            std::map<double, double> candidates{};
            for (const auto arc_index : route->arc_sequence) {
                const auto& arc = graph.arcs().at(arc_index);
                const auto local_gap = std::max(0.0, arc.nominal_cost - arc.lower_bound);
                if (local_gap <= 0.0) {
                    continue;
                }
                const auto& departure = graph.nodes().at(arc.from);
                const auto& arrival = graph.nodes().at(arc.to);
                dynamic_discretisation_detail::propose_adjacent_midpoints(
                    result.epochs,
                    departure.epoch_index,
                    local_gap,
                    config,
                    candidates
                );
                dynamic_discretisation_detail::propose_adjacent_midpoints(
                    result.epochs,
                    arrival.epoch_index,
                    local_gap,
                    config,
                    candidates
                );
            }
            additions = dynamic_discretisation_detail::select_candidates(
                candidates,
                config,
                result.epochs.size()
            );
        }

        record.inserted_epochs = additions;
        result.history.push_back(std::move(record));
        if (additions.empty()) {
            return result;
        }
        dynamic_discretisation_detail::insert_epochs(result.epochs, additions);
    }

    const auto start_index = dynamic_discretisation_detail::epoch_index(
        result.epochs,
        start_epoch
    );
    const auto final_graph = build_time_expanded_graph(target_count, result.epochs, oracle);
    result.route = minimum_cost_elementary_route(
        final_graph,
        start_target,
        start_index,
        required_legs
    );
    if (result.route.has_value()) {
        result.converged = dynamic_discretisation_detail::route_gap(*result.route)
                           <= dynamic_discretisation_detail::gap_tolerance(
                               *result.route,
                               config
                           );
    }
    return result;
}

}  // namespace spacepdhcg::orbitweaver
