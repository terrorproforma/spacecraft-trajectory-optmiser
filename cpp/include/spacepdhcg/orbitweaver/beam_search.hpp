#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

struct ArcEstimate {
    bool feasible{false};
    double cost{std::numeric_limits<double>::infinity()};
    double duration{0.0};
    double delta_v{0.0};
    double final_mass{0.0};
    double lower_bound{0.0};
};

struct RouteState {
    std::vector<std::size_t> sequence{};
    std::vector<bool> visited{};
    double epoch{0.0};
    double mass{0.0};
    double accumulated_cost{0.0};
    double accumulated_delta_v{0.0};
    double ranking_score{0.0};

    [[nodiscard]] std::size_t current_target() const {
        if (sequence.empty()) {
            throw std::logic_error("route sequence may not be empty");
        }
        return sequence.back();
    }
};

struct BeamSearchConfig {
    std::size_t beam_width{64U};
    std::size_t destinations_to_visit{1U};
    double minimum_mass{0.0};
    double maximum_epoch{std::numeric_limits<double>::infinity()};

    void validate(std::size_t target_count) const {
        if (target_count == 0U) {
            throw std::invalid_argument("beam search requires at least one target");
        }
        if (beam_width == 0U) {
            throw std::invalid_argument("beam width must be positive");
        }
        if (destinations_to_visit >= target_count) {
            throw std::invalid_argument(
                "destinations_to_visit must leave room for the starting target"
            );
        }
        if (!std::isfinite(minimum_mass) || minimum_mass < 0.0) {
            throw std::invalid_argument("minimum mass must be finite and non-negative");
        }
        if (std::isnan(maximum_epoch)) {
            throw std::invalid_argument("maximum epoch may be infinite but not NaN");
        }
    }
};

namespace detail {

inline bool route_less(const RouteState& left, const RouteState& right) {
    return std::tie(left.ranking_score, left.accumulated_cost, left.epoch, left.sequence)
           < std::tie(right.ranking_score, right.accumulated_cost, right.epoch, right.sequence);
}

inline void validate_arc(const ArcEstimate& arc) {
    if (!arc.feasible) {
        return;
    }
    for (const auto value : {
             arc.cost,
             arc.duration,
             arc.delta_v,
             arc.final_mass,
             arc.lower_bound,
         }) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("a feasible arc estimate must contain finite values");
        }
    }
    if (arc.cost < 0.0 || arc.duration <= 0.0 || arc.delta_v < 0.0 || arc.final_mass < 0.0
        || arc.lower_bound < 0.0) {
        throw std::runtime_error("a feasible arc estimate contains an invalid negative value");
    }
}

}  // namespace detail

/// Deterministic route expansion over a time- and mass-dependent arc oracle.
///
/// The oracle must be callable as
/// `ArcEstimate(from, to, departure_epoch, current_mass)`. The result ordering is stable across
/// platforms because ties are broken by accumulated cost, epoch, and the full target sequence.
template <typename ArcOracle>
std::vector<RouteState> beam_search(
    std::size_t target_count,
    std::size_t start_target,
    double start_epoch,
    double initial_mass,
    const BeamSearchConfig& config,
    ArcOracle&& oracle
) {
    config.validate(target_count);
    if (start_target >= target_count) {
        throw std::invalid_argument("start target is outside the target set");
    }
    if (!std::isfinite(start_epoch) || !std::isfinite(initial_mass) || initial_mass <= 0.0) {
        throw std::invalid_argument("start epoch and initial mass must be finite and valid");
    }
    if (initial_mass < config.minimum_mass) {
        throw std::invalid_argument("initial mass is below the configured route reserve");
    }

    RouteState initial{};
    initial.sequence.push_back(start_target);
    initial.visited.assign(target_count, false);
    initial.visited[start_target] = true;
    initial.epoch = start_epoch;
    initial.mass = initial_mass;
    std::vector<RouteState> beam{std::move(initial)};

    for (std::size_t depth = 0; depth < config.destinations_to_visit; ++depth) {
        std::vector<RouteState> expanded{};
        expanded.reserve(beam.size() * std::max<std::size_t>(1U, target_count - depth - 1U));
        for (const auto& route : beam) {
            for (std::size_t target = 0; target < target_count; ++target) {
                if (route.visited[target]) {
                    continue;
                }
                const auto arc = oracle(
                    route.current_target(),
                    target,
                    route.epoch,
                    route.mass
                );
                detail::validate_arc(arc);
                if (!arc.feasible || arc.final_mass < config.minimum_mass) {
                    continue;
                }
                const auto arrival_epoch = route.epoch + arc.duration;
                if (arrival_epoch > config.maximum_epoch) {
                    continue;
                }
                auto candidate = route;
                candidate.sequence.push_back(target);
                candidate.visited[target] = true;
                candidate.epoch = arrival_epoch;
                candidate.mass = arc.final_mass;
                candidate.accumulated_cost += arc.cost;
                candidate.accumulated_delta_v += arc.delta_v;
                candidate.ranking_score = candidate.accumulated_cost + arc.lower_bound;
                expanded.push_back(std::move(candidate));
            }
        }
        if (expanded.empty()) {
            return {};
        }
        std::sort(expanded.begin(), expanded.end(), detail::route_less);
        if (expanded.size() > config.beam_width) {
            expanded.resize(config.beam_width);
        }
        beam = std::move(expanded);
    }
    std::sort(beam.begin(), beam.end(), detail::route_less);
    return beam;
}

}  // namespace spacepdhcg::orbitweaver
