#pragma once

#include "spacepdhcg/orbitweaver/oracle.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

struct ServiceTarget {
    CircularOrbitTarget orbit;
    EpochWindow service_window;
    double service_duration{0.0};

    void validate() const {
        orbit.validate();
        service_window.validate();
        if (!std::isfinite(service_duration) || service_duration < 0.0) {
            throw std::invalid_argument("service duration must be finite and non-negative");
        }
    }
};

struct RouteRequest {
    CircularOrbitTarget start;
    std::vector<ServiceTarget> targets;
    SpacecraftResources spacecraft;
    double start_epoch{0.0};
    double end_epoch{0.0};
    std::size_t maximum_visits{0};
    std::size_t beam_width{64};
    double time_weight{0.0};
    double phase_tolerance{1.0e-8};

    void validate() const {
        start.validate();
        spacecraft.validate();
        if (!std::isfinite(start_epoch) || !std::isfinite(end_epoch) ||
            start_epoch > end_epoch) {
            throw std::invalid_argument("route epoch range is invalid");
        }
        if (maximum_visits == 0U || maximum_visits > targets.size()) {
            throw std::invalid_argument("maximum visits must lie within the target set");
        }
        if (beam_width == 0U) {
            throw std::invalid_argument("beam width must be positive");
        }
        if (!std::isfinite(time_weight) || time_weight < 0.0) {
            throw std::invalid_argument("route time weight must be non-negative");
        }
        if (!std::isfinite(phase_tolerance) || phase_tolerance <= 0.0) {
            throw std::invalid_argument("route phase tolerance must be positive");
        }
        std::vector<std::string> ids;
        ids.reserve(targets.size() + 1U);
        ids.push_back(start.id);
        for (const auto& target : targets) {
            target.validate();
            if (target.orbit.gravitational_parameter != start.gravitational_parameter) {
                throw std::invalid_argument("all route targets must orbit one central body");
            }
            ids.push_back(target.orbit.id);
        }
        std::sort(ids.begin(), ids.end());
        if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
            throw std::invalid_argument("route target identifiers must be unique");
        }
    }
};

struct RouteLeg {
    ArcResult arc;
    double service_start{0.0};
    double service_end{0.0};
    double propellant_before{0.0};
    double propellant_after{0.0};
};

struct RouteResult {
    bool feasible{false};
    std::vector<std::string> visited_targets;
    std::vector<RouteLeg> legs;
    double total_delta_v{0.0};
    double propellant_used{0.0};
    double finish_epoch{0.0};
    double score{std::numeric_limits<double>::infinity()};
    std::string status;
};

class BeamRouter {
  public:
    explicit BeamRouter(const ArcOracle& oracle) : oracle_(&oracle) {}

    [[nodiscard]] RouteResult solve(const RouteRequest& request) const {
        request.validate();
        Node initial;
        initial.current = request.start;
        initial.resources = request.spacecraft;
        initial.epoch = request.start_epoch;
        initial.visited.assign(request.targets.size(), false);
        initial.score = 0.0;

        std::vector<Node> beam{initial};
        std::vector<Node> terminal;
        for (std::size_t depth = 0; depth < request.maximum_visits; ++depth) {
            std::vector<Node> expanded;
            for (const auto& node : beam) {
                const auto children = expand(node, request);
                expanded.insert(expanded.end(), children.begin(), children.end());
            }
            if (expanded.empty()) {
                terminal.insert(terminal.end(), beam.begin(), beam.end());
                break;
            }
            std::sort(expanded.begin(), expanded.end(), node_less);
            if (expanded.size() > request.beam_width) {
                expanded.resize(request.beam_width);
            }
            beam = std::move(expanded);
        }
        terminal.insert(terminal.end(), beam.begin(), beam.end());
        if (terminal.empty()) {
            return RouteResult{
                false,
                {},
                {},
                0.0,
                0.0,
                request.start_epoch,
                std::numeric_limits<double>::infinity(),
                "no route state was generated",
            };
        }
        std::sort(terminal.begin(), terminal.end(), node_less);
        const auto best = std::max_element(
            terminal.begin(),
            terminal.end(),
            [](const Node& left, const Node& right) {
                if (left.route.size() != right.route.size()) {
                    return left.route.size() < right.route.size();
                }
                return node_less(right, left);
            }
        );
        return result_from(*best, request);
    }

  private:
    struct Node {
        CircularOrbitTarget current;
        SpacecraftResources resources;
        double epoch{0.0};
        std::vector<bool> visited;
        std::vector<std::size_t> route;
        std::vector<RouteLeg> legs;
        double total_delta_v{0.0};
        double score{0.0};
    };

    [[nodiscard]] std::vector<Node> expand(
        const Node& node,
        const RouteRequest& request
    ) const {
        std::vector<ArcRequest> arc_requests;
        std::vector<std::size_t> target_indices;
        for (std::size_t index = 0; index < request.targets.size(); ++index) {
            if (node.visited[index]) {
                continue;
            }
            const auto& target = request.targets[index];
            const EpochWindow departure_window{node.epoch, request.end_epoch};
            const auto arrival_window = target.service_window.intersect(
                EpochWindow{node.epoch, request.end_epoch}
            );
            if (!arrival_window.has_value()) {
                continue;
            }
            arc_requests.push_back(ArcRequest{
                node.current,
                target.orbit,
                departure_window,
                *arrival_window,
                node.resources,
                ArcFidelity::analytical,
                request.phase_tolerance,
            });
            target_indices.push_back(index);
        }
        const auto results = oracle_->evaluate_batch(arc_requests);
        std::vector<Node> children;
        for (std::size_t result_index = 0; result_index < results.size(); ++result_index) {
            const auto& arc = results[result_index];
            const auto target_index = target_indices[result_index];
            const auto& target = request.targets[target_index];
            if (!arc.feasible || !std::isfinite(arc.propellant_required) ||
                arc.propellant_required > node.resources.propellant_mass) {
                continue;
            }
            const double service_start = std::max(arc.arrival_epoch, target.service_window.earliest);
            const double service_end = service_start + target.service_duration;
            if (service_end > target.service_window.latest || service_end > request.end_epoch) {
                continue;
            }

            Node child = node;
            child.current = target.orbit;
            child.epoch = service_end;
            child.visited[target_index] = true;
            child.route.push_back(target_index);
            const double propellant_before = child.resources.propellant_mass;
            child.resources.propellant_mass -= arc.propellant_required;
            child.total_delta_v += arc.delta_v;
            child.score = child.total_delta_v + request.time_weight *
                (child.epoch - request.start_epoch);
            child.legs.push_back(RouteLeg{
                arc,
                service_start,
                service_end,
                propellant_before,
                child.resources.propellant_mass,
            });
            children.push_back(std::move(child));
        }
        return children;
    }

    [[nodiscard]] static bool node_less(const Node& left, const Node& right) {
        if (left.score != right.score) {
            return left.score < right.score;
        }
        if (left.epoch != right.epoch) {
            return left.epoch < right.epoch;
        }
        return left.route < right.route;
    }

    [[nodiscard]] static RouteResult result_from(
        const Node& node,
        const RouteRequest& request
    ) {
        std::vector<std::string> ids;
        ids.reserve(node.route.size());
        for (const auto index : node.route) {
            ids.push_back(request.targets[index].orbit.id);
        }
        return RouteResult{
            node.route.size() == request.maximum_visits,
            std::move(ids),
            node.legs,
            node.total_delta_v,
            request.spacecraft.propellant_mass - node.resources.propellant_mass,
            node.epoch,
            node.score,
            node.route.size() == request.maximum_visits
                ? "requested visit count achieved"
                : "best partial route returned",
        };
    }

    const ArcOracle* oracle_{nullptr};
};

}  // namespace spacepdhcg::orbitweaver
