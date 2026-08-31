#include "spacepdhcg/orbitweaver/low_thrust_screening.hpp"
#include "spacepdhcg/orbitweaver/time_expanded_graph.hpp"

#include <cmath>
#include <cstddef>
#include <set>
#include <vector>

int main() {
    using spacepdhcg::orbitweaver::ScheduledArcEstimate;
    using spacepdhcg::orbitweaver::build_time_expanded_graph;
    using spacepdhcg::orbitweaver::circular_speed_change_lower_bound;
    using spacepdhcg::orbitweaver::edelbaum_circular_transfer;
    using spacepdhcg::orbitweaver::minimum_cost_elementary_route;
    using spacepdhcg::orbitweaver::optimistic_route_lower_bound;

    const std::vector<double> epochs{0.0, 10.0, 20.0, 30.0, 40.0};
    const auto graph = build_time_expanded_graph(
        4U,
        epochs,
        [](std::size_t from, std::size_t to, double departure, double arrival) {
            const auto distance = std::abs(static_cast<double>(to) - static_cast<double>(from));
            const auto duration = arrival - departure;
            const auto nominal = distance + 0.01 * duration + 0.001 * departure;
            return ScheduledArcEstimate{true, nominal, 0.5 * nominal, distance};
        }
    );
    if (graph.nodes().size() != 20U || graph.arcs().size() != 120U) {
        return 1;
    }
    const auto route = minimum_cost_elementary_route(graph, 0U, 0U, 3U);
    if (!route.has_value() || route->node_sequence.size() != 4U
        || route->arc_sequence.size() != 3U || route->lower_bound > route->nominal_cost) {
        return 2;
    }
    std::set<std::size_t> targets{};
    for (const auto node : route->node_sequence) {
        targets.insert(graph.nodes()[node].target);
    }
    if (targets.size() != route->node_sequence.size()) {
        return 3;
    }
    const auto bound = optimistic_route_lower_bound(graph, graph.node_id(0U, 0U), 3U);
    if (!std::isfinite(bound) || bound > route->lower_bound + 1.0e-12) {
        return 4;
    }

    const auto edelbaum = edelbaum_circular_transfer(
        7'000.0,
        8'000.0,
        0.1,
        398'600.4418,
        1.0e-6,
        500.0,
        2'000.0
    );
    if (!(edelbaum.delta_v > 0.0 && edelbaum.transfer_time > 0.0
          && edelbaum.propellant > 0.0 && edelbaum.propellant < 500.0)) {
        return 5;
    }
    const auto speed_bound = circular_speed_change_lower_bound(
        7'000.0,
        8'000.0,
        398'600.4418
    );
    if (!(speed_bound > 0.0 && speed_bound <= edelbaum.delta_v)) {
        return 6;
    }
    return 0;
}
