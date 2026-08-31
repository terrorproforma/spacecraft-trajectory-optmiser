#include "spacepdhcg/orbitweaver/beam_search.hpp"

#include <cmath>
#include <cstddef>
#include <set>

int main() {
    using spacepdhcg::orbitweaver::ArcEstimate;
    using spacepdhcg::orbitweaver::BeamSearchConfig;
    using spacepdhcg::orbitweaver::beam_search;

    const BeamSearchConfig config{
        .beam_width = 8U,
        .destinations_to_visit = 3U,
        .minimum_mass = 800.0,
        .maximum_epoch = 1'000.0,
    };
    const auto oracle = [](std::size_t from, std::size_t to, double epoch, double mass) {
        const auto distance = std::abs(static_cast<double>(to) - static_cast<double>(from));
        const auto delta_v = 5.0 + distance;
        const auto duration = 10.0 + 2.0 * distance;
        const auto cost = delta_v + 1.0e-3 * epoch;
        return ArcEstimate{
            true,
            cost,
            duration,
            delta_v,
            mass - 10.0 * delta_v,
            0.25 * distance,
        };
    };

    const auto routes = beam_search(6U, 0U, 100.0, 1'200.0, config, oracle);
    if (routes.empty() || routes.front().sequence.size() != 4U) {
        return 1;
    }
    for (const auto& route : routes) {
        const std::set<std::size_t> unique(route.sequence.begin(), route.sequence.end());
        if (unique.size() != route.sequence.size() || route.mass < config.minimum_mass) {
            return 2;
        }
    }
    for (std::size_t index = 1; index < routes.size(); ++index) {
        if (routes[index].ranking_score < routes[index - 1U].ranking_score) {
            return 3;
        }
    }
    return 0;
}
