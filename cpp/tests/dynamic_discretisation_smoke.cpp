#include "spacepdhcg/orbitweaver/dynamic_discretisation.hpp"

#include <cmath>
#include <vector>

int main() {
    using spacepdhcg::orbitweaver::DynamicDiscretisationConfig;
    using spacepdhcg::orbitweaver::ScheduledArcEstimate;
    using spacepdhcg::orbitweaver::discover_time_discretisation;

    const auto oracle = [](
                            std::size_t from,
                            std::size_t to,
                            double departure,
                            double arrival
                        ) -> ScheduledArcEstimate {
        if (from != 0U || to != 1U || departure != 0.0 || arrival <= departure) {
            return {};
        }
        const auto error = arrival - 13.0;
        return ScheduledArcEstimate{
            true,
            1.0 + error * error,
            1.0,
            std::abs(error),
        };
    };

    DynamicDiscretisationConfig config{};
    config.maximum_iterations = 20U;
    config.maximum_epochs = 128U;
    config.maximum_new_epochs_per_iteration = 8U;
    config.absolute_gap_tolerance = 1.0e-5;
    config.relative_gap_tolerance = 0.0;
    config.minimum_interval = 1.0e-5;

    const auto result = discover_time_discretisation(
        2U,
        std::vector<double>{0.0, 10.0, 20.0},
        0U,
        0.0,
        1U,
        oracle,
        config
    );

    if (!result.converged || !result.route.has_value()) {
        return 1;
    }
    if (result.epochs.size() <= 3U || result.history.size() <= 1U) {
        return 2;
    }
    if (result.route->nominal_cost - result.route->lower_bound > 1.0e-5) {
        return 3;
    }

    bool inserted_off_grid_epoch{false};
    for (const auto& iteration : result.history) {
        if (!iteration.inserted_epochs.empty()) {
            inserted_off_grid_epoch = true;
            break;
        }
    }
    if (!inserted_off_grid_epoch) {
        return 4;
    }
    return 0;
}
