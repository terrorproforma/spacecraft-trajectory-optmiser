#include "spacepdhcg/orbitweaver/oracle.hpp"
#include "spacepdhcg/orbitweaver/routing.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <set>
#include <vector>

int main() {
    using namespace spacepdhcg::orbitweaver;

    constexpr double earth_mu = 3.986004418e14;
    CircularOrbitTarget source{"source", 7.0e6, 0.0, earth_mu};
    CircularOrbitTarget target{"target", 8.0e6, 0.0, earth_mu};
    const auto transfer = hohmann_transfer(source.radius, target.radius, earth_mu);
    constexpr double departure = 1'000.0;
    target.phase_at_epoch_zero =
        source.phase(departure) + std::numbers::pi -
        target.mean_motion() * (departure + transfer.flight_time);
    const SpacecraftResources resources{500.0, 500.0, 1'500.0};
    const ArcRequest request{
        source,
        target,
        EpochWindow{departure, departure},
        EpochWindow{departure + transfer.flight_time, departure + transfer.flight_time},
        resources,
        ArcFidelity::analytical,
        1.0e-7,
    };
    const AnalyticalCircularOracle oracle;
    const auto result = oracle.evaluate(request);
    if (!result.feasible || std::abs(result.departure_epoch - departure) > 1.0e-7 ||
        std::abs(result.phase_error) > 1.0e-7 || result.delta_v <= 0.0 ||
        result.propellant_required <= 0.0 || result.warm_start_token.empty()) {
        return 1;
    }
    const std::vector<ArcRequest> batch{request, request};
    const auto batch_results = oracle.evaluate_batch(batch);
    if (batch_results.size() != 2U ||
        batch_results[0].warm_start_token != batch_results[1].warm_start_token) {
        return 2;
    }

    const CircularOrbitTarget start{"start", 7.0e6, 0.0, earth_mu};
    const std::vector<ServiceTarget> targets{
        {CircularOrbitTarget{"a", 8.0e6, 0.3, earth_mu}, EpochWindow{0.0, 2.0e7}, 300.0},
        {CircularOrbitTarget{"b", 9.0e6, 1.1, earth_mu}, EpochWindow{0.0, 2.0e7}, 300.0},
        {CircularOrbitTarget{"c", 1.0e7, 2.0, earth_mu}, EpochWindow{0.0, 2.0e7}, 300.0},
    };
    const RouteRequest route_request{
        start,
        targets,
        SpacecraftResources{500.0, 1'500.0, 3'000.0},
        0.0,
        2.0e7,
        3U,
        16U,
        1.0e-7,
        1.0e-7,
    };
    const BeamRouter router{oracle};
    const auto route = router.solve(route_request);
    if (!route.feasible || route.visited_targets.size() != 3U || route.legs.size() != 3U ||
        route.total_delta_v <= 0.0 || route.propellant_used <= 0.0 ||
        route.finish_epoch > route_request.end_epoch) {
        return 3;
    }
    const std::set<std::string> unique_targets(
        route.visited_targets.begin(),
        route.visited_targets.end()
    );
    if (unique_targets.size() != route.visited_targets.size()) {
        return 4;
    }
    double previous_epoch = route_request.start_epoch;
    double previous_propellant = route_request.spacecraft.propellant_mass;
    for (const auto& leg : route.legs) {
        if (!leg.arc.feasible || leg.arc.departure_epoch < previous_epoch ||
            leg.service_start < leg.arc.arrival_epoch || leg.service_end < leg.service_start ||
            leg.propellant_before > previous_propellant + 1.0e-9 ||
            leg.propellant_after >= leg.propellant_before) {
            return 5;
        }
        previous_epoch = leg.service_end;
        previous_propellant = leg.propellant_after;
    }
    return 0;
}
