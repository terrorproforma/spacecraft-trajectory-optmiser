#include "spacepdhcg/orbitweaver/lambert.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace {

[[nodiscard]] double maximum_difference(
    const spacepdhcg::orbitweaver::Vector3& left,
    const spacepdhcg::orbitweaver::Vector3& right
) {
    double maximum = 0.0;
    for (std::size_t index = 0; index < 3U; ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

}  // namespace

int main() {
    using namespace spacepdhcg::orbitweaver;

    constexpr double mu = 3.986004418e14;
    constexpr double radius = 7.0e6;
    constexpr double angle = std::numbers::pi / 3.0;
    const double mean_motion = std::sqrt(mu / (radius * radius * radius));
    const double flight_time = angle / mean_motion;
    const double circular_speed = std::sqrt(mu / radius);
    const Vector3 departure_position{radius, 0.0, 0.0};
    const Vector3 arrival_position{
        radius * std::cos(angle),
        radius * std::sin(angle),
        0.0,
    };
    const auto solution = solve_lambert_zero_revolution(
        departure_position,
        arrival_position,
        flight_time,
        mu
    );
    const Vector3 expected_departure{0.0, circular_speed, 0.0};
    const Vector3 expected_arrival{
        -circular_speed * std::sin(angle),
        circular_speed * std::cos(angle),
        0.0,
    };
    if (maximum_difference(solution.departure_velocity, expected_departure) > 2.0e-4 ||
        maximum_difference(solution.arrival_velocity, expected_arrival) > 2.0e-4 ||
        std::abs(solution.time_residual) > 1.0e-6) {
        return 1;
    }

    const CircularOrbitTarget source{"source", radius, 0.0, mu};
    const CircularOrbitTarget target{"target", radius, 0.0, mu};
    const SpacecraftResources resources{500.0, 500.0, 3'000.0};
    const ArcRequest request{
        source,
        target,
        EpochWindow{0.0, 0.0},
        EpochWindow{flight_time, flight_time},
        resources,
        ArcFidelity::analytical,
        1.0e-8,
    };
    LambertScreeningOptions options;
    options.departure_samples = 1U;
    options.arrival_samples = 1U;
    options.minimum_flight_time = 1.0;
    const LambertCircularOracle oracle{options};
    const auto result = oracle.evaluate(request);
    if (!result.feasible || result.delta_v > 1.0e-3 ||
        result.propellant_required > 1.0e-3 || result.warm_start_token.empty()) {
        return 2;
    }
    return 0;
}
