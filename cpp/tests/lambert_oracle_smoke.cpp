#include "spacepdhcg/orbitweaver/lambert_oracle.hpp"

#include <cmath>
#include <optional>

int main() {
    using spacepdhcg::orbitweaver::ArcFidelity;
    using spacepdhcg::orbitweaver::ArcRequest;
    using spacepdhcg::orbitweaver::CartesianEphemerisState;
    using spacepdhcg::orbitweaver::LambertScreeningConfig;
    using spacepdhcg::orbitweaver::LambertScreeningOracle;

    constexpr double mu = 3.986004418e14;
    constexpr double radius = 7.0e6;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const auto circular_speed = std::sqrt(mu / radius);
    const auto quarter_period = 0.5 * pi * std::sqrt(radius * radius * radius / mu);

    const auto ephemeris = [circular_speed](std::size_t target, double) {
        if (target == 0U) {
            return CartesianEphemerisState{
                {radius, 0.0, 0.0},
                {0.0, circular_speed, 0.0},
            };
        }
        if (target == 1U) {
            return CartesianEphemerisState{
                {0.0, radius, 0.0},
                {-circular_speed, 0.0, 0.0},
            };
        }
        throw std::invalid_argument("unknown target");
    };

    LambertScreeningConfig config{};
    config.gravitational_parameter = mu;
    config.specific_impulse_seconds = 320.0;
    config.time_tolerance = 1.0e-7;
    LambertScreeningOracle oracle(ephemeris, config);

    const ArcRequest request{
        0U,
        1U,
        0.0,
        quarter_period,
        500.0,
        0U,
        1U,
        ArcFidelity::analytical_screening,
        1.0e-6,
        "circular-two-body",
        std::nullopt,
    };
    const auto result = oracle.evaluate(request);
    if (!result.feasible || result.achieved_fidelity != ArcFidelity::analytical_screening) {
        return 1;
    }
    if (result.duration != quarter_period || result.lower_bound != 0.0) {
        return 2;
    }
    if (result.delta_v > 1.0e-3 || result.propellant > 1.0e-4) {
        return 3;
    }
    if (std::abs(result.final_mass + result.propellant - request.initial_mass) > 1.0e-10) {
        return 4;
    }
    if (result.terminal_error > 1.0e-6 || result.solve_seconds < 0.0) {
        return 5;
    }

    auto missing_arrival = request;
    missing_arrival.arrival_epoch.reset();
    try {
        static_cast<void>(oracle.evaluate(missing_arrival));
        return 6;
    } catch (const std::invalid_argument&) {
    }

    auto wrong_fidelity = request;
    wrong_fidelity.fidelity = ArcFidelity::coarse_convex;
    try {
        static_cast<void>(oracle.evaluate(wrong_fidelity));
        return 7;
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
