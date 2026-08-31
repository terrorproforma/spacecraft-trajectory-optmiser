#include "spacepdhcg/orbitweaver/lambert_oracle.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <string>

int main() {
    using spacepdhcg::orbitweaver::ArcFidelity;
    using spacepdhcg::orbitweaver::ArcRequest;
    using spacepdhcg::orbitweaver::CartesianEphemerisState;
    using spacepdhcg::orbitweaver::LambertScreeningConfig;
    using spacepdhcg::orbitweaver::LambertScreeningOracle;
    using spacepdhcg::orbitweaver::Vector3;
    using spacepdhcg::orbitweaver::solve_lambert_revolution_family;

    constexpr double gravitational_parameter = 398'600.4418;
    constexpr double radius = 7'000.0;
    constexpr double time_of_flight = 10'000.0;
    const Vector3 departure_position{radius, 0.0, 0.0};
    const Vector3 arrival_position{
        0.5 * radius,
        0.86602540378443864676 * radius,
        0.0,
    };
    const auto family = solve_lambert_revolution_family(
        departure_position,
        arrival_position,
        time_of_flight,
        gravitational_parameter,
        2U,
        false
    );
    const auto selected = family.back();
    if (selected.revolutions != 2U) {
        return 1;
    }

    const auto ephemeris = [selected, departure_position, arrival_position](
                               const std::size_t target,
                               const double epoch
                           ) {
        static_cast<void>(epoch);
        if (target == 0U) {
            return CartesianEphemerisState{
                departure_position,
                selected.solution.departure_velocity,
            };
        }
        if (target == 1U) {
            return CartesianEphemerisState{
                arrival_position,
                selected.solution.arrival_velocity,
            };
        }
        throw std::invalid_argument("unexpected multi-revolution target");
    };

    LambertScreeningConfig config{};
    config.gravitational_parameter = gravitational_parameter;
    config.specific_impulse_seconds = 300.0;
    config.maximum_revolutions = 2U;
    config.time_tolerance = 1.0e-7;
    LambertScreeningOracle oracle{ephemeris, config};
    const ArcRequest request{
        0U,
        1U,
        0.0,
        time_of_flight,
        500.0,
        0U,
        1U,
        ArcFidelity::analytical_screening,
        1.0e-6,
        "multi-revolution-screen",
        std::nullopt,
    };
    const auto result = oracle.evaluate(request);
    if (!result.feasible || result.delta_v > 1.0e-8 || result.propellant > 1.0e-8) {
        return 2;
    }
    if (result.diagnostics.find("revolutions=2") == std::string::npos
        || result.diagnostics.find("short-way") == std::string::npos) {
        return 3;
    }
    return 0;
}
