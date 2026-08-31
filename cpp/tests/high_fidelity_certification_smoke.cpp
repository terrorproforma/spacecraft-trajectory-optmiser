#include "spacepdhcg/orbitweaver/high_fidelity_certification.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

int main() {
    using spacepdhcg::dynamics::LowThrustControl;
    using spacepdhcg::dynamics::LowThrustState;
    using spacepdhcg::orbitweaver::ArcFidelity;
    using spacepdhcg::orbitweaver::ArcRequest;
    using spacepdhcg::orbitweaver::CartesianEphemerisState;
    using spacepdhcg::orbitweaver::HighFidelityLowThrustConfig;
    using spacepdhcg::orbitweaver::HighFidelityLowThrustOrbitStage;
    using spacepdhcg::orbitweaver::LowThrustWarmStartStore;

    const auto ephemeris = [](const std::size_t target, const double epoch) {
        static_cast<void>(epoch);
        if (target > 1U) {
            throw std::invalid_argument("unexpected high-fidelity target");
        }
        return CartesianEphemerisState{
            {7'000.0, 0.0, 0.0},
            {0.0, 0.0, 0.0},
        };
    };
    HighFidelityLowThrustConfig config{};
    config.gravitational_parameter = 1.0e-12;
    config.equatorial_radius = 1.0;
    config.j2 = 0.0;
    config.thrust_to_acceleration = 1.0;
    config.mass_flow_coefficient = 1.0e-6;
    config.minimum_mass = 100.0;
    config.maximum_thrust = 1.0;
    config.minimum_radius = 1.0;
    config.intervals = 4U;
    config.substeps_per_interval = 2U;
    config.feasibility_tolerance = 1.0e-8;

    ArcRequest request{
        0U,
        1U,
        0.0,
        40.0,
        500.0,
        0U,
        1U,
        ArcFidelity::certified_high_fidelity,
        1.0e-8,
        "j2-certification-smoke",
        std::nullopt,
    };
    const LowThrustState state{
        7'000.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        500.0,
    };
    const LowThrustControl coast{0.0, 0.0, 0.0, 0.0};
    const std::vector<LowThrustState> states(config.intervals + 1U, state);
    const std::vector<LowThrustControl> controls(config.intervals, coast);
    const auto store = std::make_shared<LowThrustWarmStartStore>();
    const auto token = store->put(
        request,
        config.intervals,
        {states, controls}
    );
    request.warm_start_token = token;

    const HighFidelityLowThrustOrbitStage stage{
        ephemeris,
        store,
        config,
    };
    const auto result = stage.evaluate(request);
    if (!result.feasible
        || result.achieved_fidelity != ArcFidelity::certified_high_fidelity
        || result.warm_start_token != request.warm_start_token) {
        return 1;
    }
    if (result.terminal_error > 1.0e-8
        || result.maximum_constraint_violation > 1.0e-12
        || result.achieved_accuracy > 1.0e-8) {
        return 2;
    }
    if (result.delta_v > 1.0e-12 || result.propellant > 1.0e-12
        || std::abs(result.final_mass + result.propellant - request.initial_mass)
               > 1.0e-12) {
        return 3;
    }
    if (result.inner_iterations != 16U
        || result.diagnostics.find("independent J2") == std::string::npos) {
        return 4;
    }
    return 0;
}
