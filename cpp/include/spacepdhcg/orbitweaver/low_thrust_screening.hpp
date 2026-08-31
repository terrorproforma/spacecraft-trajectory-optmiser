#pragma once

#include "spacepdhcg/orbitweaver/lambert.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spacepdhcg::orbitweaver {

struct EdelbaumEstimate {
    double delta_v{0.0};
    double transfer_time{0.0};
    double propellant{0.0};
    double initial_circular_speed{0.0};
    double final_circular_speed{0.0};
};

/// Edelbaum's classical circular-orbit radius/inclination screening estimate.
///
/// Distances and gravitational parameter use a consistent kilometre system, so speeds and
/// `acceleration` are in km/s and km/s^2. The estimate is a fast continuous-thrust screen, not a
/// high-fidelity feasible trajectory and not a substitute for SCvx refinement.
inline EdelbaumEstimate edelbaum_circular_transfer(
    double initial_radius,
    double final_radius,
    double inclination_change_radians,
    double gravitational_parameter,
    double acceleration,
    double initial_mass,
    double specific_impulse_seconds
) {
    if (!std::isfinite(initial_radius) || initial_radius <= 0.0
        || !std::isfinite(final_radius) || final_radius <= 0.0
        || !std::isfinite(inclination_change_radians)
        || !std::isfinite(gravitational_parameter) || gravitational_parameter <= 0.0
        || !std::isfinite(acceleration) || acceleration <= 0.0
        || !std::isfinite(initial_mass) || initial_mass <= 0.0
        || !std::isfinite(specific_impulse_seconds) || specific_impulse_seconds <= 0.0) {
        throw std::invalid_argument("Edelbaum screening inputs are invalid");
    }
    constexpr double pi = 3.141592653589793238462643383279502884;
    const auto inclination = std::clamp(
        std::abs(inclination_change_radians),
        0.0,
        pi
    );
    const auto speed_one = std::sqrt(gravitational_parameter / initial_radius);
    const auto speed_two = std::sqrt(gravitational_parameter / final_radius);
    const auto coupling_angle = 0.5 * pi * inclination;
    const auto delta_v_squared =
        speed_one * speed_one + speed_two * speed_two
        - 2.0 * speed_one * speed_two * std::cos(coupling_angle);
    const auto delta_v = std::sqrt(std::max(delta_v_squared, 0.0));
    return EdelbaumEstimate{
        delta_v,
        delta_v / acceleration,
        propellant_required(
            initial_mass,
            1'000.0 * delta_v,
            specific_impulse_seconds
        ),
        speed_one,
        speed_two,
    };
}

/// Optimistic lower bound that ignores simultaneous radius/plane-change coupling losses.
inline double circular_speed_change_lower_bound(
    double initial_radius,
    double final_radius,
    double gravitational_parameter
) {
    if (!std::isfinite(initial_radius) || initial_radius <= 0.0
        || !std::isfinite(final_radius) || final_radius <= 0.0
        || !std::isfinite(gravitational_parameter) || gravitational_parameter <= 0.0) {
        throw std::invalid_argument("circular-speed lower-bound inputs are invalid");
    }
    return std::abs(
        std::sqrt(gravitational_parameter / final_radius)
        - std::sqrt(gravitational_parameter / initial_radius)
    );
}

}  // namespace spacepdhcg::orbitweaver
