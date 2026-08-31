#include "spacepdhcg/orbitweaver/lambert.hpp"

#include <array>
#include <cmath>

namespace {

bool close(double left, double right, double tolerance) {
    return std::abs(left - right) <= tolerance;
}

}  // namespace

int main() {
    using spacepdhcg::orbitweaver::Vector3;
    using spacepdhcg::orbitweaver::hohmann_transfer;
    using spacepdhcg::orbitweaver::propellant_required;
    using spacepdhcg::orbitweaver::solve_lambert_zero_revolution;

    const Vector3 departure{5'000.0, 10'000.0, 2'100.0};
    const Vector3 arrival{-14'600.0, 2'500.0, 7'000.0};
    const auto solution = solve_lambert_zero_revolution(
        departure,
        arrival,
        3'600.0,
        398'600.0,
        false,
        1.0e-9
    );
    const std::array<double, 3U> expected_departure{-5.9925, 1.9254, 3.2456};
    const std::array<double, 3U> expected_arrival{-3.3125, -4.1966, -0.3853};
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        if (!close(solution.departure_velocity[axis], expected_departure[axis], 5.0e-3)
            || !close(solution.arrival_velocity[axis], expected_arrival[axis], 5.0e-3)) {
            return 1;
        }
    }
    if (std::abs(solution.time_of_flight_residual) > 1.0e-7) {
        return 2;
    }

    const auto hohmann = hohmann_transfer(7'000.0, 42'164.0, 398'600.4418);
    if (!(hohmann.total_delta_v > 3.7 && hohmann.total_delta_v < 3.9)
        || hohmann.transfer_time <= 0.0) {
        return 3;
    }
    const auto propellant = propellant_required(1'000.0, 3'800.0, 450.0);
    if (!(propellant > 0.0 && propellant < 1'000.0)) {
        return 4;
    }
    return 0;
}
