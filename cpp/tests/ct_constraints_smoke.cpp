#include "spacepdhcg/core/ct_constraints.hpp"

#include <cmath>
#include <vector>

int main() {
    using namespace spacepdhcg::core;

    const std::vector<std::vector<double>> feasible{
        {-1.0, -0.2},
        {-0.5, -0.1},
        {-0.3, -0.4},
    };
    const auto feasible_certificate = certify_continuous_time_constraints(feasible, 2.0);
    if (!feasible_certificate.certified(0.0) || feasible_certificate.total_violation != 0.0) {
        return 1;
    }

    const std::vector<std::vector<double>> nodes{
        {-1.0, -1.0},
        {-1.0, -1.0},
        {-1.0, -1.0},
    };
    const std::vector<std::vector<double>> midpoints{
        {0.5, -1.0},
        {-1.0, 0.25},
    };
    const auto midpoint_certificate = certify_with_midpoints(nodes, midpoints, 3.0);
    const double expected_first = 3.0 * (4.0 * 0.25) / 6.0;
    const double expected_second = 3.0 * (4.0 * 0.0625) / 6.0;
    if (std::abs(midpoint_certificate.interval_increments[0] - expected_first) > 1.0e-14 ||
        std::abs(midpoint_certificate.interval_increments[1] - expected_second) > 1.0e-14) {
        return 2;
    }
    if (midpoint_certificate.certified(expected_first - 1.0e-6) ||
        !midpoint_certificate.certified(expected_first + 1.0e-6)) {
        return 3;
    }

    const auto absolute = certify_with_midpoints(
        nodes,
        midpoints,
        3.0,
        ViolationPenalty::absolute
    );
    if (absolute.total_violation <= midpoint_certificate.total_violation) {
        return 4;
    }
    return 0;
}
