#include "spacepdhcg/backends/dense_admm.hpp"
#include "spacepdhcg/dynamics/hcw.hpp"
#include "spacepdhcg/transcription/hcw_rendezvous.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace {

template <std::size_t Rows, std::size_t Inner, std::size_t Columns>
std::array<double, Rows * Columns> multiply(
    const std::array<double, Rows * Inner>& left,
    const std::array<double, Inner * Columns>& right
) {
    std::array<double, Rows * Columns> result{};
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t column = 0; column < Columns; ++column) {
            for (std::size_t inner = 0; inner < Inner; ++inner) {
                result[row * Columns + column] +=
                    left[row * Inner + inner] * right[inner * Columns + column];
            }
        }
    }
    return result;
}

template <typename Array>
double maximum_difference(const Array& left, const Array& right) {
    double maximum{0.0};
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

}  // namespace

int main() {
    using spacepdhcg::backends::DenseAdmmBackend;
    using spacepdhcg::backends::DenseAdmmConfig;
    using spacepdhcg::core::HostWarmStart;
    using spacepdhcg::dynamics::HcwControlMatrix;
    using spacepdhcg::dynamics::HcwState;
    using spacepdhcg::dynamics::HcwStateMatrix;
    using spacepdhcg::dynamics::discretise_hcw;
    using spacepdhcg::transcription::HcwControlSet;
    using spacepdhcg::transcription::HcwRendezvousConfig;
    using spacepdhcg::transcription::HcwRendezvousCqp;

    constexpr double mean_motion = 1.13e-3;
    const auto first = discretise_hcw(mean_motion, 12.0);
    const auto second = discretise_hcw(mean_motion, 18.0);
    const auto combined = discretise_hcw(mean_motion, 30.0);
    const auto composed_state = multiply<6U, 6U, 6U>(second.state, first.state);
    auto propagated_first_control = multiply<6U, 6U, 3U>(
        second.state,
        first.control
    );
    HcwControlMatrix composed_control{};
    for (std::size_t index = 0; index < composed_control.size(); ++index) {
        composed_control[index] = propagated_first_control[index] + second.control[index];
    }
    if (maximum_difference(composed_state, combined.state) > 2.0e-10) {
        return 1;
    }
    if (maximum_difference(composed_control, combined.control) > 2.0e-8) {
        return 2;
    }

    HcwRendezvousConfig config{};
    config.intervals = 8U;
    config.step_seconds = 25.0;
    config.mean_motion = mean_motion;
    config.maximum_acceleration = 0.05;
    config.control_set = HcwControlSet::box;
    HcwRendezvousCqp rendezvous(config);
    const HcwState initial{1.0, -0.5, 0.2, 0.0, 0.0, 0.0};
    const HcwState target{};

    DenseAdmmBackend solver(
        rendezvous.problem(initial, target),
        DenseAdmmConfig{1.0, 1.0e-10, 256U}
    );
    const auto solution = solver.solve(1.0e-7, 200'000U);
    if (!solution.solved()) {
        return 3;
    }
    const auto diagnostics = rendezvous.diagnostics(solution.primal, initial, target);
    if (diagnostics.initial_error > 2.0e-6
        || diagnostics.terminal_error > 2.0e-6
        || diagnostics.dynamics_defect > 2.0e-6
        || diagnostics.control_violation > 2.0e-6) {
        return 4;
    }

    const HcwState second_initial{0.8, -0.3, 0.1, 0.0, 0.0, 0.0};
    solver.update(rendezvous.values(second_initial, target));
    solver.warm_start(HostWarmStart{solution.primal, solution.dual});
    const auto repeated = solver.solve(1.0e-7, 200'000U);
    if (!repeated.solved() || solver.update_count() != 1U
        || solver.warm_start_count() != 1U || solver.solve_count() != 2U) {
        return 5;
    }
    const auto repeated_diagnostics = rendezvous.diagnostics(
        repeated.primal,
        second_initial,
        target
    );
    if (repeated_diagnostics.terminal_error > 2.0e-6
        || repeated_diagnostics.dynamics_defect > 2.0e-6) {
        return 6;
    }

    config.control_set = HcwControlSet::second_order_cone;
    const HcwRendezvousCqp socp(config);
    if (!socp.structure().affine_cone.has_value()
        || socp.structure().affine_cones.size() != config.intervals
        || socp.structure().affine_rows() != static_cast<spacepdhcg::Index>(4U * config.intervals)) {
        return 7;
    }
    return 0;
}
