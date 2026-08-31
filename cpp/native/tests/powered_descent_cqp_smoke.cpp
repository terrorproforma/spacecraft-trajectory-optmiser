#include "spacepdhcg/native/powered_descent_cqp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace native = spacepdhcg::native;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_reference_interpolation_and_fixed_topology() {
    native::PoweredDescentCqpConfig config{};
    config.intervals = 4;
    config.step_seconds = 2.0;
    config.trust_radius = 1.0;
    native::PoweredDescent3DofModel model;
    native::PoweredDescentCqp transcription(model, config);

    native::PoweredDescentState initial{0.0, 0.0, 100.0, 0.0, 0.0, -3.0, 2'000.0};
    native::PoweredDescentControl control{0.0, 0.0, 6'000.0, 6'000.0};
    const std::vector<native::PoweredDescentControl> controls(
        static_cast<std::size_t>(config.intervals),
        control
    );
    const auto states = model.rollout_euler(initial, controls, config.step_seconds);
    const std::array<double, 3> target_position{
        states.back()[0],
        states.back()[1],
        states.back()[2],
    };
    const std::array<double, 3> target_velocity{
        states.back()[3],
        states.back()[4],
        states.back()[5],
    };

    auto problem = transcription.make_cqp(
        states,
        controls,
        initial,
        target_position,
        target_velocity,
        config.trust_radius
    );
    require(problem.variables() == transcription.layout().variables(),
            "powered-descent CQP variable count is wrong");
    require(problem.affine_cones.size() == 3U * static_cast<std::size_t>(config.intervals) + 2U,
            "powered-descent cone count is wrong");

    std::vector<double> decision(
        static_cast<std::size_t>(transcription.layout().variables()),
        0.0
    );
    for (native::Index node = 0; node <= config.intervals; ++node) {
        const auto offset = static_cast<std::size_t>(transcription.layout().state_offset(node));
        std::copy(
            states[static_cast<std::size_t>(node)].begin(),
            states[static_cast<std::size_t>(node)].end(),
            decision.begin() + static_cast<std::ptrdiff_t>(offset)
        );
    }
    for (native::Index interval = 0; interval < config.intervals; ++interval) {
        const auto offset =
            static_cast<std::size_t>(transcription.layout().control_offset(interval));
        std::copy(
            controls[static_cast<std::size_t>(interval)].begin(),
            controls[static_cast<std::size_t>(interval)].end(),
            decision.begin() + static_cast<std::ptrdiff_t>(offset)
        );
    }

    const auto diagnostics = transcription.diagnostics(
        decision,
        problem,
        target_position,
        target_velocity
    );
    require(diagnostics.maximum_convex_violation() < 2.0e-10,
            "reference trajectory is not feasible in its convex subproblem");
    require(diagnostics.linearised_dynamics_defect < 2.0e-10,
            "linearised dynamics do not interpolate the reference");
    require(diagnostics.nonlinear_dynamics_defect < 2.0e-10,
            "reference trajectory is not Euler-dynamics consistent");
    require(diagnostics.terminal_error < 2.0e-10,
            "reference trajectory misses its own terminal state");
    require(diagnostics.virtual_control < 2.0e-10,
            "reference trajectory unexpectedly uses virtual control");
    require(std::isfinite(problem.objective(decision)),
            "powered-descent objective is not finite");

    const auto scalar_offsets = problem.scalar_constraint.offsets;
    const auto affine_indices = problem.affine_cone.indices;
    transcription.update_numerical_values(
        problem,
        states,
        controls,
        initial,
        target_position,
        target_velocity,
        0.5
    );
    require(problem.scalar_constraint.offsets == scalar_offsets,
            "powered-descent update changed scalar topology");
    require(problem.affine_cone.indices == affine_indices,
            "powered-descent update changed affine topology");
    const auto terminal_radius_index = static_cast<std::size_t>(
        transcription.layout().terminal_trust_cone_row() + 7
    );
    require(std::abs(problem.affine_offset[terminal_radius_index] - 0.5) < 1.0e-15,
            "powered-descent trust radius was not updated in place");
}

}  // namespace

int main() {
    test_reference_interpolation_and_fixed_topology();
    return 0;
}
