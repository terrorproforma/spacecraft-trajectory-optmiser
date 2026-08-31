#include "spacepdhcg/core/powered_descent_cqp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

int main() {
    using namespace spacepdhcg::core;

    PoweredDescentCQPConfig config;
    config.intervals = 6U;
    config.step_seconds = 1.0;
    config.trust_radius = 1.0;
    PoweredDescent3DOF model;
    const PoweredDescentCQP problem{model, config};

    std::vector<PoweredDescentState> states(config.intervals + 1U);
    std::vector<PoweredDescentControl> controls(config.intervals);
    states.front() = PoweredDescentState{0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 2'000.0};
    for (std::size_t interval = 0; interval < config.intervals; ++interval) {
        const double vertical_thrust = -states[interval][6] * model.config().gravity[2];
        controls[interval] = PoweredDescentControl{
            0.0,
            0.0,
            vertical_thrust,
            vertical_thrust,
        };
        states[interval + 1U] = model.euler_step(
            states[interval],
            controls[interval],
            config.step_seconds
        );
    }

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
    const auto values = problem.values(
        states,
        controls,
        states.front(),
        target_position,
        target_velocity
    );
    const auto decision = problem.reference_decision(states, controls);
    const auto diagnostics = problem.diagnostics(decision, values);

    if (problem.structure().fingerprint() == 0U ||
        problem.structure().variables() != static_cast<spacepdhcg::Index>(problem.layout().variables()) ||
        problem.structure().scalar_rows() !=
            static_cast<spacepdhcg::Index>(problem.layout().scalar_rows()) ||
        problem.structure().affine_rows() !=
            static_cast<spacepdhcg::Index>(problem.layout().affine_rows())) {
        return 1;
    }
    if (diagnostics.convex_violation() > 2.0e-10 ||
        diagnostics.linearised_dynamics_defect > 2.0e-10 ||
        diagnostics.nonlinear_dynamics_defect > 2.0e-10 ||
        diagnostics.terminal_error > 2.0e-10 ||
        diagnostics.virtual_control > 2.0e-12) {
        return 2;
    }

    const auto decoded = problem.decode(decision);
    if (decoded.states.size() != states.size() || decoded.controls.size() != controls.size() ||
        std::abs(decoded.states.back()[6] - states.back()[6]) > 1.0e-12) {
        return 3;
    }

    const auto tighter = problem.values(
        states,
        controls,
        states.front(),
        target_position,
        target_velocity,
        0.25
    );
    if (tighter.quadratic.size() != values.quadratic.size() ||
        tighter.scalar_constraint.size() != values.scalar_constraint.size() ||
        tighter.affine_cone.size() != values.affine_cone.size()) {
        return 4;
    }
    const bool offset_changed = !std::equal(
        tighter.affine_offset.begin(),
        tighter.affine_offset.end(),
        values.affine_offset.begin()
    );
    if (!offset_changed) {
        return 5;
    }

    const auto epigraph_begin = values.variable_lower.begin() +
        static_cast<std::ptrdiff_t>(problem.layout().epigraph_offset());
    if (!std::all_of(epigraph_begin, values.variable_lower.end(), [](const double value) {
            return value == 0.0;
        })) {
        return 6;
    }
    return 0;
}
