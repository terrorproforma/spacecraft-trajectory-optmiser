#include "spacepdhcg/core/powered_descent_6dof_cqp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

int main() {
    using namespace spacepdhcg::core;

    PoweredDescent6DOFCQPConfig config;
    config.intervals = 3U;
    config.step_seconds = 0.25;
    config.trust_radius = 1.0;
    PoweredDescent6DOF model{PoweredDescent6DOFConfig{}};
    const PoweredDescent6DOFCQP problem{model, config};

    std::vector<PoweredDescent6DOFState> states(config.intervals + 1U);
    std::vector<PoweredDescent6DOFControl> controls(config.intervals);
    states.front() = PoweredDescent6DOFState{
        0.0,
        0.0,
        100.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2'000.0,
    };
    for (std::size_t interval = 0; interval < config.intervals; ++interval) {
        const double hover = -states[interval][13] * model.config().gravity[2];
        controls[interval] = PoweredDescent6DOFControl{
            0.0,
            0.0,
            hover,
            0.0,
            0.0,
            0.0,
            hover,
        };
        const auto derivative = model.dynamics(states[interval], controls[interval]);
        states[interval + 1U] = states[interval];
        for (std::size_t component = 0;
             component < powered_descent_6dof_state_dimension;
             ++component) {
            states[interval + 1U][component] +=
                config.step_seconds * derivative[component];
        }
    }

    const auto values = problem.values(
        states,
        controls,
        states.front(),
        states.back()
    );
    const auto decision = problem.reference_decision(states, controls);
    const auto diagnostics = problem.diagnostics(decision, values);

    if (problem.structure().fingerprint() == 0U ||
        problem.structure().variables() !=
            static_cast<spacepdhcg::Index>(problem.layout().variables()) ||
        problem.structure().scalar_rows() !=
            static_cast<spacepdhcg::Index>(problem.layout().scalar_rows()) ||
        problem.structure().affine_rows() !=
            static_cast<spacepdhcg::Index>(problem.layout().affine_rows())) {
        return 1;
    }
    if (diagnostics.convex_violation() > 2.0e-8 ||
        diagnostics.linearised_dynamics_defect > 2.0e-8 ||
        diagnostics.nonlinear_euler_defect > 2.0e-8 ||
        diagnostics.terminal_error > 2.0e-10 ||
        diagnostics.quaternion_tangent_error > 2.0e-10 ||
        diagnostics.virtual_control > 2.0e-12) {
        return 2;
    }

    const auto decoded = problem.decode(decision);
    if (decoded.states.size() != states.size() ||
        decoded.controls.size() != controls.size() ||
        std::abs(decoded.states.back()[13] - states.back()[13]) > 1.0e-12 ||
        std::abs(quaternion_norm(Quaternion{
            decoded.states.back()[6],
            decoded.states.back()[7],
            decoded.states.back()[8],
            decoded.states.back()[9],
        }) - 1.0) > 1.0e-12) {
        return 3;
    }

    const auto tighter = problem.values(
        states,
        controls,
        states.front(),
        states.back(),
        0.25
    );
    if (tighter.quadratic.size() != values.quadratic.size() ||
        tighter.scalar_constraint.size() != values.scalar_constraint.size() ||
        tighter.affine_cone.size() != values.affine_cone.size()) {
        return 4;
    }
    if (std::equal(
            tighter.affine_offset.begin(),
            tighter.affine_offset.end(),
            values.affine_offset.begin()
        )) {
        return 5;
    }

    const auto path = model.path_diagnostics(states, controls);
    if (path.maximum_violation() > model.config().quaternion_norm_tolerance) {
        return 6;
    }
    return 0;
}
