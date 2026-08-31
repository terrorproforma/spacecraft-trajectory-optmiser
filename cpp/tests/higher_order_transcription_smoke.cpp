#include "spacepdhcg/transcription/low_thrust.hpp"
#include "spacepdhcg/transcription/powered_descent_3dof.hpp"
#include "spacepdhcg/transcription/powered_descent_6dof.hpp"

#include <array>
#include <cmath>
#include <vector>

int main() {
    using spacepdhcg::transcription::DiscretisationMethod;

    {
        using spacepdhcg::dynamics::PoweredDescent3DofModel;
        using spacepdhcg::dynamics::PoweredDescentControl;
        using spacepdhcg::dynamics::PoweredDescentState;
        using spacepdhcg::transcription::PoweredDescent3DofSubproblem;
        using spacepdhcg::transcription::PoweredDescentScvxConfig;

        const PoweredDescent3DofModel model{};
        PoweredDescentScvxConfig rk4_config{};
        rk4_config.intervals = 2U;
        rk4_config.step_seconds = 0.1;
        rk4_config.discretisation = DiscretisationMethod::rk4_finite_difference;
        const PoweredDescent3DofSubproblem rk4(model, rk4_config);
        auto euler_config = rk4_config;
        euler_config.discretisation = DiscretisationMethod::forward_euler;
        const PoweredDescent3DofSubproblem euler(model, euler_config);
        if (rk4.structure().fingerprint() != euler.structure().fingerprint()) {
            return 1;
        }
        const PoweredDescentState initial{0.0, 0.0, 80.0, 0.0, 0.0, -1.0, 2'000.0};
        const PoweredDescentControl control{0.0, 0.0, 7'500.0, 7'500.0};
        const std::vector<PoweredDescentControl> controls(rk4_config.intervals, control);
        const auto states = model.rollout(initial, controls, rk4_config.step_seconds, true);
        const std::array<double, 3U> target_position{
            states.back()[0U], states.back()[1U], states.back()[2U]
        };
        const std::array<double, 3U> target_velocity{
            states.back()[3U], states.back()[4U], states.back()[5U]
        };
        const auto problem = rk4.problem(
            states,
            controls,
            initial,
            target_position,
            target_velocity
        );
        const auto diagnostics = rk4.diagnostics(
            rk4.reference_decision(states, controls),
            problem.values()
        );
        if (diagnostics.linearised_dynamics_defect_inf > 2.0e-8
            || diagnostics.terminal_error_inf > 1.0e-10
            || diagnostics.maximum_violation() > 2.0e-8) {
            return 2;
        }
    }

    {
        using spacepdhcg::dynamics::PoweredDescent6DofControl;
        using spacepdhcg::dynamics::PoweredDescent6DofModel;
        using spacepdhcg::dynamics::PoweredDescent6DofState;
        using spacepdhcg::transcription::PoweredDescent6DofScvxConfig;
        using spacepdhcg::transcription::PoweredDescent6DofSubproblem;

        const PoweredDescent6DofModel model{};
        PoweredDescent6DofScvxConfig rk4_config{};
        rk4_config.intervals = 2U;
        rk4_config.step_seconds = 0.05;
        rk4_config.discretisation = DiscretisationMethod::rk4_finite_difference;
        const PoweredDescent6DofSubproblem rk4(model, rk4_config);
        auto euler_config = rk4_config;
        euler_config.discretisation = DiscretisationMethod::forward_euler;
        const PoweredDescent6DofSubproblem euler(model, euler_config);
        if (rk4.structure().fingerprint() != euler.structure().fingerprint()) {
            return 3;
        }
        const PoweredDescent6DofState initial{
            0.0, 0.0, 80.0, 0.0, 0.0, -1.0,
            1.0, 0.0, 0.0, 0.0,
            0.01, -0.02, 0.015,
            2'000.0
        };
        const PoweredDescent6DofControl control{
            0.0, 0.0, 7'500.0,
            0.0, 0.0, 0.0,
            7'500.0
        };
        const std::vector<PoweredDescent6DofControl> controls(rk4_config.intervals, control);
        const auto states = model.rollout(initial, controls, rk4_config.step_seconds);
        const auto problem = rk4.problem(states, controls, initial, states.back());
        const auto diagnostics = rk4.diagnostics(
            rk4.reference_decision(states, controls),
            problem.values()
        );
        if (diagnostics.linearised_dynamics_defect_inf > 5.0e-7
            || diagnostics.terminal_error_inf > 1.0e-10
            || diagnostics.quaternion_linearisation_error_inf > 1.0e-8
            || diagnostics.maximum_violation() > 5.0e-7) {
            return 4;
        }
    }

    {
        using spacepdhcg::dynamics::LowThrustControl;
        using spacepdhcg::dynamics::LowThrustState;
        using spacepdhcg::dynamics::LowThrustTwoBodyModel;
        using spacepdhcg::transcription::LowThrustScvxConfig;
        using spacepdhcg::transcription::LowThrustSubproblem;

        const LowThrustTwoBodyModel model{};
        LowThrustScvxConfig rk4_config{};
        rk4_config.intervals = 2U;
        rk4_config.step_seconds = 1.0;
        rk4_config.discretisation = DiscretisationMethod::rk4_finite_difference;
        const LowThrustSubproblem rk4(model, rk4_config);
        auto euler_config = rk4_config;
        euler_config.discretisation = DiscretisationMethod::forward_euler;
        const LowThrustSubproblem euler(model, euler_config);
        if (rk4.structure().fingerprint() != euler.structure().fingerprint()) {
            return 5;
        }
        const LowThrustState initial{
            7'000.0,
            0.0,
            0.0,
            0.0,
            std::sqrt(model.config().gravitational_parameter / 7'000.0),
            0.0,
            500.0,
        };
        const LowThrustControl control{0.1, 0.0, 0.0, 0.1};
        const std::vector<LowThrustControl> controls(rk4_config.intervals, control);
        const auto states = model.rollout(initial, controls, rk4_config.step_seconds, true);
        const auto problem = rk4.problem(states, controls, initial, states.back());
        const auto diagnostics = rk4.diagnostics(
            rk4.reference_decision(states, controls),
            problem.values()
        );
        if (diagnostics.linearised_dynamics_defect_inf > 2.0e-7
            || diagnostics.terminal_error_inf > 1.0e-10
            || diagnostics.maximum_violation() > 2.0e-7) {
            return 6;
        }
    }
    return 0;
}
