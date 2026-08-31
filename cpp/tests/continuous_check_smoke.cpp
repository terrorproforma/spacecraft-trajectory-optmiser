#include "spacepdhcg/dynamics/continuous_check.hpp"
#include "spacepdhcg/dynamics/low_thrust_two_body.hpp"
#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"

#include <cmath>
#include <vector>

int main() {
    using spacepdhcg::dynamics::check_piecewise_constant_trajectory;

    {
        using spacepdhcg::dynamics::PoweredDescent3DofModel;
        using spacepdhcg::dynamics::PoweredDescentControl;
        using spacepdhcg::dynamics::PoweredDescentState;
        const PoweredDescent3DofModel model{};
        const PoweredDescentState initial{0.0, 0.0, 80.0, 0.0, 0.0, -1.0, 2'000.0};
        const PoweredDescentControl control{0.0, 0.0, 7'500.0, 7'500.0};
        const std::vector<PoweredDescentControl> controls(3U, control);
        auto states = model.rollout(initial, controls, 0.1, true);
        const auto diagnostics = check_piecewise_constant_trajectory(
            model,
            states,
            controls,
            0.1,
            8U
        );
        if (diagnostics.maximum_node_mismatch > 1.0e-7
            || diagnostics.maximum_path_violation > 1.0e-10
            || diagnostics.propagated_substeps != 24U) {
            return 1;
        }
        states[1U][2U] += 0.01;
        const auto tampered = check_piecewise_constant_trajectory(
            model,
            states,
            controls,
            0.1,
            8U
        );
        if (tampered.maximum_node_mismatch < 0.009) {
            return 2;
        }
    }

    {
        using spacepdhcg::dynamics::PoweredDescent6DofControl;
        using spacepdhcg::dynamics::PoweredDescent6DofModel;
        using spacepdhcg::dynamics::PoweredDescent6DofState;
        const PoweredDescent6DofModel model{};
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
        const std::vector<PoweredDescent6DofControl> controls(2U, control);
        const auto states = model.rollout(initial, controls, 0.05);
        const auto diagnostics = check_piecewise_constant_trajectory(
            model,
            states,
            controls,
            0.05,
            8U
        );
        if (diagnostics.maximum_node_mismatch > 2.0e-7
            || diagnostics.maximum_path_violation > 1.0e-10) {
            return 3;
        }
    }

    {
        using spacepdhcg::dynamics::LowThrustControl;
        using spacepdhcg::dynamics::LowThrustState;
        using spacepdhcg::dynamics::LowThrustTwoBodyModel;
        const LowThrustTwoBodyModel model{};
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
        const std::vector<LowThrustControl> controls(2U, control);
        const auto states = model.rollout(initial, controls, 1.0, true);
        const auto diagnostics = check_piecewise_constant_trajectory(
            model,
            states,
            controls,
            1.0,
            8U
        );
        if (diagnostics.maximum_node_mismatch > 1.0e-7
            || diagnostics.maximum_path_violation > 1.0e-10) {
            return 4;
        }
    }
    return 0;
}
