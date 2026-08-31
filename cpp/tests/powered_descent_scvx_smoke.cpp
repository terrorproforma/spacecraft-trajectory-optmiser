#include "spacepdhcg/core/powered_descent_scvx.hpp"

#include <array>
#include <cstddef>
#include <vector>

int main() {
    using namespace spacepdhcg::core;

    PoweredDescentCQPConfig cqp_config;
    cqp_config.intervals = 4U;
    cqp_config.step_seconds = 1.0;
    cqp_config.fuel_weight = 0.0;
    cqp_config.virtual_l1_weight = 0.0;
    cqp_config.trust_radius = 1.0;
    PoweredDescent3DOF model;
    PoweredDescentCQP subproblem{model, cqp_config};

    std::vector<PoweredDescentState> states(cqp_config.intervals + 1U);
    std::vector<PoweredDescentControl> controls(cqp_config.intervals);
    states.front() = PoweredDescentState{0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 2'000.0};
    for (std::size_t interval = 0; interval < cqp_config.intervals; ++interval) {
        const double hover = -states[interval][6] * model.config().gravity[2];
        controls[interval] = PoweredDescentControl{0.0, 0.0, hover, hover};
        states[interval + 1U] = model.euler_step(
            states[interval],
            controls[interval],
            cqp_config.step_seconds
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

    PoweredDescentHostSCvxConfig outer;
    outer.maximum_iterations = 3U;
    outer.minimum_iterations = 1U;
    outer.convergence_tolerance = 1.0e-6;
    outer.step_tolerance = 1.0e-5;
    outer.residual_check_interval = 5U;
    outer.norm_iterations = 10U;
    ForcingRuleConfig forcing;
    forcing.epsilon_max = 1.0e-6;
    forcing.epsilon_0 = 1.0e-6;
    forcing.epsilon_floor = 1.0e-8;
    forcing.exploration_iteration_limit = 100U;
    forcing.convergence_iteration_limit = 200U;

    PoweredDescentHostSCvx solver{
        std::move(subproblem),
        outer,
        forcing,
        TrustRegionConfig{},
    };
    const auto result = solver.solve(
        states.front(),
        target_position,
        target_velocity,
        states,
        controls
    );

    if (!result.converged() || result.iterations.empty() ||
        result.accepted_iterations == 0U || result.inner_solves == 0U ||
        result.warm_starts == 0U) {
        return 1;
    }
    if (result.residual.maximum() > outer.convergence_tolerance ||
        result.path.maximum_violation() > 1.0e-8) {
        return 2;
    }
    if (result.iterations.front().convex_diagnostics.convex_violation() > 1.0e-6 ||
        result.iterations.front().solver_status != "solved") {
        return 3;
    }
    return 0;
}
