#include "spacepdhcg/core/csc_operator.hpp"
#include "spacepdhcg/core/cw_cqp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

int main() {
    using namespace spacepdhcg::core;

    CWRendezvousCQPConfig box_config;
    box_config.intervals = 6U;
    box_config.step_seconds = 15.0;
    box_config.thrust_constraint = CWThrustConstraint::box;
    const CWRendezvousCQP box_problem{box_config};
    const HCWState initial{100.0, -25.0, 10.0, 0.01, -0.02, 0.005};
    const std::vector<HCWControl> zero_controls(box_config.intervals, HCWControl{});
    const auto states = box_problem.rollout(initial, zero_controls);
    const auto decision = box_problem.encode(states, zero_controls);
    const auto target = states.back();
    const auto values = box_problem.values(initial, target);
    const auto diagnostics = box_problem.diagnostics(decision, initial, target);
    if (diagnostics.maximum_violation() > 2.0e-12) {
        return 1;
    }

    const CscOperator scalar{
        box_problem.structure().scalar_constraint(),
        values.scalar_constraint,
    };
    const auto activity = scalar.multiply(decision);
    double equality_error = 0.0;
    const std::size_t equality_rows = hcw_state_dimension * (box_config.intervals + 2U);
    for (std::size_t row = 0; row < equality_rows; ++row) {
        equality_error = std::max(equality_error, std::abs(activity[row] - values.scalar_lower[row]));
    }
    if (equality_error > 2.0e-12) {
        return 2;
    }

    auto shifted_target = target;
    shifted_target[0] += 1.0;
    const auto updated = box_problem.values(initial, shifted_target);
    if (updated.linear_objective == values.linear_objective ||
        box_problem.structure().fingerprint() == 0U) {
        return 3;
    }

    CWRendezvousCQPConfig soc_config = box_config;
    soc_config.thrust_constraint = CWThrustConstraint::second_order_cone;
    const CWRendezvousCQP soc_problem{soc_config};
    const auto soc_states = soc_problem.rollout(initial, zero_controls);
    const auto soc_values = soc_problem.values(initial, soc_states.back());
    if (!soc_problem.structure().affine_cone().has_value() ||
        soc_problem.structure().affine_cones().size() != soc_config.intervals ||
        soc_values.affine_offset.size() != soc_config.intervals * 4U) {
        return 4;
    }
    const auto soc_decision = soc_problem.encode(soc_states, zero_controls);
    const CscOperator affine{
        *soc_problem.structure().affine_cone(),
        soc_values.affine_cone,
    };
    const auto cone_activity = affine.multiply(soc_decision);
    for (std::size_t interval = 0; interval < soc_config.intervals; ++interval) {
        const std::size_t start = interval * 4U;
        const double norm = std::sqrt(
            cone_activity[start] * cone_activity[start] +
            cone_activity[start + 1U] * cone_activity[start + 1U] +
            cone_activity[start + 2U] * cone_activity[start + 2U]
        );
        const double radius = cone_activity[start + 3U] + soc_values.affine_offset[start + 3U];
        if (norm > radius + 1.0e-14) {
            return 5;
        }
    }
    return 0;
}
