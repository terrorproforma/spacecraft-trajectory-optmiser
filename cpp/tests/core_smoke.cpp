#include "spacepdhcg/core/cqp.hpp"
#include "spacepdhcg/core/csc_operator.hpp"
#include "spacepdhcg/core/hcw.hpp"
#include "spacepdhcg/core/powered_descent.hpp"
#include "spacepdhcg/core/scenario.hpp"
#include "spacepdhcg/core/scvx_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

[[nodiscard]] bool near(const double left, const double right, const double tolerance) {
    return std::abs(left - right) <= tolerance;
}

[[nodiscard]] double maximum_difference(
    const std::span<const double> left,
    const std::span<const double> right
) {
    if (left.size() != right.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double maximum = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

[[nodiscard]] int test_cqp_and_sparse_operator() {
    using namespace spacepdhcg;
    using namespace spacepdhcg::core;

    CscStructure identity{2, 2, {0, 1, 2}, {0, 1}};
    CscOperator sparse{identity, {2.0, 3.0}};
    const std::array<double, 2> input{4.0, 5.0};
    const auto forward = sparse.multiply(input);
    const auto transpose = sparse.transpose_multiply(input);
    if (!near(forward[0], 8.0, 1.0e-14) || !near(forward[1], 15.0, 1.0e-14) ||
        maximum_difference(forward, transpose) > 1.0e-14) {
        return 1;
    }
    const std::array<double, 2> updated{1.0, -1.0};
    sparse.update_values(updated);
    if (sparse.update_count() != 1U || !near(sparse.multiply(input)[1], -5.0, 1.0e-14)) {
        return 2;
    }

    CQPStructure structure{identity, CscStructure{2, 2, {0, 1, 2}, {0, 1}}};
    CQPValues values{
        {1.0, 1.0},
        {1.0, 1.0},
        {},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0},
        {},
        {-std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity()},
        {std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()},
    };
    validate_values(structure, values);
    if (structure.variables() != 2 || structure.duals() != 2 || structure.fingerprint() == 0U) {
        return 3;
    }
    return 0;
}

[[nodiscard]] int test_hcw_semigroup() {
    using namespace spacepdhcg::core;
    constexpr double mean_motion = 1.13e-3;
    const auto first = discretise_hcw(mean_motion, 17.0);
    const auto second = discretise_hcw(mean_motion, 29.0);
    const auto combined = discretise_hcw(mean_motion, 46.0);
    const auto composed_state = multiply_hcw_state_matrices(second.state, first.state);
    const auto composed_control = compose_hcw_control_matrices(
        second.state,
        first.control,
        second.control
    );
    if (maximum_difference(combined.state, composed_state) > 2.0e-10 ||
        maximum_difference(combined.control, composed_control) > 2.0e-9) {
        return 10;
    }

    const HCWState state{100.0, -25.0, 15.0, 0.1, -0.05, 0.02};
    const HCWControl control{1.0e-3, -5.0e-4, 2.0e-4};
    const auto one_step = propagate_hcw(combined, state, control);
    const auto first_step = propagate_hcw(first, state, control);
    const auto two_step = propagate_hcw(second, first_step, control);
    if (maximum_difference(one_step, two_step) > 2.0e-9) {
        return 11;
    }
    return 0;
}

[[nodiscard]] int test_powered_descent() {
    using namespace spacepdhcg::core;
    PoweredDescent3DOF model;
    const PoweredDescentState state{10.0, -5.0, 120.0, 1.0, -2.0, -6.0, 2'000.0};
    const double thrust_norm = std::sqrt(100.0 * 100.0 + 50.0 * 50.0 + 8'000.0 * 8'000.0);
    const PoweredDescentControl control{100.0, 50.0, 8'000.0, thrust_norm};
    const auto derivatives = model.jacobians(state, control);
    constexpr double epsilon = 1.0e-5;
    for (std::size_t column = 0; column < powered_descent_state_dimension; ++column) {
        auto plus = state;
        auto minus = state;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        const auto forward = model.dynamics(plus, control);
        const auto backward = model.dynamics(minus, control);
        for (std::size_t row = 0; row < powered_descent_state_dimension; ++row) {
            const double numerical = (forward[row] - backward[row]) / (2.0 * epsilon);
            const double analytic = derivatives.state[powered_descent_state_index(row, column)];
            if (!near(numerical, analytic, 2.0e-7)) {
                return 20;
            }
        }
    }
    for (std::size_t column = 0; column < powered_descent_control_dimension; ++column) {
        auto plus = control;
        auto minus = control;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        const auto forward = model.dynamics(state, plus);
        const auto backward = model.dynamics(state, minus);
        for (std::size_t row = 0; row < powered_descent_state_dimension; ++row) {
            const double numerical = (forward[row] - backward[row]) / (2.0 * epsilon);
            const double analytic = derivatives.control[powered_descent_control_index(row, column)];
            if (!near(numerical, analytic, 2.0e-7)) {
                return 21;
            }
        }
    }

    const std::vector<PoweredDescentControl> controls(5U, control);
    const auto euler = model.rollout(state, controls, 0.1, false);
    const auto rk4 = model.rollout(state, controls, 0.1, true);
    if (euler.size() != 6U || rk4.size() != 6U || euler.back()[6] >= state[6]) {
        return 22;
    }
    const auto path = model.path_diagnostics(rk4, controls);
    if (!path.feasible(1.0e-10)) {
        return 23;
    }
    return 0;
}

[[nodiscard]] int test_scenarios() {
    using namespace spacepdhcg::core;
    const auto tree = ScenarioTree::common_open_loop(4U, 5U, 2U);
    if (tree.scenario_count() != 4U || tree.horizon() != 5U ||
        tree.shared_nodes().size() != 2U) {
        return 30;
    }
    const BlockArrowLayout layout{tree, 7U, 4U, 10U};
    if (layout.consensus_dimension() != 8U || layout.nonanticipativity_rows() != 32U ||
        layout.nonanticipativity_triplets().size() != 64U) {
        return 31;
    }
    const auto profile = layout.communication_profile(4U);
    if (profile.payload_bytes != 8U * sizeof(double) || profile.bytes_per_device <= 0.0) {
        return 32;
    }
    const auto partition = partition_scenarios({7.0, 6.0, 4.0, 3.0, 2.0}, 2U);
    if (partition.assignments.size() != 2U || partition.imbalance() > 1.1) {
        return 33;
    }
    return 0;
}

[[nodiscard]] int test_scvx_policies() {
    using namespace spacepdhcg::core;
    const AdaptiveForcingRule forcing;
    const auto exploration = forcing.request(0U, OuterResidual{0.2, 0.1, 0.05, 0.8});
    if (exploration.phase != SolvePhase::exploration || exploration.tolerance < 1.0e-4) {
        return 40;
    }
    const auto polish = forcing.request(
        6U,
        OuterResidual{1.0e-5, 1.0e-5, 1.0e-5, 1.0e-3},
        2U,
        0.9
    );
    if (polish.phase != SolvePhase::polish ||
        make_hybrid_plan(polish, false).polish != SolverFamily::qoco_gpu ||
        make_hybrid_plan(polish, true).polish != SolverFamily::cuclarabel) {
        return 41;
    }
    if (!forcing.should_resolve(false, 1.0e-2, 1.0e-3, 1.0e-4) ||
        forcing.should_resolve(true, 1.0, 1.0, 1.0e-4)) {
        return 42;
    }

    TrustRegionController trust;
    const auto grown = trust.update(true, 0.9, 0.9);
    if (grown.action != RadiusAction::grow || grown.radius_after <= grown.radius_before) {
        return 43;
    }
    const auto shrunk = trust.update(false, 0.0, 0.0);
    if (shrunk.action != RadiusAction::shrink || shrunk.radius_after >= shrunk.radius_before) {
        return 44;
    }
    return 0;
}

}  // namespace

int main() {
    const int tests[]{
        test_cqp_and_sparse_operator(),
        test_hcw_semigroup(),
        test_powered_descent(),
        test_scenarios(),
        test_scvx_policies(),
    };
    for (const auto result : tests) {
        if (result != 0) {
            return result;
        }
    }
    return 0;
}
