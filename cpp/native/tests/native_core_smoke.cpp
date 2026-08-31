#include "spacepdhcg/native/cqp.hpp"
#include "spacepdhcg/native/cw.hpp"
#include "spacepdhcg/native/cw_problem.hpp"
#include "spacepdhcg/native/scenario_partition.hpp"
#include "spacepdhcg/native/scvx_policy.hpp"
#include "spacepdhcg/native_c_api.h"

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

void test_sparse_cqp() {
    native::CscBuilder quadratic_builder(2, 2);
    quadratic_builder.add(0, 0, 2.0);
    quadratic_builder.add(1, 1, 4.0);

    native::CscBuilder scalar_builder(1, 2);
    scalar_builder.add(0, 0, 1.0);
    scalar_builder.add(0, 1, 1.0);

    native::CscBuilder affine_builder(3, 2);
    affine_builder.add(0, 0, 1.0);
    affine_builder.add(1, 1, 1.0);

    native::OwnedCqp problem{
        quadratic_builder.build(),
        scalar_builder.build(),
        affine_builder.build(),
        {0.0, 0.0},
        {0.0},
        {0.0},
        {0.0, 0.0, 2.0},
        {-10.0, -10.0},
        {10.0, 10.0},
        {{spacepdhcg::ConeKind::second_order, 0, 1, 0.0}},
        {},
    };
    problem.validate();

    const std::array<double, 2> decision{0.0, 0.0};
    require(std::abs(problem.objective(decision)) < 1.0e-15, "zero CQP objective is wrong");
    require(
        problem.diagnostics(decision).maximum_violation() < 1.0e-15,
        "feasible CQP was reported infeasible"
    );

    const auto transpose = problem.scalar_constraint.transpose_multiply(std::array{3.0});
    require(transpose.size() == 2 && transpose[0] == 3.0 && transpose[1] == 3.0,
            "CSC transpose product is wrong");
}

void test_cw_semigroup_and_c_api() {
    constexpr double mean_motion = 1.13e-3;
    const auto first = native::discretise_cw(mean_motion, 12.5);
    const auto second = native::discretise_cw(mean_motion, 7.5);
    const auto combined = native::discretise_cw(mean_motion, 20.0);
    const auto composed_state = native::multiply_state_matrices(second.state, first.state);
    const auto composed_control = native::compose_control_matrices(
        second.state,
        first.control,
        second.control
    );
    require(
        native::maximum_absolute_difference(composed_state, combined.state) < 2.0e-12,
        "HCW state transition violates the semigroup property"
    );
    require(
        native::maximum_absolute_difference(composed_control, combined.control) < 2.0e-10,
        "HCW input transition violates zero-order-hold composition"
    );

    native::CwState zero_state{};
    native::CwControl zero_control{};
    const auto propagated = native::propagate_cw(combined, zero_state, zero_control);
    require(
        native::maximum_absolute_difference(propagated, zero_state) < 1.0e-15,
        "zero HCW state did not remain zero"
    );

    std::array<double, 36> state_matrix{};
    std::array<double, 18> control_matrix{};
    require(spacepdhcg_native_abi_version() == 1, "native ABI version is wrong");
    require(
        spacepdhcg_cw_discretise(
            mean_motion,
            20.0,
            state_matrix.data(),
            state_matrix.size(),
            control_matrix.data(),
            control_matrix.size()
        ) == SPACEPDHCG_NATIVE_SUCCESS,
        "C ABI HCW call failed"
    );
    require(
        native::maximum_absolute_difference(state_matrix, combined.state) < 1.0e-15,
        "C ABI state matrix differs from C++ result"
    );
    require(
        native::maximum_absolute_difference(control_matrix, combined.control) < 1.0e-15,
        "C ABI control matrix differs from C++ result"
    );
}

void test_cw_cqp() {
    native::CwRendezvousConfig config{};
    config.intervals = 6;
    config.thrust_constraint = native::CwThrustConstraint::second_order_cone;
    native::CwRendezvousProblem model(config);

    native::CwState initial{};
    native::CwState target{};
    auto problem = model.make_cqp(initial, target);
    require(problem.variables() == model.layout().variables(), "CW CQP variable count is wrong");
    require(
        problem.affine_cones.size() == static_cast<std::size_t>(config.intervals),
        "CW CQP cone count is wrong"
    );

    std::vector<double> decision(static_cast<std::size_t>(model.layout().variables()), 0.0);
    require(problem.diagnostics(decision).maximum_violation() < 1.0e-15,
            "zero CW rendezvous should be conically feasible");
    require(model.diagnostics(decision, initial, target).maximum_violation() < 1.0e-15,
            "zero CW rendezvous should satisfy dynamics");

    const auto quadratic_offsets = problem.quadratic.offsets;
    const auto scalar_indices = problem.scalar_constraint.indices;
    target[0] = 10.0;
    target[4] = -0.01;
    model.update_numerical_values(problem, initial, target);
    require(problem.quadratic.offsets == quadratic_offsets,
            "CW numerical update changed quadratic topology");
    require(problem.scalar_constraint.indices == scalar_indices,
            "CW numerical update changed scalar topology");
    const auto terminal_state = static_cast<std::size_t>(
        model.layout().state_offset(config.intervals)
    );
    require(std::abs(problem.linear[terminal_state]) > 0.0,
            "CW target update did not change the linear objective");
}

void test_scvx_policies() {
    native::AdaptiveForcingRule forcing;
    const auto repair = forcing.request(
        0,
        native::OuterResidual{0.1, 0.02, 0.01, 1.0},
        0
    );
    require(repair.phase == native::SolvePhase::repair, "large defect did not request repair");

    const auto polish = forcing.request(
        8,
        native::OuterResidual{1.0e-6, 1.0e-6, 1.0e-6, 1.0e-3},
        3,
        0.95
    );
    require(polish.phase == native::SolvePhase::polish, "small stable defect did not polish");
    require(
        polish.tolerance == forcing.config().minimum_tolerance,
        "polish tolerance is not the configured minimum"
    );

    native::TrustRegionController trust;
    const auto contracted = trust.update(false, 0.0, 1.0);
    require(contracted.action == native::TrustAction::contract,
            "rejected step did not contract trust region");
    const auto expanded = trust.update(true, 0.99, 0.95);
    require(expanded.action == native::TrustAction::expand,
            "high-quality boundary step did not expand trust region");

    native::InexactErrorLedger ledger;
    ledger.record({1.0e-2, 1.0e-3, 5.0e-4});
    ledger.record({1.0e-3, 1.0e-4, 8.0e-5});
    require(ledger.respects_requested_tolerances(), "inexact solve exceeded requested tolerance");
    require(ledger.maximum_relative_forcing() < 0.1, "relative forcing diagnostic is wrong");
}

void test_scenario_partition() {
    const std::array<double, 6> weights{9.0, 8.0, 7.0, 2.0, 2.0, 1.0};
    const auto partition = native::partition_scenarios(weights, 3);
    require(partition.device_count() == 3, "scenario partition device count is wrong");
    require(partition.owner(0) < 3, "heavy scenario has no owner");
    require(partition.imbalance() < 1.25, "deterministic LPT partition is unexpectedly imbalanced");

    const native::LogicalGpuGrid grid{2, 3};
    require(grid.device_count() == 6 && grid.rank(1, 2) == 5, "logical GPU grid is wrong");

    const auto profile = native::ring_allreduce_profile(100, 4, sizeof(double), 2);
    require(profile.payload_bytes == 800, "collective payload is wrong");
    require(profile.bytes_per_device == 2400.0, "ring allreduce traffic is wrong");
    require(profile.aggregate_bytes == 9600.0, "aggregate collective traffic is wrong");
}

}  // namespace

int main() {
    test_sparse_cqp();
    test_cw_semigroup_and_c_api();
    test_cw_cqp();
    test_scvx_policies();
    test_scenario_partition();
    return 0;
}
