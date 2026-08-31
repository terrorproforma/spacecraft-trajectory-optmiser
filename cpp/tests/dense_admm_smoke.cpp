#include "spacepdhcg/backends/dense_admm.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

spacepdhcg::core::FixedCQP equality_qp() {
    using spacepdhcg::core::CscPattern;
    using spacepdhcg::core::FixedStructure;
    using spacepdhcg::core::NumericValues;
    const FixedStructure structure{
        CscPattern{2, 2, {0, 1, 2}, {0, 1}},
        CscPattern{1, 2, {0, 1, 2}, {0, 0}},
        std::nullopt,
        {},
        {},
    };
    const auto infinity = std::numeric_limits<double>::infinity();
    return spacepdhcg::core::FixedCQP(
        structure,
        NumericValues{
            {2.0, 2.0},
            {1.0, 1.0},
            {},
            {-2.0, -4.0},
            {3.0},
            {3.0},
            {},
            {0.0, 0.0},
            {infinity, infinity},
        }
    );
}

spacepdhcg::core::FixedCQP unit_ball_socp() {
    using spacepdhcg::ConeBlockDescriptor;
    using spacepdhcg::ConeKind;
    using spacepdhcg::core::CscPattern;
    using spacepdhcg::core::FixedStructure;
    using spacepdhcg::core::NumericValues;
    const FixedStructure structure{
        CscPattern{2, 2, {0, 1, 2}, {0, 1}},
        CscPattern{0, 2, {0, 0, 0}, {}},
        CscPattern{3, 2, {0, 1, 2}, {0, 1}},
        {ConeBlockDescriptor{ConeKind::second_order, 0, 1, 0.0}},
        {},
    };
    const auto infinity = std::numeric_limits<double>::infinity();
    return spacepdhcg::core::FixedCQP(
        structure,
        NumericValues{
            {2.0, 2.0},
            {},
            {1.0, 1.0},
            {-4.0, 0.0},
            {},
            {},
            {0.0, 0.0, 1.0},
            {-infinity, -infinity},
            {infinity, infinity},
        }
    );
}

bool near(double value, double expected, double tolerance = 2.0e-5) {
    return std::abs(value - expected) <= tolerance;
}

}  // namespace

int main() {
    using spacepdhcg::backends::DenseAdmmBackend;
    using spacepdhcg::backends::DenseAdmmConfig;
    using spacepdhcg::core::HostWarmStart;

    DenseAdmmBackend qp(equality_qp(), DenseAdmmConfig{1.0, 1.0e-10, 128U});
    const auto qp_solution = qp.solve(1.0e-8, 20'000U);
    if (!qp_solution.solved() || !near(qp_solution.primal[0U], 1.0)
        || !near(qp_solution.primal[1U], 2.0)
        || qp_solution.primal_residual > 1.0e-7
        || qp_solution.dual_residual > 1.0e-7) {
        return 1;
    }

    auto soc_problem = unit_ball_socp();
    DenseAdmmBackend socp(soc_problem, DenseAdmmConfig{1.0, 1.0e-10, 128U});
    const auto first = socp.solve(1.0e-7, 30'000U);
    if (!first.solved() || !near(first.primal[0U], 1.0, 1.0e-4)
        || !near(first.primal[1U], 0.0, 1.0e-4)) {
        return 2;
    }

    auto updated = soc_problem.values();
    updated.linear_objective = {0.0, -4.0};
    socp.update(updated);
    socp.warm_start(HostWarmStart{first.primal, first.dual});
    const auto second = socp.solve(1.0e-7, 30'000U);
    if (!second.solved() || !near(second.primal[0U], 0.0, 1.0e-4)
        || !near(second.primal[1U], 1.0, 1.0e-4)
        || socp.update_count() != 1U || socp.warm_start_count() != 1U
        || socp.solve_count() != 2U) {
        return 3;
    }
    return 0;
}
