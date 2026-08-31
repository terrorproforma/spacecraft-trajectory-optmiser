#include "spacepdhcg/backends/dense_admm.hpp"
#include "spacepdhcg/distributed/risk_cqp.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace {

spacepdhcg::core::FixedCQP make_base_problem() {
    spacepdhcg::core::FixedStructure structure{};
    structure.quadratic = spacepdhcg::core::CscPattern{1, 1, {0, 1}, {0}};
    structure.scalar_constraint = spacepdhcg::core::CscPattern{0, 1, {0, 0}, {}};

    spacepdhcg::core::NumericValues values{};
    values.quadratic = {1.0};
    values.scalar_constraint = {};
    values.affine_cone = {};
    values.linear_objective = {0.0};
    values.scalar_lower = {};
    values.scalar_upper = {};
    values.affine_offset = {};
    values.variable_lower = {-10.0};
    values.variable_upper = {10.0};
    return spacepdhcg::core::FixedCQP(std::move(structure), std::move(values));
}

bool close(double left, double right, double tolerance = 2.0e-4) {
    return std::abs(left - right) <= tolerance;
}

}  // namespace

int main() {
    using spacepdhcg::backends::DenseAdmmBackend;
    using spacepdhcg::distributed::AffineScenarioLoss;
    using spacepdhcg::distributed::RiskAugmentedCqp;
    using spacepdhcg::distributed::RiskMeasure;

    const auto base = make_base_problem();
    const std::vector<std::vector<std::size_t>> patterns{{0U}, {0U}};
    const std::vector<AffineScenarioLoss> losses{
        AffineScenarioLoss{{0U}, {1.0}, 1.0},
        AffineScenarioLoss{{0U}, {-1.0}, 3.0},
    };
    const std::vector<double> probabilities{0.5, 0.5};

    const RiskAugmentedCqp expected(
        base.structure(),
        patterns,
        RiskMeasure::expected,
        0.5
    );
    if (expected.structure().variables() != 1 || expected.structure().scalar_rows() != 0) {
        return 1;
    }
    DenseAdmmBackend expected_solver(
        expected.problem(base.values(), losses, probabilities, 1.0)
    );
    const auto expected_solution = expected_solver.solve(1.0e-7, 100'000U);
    if (!expected_solution.solved() || !close(expected_solution.primal[0], 0.0)) {
        return 2;
    }

    const RiskAugmentedCqp worst(
        base.structure(),
        patterns,
        RiskMeasure::worst_case,
        0.5
    );
    if (worst.structure().variables() != 2 || worst.structure().scalar_rows() != 2) {
        return 3;
    }
    DenseAdmmBackend worst_solver(
        worst.problem(base.values(), losses, probabilities, 1.0)
    );
    const auto worst_solution = worst_solver.solve(1.0e-7, 200'000U);
    if (!worst_solution.solved()
        || !close(worst_solution.primal[0], 1.0)
        || !close(worst_solution.primal[*worst.threshold_index()], 2.0)) {
        return 4;
    }
    const auto worst_diagnostics = worst.diagnostics(
        worst_solution.primal,
        losses,
        probabilities
    );
    if (!close(worst_diagnostics.summary.worst, 2.0)
        || worst_diagnostics.maximum_epigraph_violation > 2.0e-5) {
        return 5;
    }

    const RiskAugmentedCqp cvar(
        base.structure(),
        patterns,
        RiskMeasure::conditional_value_at_risk,
        0.5
    );
    if (cvar.structure().variables() != 4 || cvar.structure().scalar_rows() != 2) {
        return 6;
    }
    DenseAdmmBackend cvar_solver(
        cvar.problem(base.values(), losses, probabilities, 1.0)
    );
    const auto cvar_solution = cvar_solver.solve(1.0e-7, 200'000U);
    if (!cvar_solution.solved()
        || !close(cvar_solution.primal[0], 1.0)
        || !close(cvar_solution.primal[*cvar.threshold_index()], 2.0)) {
        return 7;
    }
    const auto decoded = cvar.decode_risk_variables(cvar_solution.primal);
    if (decoded.excess.size() != 2U
        || decoded.excess[0] < -2.0e-5
        || decoded.excess[1] < -2.0e-5) {
        return 8;
    }
    const auto cvar_diagnostics = cvar.diagnostics(
        cvar_solution.primal,
        losses,
        probabilities
    );
    if (!close(cvar_diagnostics.summary.conditional_value_at_risk, 2.0)
        || cvar_diagnostics.maximum_epigraph_violation > 2.0e-5) {
        return 9;
    }

    return 0;
}
