#include "spacepdhcg/core/host_pdhg.hpp"
#include "spacepdhcg/core/risk_cqp.hpp"
#include "spacepdhcg/core/scenario_cqp.hpp"
#include "spacepdhcg/core/sparse_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace {

using namespace spacepdhcg::core;

struct LocalFixture {
    CQPStructure structure;
    CQPValues values;
};

LocalFixture make_local_fixture(const double control_bias) {
    CscBuilder quadratic{3, 3};
    quadratic.add(0, 0, 0.0);
    quadratic.add(1, 1, 0.0);
    quadratic.add(2, 2, 2.0);

    CscBuilder scalar{2, 3};
    scalar.add(0, 0, 1.0);
    scalar.add(1, 0, -1.0);
    scalar.add(1, 1, 1.0);
    scalar.add(1, 2, -1.0);

    const auto q = quadratic.build();
    const auto a = scalar.build();
    CQPStructure structure{q.structure, a.structure};
    const double infinity = std::numeric_limits<double>::infinity();
    CQPValues values{
        q.values,
        a.values,
        {},
        {0.0, 0.0, control_bias},
        {0.0, 0.0},
        {0.0, 0.0},
        {},
        {-infinity, -infinity, -infinity},
        {infinity, infinity, infinity},
    };
    validate_values(structure, values);
    return LocalFixture{std::move(structure), std::move(values)};
}

HostPDHGOptions solve_options() {
    HostPDHGOptions options;
    options.tolerance = 2.0e-6;
    options.iteration_limit = 500'000U;
    options.check_interval = 25U;
    options.norm_iterations = 60U;
    options.step_safety = 0.9;
    return options;
}

bool near(const double actual, const double expected, const double tolerance) {
    return std::isfinite(actual) && std::abs(actual - expected) <= tolerance;
}

int test_worst_case() {
    const auto first = make_local_fixture(-2.0);
    const auto second = make_local_fixture(1.0);
    const auto tree = ScenarioTree::common_open_loop(2U, 1U);
    const ScenarioCQPBundle base{tree, first.structure, 1U, 1U};
    const ScenarioRiskCQPBundle risk{base, ScenarioRiskMeasure::worst_case};
    const std::vector<CQPValues> local_values{first.values, second.values};
    const auto values = risk.values(local_values);
    PersistentHostPDHG solver{risk.structure(), values};
    const auto solution = solver.solve(solve_options());
    if (!solution.solved() || solution.primal_residual > 2.0e-5 ||
        solution.dual_residual > 2.0e-5) {
        return 1;
    }
    const auto decoded = risk.decode_primal(solution.primal);
    if (!near(decoded.scenario.local[0][2], 0.0, 5.0e-4) ||
        !near(decoded.scenario.local[1][2], 0.0, 5.0e-4) ||
        !near(decoded.threshold, 0.0, 7.0e-4) ||
        !near(solution.objective, 0.0, 7.0e-4)) {
        return 2;
    }
    if (risk.maximum_cost_epigraph_violation(solution.primal, local_values) >
        2.0e-5) {
        return 3;
    }
    const std::span<const double> base_primal{
        solution.primal.data(),
        risk.base_variable_count(),
    };
    if (base.maximum_nonanticipativity_violation(base_primal) > 2.0e-5) {
        return 4;
    }
    return 0;
}

int test_cvar() {
    const auto first = make_local_fixture(-2.0);
    const auto second = make_local_fixture(-1.0);
    const auto tree = ScenarioTree::common_open_loop(2U, 1U);
    const ScenarioCQPBundle base{tree, first.structure, 1U, 1U};
    const ScenarioRiskCQPBundle risk{
        base,
        ScenarioRiskMeasure::conditional_value_at_risk,
    };
    const std::vector<CQPValues> local_values{first.values, second.values};
    constexpr double alpha = 0.5;
    const auto values = risk.values(local_values, alpha);
    PersistentHostPDHG solver{risk.structure(), values};
    const auto solution = solver.solve(solve_options());
    if (!solution.solved() || solution.primal_residual > 3.0e-5 ||
        solution.dual_residual > 3.0e-5) {
        return 5;
    }
    const auto decoded = risk.decode_primal(solution.primal);
    if (!near(decoded.scenario.local[0][2], 0.5, 8.0e-4) ||
        !near(decoded.scenario.local[1][2], 0.5, 8.0e-4) ||
        decoded.excesses.size() != 2U) {
        return 6;
    }
    const double evaluated = risk.evaluated_risk(
        solution.primal,
        local_values,
        alpha
    );
    if (!near(evaluated, -0.25, 1.5e-3) ||
        !near(solution.objective, evaluated, 2.0e-3)) {
        return 7;
    }
    if (risk.maximum_cost_epigraph_violation(solution.primal, local_values) >
        3.0e-5) {
        return 8;
    }

    const std::vector<double> costs{-0.75, -0.25};
    const std::vector<double> probabilities{0.5, 0.5};
    if (!near(
            conditional_value_at_risk(costs, probabilities, alpha),
            -0.25,
            1.0e-12
        )) {
        return 9;
    }
    return 0;
}

}  // namespace

int main() {
    const int worst_case = test_worst_case();
    if (worst_case != 0) {
        return worst_case;
    }
    return test_cvar();
}
