#include "spacepdhcg/native/robust_cqp.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace native = spacepdhcg::native;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

native::OwnedCqp make_local_problem() {
    native::CscBuilder quadratic(3, 3);
    quadratic.add(0, 0, 2.0);
    quadratic.add(1, 1, 4.0);
    quadratic.add(2, 2, 6.0);
    native::CscBuilder scalar(0, 3);
    native::CscBuilder affine(0, 3);
    return native::OwnedCqp{
        quadratic.build(),
        scalar.build(),
        affine.build(),
        {-1.0, 0.5, 2.0},
        {},
        {},
        {},
        {
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
        },
        {
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
        },
        {},
        {},
    };
}

std::vector<double> base_primal(
    const native::ScenarioCqpAssembler& assembler,
    const std::array<double, 3>& local
) {
    std::vector<double> result(assembler.block_arrow().total_variables(), 0.0);
    for (std::size_t scenario = 0; scenario < assembler.tree().scenario_count(); ++scenario) {
        const auto slice = assembler.block_arrow().scenario_slice(scenario);
        for (std::size_t index = 0; index < local.size(); ++index) {
            result[slice.offset + index] = local[index];
        }
    }
    for (const auto& block : assembler.block_arrow().consensus_blocks()) {
        result[block.variables.offset] = local[2];
    }
    return result;
}

void test_expected_and_epigraph_risks() {
    const auto local_problem = make_local_problem();
    const auto tree = native::ScenarioTree::common_open_loop(2, 1, 1);
    native::ScenarioCqpAssembler assembler(tree, local_problem, 1, 1, 0);
    const std::array<native::OwnedCqp, 2> locals{local_problem, local_problem};
    const std::array<double, 3> local_primal{1.0, 2.0, 0.5};
    const double local_objective = local_problem.objective(local_primal);

    const auto expected = assembler.assemble(
        locals,
        native::ScenarioRiskMode::expected_value
    );
    const auto expected_primal = base_primal(assembler, local_primal);
    require(expected.diagnostics(expected_primal).maximum_violation() < 1.0e-14,
            "expected-value robust primal is infeasible");
    require(std::abs(expected.objective(expected_primal) - local_objective) < 1.0e-13,
            "expected-value objective does not match identical local objectives");

    const auto worst = assembler.assemble(locals, native::ScenarioRiskMode::worst_case);
    auto worst_primal = base_primal(assembler, local_primal);
    worst_primal.resize(worst_primal.size() + 1U, 0.0);
    const auto worst_layout = assembler.layout(native::ScenarioRiskMode::worst_case);
    worst_primal[worst_layout.threshold_index()] = local_objective;
    require(worst.diagnostics(worst_primal).maximum_violation() < 2.0e-12,
            "worst-case quadratic epigraph is infeasible at the exact scenario cost");
    require(std::abs(worst.objective(worst_primal) - local_objective) < 1.0e-13,
            "worst-case objective is not the epigraph threshold");

    const auto cvar = assembler.assemble(
        locals,
        native::ScenarioRiskMode::conditional_value_at_risk,
        0.9
    );
    auto cvar_primal = base_primal(assembler, local_primal);
    cvar_primal.resize(cvar_primal.size() + 3U, 0.0);
    const auto cvar_layout = assembler.layout(
        native::ScenarioRiskMode::conditional_value_at_risk
    );
    cvar_primal[cvar_layout.threshold_index()] = local_objective;
    require(cvar.diagnostics(cvar_primal).maximum_violation() < 2.0e-12,
            "CVaR quadratic epigraph is infeasible at zero excess");
    require(std::abs(cvar.objective(cvar_primal) - local_objective) < 1.0e-13,
            "CVaR objective is wrong for identical scenarios");

    const auto decoded = assembler.decode(
        cvar_primal,
        native::ScenarioRiskMode::conditional_value_at_risk
    );
    const auto decoded_objectives = assembler.local_objectives(decoded, locals);
    require(decoded_objectives.size() == 2,
            "robust primal decoding lost a scenario");
    require(std::abs(decoded_objectives[0] - local_objective) < 1.0e-13 &&
                std::abs(decoded_objectives[1] - local_objective) < 1.0e-13,
            "decoded local objectives are wrong");
    require(assembler.block_arrow().nonanticipativity_violation(cvar_primal) < 1.0e-14,
            "robust primal violates shared controls");
}

void test_risk_evaluation() {
    const auto local_problem = make_local_problem();
    const std::array<double, 3> probabilities{0.2, 0.3, 0.5};
    const auto tree = native::ScenarioTree::common_open_loop(
        3,
        1,
        1,
        probabilities
    );
    native::ScenarioCqpAssembler assembler(tree, local_problem, 1, 1, 0);
    const std::array<double, 3> losses{1.0, 4.0, 10.0};
    require(std::abs(
                assembler.aggregate_risk(
                    losses,
                    native::ScenarioRiskMode::expected_value
                ) - 6.4
            ) < 1.0e-14,
            "weighted expected loss is wrong");
    require(assembler.aggregate_risk(losses, native::ScenarioRiskMode::worst_case) == 10.0,
            "worst-case loss is wrong");
    require(std::abs(
                assembler.aggregate_risk(
                    losses,
                    native::ScenarioRiskMode::conditional_value_at_risk,
                    0.5
                ) - 10.0
            ) < 1.0e-14,
            "weighted CVaR is wrong");
}

}  // namespace

int main() {
    test_expected_and_epigraph_risks();
    test_risk_evaluation();
    return 0;
}
