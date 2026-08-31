#include "spacepdhcg/native/robust_cqp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace spacepdhcg::native {
namespace {

[[nodiscard]] Index checked_index(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        throw std::overflow_error(std::string(name) + " exceeds native Index capacity");
    }
    return static_cast<Index>(value);
}

template <typename Function>
void for_each_nonzero(const CscMatrix& matrix, Function&& function) {
    for (Index column = 0; column < matrix.columns; ++column) {
        for (Index position = matrix.offsets[static_cast<std::size_t>(column)];
             position < matrix.offsets[static_cast<std::size_t>(column) + 1U];
             ++position) {
            const auto index = static_cast<std::size_t>(position);
            function(matrix.indices[index], column, matrix.values[index]);
        }
    }
}

[[nodiscard]] bool same_pattern(const CscMatrix& left, const CscMatrix& right) {
    return left.rows == right.rows && left.columns == right.columns &&
           left.offsets == right.offsets && left.indices == right.indices;
}

[[nodiscard]] bool same_cones(
    const std::vector<ConeBlockDescriptor>& left,
    const std::vector<ConeBlockDescriptor>& right
) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto& lhs = left[index];
        const auto& rhs = right[index];
        if (lhs.kind != rhs.kind || lhs.start != rhs.start ||
            lhs.vector_dimension != rhs.vector_dimension ||
            lhs.power_alpha != rhs.power_alpha) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::vector<double> diagonal_quadratic(const OwnedCqp& problem) {
    const auto dimension = static_cast<std::size_t>(problem.variables());
    std::vector<double> diagonal(dimension, 0.0);
    for_each_nonzero(problem.quadratic, [&](Index row, Index column, double value) {
        if (row != column) {
            if (std::abs(value) > 1.0e-14) {
                throw std::invalid_argument(
                    "worst-case and CVaR assembly currently require diagonal PSD objectives"
                );
            }
            return;
        }
        if (value < -1.0e-14) {
            throw std::invalid_argument("risk epigraph requires a positive-semidefinite objective");
        }
        diagonal[static_cast<std::size_t>(row)] += std::max(value, 0.0);
    });
    return diagonal;
}

void append_local_scalar_blocks(
    CscBuilder& builder,
    std::span<const OwnedCqp> local_problems,
    std::size_t local_variables,
    std::size_t local_scalar_rows
) {
    for (std::size_t scenario = 0; scenario < local_problems.size(); ++scenario) {
        for_each_nonzero(
            local_problems[scenario].scalar_constraint,
            [&](Index row, Index column, double value) {
                builder.add(
                    checked_index(
                        scenario * local_scalar_rows + static_cast<std::size_t>(row),
                        "global scalar row"
                    ),
                    checked_index(
                        scenario * local_variables + static_cast<std::size_t>(column),
                        "global local-variable column"
                    ),
                    value
                );
            }
        );
    }
}

void append_nonanticipativity(
    CscBuilder& builder,
    const BlockArrowLayout& layout,
    std::size_t row_offset
) {
    const auto nonanticipativity = layout.nonanticipativity_operator();
    for_each_nonzero(nonanticipativity, [&](Index row, Index column, double value) {
        builder.add(
            checked_index(row_offset + static_cast<std::size_t>(row), "global consensus row"),
            column,
            value
        );
    });
}

void append_local_affine_blocks(
    CscBuilder& builder,
    std::span<const OwnedCqp> local_problems,
    std::size_t local_variables,
    std::size_t local_affine_rows
) {
    for (std::size_t scenario = 0; scenario < local_problems.size(); ++scenario) {
        for_each_nonzero(
            local_problems[scenario].affine_cone,
            [&](Index row, Index column, double value) {
                builder.add(
                    checked_index(
                        scenario * local_affine_rows + static_cast<std::size_t>(row),
                        "global affine row"
                    ),
                    checked_index(
                        scenario * local_variables + static_cast<std::size_t>(column),
                        "global local-variable column"
                    ),
                    value
                );
            }
        );
    }
}

[[nodiscard]] std::vector<ConeBlockDescriptor> repeated_affine_cones(
    std::span<const OwnedCqp> local_problems
) {
    std::vector<ConeBlockDescriptor> result;
    if (local_problems.empty()) {
        return result;
    }
    const auto local_rows = static_cast<std::size_t>(local_problems.front().affine_cone.rows);
    result.reserve(local_problems.size() * local_problems.front().affine_cones.size());
    for (std::size_t scenario = 0; scenario < local_problems.size(); ++scenario) {
        for (const auto& cone : local_problems[scenario].affine_cones) {
            result.push_back(ConeBlockDescriptor{
                cone.kind,
                checked_index(
                    scenario * local_rows + static_cast<std::size_t>(cone.start),
                    "global affine cone start"
                ),
                cone.vector_dimension,
                cone.power_alpha,
            });
        }
    }
    return result;
}

[[nodiscard]] std::vector<ConeBlockDescriptor> repeated_variable_cones(
    std::span<const OwnedCqp> local_problems
) {
    std::vector<ConeBlockDescriptor> result;
    if (local_problems.empty()) {
        return result;
    }
    const auto local_variables = static_cast<std::size_t>(local_problems.front().variables());
    result.reserve(local_problems.size() * local_problems.front().variable_cones.size());
    for (std::size_t scenario = 0; scenario < local_problems.size(); ++scenario) {
        for (const auto& cone : local_problems[scenario].variable_cones) {
            result.push_back(ConeBlockDescriptor{
                cone.kind,
                checked_index(
                    scenario * local_variables + static_cast<std::size_t>(cone.start),
                    "global variable cone start"
                ),
                cone.vector_dimension,
                cone.power_alpha,
            });
        }
    }
    return result;
}

}  // namespace

std::size_t RobustCqpLayout::threshold_index() const {
    if (mode == ScenarioRiskMode::expected_value) {
        throw std::logic_error("expected-value CQP has no risk threshold variable");
    }
    return risk_offset;
}

std::size_t RobustCqpLayout::excess_index(std::size_t scenario) const {
    if (mode != ScenarioRiskMode::conditional_value_at_risk) {
        throw std::logic_error("only CVaR CQP has scenario excess variables");
    }
    if (scenario >= scenario_count) {
        throw std::out_of_range("CVaR scenario index lies outside the risk layout");
    }
    return risk_offset + 1U + scenario;
}

ScenarioCqpAssembler::ScenarioCqpAssembler(
    ScenarioTree tree,
    OwnedCqp local_prototype,
    std::size_t state_dimension,
    std::size_t control_dimension,
    std::size_t local_auxiliary_dimension
)
    : tree_(std::move(tree)),
      local_prototype_(std::move(local_prototype)),
      block_arrow_(
          tree_,
          state_dimension,
          control_dimension,
          local_auxiliary_dimension
      ) {
    local_prototype_.validate();
    if (block_arrow_.local_variables_per_scenario() !=
        static_cast<std::size_t>(local_prototype_.variables())) {
        throw std::invalid_argument(
            "scenario state/control/auxiliary layout does not match the local CQP"
        );
    }
}

void ScenarioCqpAssembler::validate_local_problems(
    std::span<const OwnedCqp> local_problems
) const {
    if (local_problems.size() != tree_.scenario_count()) {
        throw std::invalid_argument("one local CQP is required per scenario");
    }
    for (const auto& problem : local_problems) {
        problem.validate();
        if (!same_pattern(problem.quadratic, local_prototype_.quadratic) ||
            !same_pattern(problem.scalar_constraint, local_prototype_.scalar_constraint) ||
            !same_pattern(problem.affine_cone, local_prototype_.affine_cone) ||
            !same_cones(problem.affine_cones, local_prototype_.affine_cones) ||
            !same_cones(problem.variable_cones, local_prototype_.variable_cones)) {
            throw std::invalid_argument("scenario-local CQP topology differs from the prototype");
        }
    }
}

RobustCqpLayout ScenarioCqpAssembler::layout(ScenarioRiskMode mode) const {
    const std::size_t base = block_arrow_.total_variables();
    std::size_t risk_variables = 0;
    if (mode == ScenarioRiskMode::worst_case) {
        risk_variables = 1;
    } else if (mode == ScenarioRiskMode::conditional_value_at_risk) {
        risk_variables = 1U + tree_.scenario_count();
    }
    return RobustCqpLayout{
        base,
        base,
        risk_variables,
        tree_.scenario_count(),
        mode,
    };
}

OwnedCqp ScenarioCqpAssembler::assemble(
    std::span<const OwnedCqp> local_problems,
    ScenarioRiskMode mode,
    double cvar_alpha
) const {
    validate_local_problems(local_problems);
    if (mode == ScenarioRiskMode::expected_value) {
        return assemble_expected(local_problems);
    }
    return assemble_epigraph_risk(local_problems, mode, cvar_alpha);
}

OwnedCqp ScenarioCqpAssembler::assemble_expected(
    std::span<const OwnedCqp> local_problems
) const {
    const auto probabilities = tree_.probabilities();
    const std::size_t scenario_count = tree_.scenario_count();
    const std::size_t local_variables = static_cast<std::size_t>(local_prototype_.variables());
    const std::size_t local_scalar_rows =
        static_cast<std::size_t>(local_prototype_.scalar_constraint.rows);
    const std::size_t local_affine_rows =
        static_cast<std::size_t>(local_prototype_.affine_cone.rows);
    const std::size_t nonanticipativity_rows = block_arrow_.nonanticipativity_rows();
    const std::size_t variables = block_arrow_.total_variables();

    CscBuilder quadratic(
        checked_index(variables, "expected-value variable count"),
        checked_index(variables, "expected-value variable count")
    );
    for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
        for_each_nonzero(
            local_problems[scenario].quadratic,
            [&](Index row, Index column, double value) {
                quadratic.add(
                    checked_index(
                        scenario * local_variables + static_cast<std::size_t>(row),
                        "expected-value quadratic row"
                    ),
                    checked_index(
                        scenario * local_variables + static_cast<std::size_t>(column),
                        "expected-value quadratic column"
                    ),
                    probabilities[scenario] * value
                );
            }
        );
    }

    const std::size_t scalar_rows = scenario_count * local_scalar_rows +
                                    nonanticipativity_rows;
    CscBuilder scalar(
        checked_index(scalar_rows, "expected-value scalar row count"),
        checked_index(variables, "expected-value variable count")
    );
    append_local_scalar_blocks(scalar, local_problems, local_variables, local_scalar_rows);
    append_nonanticipativity(
        scalar,
        block_arrow_,
        scenario_count * local_scalar_rows
    );

    const std::size_t affine_rows = scenario_count * local_affine_rows;
    CscBuilder affine(
        checked_index(affine_rows, "expected-value affine row count"),
        checked_index(variables, "expected-value variable count")
    );
    append_local_affine_blocks(affine, local_problems, local_variables, local_affine_rows);

    std::vector<double> linear(variables, 0.0);
    std::vector<double> scalar_lower;
    std::vector<double> scalar_upper;
    std::vector<double> affine_offset;
    std::vector<double> variable_lower(
        variables,
        -std::numeric_limits<double>::infinity()
    );
    std::vector<double> variable_upper(
        variables,
        std::numeric_limits<double>::infinity()
    );
    scalar_lower.reserve(scalar_rows);
    scalar_upper.reserve(scalar_rows);
    affine_offset.reserve(affine_rows);

    for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
        const auto& local = local_problems[scenario];
        for (std::size_t index = 0; index < local_variables; ++index) {
            linear[scenario * local_variables + index] =
                probabilities[scenario] * local.linear[index];
            variable_lower[scenario * local_variables + index] =
                local.variable_lower[index];
            variable_upper[scenario * local_variables + index] =
                local.variable_upper[index];
        }
        scalar_lower.insert(
            scalar_lower.end(),
            local.scalar_lower.begin(),
            local.scalar_lower.end()
        );
        scalar_upper.insert(
            scalar_upper.end(),
            local.scalar_upper.begin(),
            local.scalar_upper.end()
        );
        affine_offset.insert(
            affine_offset.end(),
            local.affine_offset.begin(),
            local.affine_offset.end()
        );
    }
    scalar_lower.insert(scalar_lower.end(), nonanticipativity_rows, 0.0);
    scalar_upper.insert(scalar_upper.end(), nonanticipativity_rows, 0.0);

    OwnedCqp result{
        quadratic.build(),
        scalar.build(),
        affine.build(),
        std::move(linear),
        std::move(scalar_lower),
        std::move(scalar_upper),
        std::move(affine_offset),
        std::move(variable_lower),
        std::move(variable_upper),
        repeated_affine_cones(local_problems),
        repeated_variable_cones(local_problems),
    };
    result.validate();
    return result;
}

OwnedCqp ScenarioCqpAssembler::assemble_epigraph_risk(
    std::span<const OwnedCqp> local_problems,
    ScenarioRiskMode mode,
    double cvar_alpha
) const {
    if (mode == ScenarioRiskMode::expected_value) {
        throw std::logic_error("epigraph assembler requires worst-case or CVaR mode");
    }
    if (mode == ScenarioRiskMode::conditional_value_at_risk &&
        (!std::isfinite(cvar_alpha) || !(cvar_alpha > 0.0 && cvar_alpha < 1.0))) {
        throw std::invalid_argument("CVaR alpha must lie strictly between zero and one");
    }

    const auto robust_layout = layout(mode);
    const auto probabilities = tree_.probabilities();
    const std::size_t scenario_count = tree_.scenario_count();
    const std::size_t local_variables = static_cast<std::size_t>(local_prototype_.variables());
    const std::size_t local_scalar_rows =
        static_cast<std::size_t>(local_prototype_.scalar_constraint.rows);
    const std::size_t local_affine_rows =
        static_cast<std::size_t>(local_prototype_.affine_cone.rows);
    const std::size_t nonanticipativity_rows = block_arrow_.nonanticipativity_rows();
    const std::size_t variables = robust_layout.variables();
    const std::size_t scalar_rows = scenario_count * local_scalar_rows +
                                    nonanticipativity_rows;
    const std::size_t risk_slots = local_variables + 2U;
    const std::size_t affine_rows = scenario_count * local_affine_rows +
                                    scenario_count * risk_slots;

    CscBuilder quadratic(
        checked_index(variables, "risk variable count"),
        checked_index(variables, "risk variable count")
    );
    CscBuilder scalar(
        checked_index(scalar_rows, "risk scalar row count"),
        checked_index(variables, "risk variable count")
    );
    append_local_scalar_blocks(scalar, local_problems, local_variables, local_scalar_rows);
    append_nonanticipativity(
        scalar,
        block_arrow_,
        scenario_count * local_scalar_rows
    );

    CscBuilder affine(
        checked_index(affine_rows, "risk affine row count"),
        checked_index(variables, "risk variable count")
    );
    append_local_affine_blocks(affine, local_problems, local_variables, local_affine_rows);

    std::vector<ConeBlockDescriptor> affine_cones = repeated_affine_cones(local_problems);
    std::vector<double> affine_offset(affine_rows, 0.0);
    for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
        std::copy(
            local_problems[scenario].affine_offset.begin(),
            local_problems[scenario].affine_offset.end(),
            affine_offset.begin() + static_cast<std::ptrdiff_t>(scenario * local_affine_rows)
        );

        const auto diagonal = diagonal_quadratic(local_problems[scenario]);
        const std::size_t risk_start = scenario_count * local_affine_rows +
                                       scenario * risk_slots;
        const std::size_t local_start = scenario * local_variables;
        for (std::size_t index = 0; index < local_variables; ++index) {
            affine.add(
                checked_index(risk_start + index, "risk vector row"),
                checked_index(local_start + index, "risk local column"),
                std::sqrt(diagonal[index])
            );
        }

        const std::size_t epigraph_row = risk_start + local_variables;
        for (std::size_t index = 0; index < local_variables; ++index) {
            affine.add(
                checked_index(epigraph_row, "risk epigraph row"),
                checked_index(local_start + index, "risk local column"),
                -local_problems[scenario].linear[index]
            );
        }
        affine.add(
            checked_index(epigraph_row, "risk epigraph row"),
            checked_index(robust_layout.threshold_index(), "risk threshold column"),
            1.0
        );
        if (mode == ScenarioRiskMode::conditional_value_at_risk) {
            affine.add(
                checked_index(epigraph_row, "risk epigraph row"),
                checked_index(robust_layout.excess_index(scenario), "CVaR excess column"),
                1.0
            );
        }
        affine_offset[risk_start + local_variables + 1U] = 1.0;
        affine_cones.push_back(ConeBlockDescriptor{
            ConeKind::rotated_second_order,
            checked_index(risk_start, "risk cone start"),
            checked_index(local_variables, "risk cone vector dimension"),
            0.0,
        });
    }

    std::vector<double> linear(variables, 0.0);
    linear[robust_layout.threshold_index()] = 1.0;
    if (mode == ScenarioRiskMode::conditional_value_at_risk) {
        const double scale = 1.0 / (1.0 - cvar_alpha);
        for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
            linear[robust_layout.excess_index(scenario)] =
                scale * probabilities[scenario];
        }
    }

    std::vector<double> scalar_lower;
    std::vector<double> scalar_upper;
    scalar_lower.reserve(scalar_rows);
    scalar_upper.reserve(scalar_rows);
    std::vector<double> variable_lower(
        variables,
        -std::numeric_limits<double>::infinity()
    );
    std::vector<double> variable_upper(
        variables,
        std::numeric_limits<double>::infinity()
    );
    for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
        const auto& local = local_problems[scenario];
        scalar_lower.insert(
            scalar_lower.end(),
            local.scalar_lower.begin(),
            local.scalar_lower.end()
        );
        scalar_upper.insert(
            scalar_upper.end(),
            local.scalar_upper.begin(),
            local.scalar_upper.end()
        );
        for (std::size_t index = 0; index < local_variables; ++index) {
            variable_lower[scenario * local_variables + index] =
                local.variable_lower[index];
            variable_upper[scenario * local_variables + index] =
                local.variable_upper[index];
        }
    }
    scalar_lower.insert(scalar_lower.end(), nonanticipativity_rows, 0.0);
    scalar_upper.insert(scalar_upper.end(), nonanticipativity_rows, 0.0);
    if (mode == ScenarioRiskMode::conditional_value_at_risk) {
        for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
            variable_lower[robust_layout.excess_index(scenario)] = 0.0;
        }
    }

    OwnedCqp result{
        quadratic.build(),
        scalar.build(),
        affine.build(),
        std::move(linear),
        std::move(scalar_lower),
        std::move(scalar_upper),
        std::move(affine_offset),
        std::move(variable_lower),
        std::move(variable_upper),
        std::move(affine_cones),
        repeated_variable_cones(local_problems),
    };
    result.validate();
    return result;
}

RobustPrimal ScenarioCqpAssembler::decode(
    std::span<const double> primal,
    ScenarioRiskMode mode
) const {
    const auto robust_layout = layout(mode);
    if (primal.size() != robust_layout.variables()) {
        throw std::invalid_argument("robust primal has the wrong dimension");
    }
    for (double value : primal) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("robust primal must be finite");
        }
    }

    RobustPrimal result{};
    result.local.reserve(tree_.scenario_count());
    for (std::size_t scenario = 0; scenario < tree_.scenario_count(); ++scenario) {
        const auto slice = block_arrow_.scenario_slice(scenario);
        result.local.emplace_back(
            primal.begin() + static_cast<std::ptrdiff_t>(slice.offset),
            primal.begin() + static_cast<std::ptrdiff_t>(slice.stop())
        );
    }
    result.consensus.reserve(block_arrow_.consensus_blocks().size());
    for (const auto& block : block_arrow_.consensus_blocks()) {
        result.consensus.emplace_back(
            primal.begin() + static_cast<std::ptrdiff_t>(block.variables.offset),
            primal.begin() + static_cast<std::ptrdiff_t>(block.variables.stop())
        );
    }
    if (mode != ScenarioRiskMode::expected_value) {
        result.threshold = primal[robust_layout.threshold_index()];
    }
    if (mode == ScenarioRiskMode::conditional_value_at_risk) {
        result.excess.reserve(tree_.scenario_count());
        for (std::size_t scenario = 0; scenario < tree_.scenario_count(); ++scenario) {
            result.excess.push_back(primal[robust_layout.excess_index(scenario)]);
        }
    }
    return result;
}

std::vector<double> ScenarioCqpAssembler::local_objectives(
    const RobustPrimal& primal,
    std::span<const OwnedCqp> local_problems
) const {
    validate_local_problems(local_problems);
    if (primal.local.size() != tree_.scenario_count()) {
        throw std::invalid_argument("decoded robust primal has the wrong scenario count");
    }
    std::vector<double> result(tree_.scenario_count(), 0.0);
    for (std::size_t scenario = 0; scenario < tree_.scenario_count(); ++scenario) {
        result[scenario] = local_problems[scenario].objective(primal.local[scenario]);
    }
    return result;
}

double ScenarioCqpAssembler::aggregate_risk(
    std::span<const double> scenario_losses,
    ScenarioRiskMode mode,
    double cvar_alpha
) const {
    if (scenario_losses.size() != tree_.scenario_count()) {
        throw std::invalid_argument("one loss is required per scenario");
    }
    for (double loss : scenario_losses) {
        if (!std::isfinite(loss)) {
            throw std::invalid_argument("scenario losses must be finite");
        }
    }
    const auto probabilities = tree_.probabilities();
    if (mode == ScenarioRiskMode::expected_value) {
        return std::inner_product(
            scenario_losses.begin(),
            scenario_losses.end(),
            probabilities.begin(),
            0.0
        );
    }
    if (mode == ScenarioRiskMode::worst_case) {
        return *std::max_element(scenario_losses.begin(), scenario_losses.end());
    }
    if (!std::isfinite(cvar_alpha) || !(cvar_alpha > 0.0 && cvar_alpha < 1.0)) {
        throw std::invalid_argument("CVaR alpha must lie strictly between zero and one");
    }

    double best = std::numeric_limits<double>::infinity();
    for (double threshold : scenario_losses) {
        double value = threshold;
        for (std::size_t scenario = 0; scenario < scenario_losses.size(); ++scenario) {
            value += probabilities[scenario] /
                     (1.0 - cvar_alpha) *
                     std::max(scenario_losses[scenario] - threshold, 0.0);
        }
        best = std::min(best, value);
    }
    return best;
}

}  // namespace spacepdhcg::native
