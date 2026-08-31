#pragma once

#include "spacepdhcg/core/cqp.hpp"
#include "spacepdhcg/core/csc_operator.hpp"
#include "spacepdhcg/core/scenario_cqp.hpp"
#include "spacepdhcg/core/sparse_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spacepdhcg::core {

enum class ScenarioRiskMeasure {
    worst_case,
    conditional_value_at_risk,
};

struct ScenarioRiskPrimal {
    ScenarioCQPPrimal scenario;
    std::vector<double> cost_epigraphs;
    double threshold{0.0};
    std::vector<double> excesses;
};

[[nodiscard]] inline double worst_case_cost(std::span<const double> costs) {
    if (costs.empty()) {
        throw std::invalid_argument("worst-case risk requires at least one cost");
    }
    if (!std::all_of(costs.begin(), costs.end(), [](const double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument("scenario costs must be finite");
    }
    return *std::max_element(costs.begin(), costs.end());
}

[[nodiscard]] inline double conditional_value_at_risk(
    std::span<const double> costs,
    std::span<const double> probabilities,
    const double alpha
) {
    if (costs.empty() || costs.size() != probabilities.size()) {
        throw std::invalid_argument(
            "CVaR requires equally sized non-empty cost and probability vectors"
        );
    }
    if (!std::isfinite(alpha) || alpha < 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("CVaR alpha must lie in [0,1)");
    }
    double probability_sum = 0.0;
    std::vector<std::pair<double, double>> distribution;
    distribution.reserve(costs.size());
    for (std::size_t index = 0; index < costs.size(); ++index) {
        if (!std::isfinite(costs[index]) || !std::isfinite(probabilities[index]) ||
            probabilities[index] <= 0.0) {
            throw std::invalid_argument(
                "CVaR costs must be finite and probabilities finite and positive"
            );
        }
        probability_sum += probabilities[index];
        distribution.emplace_back(costs[index], probabilities[index]);
    }
    if (std::abs(probability_sum - 1.0) > 1.0e-12) {
        throw std::invalid_argument("CVaR probabilities must sum to one");
    }
    std::sort(distribution.begin(), distribution.end());
    double cumulative = 0.0;
    double threshold = distribution.back().first;
    for (const auto& [cost, probability] : distribution) {
        cumulative += probability;
        if (cumulative + 1.0e-15 >= alpha) {
            threshold = cost;
            break;
        }
    }
    double excess = 0.0;
    for (const auto& [cost, probability] : distribution) {
        excess += probability * std::max(0.0, cost - threshold);
    }
    return threshold + excess / (1.0 - alpha);
}

/// Exact epigraph formulation for worst-case and finite-scenario CVaR risk.
///
/// The wrapped ScenarioCQPBundle supplies feasibility and non-anticipativity rows.
/// Scenario-local quadratic objectives are removed from the global objective and
/// represented exactly by one rotated-SOC epigraph per scenario:
///
///     [sqrt(Q_s) z_s, t_s - c_s^T z_s, 1] in Q_r.
///
/// Therefore t_s >= 0.5 z_s^T Q_s z_s + c_s^T z_s.  The current CPU
/// implementation requires each local Q_s to be diagonal and positive semidefinite,
/// matching the dependency-free host PDHG backend.  The bundle is non-owning; the
/// referenced ScenarioCQPBundle must outlive it.
class ScenarioRiskCQPBundle {
  public:
    ScenarioRiskCQPBundle(
        const ScenarioCQPBundle& base,
        const ScenarioRiskMeasure measure
    )
        : base_(&base), measure_(measure) {
        validate_dimensions();
        structure_.emplace(build_structure());
    }

    [[nodiscard]] const ScenarioCQPBundle& base() const noexcept { return *base_; }
    [[nodiscard]] ScenarioRiskMeasure measure() const noexcept { return measure_; }
    [[nodiscard]] const CQPStructure& structure() const noexcept { return *structure_; }

    [[nodiscard]] std::size_t scenario_count() const noexcept {
        return base().scenario_count();
    }
    [[nodiscard]] std::size_t base_variable_count() const noexcept {
        return static_cast<std::size_t>(base().structure().variables());
    }
    [[nodiscard]] std::size_t local_variable_count() const noexcept {
        return static_cast<std::size_t>(base().local_structure().variables());
    }
    [[nodiscard]] std::size_t cost_epigraph_start() const noexcept {
        return base_variable_count();
    }
    [[nodiscard]] std::size_t threshold_index() const noexcept {
        return cost_epigraph_start() + scenario_count();
    }
    [[nodiscard]] std::size_t excess_start() const noexcept {
        return threshold_index() + 1U;
    }

    [[nodiscard]] CQPValues values(
        std::span<const CQPValues> local_values,
        const double alpha = 0.9
    ) const {
        validate_alpha(alpha);
        validate_local_values(local_values);
        const auto base_values = base().values(local_values);
        const auto matrices = assemble(base_values, local_values, false);
        if (matrices.quadratic.structure.fingerprint() !=
                structure().quadratic().fingerprint() ||
            matrices.scalar.structure.fingerprint() !=
                structure().scalar_constraint().fingerprint() ||
            matrices.affine.structure.fingerprint() !=
                structure().affine_cone()->fingerprint()) {
            throw std::logic_error("risk numerical assembly changed fixed sparse topology");
        }

        const double infinity = std::numeric_limits<double>::infinity();
        CQPValues result;
        result.quadratic = matrices.quadratic.values;
        result.scalar_constraint = matrices.scalar.values;
        result.affine_cone = matrices.affine.values;
        result.linear_objective.assign(total_variables(), 0.0);
        result.scalar_lower = base_values.scalar_lower;
        result.scalar_upper = base_values.scalar_upper;
        result.scalar_lower.insert(result.scalar_lower.end(), scenario_count(), -infinity);
        result.scalar_upper.insert(result.scalar_upper.end(), scenario_count(), 0.0);
        result.affine_offset = base_values.affine_offset;
        result.affine_offset.resize(total_affine_rows(), 0.0);
        result.variable_lower.assign(total_variables(), -infinity);
        result.variable_upper.assign(total_variables(), infinity);
        std::copy(
            base_values.variable_lower.begin(),
            base_values.variable_lower.end(),
            result.variable_lower.begin()
        );
        std::copy(
            base_values.variable_upper.begin(),
            base_values.variable_upper.end(),
            result.variable_upper.begin()
        );

        result.linear_objective[threshold_index()] = 1.0;
        if (measure_ == ScenarioRiskMeasure::conditional_value_at_risk) {
            const auto probabilities = base().tree().probabilities();
            for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
                result.linear_objective[excess_start() + scenario] =
                    probabilities[scenario] / (1.0 - alpha);
                result.variable_lower[excess_start() + scenario] = 0.0;
            }
        }
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto cone_start = risk_affine_row_start() +
                scenario * risk_cone_slots();
            result.affine_offset[cone_start + local_variable_count() + 1U] = 1.0;
        }
        validate_values(structure(), result);
        return result;
    }

    [[nodiscard]] ScenarioRiskPrimal decode_primal(
        std::span<const double> primal
    ) const {
        if (primal.size() != total_variables()) {
            throw std::invalid_argument("risk primal has an incompatible size");
        }
        ScenarioRiskPrimal result;
        result.scenario = base().decode_primal(primal.first(base_variable_count()));
        result.cost_epigraphs.assign(
            primal.begin() + static_cast<std::ptrdiff_t>(cost_epigraph_start()),
            primal.begin() + static_cast<std::ptrdiff_t>(threshold_index())
        );
        result.threshold = primal[threshold_index()];
        if (measure_ == ScenarioRiskMeasure::conditional_value_at_risk) {
            result.excesses.assign(
                primal.begin() + static_cast<std::ptrdiff_t>(excess_start()),
                primal.end()
            );
        }
        return result;
    }

    [[nodiscard]] std::vector<double> local_objectives(
        std::span<const std::vector<double>> local_primals,
        std::span<const CQPValues> local_values
    ) const {
        if (local_primals.size() != scenario_count()) {
            throw std::invalid_argument("one local primal is required per scenario");
        }
        validate_local_values(local_values);
        std::vector<double> result(scenario_count(), 0.0);
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto& primal = local_primals[scenario];
            if (primal.size() != local_variable_count()) {
                throw std::invalid_argument("local primal has an incompatible size");
            }
            CscOperator quadratic{
                base().local_structure().quadratic(),
                local_values[scenario].quadratic
            };
            const auto product = quadratic.multiply(primal);
            for (std::size_t index = 0; index < primal.size(); ++index) {
                result[scenario] += 0.5 * primal[index] * product[index] +
                    local_values[scenario].linear_objective[index] * primal[index];
            }
        }
        return result;
    }

    [[nodiscard]] double maximum_cost_epigraph_violation(
        std::span<const double> primal,
        std::span<const CQPValues> local_values
    ) const {
        const auto decoded = decode_primal(primal);
        const auto costs = local_objectives(decoded.scenario.local, local_values);
        double maximum = 0.0;
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            maximum = std::max(
                maximum,
                costs[scenario] - decoded.cost_epigraphs[scenario]
            );
        }
        return std::max(0.0, maximum);
    }

    [[nodiscard]] double evaluated_risk(
        std::span<const double> primal,
        std::span<const CQPValues> local_values,
        const double alpha = 0.9
    ) const {
        validate_alpha(alpha);
        const auto decoded = decode_primal(primal);
        const auto costs = local_objectives(decoded.scenario.local, local_values);
        if (measure_ == ScenarioRiskMeasure::worst_case) {
            return worst_case_cost(costs);
        }
        const auto probabilities = base().tree().probabilities();
        return conditional_value_at_risk(costs, probabilities, alpha);
    }

  private:
    struct MatrixBundle {
        CscMatrixData quadratic;
        CscMatrixData scalar;
        CscMatrixData affine;
    };

    [[nodiscard]] std::size_t risk_variable_count() const noexcept {
        return scenario_count() + 1U +
            (measure_ == ScenarioRiskMeasure::conditional_value_at_risk
                 ? scenario_count()
                 : 0U);
    }
    [[nodiscard]] std::size_t total_variables() const noexcept {
        return base_variable_count() + risk_variable_count();
    }
    [[nodiscard]] std::size_t risk_scalar_row_start() const noexcept {
        return static_cast<std::size_t>(base().structure().scalar_rows());
    }
    [[nodiscard]] std::size_t risk_affine_row_start() const noexcept {
        return static_cast<std::size_t>(base().structure().affine_rows());
    }
    [[nodiscard]] std::size_t risk_cone_slots() const noexcept {
        return local_variable_count() + 2U;
    }
    [[nodiscard]] std::size_t total_affine_rows() const noexcept {
        return risk_affine_row_start() + scenario_count() * risk_cone_slots();
    }

    void validate_dimensions() const {
        const auto maximum = static_cast<std::size_t>(
            std::numeric_limits<Index>::max()
        );
        const auto scalar_rows = risk_scalar_row_start() + scenario_count();
        if (total_variables() > maximum || scalar_rows > maximum ||
            total_affine_rows() > maximum) {
            throw std::invalid_argument("risk CQP exceeds 32-bit index capacity");
        }
    }

    void validate_alpha(const double alpha) const {
        if (measure_ == ScenarioRiskMeasure::conditional_value_at_risk &&
            (!std::isfinite(alpha) || alpha < 0.0 || alpha >= 1.0)) {
            throw std::invalid_argument("CVaR alpha must lie in [0,1)");
        }
    }

    void validate_local_values(std::span<const CQPValues> local_values) const {
        if (local_values.size() != scenario_count()) {
            throw std::invalid_argument("one local CQP value set is required per scenario");
        }
        for (const auto& values : local_values) {
            validate_values(base().local_structure(), values);
            static_cast<void>(quadratic_diagonal(values));
        }
    }

    [[nodiscard]] CQPStructure build_structure() const {
        std::vector<CQPValues> symbolic;
        symbolic.reserve(scenario_count());
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            CQPValues values;
            values.quadratic.assign(base().local_structure().quadratic().nonzeros(), 1.0);
            values.scalar_constraint.assign(
                base().local_structure().scalar_constraint().nonzeros(),
                1.0
            );
            values.affine_cone.assign(
                base().local_structure().affine_cone().has_value()
                    ? base().local_structure().affine_cone()->nonzeros()
                    : 0U,
                1.0
            );
            values.linear_objective.assign(local_variable_count(), 1.0);
            values.scalar_lower.assign(
                static_cast<std::size_t>(base().local_structure().scalar_rows()),
                0.0
            );
            values.scalar_upper = values.scalar_lower;
            values.affine_offset.assign(
                static_cast<std::size_t>(base().local_structure().affine_rows()),
                0.0
            );
            values.variable_lower.assign(
                local_variable_count(),
                -std::numeric_limits<double>::infinity()
            );
            values.variable_upper.assign(
                local_variable_count(),
                std::numeric_limits<double>::infinity()
            );
            symbolic.push_back(std::move(values));
        }
        const auto base_values = base().values(symbolic);
        const auto matrices = assemble(base_values, symbolic, true);
        std::vector<ConeBlock> affine_cones(
            base().structure().affine_cones().begin(),
            base().structure().affine_cones().end()
        );
        affine_cones.reserve(affine_cones.size() + scenario_count());
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            affine_cones.emplace_back(
                ConeKind::rotated_second_order,
                static_cast<Index>(
                    risk_affine_row_start() + scenario * risk_cone_slots()
                ),
                static_cast<Index>(local_variable_count())
            );
        }
        std::vector<ConeBlock> variable_cones(
            base().structure().variable_cones().begin(),
            base().structure().variable_cones().end()
        );
        return CQPStructure{
            matrices.quadratic.structure,
            matrices.scalar.structure,
            std::optional<CscStructure>{matrices.affine.structure},
            std::move(affine_cones),
            std::move(variable_cones),
        };
    }

    [[nodiscard]] MatrixBundle assemble(
        const CQPValues& base_values,
        std::span<const CQPValues> local_values,
        const bool symbolic
    ) const {
        CscBuilder quadratic{
            static_cast<Index>(total_variables()),
            static_cast<Index>(total_variables()),
        };
        for (std::size_t variable = 0; variable < total_variables(); ++variable) {
            quadratic.add(
                static_cast<Index>(variable),
                static_cast<Index>(variable),
                0.0
            );
        }
        CscBuilder scalar{
            static_cast<Index>(risk_scalar_row_start() + scenario_count()),
            static_cast<Index>(total_variables()),
        };
        append_matrix(
            scalar,
            base().structure().scalar_constraint(),
            base_values.scalar_constraint,
            0,
            0
        );
        CscBuilder affine{
            static_cast<Index>(total_affine_rows()),
            static_cast<Index>(total_variables()),
        };
        if (base().structure().affine_cone().has_value()) {
            append_matrix(
                affine,
                *base().structure().affine_cone(),
                base_values.affine_cone,
                0,
                0
            );
        }

        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto scalar_row = static_cast<Index>(
                risk_scalar_row_start() + scenario
            );
            scalar.add(
                scalar_row,
                static_cast<Index>(cost_epigraph_start() + scenario),
                1.0
            );
            scalar.add(
                scalar_row,
                static_cast<Index>(threshold_index()),
                -1.0
            );
            if (measure_ == ScenarioRiskMeasure::conditional_value_at_risk) {
                scalar.add(
                    scalar_row,
                    static_cast<Index>(excess_start() + scenario),
                    -1.0
                );
            }

            const auto local_range = base().layout().scenario_range(scenario);
            const auto cone_start = risk_affine_row_start() +
                scenario * risk_cone_slots();
            const auto diagonal = symbolic
                ? std::vector<double>(local_variable_count(), 1.0)
                : quadratic_diagonal(local_values[scenario]);
            for (std::size_t local = 0; local < local_variable_count(); ++local) {
                affine.add(
                    static_cast<Index>(cone_start + local),
                    static_cast<Index>(local_range.first + local),
                    symbolic ? 1.0 : std::sqrt(diagonal[local])
                );
                affine.add(
                    static_cast<Index>(cone_start + local_variable_count()),
                    static_cast<Index>(local_range.first + local),
                    symbolic ? 1.0 : -local_values[scenario].linear_objective[local]
                );
            }
            affine.add(
                static_cast<Index>(cone_start + local_variable_count()),
                static_cast<Index>(cost_epigraph_start() + scenario),
                1.0
            );
        }
        return MatrixBundle{quadratic.build(), scalar.build(), affine.build()};
    }

    [[nodiscard]] std::vector<double> quadratic_diagonal(
        const CQPValues& values
    ) const {
        const auto& structure = base().local_structure().quadratic();
        std::vector<double> diagonal(local_variable_count(), 0.0);
        const auto offsets = structure.offsets();
        const auto indices = structure.indices();
        for (Index column = 0; column < structure.columns(); ++column) {
            const auto begin = static_cast<std::size_t>(
                offsets[static_cast<std::size_t>(column)]
            );
            const auto end = static_cast<std::size_t>(
                offsets[static_cast<std::size_t>(column) + 1U]
            );
            for (std::size_t position = begin; position < end; ++position) {
                const double value = values.quadratic[position];
                if (indices[position] == column) {
                    diagonal[static_cast<std::size_t>(column)] += value;
                } else if (std::abs(value) > 1.0e-14) {
                    throw std::invalid_argument(
                        "risk epigraph currently requires diagonal local quadratics"
                    );
                }
            }
            if (!std::isfinite(diagonal[static_cast<std::size_t>(column)]) ||
                diagonal[static_cast<std::size_t>(column)] < -1.0e-14) {
                throw std::invalid_argument(
                    "risk epigraph requires a positive-semidefinite diagonal quadratic"
                );
            }
            diagonal[static_cast<std::size_t>(column)] =
                std::max(0.0, diagonal[static_cast<std::size_t>(column)]);
        }
        return diagonal;
    }

    static void append_matrix(
        CscBuilder& builder,
        const CscStructure& structure,
        std::span<const double> values,
        const Index row_offset,
        const Index column_offset
    ) {
        if (values.size() != structure.nonzeros()) {
            throw std::invalid_argument("risk matrix values have an incompatible size");
        }
        const auto offsets = structure.offsets();
        const auto indices = structure.indices();
        for (Index column = 0; column < structure.columns(); ++column) {
            const auto begin = static_cast<std::size_t>(
                offsets[static_cast<std::size_t>(column)]
            );
            const auto end = static_cast<std::size_t>(
                offsets[static_cast<std::size_t>(column) + 1U]
            );
            for (std::size_t position = begin; position < end; ++position) {
                builder.add(
                    row_offset + indices[position],
                    column_offset + column,
                    values[position]
                );
            }
        }
    }

    const ScenarioCQPBundle* base_{nullptr};
    ScenarioRiskMeasure measure_{ScenarioRiskMeasure::worst_case};
    std::optional<CQPStructure> structure_;
};

}  // namespace spacepdhcg::core
