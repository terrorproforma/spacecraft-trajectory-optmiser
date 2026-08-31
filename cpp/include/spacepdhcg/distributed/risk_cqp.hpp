#pragma once

#include "spacepdhcg/core/fixed_cqp.hpp"
#include "spacepdhcg/distributed/risk.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spacepdhcg::distributed {

enum class RiskMeasure : std::uint8_t {
    expected,
    worst_case,
    conditional_value_at_risk,
};

/// One affine scenario loss `a^T x + b` over the unaugmented primal vector.
struct AffineScenarioLoss {
    std::vector<std::size_t> indices{};
    std::vector<double> coefficients{};
    double offset{0.0};

    void validate(std::size_t variables) const {
        if (indices.size() != coefficients.size()) {
            throw std::invalid_argument("loss indices and coefficients must have equal size");
        }
        if (!std::isfinite(offset)) {
            throw std::invalid_argument("loss offset must be finite");
        }
        std::size_t previous{0U};
        for (std::size_t position = 0; position < indices.size(); ++position) {
            const auto index = indices[position];
            if (index >= variables) {
                throw std::invalid_argument("loss coefficient index is outside the primal vector");
            }
            if (position > 0U && index <= previous) {
                throw std::invalid_argument("loss coefficient indices must be strictly increasing");
            }
            if (!std::isfinite(coefficients[position])) {
                throw std::invalid_argument("loss coefficients must be finite");
            }
            previous = index;
        }
    }

    [[nodiscard]] double evaluate(const std::vector<double>& primal) const {
        validate(primal.size());
        double value = offset;
        for (std::size_t position = 0; position < indices.size(); ++position) {
            value += coefficients[position] * primal[indices[position]];
        }
        return value;
    }
};

struct RiskEpigraphVariables {
    std::optional<double> threshold{};
    std::vector<double> excess{};
};

struct RiskCqpDiagnostics {
    std::vector<double> losses{};
    RiskSummary summary{};
    double maximum_epigraph_violation{0.0};
};

namespace risk_cqp_detail {

using EntrySet = std::set<std::pair<std::size_t, std::size_t>>;  // (column, row)

inline void append_pattern(
    EntrySet& destination,
    const core::CscPattern& source,
    std::size_t row_shift,
    std::size_t column_shift
) {
    for (Index column = 0; column < source.columns; ++column) {
        const auto begin = static_cast<std::size_t>(source.offsets[static_cast<std::size_t>(column)]);
        const auto end = static_cast<std::size_t>(
            source.offsets[static_cast<std::size_t>(column) + 1U]
        );
        for (std::size_t position = begin; position < end; ++position) {
            destination.emplace(
                column_shift + static_cast<std::size_t>(column),
                row_shift + static_cast<std::size_t>(source.indices[position])
            );
        }
    }
}

inline core::CscPattern make_pattern(
    std::size_t rows,
    std::size_t columns,
    const EntrySet& entries
) {
    if (rows > static_cast<std::size_t>(std::numeric_limits<Index>::max())
        || columns > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        throw std::overflow_error("risk CQP dimensions exceed the native index range");
    }
    core::CscPattern pattern{};
    pattern.rows = static_cast<Index>(rows);
    pattern.columns = static_cast<Index>(columns);
    pattern.offsets.resize(columns + 1U, 0);
    for (std::size_t column = 0; column < columns; ++column) {
        pattern.offsets[column] = static_cast<Index>(pattern.indices.size());
        auto iterator = entries.lower_bound({column, 0U});
        while (iterator != entries.end() && iterator->first == column) {
            if (iterator->second >= rows) {
                throw std::logic_error("risk CQP pattern row is outside the matrix");
            }
            pattern.indices.push_back(static_cast<Index>(iterator->second));
            ++iterator;
        }
    }
    pattern.offsets[columns] = static_cast<Index>(pattern.indices.size());
    pattern.validate();
    return pattern;
}

inline std::map<std::pair<std::size_t, std::size_t>, std::size_t> position_lookup(
    const core::CscPattern& pattern
) {
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> lookup{};
    for (Index column = 0; column < pattern.columns; ++column) {
        const auto begin = static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column)]);
        const auto end = static_cast<std::size_t>(
            pattern.offsets[static_cast<std::size_t>(column) + 1U]
        );
        for (std::size_t position = begin; position < end; ++position) {
            lookup.emplace(
                std::pair{
                    static_cast<std::size_t>(pattern.indices[position]),
                    static_cast<std::size_t>(column),
                },
                position
            );
        }
    }
    return lookup;
}

inline std::vector<std::size_t> map_positions(
    const core::CscPattern& source,
    const std::map<std::pair<std::size_t, std::size_t>, std::size_t>& destination,
    std::size_t row_shift = 0U,
    std::size_t column_shift = 0U
) {
    std::vector<std::size_t> positions{};
    positions.reserve(source.nonzeros());
    for (Index column = 0; column < source.columns; ++column) {
        const auto begin = static_cast<std::size_t>(source.offsets[static_cast<std::size_t>(column)]);
        const auto end = static_cast<std::size_t>(
            source.offsets[static_cast<std::size_t>(column) + 1U]
        );
        for (std::size_t position = begin; position < end; ++position) {
            const auto key = std::pair{
                row_shift + static_cast<std::size_t>(source.indices[position]),
                column_shift + static_cast<std::size_t>(column),
            };
            const auto iterator = destination.find(key);
            if (iterator == destination.end()) {
                throw std::logic_error("risk CQP failed to map an inherited sparse entry");
            }
            positions.push_back(iterator->second);
        }
    }
    return positions;
}

}  // namespace risk_cqp_detail

/// Fixed-topology risk augmentation for any canonical CQP with affine scenario losses.
///
/// The base objective is retained. `risk_weight * R(loss)` is added using either a direct
/// expected-loss linear term, one worst-case epigraph variable, or the standard CVaR
/// threshold/excess formulation. Loss sparsity is frozen at construction so repeated SCvx
/// iterations update only numerical coefficients and offsets.
class RiskAugmentedCqp {
  public:
    RiskAugmentedCqp(
        core::FixedStructure base_structure,
        std::vector<std::vector<std::size_t>> loss_patterns,
        RiskMeasure measure,
        double confidence = 0.95
    )
        : base_structure_(std::move(base_structure)),
          loss_patterns_(std::move(loss_patterns)),
          measure_(measure),
          confidence_(confidence) {
        base_structure_.validate();
        if (loss_patterns_.empty()) {
            throw std::invalid_argument("risk augmentation requires at least one scenario loss");
        }
        if (!std::isfinite(confidence_) || confidence_ < 0.0 || confidence_ >= 1.0) {
            throw std::invalid_argument("CVaR confidence must lie in [0, 1)");
        }
        const auto base_variables = static_cast<std::size_t>(base_structure_.variables());
        for (const auto& pattern : loss_patterns_) {
            std::size_t previous{0U};
            for (std::size_t position = 0; position < pattern.size(); ++position) {
                if (pattern[position] >= base_variables) {
                    throw std::invalid_argument("loss pattern index is outside the base primal");
                }
                if (position > 0U && pattern[position] <= previous) {
                    throw std::invalid_argument("loss patterns must be strictly increasing");
                }
                previous = pattern[position];
            }
        }
        structure_ = build_structure();
        structure_.validate();
        build_position_maps();
    }

    [[nodiscard]] const core::FixedStructure& base_structure() const noexcept {
        return base_structure_;
    }
    [[nodiscard]] const core::FixedStructure& structure() const noexcept { return structure_; }
    [[nodiscard]] RiskMeasure measure() const noexcept { return measure_; }
    [[nodiscard]] double confidence() const noexcept { return confidence_; }
    [[nodiscard]] std::size_t scenario_count() const noexcept { return loss_patterns_.size(); }
    [[nodiscard]] std::size_t base_variables() const noexcept {
        return static_cast<std::size_t>(base_structure_.variables());
    }
    [[nodiscard]] std::optional<std::size_t> threshold_index() const noexcept {
        if (measure_ == RiskMeasure::expected) {
            return std::nullopt;
        }
        return base_variables();
    }
    [[nodiscard]] std::optional<std::size_t> excess_index(std::size_t scenario) const {
        if (scenario >= scenario_count()) {
            throw std::out_of_range("risk scenario index is outside the loss set");
        }
        if (measure_ != RiskMeasure::conditional_value_at_risk) {
            return std::nullopt;
        }
        return base_variables() + 1U + scenario;
    }

    [[nodiscard]] core::NumericValues values(
        const core::NumericValues& base_values,
        const std::vector<AffineScenarioLoss>& losses,
        const std::vector<double>& probabilities,
        double risk_weight
    ) const {
        base_values.validate(base_structure_);
        validate_probability_distribution(probabilities);
        if (probabilities.size() != scenario_count() || losses.size() != scenario_count()) {
            throw std::invalid_argument("risk values require one probability and loss per scenario");
        }
        if (!std::isfinite(risk_weight) || risk_weight <= 0.0) {
            throw std::invalid_argument("risk weight must be finite and positive");
        }
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            losses[scenario].validate(base_variables());
            if (losses[scenario].indices != loss_patterns_[scenario]) {
                throw std::invalid_argument("loss sparsity changed after risk CQP initialisation");
            }
        }

        core::NumericValues result{};
        result.quadratic.assign(structure_.quadratic.nonzeros(), 0.0);
        result.scalar_constraint.assign(structure_.scalar_constraint.nonzeros(), 0.0);
        result.affine_cone.assign(
            structure_.affine_cone.has_value() ? structure_.affine_cone->nonzeros() : 0U,
            0.0
        );
        result.linear_objective.assign(static_cast<std::size_t>(structure_.variables()), 0.0);
        result.scalar_lower.assign(
            static_cast<std::size_t>(structure_.scalar_rows()),
            -std::numeric_limits<double>::infinity()
        );
        result.scalar_upper.assign(
            static_cast<std::size_t>(structure_.scalar_rows()),
            std::numeric_limits<double>::infinity()
        );
        result.affine_offset = base_values.affine_offset;
        result.variable_lower.assign(
            static_cast<std::size_t>(structure_.variables()),
            -std::numeric_limits<double>::infinity()
        );
        result.variable_upper.assign(
            static_cast<std::size_t>(structure_.variables()),
            std::numeric_limits<double>::infinity()
        );

        copy_sparse_values(base_values.quadratic, quadratic_positions_, result.quadratic);
        copy_sparse_values(
            base_values.scalar_constraint,
            scalar_positions_,
            result.scalar_constraint
        );
        copy_sparse_values(base_values.affine_cone, affine_positions_, result.affine_cone);
        std::copy(
            base_values.linear_objective.begin(),
            base_values.linear_objective.end(),
            result.linear_objective.begin()
        );
        std::copy(
            base_values.scalar_lower.begin(),
            base_values.scalar_lower.end(),
            result.scalar_lower.begin()
        );
        std::copy(
            base_values.scalar_upper.begin(),
            base_values.scalar_upper.end(),
            result.scalar_upper.begin()
        );
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

        if (measure_ == RiskMeasure::expected) {
            for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
                const auto scale = risk_weight * probabilities[scenario];
                for (std::size_t position = 0;
                     position < losses[scenario].indices.size();
                     ++position) {
                    result.linear_objective[losses[scenario].indices[position]] +=
                        scale * losses[scenario].coefficients[position];
                }
            }
        } else {
            const auto row_base = static_cast<std::size_t>(base_structure_.scalar_rows());
            const auto threshold = *threshold_index();
            result.linear_objective[threshold] += risk_weight;
            for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
                for (std::size_t position = 0;
                     position < losses[scenario].coefficients.size();
                     ++position) {
                    result.scalar_constraint[loss_positions_[scenario][position]] =
                        losses[scenario].coefficients[position];
                }
                result.scalar_constraint[threshold_positions_[scenario]] = -1.0;
                if (measure_ == RiskMeasure::conditional_value_at_risk) {
                    const auto excess = *excess_index(scenario);
                    result.scalar_constraint[excess_positions_[scenario]] = -1.0;
                    result.variable_lower[excess] = 0.0;
                    result.linear_objective[excess] +=
                        risk_weight * probabilities[scenario] / (1.0 - confidence_);
                }
                result.scalar_upper[row_base + scenario] = -losses[scenario].offset;
            }
        }
        result.validate(structure_);
        return result;
    }

    [[nodiscard]] core::FixedCQP problem(
        const core::NumericValues& base_values,
        const std::vector<AffineScenarioLoss>& losses,
        const std::vector<double>& probabilities,
        double risk_weight
    ) const {
        return core::FixedCQP(
            structure_,
            values(base_values, losses, probabilities, risk_weight)
        );
    }

    [[nodiscard]] std::vector<double> base_primal(const std::vector<double>& primal) const {
        if (primal.size() != static_cast<std::size_t>(structure_.variables())) {
            throw std::invalid_argument("risk-augmented primal has the wrong size");
        }
        return std::vector<double>(
            primal.begin(),
            primal.begin() + static_cast<std::ptrdiff_t>(base_variables())
        );
    }

    [[nodiscard]] RiskEpigraphVariables decode_risk_variables(
        const std::vector<double>& primal
    ) const {
        if (primal.size() != static_cast<std::size_t>(structure_.variables())) {
            throw std::invalid_argument("risk-augmented primal has the wrong size");
        }
        RiskEpigraphVariables decoded{};
        if (const auto threshold = threshold_index(); threshold.has_value()) {
            decoded.threshold = primal[*threshold];
        }
        if (measure_ == RiskMeasure::conditional_value_at_risk) {
            decoded.excess.reserve(scenario_count());
            for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
                decoded.excess.push_back(primal[*excess_index(scenario)]);
            }
        }
        return decoded;
    }

    [[nodiscard]] RiskCqpDiagnostics diagnostics(
        const std::vector<double>& primal,
        const std::vector<AffineScenarioLoss>& losses,
        const std::vector<double>& probabilities
    ) const {
        if (losses.size() != scenario_count() || probabilities.size() != scenario_count()) {
            throw std::invalid_argument("risk diagnostics require one loss per scenario");
        }
        const auto base = base_primal(primal);
        RiskCqpDiagnostics result{};
        result.losses.reserve(scenario_count());
        for (const auto& loss : losses) {
            result.losses.push_back(loss.evaluate(base));
        }
        result.summary = aggregate_scenario_risk(result.losses, probabilities, confidence_);
        const auto decoded = decode_risk_variables(primal);
        if (measure_ == RiskMeasure::worst_case) {
            for (const auto loss : result.losses) {
                result.maximum_epigraph_violation = std::max(
                    result.maximum_epigraph_violation,
                    loss - *decoded.threshold
                );
            }
        } else if (measure_ == RiskMeasure::conditional_value_at_risk) {
            for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
                result.maximum_epigraph_violation = std::max(
                    result.maximum_epigraph_violation,
                    result.losses[scenario] - *decoded.threshold - decoded.excess[scenario]
                );
                result.maximum_epigraph_violation = std::max(
                    result.maximum_epigraph_violation,
                    -decoded.excess[scenario]
                );
            }
        }
        result.maximum_epigraph_violation = std::max(
            0.0,
            result.maximum_epigraph_violation
        );
        return result;
    }

  private:
    core::FixedStructure base_structure_{};
    std::vector<std::vector<std::size_t>> loss_patterns_{};
    RiskMeasure measure_{RiskMeasure::expected};
    double confidence_{0.95};
    core::FixedStructure structure_{};
    std::vector<std::size_t> quadratic_positions_{};
    std::vector<std::size_t> scalar_positions_{};
    std::vector<std::size_t> affine_positions_{};
    std::vector<std::vector<std::size_t>> loss_positions_{};
    std::vector<std::size_t> threshold_positions_{};
    std::vector<std::size_t> excess_positions_{};

    [[nodiscard]] std::size_t extra_variables() const noexcept {
        switch (measure_) {
            case RiskMeasure::expected:
                return 0U;
            case RiskMeasure::worst_case:
                return 1U;
            case RiskMeasure::conditional_value_at_risk:
                return 1U + scenario_count();
        }
        return 0U;
    }

    [[nodiscard]] std::size_t extra_rows() const noexcept {
        return measure_ == RiskMeasure::expected ? 0U : scenario_count();
    }

    [[nodiscard]] core::FixedStructure build_structure() const {
        using risk_cqp_detail::EntrySet;
        const auto variables = base_variables() + extra_variables();
        const auto scalar_rows = static_cast<std::size_t>(base_structure_.scalar_rows())
                                 + extra_rows();
        EntrySet quadratic_entries{};
        EntrySet scalar_entries{};
        EntrySet affine_entries{};
        risk_cqp_detail::append_pattern(quadratic_entries, base_structure_.quadratic, 0U, 0U);
        risk_cqp_detail::append_pattern(
            scalar_entries,
            base_structure_.scalar_constraint,
            0U,
            0U
        );
        if (base_structure_.affine_cone.has_value()) {
            risk_cqp_detail::append_pattern(
                affine_entries,
                *base_structure_.affine_cone,
                0U,
                0U
            );
        }
        if (measure_ != RiskMeasure::expected) {
            const auto row_base = static_cast<std::size_t>(base_structure_.scalar_rows());
            const auto threshold = base_variables();
            for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
                const auto row = row_base + scenario;
                for (const auto variable : loss_patterns_[scenario]) {
                    scalar_entries.emplace(variable, row);
                }
                scalar_entries.emplace(threshold, row);
                if (measure_ == RiskMeasure::conditional_value_at_risk) {
                    scalar_entries.emplace(base_variables() + 1U + scenario, row);
                }
            }
        }

        core::FixedStructure result{};
        result.quadratic = risk_cqp_detail::make_pattern(
            variables,
            variables,
            quadratic_entries
        );
        result.scalar_constraint = risk_cqp_detail::make_pattern(
            scalar_rows,
            variables,
            scalar_entries
        );
        if (base_structure_.affine_cone.has_value()) {
            result.affine_cone = risk_cqp_detail::make_pattern(
                static_cast<std::size_t>(base_structure_.affine_rows()),
                variables,
                affine_entries
            );
            result.affine_cones = base_structure_.affine_cones;
        }
        result.variable_cones = base_structure_.variable_cones;
        return result;
    }

    void build_position_maps() {
        const auto q_lookup = risk_cqp_detail::position_lookup(structure_.quadratic);
        const auto a_lookup = risk_cqp_detail::position_lookup(structure_.scalar_constraint);
        quadratic_positions_ = risk_cqp_detail::map_positions(
            base_structure_.quadratic,
            q_lookup
        );
        scalar_positions_ = risk_cqp_detail::map_positions(
            base_structure_.scalar_constraint,
            a_lookup
        );
        if (base_structure_.affine_cone.has_value()) {
            const auto f_lookup = risk_cqp_detail::position_lookup(*structure_.affine_cone);
            affine_positions_ = risk_cqp_detail::map_positions(
                *base_structure_.affine_cone,
                f_lookup
            );
        }
        if (measure_ == RiskMeasure::expected) {
            return;
        }
        const auto row_base = static_cast<std::size_t>(base_structure_.scalar_rows());
        const auto threshold = base_variables();
        loss_positions_.resize(scenario_count());
        threshold_positions_.resize(scenario_count());
        if (measure_ == RiskMeasure::conditional_value_at_risk) {
            excess_positions_.resize(scenario_count());
        }
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto row = row_base + scenario;
            for (const auto variable : loss_patterns_[scenario]) {
                loss_positions_[scenario].push_back(a_lookup.at({row, variable}));
            }
            threshold_positions_[scenario] = a_lookup.at({row, threshold});
            if (measure_ == RiskMeasure::conditional_value_at_risk) {
                excess_positions_[scenario] = a_lookup.at(
                    {row, base_variables() + 1U + scenario}
                );
            }
        }
    }

    static void copy_sparse_values(
        const std::vector<double>& source,
        const std::vector<std::size_t>& positions,
        std::vector<double>& destination
    ) {
        if (source.size() != positions.size()) {
            throw std::logic_error("risk CQP inherited sparse-value map has the wrong size");
        }
        for (std::size_t position = 0; position < source.size(); ++position) {
            destination[positions[position]] = source[position];
        }
    }
};

}  // namespace spacepdhcg::distributed
