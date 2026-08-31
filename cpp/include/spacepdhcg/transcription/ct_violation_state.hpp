#pragma once

#include "spacepdhcg/core/fixed_cqp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spacepdhcg::transcription {

struct AffinePathSample {
    std::vector<std::size_t> indices{};
    std::vector<double> coefficients{};
    double offset{0.0};

    void validate(std::size_t base_variables) const {
        if (indices.size() != coefficients.size()) {
            throw std::invalid_argument(
                "path-sample indices and coefficients must have equal size"
            );
        }
        if (!std::isfinite(offset)) {
            throw std::invalid_argument("path-sample offset must be finite");
        }
        std::size_t previous{0U};
        for (std::size_t position = 0; position < indices.size(); ++position) {
            if (indices[position] >= base_variables) {
                throw std::invalid_argument("path-sample index is outside the base primal");
            }
            if (position > 0U && indices[position] <= previous) {
                throw std::invalid_argument(
                    "path-sample indices must be strictly increasing"
                );
            }
            if (!std::isfinite(coefficients[position])) {
                throw std::invalid_argument("path-sample coefficients must be finite");
            }
            previous = indices[position];
        }
    }

    [[nodiscard]] double evaluate(const std::vector<double>& base_primal) const {
        validate(base_primal.size());
        double value = offset;
        for (std::size_t position = 0; position < indices.size(); ++position) {
            value += coefficients[position] * base_primal[indices[position]];
        }
        return value;
    }
};

struct CtQuadratureInterval {
    std::vector<std::size_t> sample_indices{};
    std::vector<double> weights{};

    void validate(std::size_t sample_count) const {
        if (sample_indices.empty() || sample_indices.size() != weights.size()) {
            throw std::invalid_argument(
                "CT quadrature interval requires equally sized non-empty samples and weights"
            );
        }
        std::size_t previous{0U};
        for (std::size_t position = 0; position < sample_indices.size(); ++position) {
            if (sample_indices[position] >= sample_count) {
                throw std::invalid_argument("CT quadrature sample index is outside the grid");
            }
            if (position > 0U && sample_indices[position] <= previous) {
                throw std::invalid_argument(
                    "CT quadrature sample indices must be strictly increasing"
                );
            }
            if (!std::isfinite(weights[position]) || weights[position] <= 0.0) {
                throw std::invalid_argument("CT quadrature weights must be finite and positive");
            }
            previous = sample_indices[position];
        }
    }
};

struct CtViolationDiagnostics {
    std::vector<double> path_values{};
    std::vector<double> actual_interval_integrals{};
    std::vector<double> state_interval_increments{};
    double maximum_positive_sample{0.0};
    double maximum_integral_budget_violation{0.0};
    double final_violation_state{0.0};
};

namespace ct_violation_detail {

using Entries = std::set<std::pair<std::size_t, std::size_t>>;  // column, row

inline void append_pattern(
    Entries& entries,
    const core::CscPattern& pattern,
    std::size_t row_shift = 0U,
    std::size_t column_shift = 0U
) {
    for (Index column = 0; column < pattern.columns; ++column) {
        const auto begin = static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column)]);
        const auto end = static_cast<std::size_t>(
            pattern.offsets[static_cast<std::size_t>(column) + 1U]
        );
        for (std::size_t position = begin; position < end; ++position) {
            entries.emplace(
                column_shift + static_cast<std::size_t>(column),
                row_shift + static_cast<std::size_t>(pattern.indices[position])
            );
        }
    }
}

inline core::CscPattern make_pattern(
    std::size_t rows,
    std::size_t columns,
    const Entries& entries
) {
    if (rows > static_cast<std::size_t>(std::numeric_limits<Index>::max())
        || columns > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        throw std::overflow_error("CT violation CQP exceeds the native index range");
    }
    core::CscPattern result{};
    result.rows = static_cast<Index>(rows);
    result.columns = static_cast<Index>(columns);
    result.offsets.resize(columns + 1U, 0);
    for (std::size_t column = 0; column < columns; ++column) {
        result.offsets[column] = static_cast<Index>(result.indices.size());
        auto iterator = entries.lower_bound({column, 0U});
        while (iterator != entries.end() && iterator->first == column) {
            result.indices.push_back(static_cast<Index>(iterator->second));
            ++iterator;
        }
    }
    result.offsets[columns] = static_cast<Index>(result.indices.size());
    result.validate();
    return result;
}

inline std::map<std::pair<std::size_t, std::size_t>, std::size_t> lookup(
    const core::CscPattern& pattern
) {
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> result{};
    for (Index column = 0; column < pattern.columns; ++column) {
        const auto begin = static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column)]);
        const auto end = static_cast<std::size_t>(
            pattern.offsets[static_cast<std::size_t>(column) + 1U]
        );
        for (std::size_t position = begin; position < end; ++position) {
            result.emplace(
                std::pair{
                    static_cast<std::size_t>(pattern.indices[position]),
                    static_cast<std::size_t>(column),
                },
                position
            );
        }
    }
    return result;
}

inline std::vector<std::size_t> inherited_positions(
    const core::CscPattern& source,
    const std::map<std::pair<std::size_t, std::size_t>, std::size_t>& destination
) {
    std::vector<std::size_t> positions{};
    positions.reserve(source.nonzeros());
    for (Index column = 0; column < source.columns; ++column) {
        const auto begin = static_cast<std::size_t>(source.offsets[static_cast<std::size_t>(column)]);
        const auto end = static_cast<std::size_t>(
            source.offsets[static_cast<std::size_t>(column) + 1U]
        );
        for (std::size_t position = begin; position < end; ++position) {
            positions.push_back(
                destination.at(
                    {
                        static_cast<std::size_t>(source.indices[position]),
                        static_cast<std::size_t>(column),
                    }
                )
            );
        }
    }
    return positions;
}

inline void copy_values(
    const std::vector<double>& source,
    const std::vector<std::size_t>& positions,
    std::vector<double>& destination
) {
    if (source.size() != positions.size()) {
        throw std::logic_error("CT violation inherited sparse map has the wrong size");
    }
    for (std::size_t index = 0; index < source.size(); ++index) {
        destination[positions[index]] = source[index];
    }
}

}  // namespace ct_violation_detail

/// Augment a canonical CQP with nonnegative violation samples and cumulative violation states.
///
/// Each sample models `g_j(x) <= lambda_j`, with `lambda_j >= 0`. For interval `k`,
/// `y_(k+1)-y_k = sum_j w_(k,j) lambda_j` and the increment is constrained by a supplied
/// continuous-time budget. The topology, sample sparsity and quadrature stencil are immutable;
/// SCvx iterations update only sample coefficients, offsets, budgets and the terminal penalty.
class CtViolationStateCqp {
  public:
    CtViolationStateCqp(
        core::FixedStructure base_structure,
        std::vector<std::vector<std::size_t>> sample_patterns,
        std::vector<CtQuadratureInterval> intervals
    )
        : base_structure_(std::move(base_structure)),
          sample_patterns_(std::move(sample_patterns)),
          intervals_(std::move(intervals)) {
        base_structure_.validate();
        if (sample_patterns_.empty() || intervals_.empty()) {
            throw std::invalid_argument(
                "CT violation augmentation requires samples and intervals"
            );
        }
        const auto base_variables = static_cast<std::size_t>(base_structure_.variables());
        for (const auto& pattern : sample_patterns_) {
            std::size_t previous{0U};
            for (std::size_t position = 0; position < pattern.size(); ++position) {
                if (pattern[position] >= base_variables) {
                    throw std::invalid_argument(
                        "CT violation sample pattern is outside the base primal"
                    );
                }
                if (position > 0U && pattern[position] <= previous) {
                    throw std::invalid_argument(
                        "CT violation sample patterns must be strictly increasing"
                    );
                }
                previous = pattern[position];
            }
        }
        std::vector<std::size_t> ownership(sample_count(), 0U);
        for (const auto& interval : intervals_) {
            interval.validate(sample_count());
            for (const auto sample : interval.sample_indices) {
                ++ownership[sample];
            }
        }
        if (!std::all_of(ownership.begin(), ownership.end(), [](std::size_t count) {
                return count == 1U;
            })) {
            throw std::invalid_argument(
                "each CT violation sample must belong to exactly one interval"
            );
        }
        structure_ = build_structure();
        structure_.validate();
        build_position_maps();
    }

    [[nodiscard]] const core::FixedStructure& structure() const noexcept { return structure_; }
    [[nodiscard]] const core::FixedStructure& base_structure() const noexcept {
        return base_structure_;
    }
    [[nodiscard]] std::size_t base_variables() const noexcept {
        return static_cast<std::size_t>(base_structure_.variables());
    }
    [[nodiscard]] std::size_t sample_count() const noexcept { return sample_patterns_.size(); }
    [[nodiscard]] std::size_t interval_count() const noexcept { return intervals_.size(); }
    [[nodiscard]] std::size_t lambda_index(std::size_t sample) const {
        if (sample >= sample_count()) {
            throw std::out_of_range("CT violation sample index is outside the grid");
        }
        return base_variables() + sample;
    }
    [[nodiscard]] std::size_t state_index(std::size_t node) const {
        if (node > interval_count()) {
            throw std::out_of_range("CT violation-state node is outside the grid");
        }
        return base_variables() + sample_count() + node;
    }

    [[nodiscard]] core::NumericValues values(
        const core::NumericValues& base_values,
        const std::vector<AffinePathSample>& samples,
        const std::vector<double>& interval_budgets,
        double terminal_penalty
    ) const {
        base_values.validate(base_structure_);
        if (samples.size() != sample_count()
            || interval_budgets.size() != interval_count()) {
            throw std::invalid_argument(
                "CT violation values require one sample and one budget per fixed slot"
            );
        }
        if (!std::isfinite(terminal_penalty) || terminal_penalty <= 0.0) {
            throw std::invalid_argument(
                "CT violation terminal penalty must be finite and positive"
            );
        }
        for (std::size_t sample = 0; sample < sample_count(); ++sample) {
            samples[sample].validate(base_variables());
            if (samples[sample].indices != sample_patterns_[sample]) {
                throw std::invalid_argument(
                    "CT violation sample sparsity changed after initialisation"
                );
            }
        }
        for (const auto budget : interval_budgets) {
            if (!std::isfinite(budget) || budget < 0.0) {
                throw std::invalid_argument(
                    "CT violation interval budgets must be finite and non-negative"
                );
            }
        }

        core::NumericValues result{};
        result.quadratic.assign(structure_.quadratic.nonzeros(), 0.0);
        result.scalar_constraint.assign(structure_.scalar_constraint.nonzeros(), 0.0);
        result.affine_cone.assign(
            structure_.affine_cone.has_value() ? structure_.affine_cone->nonzeros() : 0U,
            0.0
        );
        result.linear_objective.assign(
            static_cast<std::size_t>(structure_.variables()),
            0.0
        );
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

        ct_violation_detail::copy_values(
            base_values.quadratic,
            quadratic_positions_,
            result.quadratic
        );
        ct_violation_detail::copy_values(
            base_values.scalar_constraint,
            scalar_positions_,
            result.scalar_constraint
        );
        ct_violation_detail::copy_values(
            base_values.affine_cone,
            affine_positions_,
            result.affine_cone
        );
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

        const auto base_rows = static_cast<std::size_t>(base_structure_.scalar_rows());
        for (std::size_t sample = 0; sample < sample_count(); ++sample) {
            for (std::size_t position = 0;
                 position < samples[sample].coefficients.size();
                 ++position) {
                result.scalar_constraint[sample_positions_[sample][position]] =
                    samples[sample].coefficients[position];
            }
            result.scalar_constraint[lambda_positions_[sample]] = -1.0;
            result.scalar_upper[base_rows + sample] = -samples[sample].offset;
            result.variable_lower[lambda_index(sample)] = 0.0;
        }

        const auto initial_row = base_rows + sample_count();
        result.scalar_constraint[initial_state_position_] = 1.0;
        result.scalar_lower[initial_row] = 0.0;
        result.scalar_upper[initial_row] = 0.0;

        const auto integration_row = initial_row + 1U;
        const auto budget_row = integration_row + interval_count();
        for (std::size_t interval = 0; interval < interval_count(); ++interval) {
            result.scalar_constraint[integration_previous_positions_[interval]] = -1.0;
            result.scalar_constraint[integration_next_positions_[interval]] = 1.0;
            result.scalar_constraint[budget_previous_positions_[interval]] = -1.0;
            result.scalar_constraint[budget_next_positions_[interval]] = 1.0;
            for (std::size_t position = 0;
                 position < intervals_[interval].sample_indices.size();
                 ++position) {
                result.scalar_constraint[quadrature_positions_[interval][position]] =
                    -intervals_[interval].weights[position];
            }
            result.scalar_lower[integration_row + interval] = 0.0;
            result.scalar_upper[integration_row + interval] = 0.0;
            result.scalar_upper[budget_row + interval] = interval_budgets[interval];
        }
        for (std::size_t node = 0; node <= interval_count(); ++node) {
            result.variable_lower[state_index(node)] = 0.0;
        }
        result.linear_objective[state_index(interval_count())] += terminal_penalty;
        result.validate(structure_);
        return result;
    }

    [[nodiscard]] core::FixedCQP problem(
        const core::NumericValues& base_values,
        const std::vector<AffinePathSample>& samples,
        const std::vector<double>& interval_budgets,
        double terminal_penalty
    ) const {
        return core::FixedCQP(
            structure_,
            values(base_values, samples, interval_budgets, terminal_penalty)
        );
    }

    [[nodiscard]] std::vector<double> base_primal(const std::vector<double>& primal) const {
        validate_primal(primal);
        return std::vector<double>(
            primal.begin(),
            primal.begin() + static_cast<std::ptrdiff_t>(base_variables())
        );
    }

    [[nodiscard]] CtViolationDiagnostics diagnostics(
        const std::vector<double>& primal,
        const std::vector<AffinePathSample>& samples,
        const std::vector<double>& interval_budgets
    ) const {
        validate_primal(primal);
        if (samples.size() != sample_count()
            || interval_budgets.size() != interval_count()) {
            throw std::invalid_argument("CT violation diagnostics dimensions are invalid");
        }
        const auto base = base_primal(primal);
        CtViolationDiagnostics result{};
        result.path_values.reserve(sample_count());
        for (const auto& sample : samples) {
            const auto value = sample.evaluate(base);
            result.path_values.push_back(value);
            result.maximum_positive_sample = std::max(
                result.maximum_positive_sample,
                value
            );
        }
        result.actual_interval_integrals.reserve(interval_count());
        result.state_interval_increments.reserve(interval_count());
        for (std::size_t interval = 0; interval < interval_count(); ++interval) {
            double actual{0.0};
            for (std::size_t position = 0;
                 position < intervals_[interval].sample_indices.size();
                 ++position) {
                const auto sample = intervals_[interval].sample_indices[position];
                actual += intervals_[interval].weights[position]
                          * std::max(0.0, result.path_values[sample]);
            }
            const auto increment = primal[state_index(interval + 1U)]
                                   - primal[state_index(interval)];
            result.actual_interval_integrals.push_back(actual);
            result.state_interval_increments.push_back(increment);
            result.maximum_integral_budget_violation = std::max(
                result.maximum_integral_budget_violation,
                actual - interval_budgets[interval]
            );
        }
        result.maximum_positive_sample = std::max(0.0, result.maximum_positive_sample);
        result.maximum_integral_budget_violation = std::max(
            0.0,
            result.maximum_integral_budget_violation
        );
        result.final_violation_state = primal[state_index(interval_count())];
        return result;
    }

  private:
    core::FixedStructure base_structure_{};
    std::vector<std::vector<std::size_t>> sample_patterns_{};
    std::vector<CtQuadratureInterval> intervals_{};
    core::FixedStructure structure_{};
    std::vector<std::size_t> quadratic_positions_{};
    std::vector<std::size_t> scalar_positions_{};
    std::vector<std::size_t> affine_positions_{};
    std::vector<std::vector<std::size_t>> sample_positions_{};
    std::vector<std::size_t> lambda_positions_{};
    std::size_t initial_state_position_{0U};
    std::vector<std::size_t> integration_previous_positions_{};
    std::vector<std::size_t> integration_next_positions_{};
    std::vector<std::vector<std::size_t>> quadrature_positions_{};
    std::vector<std::size_t> budget_previous_positions_{};
    std::vector<std::size_t> budget_next_positions_{};

    [[nodiscard]] core::FixedStructure build_structure() const {
        const auto variables = base_variables() + sample_count() + interval_count() + 1U;
        const auto base_rows = static_cast<std::size_t>(base_structure_.scalar_rows());
        const auto scalar_rows = base_rows + sample_count() + 1U + 2U * interval_count();
        ct_violation_detail::Entries quadratic_entries{};
        ct_violation_detail::Entries scalar_entries{};
        ct_violation_detail::Entries affine_entries{};
        ct_violation_detail::append_pattern(
            quadratic_entries,
            base_structure_.quadratic
        );
        ct_violation_detail::append_pattern(
            scalar_entries,
            base_structure_.scalar_constraint
        );
        if (base_structure_.affine_cone.has_value()) {
            ct_violation_detail::append_pattern(
                affine_entries,
                *base_structure_.affine_cone
            );
        }

        for (std::size_t sample = 0; sample < sample_count(); ++sample) {
            const auto row = base_rows + sample;
            for (const auto variable : sample_patterns_[sample]) {
                scalar_entries.emplace(variable, row);
            }
            scalar_entries.emplace(lambda_index(sample), row);
        }
        const auto initial_row = base_rows + sample_count();
        scalar_entries.emplace(state_index(0U), initial_row);
        const auto integration_row = initial_row + 1U;
        const auto budget_row = integration_row + interval_count();
        for (std::size_t interval = 0; interval < interval_count(); ++interval) {
            scalar_entries.emplace(state_index(interval), integration_row + interval);
            scalar_entries.emplace(state_index(interval + 1U), integration_row + interval);
            for (const auto sample : intervals_[interval].sample_indices) {
                scalar_entries.emplace(lambda_index(sample), integration_row + interval);
            }
            scalar_entries.emplace(state_index(interval), budget_row + interval);
            scalar_entries.emplace(state_index(interval + 1U), budget_row + interval);
        }

        core::FixedStructure result{};
        result.quadratic = ct_violation_detail::make_pattern(
            variables,
            variables,
            quadratic_entries
        );
        result.scalar_constraint = ct_violation_detail::make_pattern(
            scalar_rows,
            variables,
            scalar_entries
        );
        if (base_structure_.affine_cone.has_value()) {
            result.affine_cone = ct_violation_detail::make_pattern(
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
        const auto q_lookup = ct_violation_detail::lookup(structure_.quadratic);
        const auto a_lookup = ct_violation_detail::lookup(structure_.scalar_constraint);
        quadratic_positions_ = ct_violation_detail::inherited_positions(
            base_structure_.quadratic,
            q_lookup
        );
        scalar_positions_ = ct_violation_detail::inherited_positions(
            base_structure_.scalar_constraint,
            a_lookup
        );
        if (base_structure_.affine_cone.has_value()) {
            const auto f_lookup = ct_violation_detail::lookup(*structure_.affine_cone);
            affine_positions_ = ct_violation_detail::inherited_positions(
                *base_structure_.affine_cone,
                f_lookup
            );
        }

        const auto base_rows = static_cast<std::size_t>(base_structure_.scalar_rows());
        sample_positions_.resize(sample_count());
        lambda_positions_.resize(sample_count());
        for (std::size_t sample = 0; sample < sample_count(); ++sample) {
            const auto row = base_rows + sample;
            for (const auto variable : sample_patterns_[sample]) {
                sample_positions_[sample].push_back(a_lookup.at({row, variable}));
            }
            lambda_positions_[sample] = a_lookup.at({row, lambda_index(sample)});
        }
        const auto initial_row = base_rows + sample_count();
        initial_state_position_ = a_lookup.at({initial_row, state_index(0U)});
        const auto integration_row = initial_row + 1U;
        const auto budget_row = integration_row + interval_count();
        integration_previous_positions_.resize(interval_count());
        integration_next_positions_.resize(interval_count());
        quadrature_positions_.resize(interval_count());
        budget_previous_positions_.resize(interval_count());
        budget_next_positions_.resize(interval_count());
        for (std::size_t interval = 0; interval < interval_count(); ++interval) {
            integration_previous_positions_[interval] = a_lookup.at(
                {integration_row + interval, state_index(interval)}
            );
            integration_next_positions_[interval] = a_lookup.at(
                {integration_row + interval, state_index(interval + 1U)}
            );
            for (const auto sample : intervals_[interval].sample_indices) {
                quadrature_positions_[interval].push_back(
                    a_lookup.at({integration_row + interval, lambda_index(sample)})
                );
            }
            budget_previous_positions_[interval] = a_lookup.at(
                {budget_row + interval, state_index(interval)}
            );
            budget_next_positions_[interval] = a_lookup.at(
                {budget_row + interval, state_index(interval + 1U)}
            );
        }
    }

    void validate_primal(const std::vector<double>& primal) const {
        if (primal.size() != static_cast<std::size_t>(structure_.variables())) {
            throw std::invalid_argument("CT violation primal has the wrong size");
        }
        if (!std::all_of(primal.begin(), primal.end(), [](double value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument("CT violation primal must be finite");
        }
    }
};

}  // namespace spacepdhcg::transcription
