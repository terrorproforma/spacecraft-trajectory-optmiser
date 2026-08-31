#pragma once

#include "spacepdhcg/core/fixed_cqp.hpp"
#include "spacepdhcg/distributed/scenario_layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace spacepdhcg::distributed {

struct ScenarioPrimal {
    std::vector<std::vector<double>> local{};
    std::vector<std::vector<double>> consensus{};
};

struct ScenarioDual {
    std::vector<std::vector<double>> local_scalar{};
    std::vector<double> nonanticipativity{};
    std::vector<std::vector<double>> local_affine{};
};

/// Monolithic correctness oracle for same-topology scenario CQPs.
///
/// The production multi-GPU implementation consumes the same local patterns and layout but
/// never materialises this global matrix. This class exists to freeze global ordering, verify
/// distributed products, and provide exact host-side update buffers for cross-backend tests.
class ScenarioCqpBundle {
  public:
    ScenarioCqpBundle(
        ScenarioTree tree,
        core::FixedStructure local_structure,
        std::size_t state_dimension,
        std::size_t control_dimension,
        std::size_t local_auxiliary_dimension = 0U
    )
        : tree_(std::move(tree)),
          local_structure_(std::move(local_structure)),
          layout_(
              tree_,
              state_dimension,
              control_dimension,
              local_auxiliary_dimension
          ) {
        local_structure_.validate();
        if (layout_.local_variables_per_scenario()
            != static_cast<std::size_t>(local_structure_.variables())) {
            throw std::invalid_argument(
                "scenario local state/control/auxiliary layout does not match the CQP"
            );
        }
        structure_ = build_structure();
        structure_.validate();
        build_position_maps();
    }

    [[nodiscard]] const ScenarioTree& tree() const noexcept { return tree_; }
    [[nodiscard]] const BlockArrowLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] const core::FixedStructure& local_structure() const noexcept {
        return local_structure_;
    }
    [[nodiscard]] const core::FixedStructure& structure() const noexcept { return structure_; }
    [[nodiscard]] std::size_t scenario_count() const noexcept { return tree_.scenario_count(); }
    [[nodiscard]] std::size_t nonanticipativity_rows() const noexcept {
        return layout_.nonanticipativity_rows();
    }

    [[nodiscard]] core::NumericValues values(
        const std::vector<core::NumericValues>& local_values
    ) const {
        const auto validated = validate_local_values(local_values);
        core::NumericValues result{};
        result.quadratic.assign(structure_.quadratic.nonzeros(), 0.0);
        result.scalar_constraint.assign(structure_.scalar_constraint.nonzeros(), 0.0);
        result.affine_cone.assign(
            structure_.affine_cone.has_value() ? structure_.affine_cone->nonzeros() : 0U,
            0.0
        );
        result.linear_objective.assign(static_cast<std::size_t>(structure_.variables()), 0.0);
        result.scalar_lower.assign(static_cast<std::size_t>(structure_.scalar_rows()), 0.0);
        result.scalar_upper.assign(static_cast<std::size_t>(structure_.scalar_rows()), 0.0);
        result.affine_offset.assign(static_cast<std::size_t>(structure_.affine_rows()), 0.0);
        result.variable_lower.assign(
            static_cast<std::size_t>(structure_.variables()),
            -std::numeric_limits<double>::infinity()
        );
        result.variable_upper.assign(
            static_cast<std::size_t>(structure_.variables()),
            std::numeric_limits<double>::infinity()
        );

        const auto probabilities = tree_.probabilities();
        const auto local_variables = static_cast<std::size_t>(local_structure_.variables());
        const auto local_scalar_rows = static_cast<std::size_t>(local_structure_.scalar_rows());
        const auto local_affine_rows = static_cast<std::size_t>(local_structure_.affine_rows());
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto& values = validated[scenario];
            const auto variable_base = scenario * local_variables;
            const auto scalar_base = scenario * local_scalar_rows;
            const auto affine_base = scenario * local_affine_rows;
            for (std::size_t local_position = 0;
                 local_position < values.quadratic.size();
                 ++local_position) {
                result.quadratic[quadratic_positions_[scenario][local_position]] =
                    probabilities[scenario] * values.quadratic[local_position];
            }
            for (std::size_t local_position = 0;
                 local_position < values.scalar_constraint.size();
                 ++local_position) {
                result.scalar_constraint[scalar_positions_[scenario][local_position]] =
                    values.scalar_constraint[local_position];
            }
            for (std::size_t local_position = 0;
                 local_position < values.affine_cone.size();
                 ++local_position) {
                result.affine_cone[affine_positions_[scenario][local_position]] =
                    values.affine_cone[local_position];
            }
            for (std::size_t variable = 0; variable < local_variables; ++variable) {
                result.linear_objective[variable_base + variable] =
                    probabilities[scenario] * values.linear_objective[variable];
                result.variable_lower[variable_base + variable] = values.variable_lower[variable];
                result.variable_upper[variable_base + variable] = values.variable_upper[variable];
            }
            std::copy(
                values.scalar_lower.begin(),
                values.scalar_lower.end(),
                result.scalar_lower.begin() + static_cast<std::ptrdiff_t>(scalar_base)
            );
            std::copy(
                values.scalar_upper.begin(),
                values.scalar_upper.end(),
                result.scalar_upper.begin() + static_cast<std::ptrdiff_t>(scalar_base)
            );
            std::copy(
                values.affine_offset.begin(),
                values.affine_offset.end(),
                result.affine_offset.begin() + static_cast<std::ptrdiff_t>(affine_base)
            );
        }

        const auto nonanticipativity_base = scenario_count() * local_scalar_rows;
        for (std::size_t row = 0; row < nonanticipativity_rows(); ++row) {
            result.scalar_lower[nonanticipativity_base + row] = 0.0;
            result.scalar_upper[nonanticipativity_base + row] = 0.0;
        }
        for (std::size_t entry = 0; entry < nonanticipativity_positions_.size(); ++entry) {
            result.scalar_constraint[nonanticipativity_positions_[entry]] =
                nonanticipativity_values_[entry];
        }
        result.validate(structure_);
        return result;
    }

    [[nodiscard]] core::FixedCQP problem(
        const std::vector<core::NumericValues>& local_values
    ) const {
        return core::FixedCQP(structure_, values(local_values));
    }

    [[nodiscard]] ScenarioPrimal decode_primal(const std::vector<double>& primal) const {
        if (primal.size() != static_cast<std::size_t>(structure_.variables())) {
            throw std::invalid_argument("scenario primal has the wrong size");
        }
        ScenarioPrimal decoded{};
        decoded.local.reserve(scenario_count());
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto [begin, end] = layout_.scenario_range(scenario);
            decoded.local.emplace_back(
                primal.begin() + static_cast<std::ptrdiff_t>(begin),
                primal.begin() + static_cast<std::ptrdiff_t>(end)
            );
        }
        decoded.consensus.reserve(layout_.consensus_blocks().size());
        for (const auto& block : layout_.consensus_blocks()) {
            const auto [begin, end] = block.range();
            decoded.consensus.emplace_back(
                primal.begin() + static_cast<std::ptrdiff_t>(begin),
                primal.begin() + static_cast<std::ptrdiff_t>(end)
            );
        }
        return decoded;
    }

    [[nodiscard]] ScenarioDual decode_dual(const std::vector<double>& dual) const {
        if (dual.size() != static_cast<std::size_t>(structure_.duals())) {
            throw std::invalid_argument("scenario dual has the wrong size");
        }
        const auto local_scalar_rows = static_cast<std::size_t>(local_structure_.scalar_rows());
        const auto local_affine_rows = static_cast<std::size_t>(local_structure_.affine_rows());
        ScenarioDual decoded{};
        decoded.local_scalar.reserve(scenario_count());
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto begin = scenario * local_scalar_rows;
            decoded.local_scalar.emplace_back(
                dual.begin() + static_cast<std::ptrdiff_t>(begin),
                dual.begin() + static_cast<std::ptrdiff_t>(begin + local_scalar_rows)
            );
        }
        const auto nonanticipativity_begin = scenario_count() * local_scalar_rows;
        decoded.nonanticipativity.assign(
            dual.begin() + static_cast<std::ptrdiff_t>(nonanticipativity_begin),
            dual.begin()
                + static_cast<std::ptrdiff_t>(
                    nonanticipativity_begin + nonanticipativity_rows()
                )
        );
        const auto affine_begin = static_cast<std::size_t>(structure_.scalar_rows());
        decoded.local_affine.reserve(scenario_count());
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto begin = affine_begin + scenario * local_affine_rows;
            decoded.local_affine.emplace_back(
                dual.begin() + static_cast<std::ptrdiff_t>(begin),
                dual.begin() + static_cast<std::ptrdiff_t>(begin + local_affine_rows)
            );
        }
        return decoded;
    }

    [[nodiscard]] double maximum_nonanticipativity_violation(
        const std::vector<double>& primal
    ) const {
        const auto decoded = decode_primal(primal);
        double maximum{0.0};
        for (std::size_t block_index = 0;
             block_index < layout_.consensus_blocks().size();
             ++block_index) {
            const auto& block = layout_.consensus_blocks()[block_index];
            const auto& consensus = decoded.consensus[block_index];
            for (const auto scenario : block.node.scenario_indices) {
                const auto [begin, end] = layout_.control_range(scenario, block.node.stage);
                static_cast<void>(end);
                const auto [scenario_begin, scenario_end] = layout_.scenario_range(scenario);
                static_cast<void>(scenario_end);
                const auto local_begin = begin - scenario_begin;
                for (std::size_t component = 0; component < consensus.size(); ++component) {
                    maximum = std::max(
                        maximum,
                        std::abs(decoded.local[scenario][local_begin + component]
                                 - consensus[component])
                    );
                }
            }
        }
        return maximum;
    }

    [[nodiscard]] std::vector<double> local_objectives(
        const std::vector<std::vector<double>>& local_primals,
        const std::vector<core::NumericValues>& local_values
    ) const {
        if (local_primals.size() != scenario_count()) {
            throw std::invalid_argument("one local primal is required per scenario");
        }
        const auto validated = validate_local_values(local_values);
        std::vector<double> objectives(scenario_count(), 0.0);
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto& primal = local_primals[scenario];
            if (primal.size() != static_cast<std::size_t>(local_structure_.variables())) {
                throw std::invalid_argument("a local scenario primal has the wrong size");
            }
            objectives[scenario] = objective(local_structure_, validated[scenario], primal);
        }
        return objectives;
    }

    [[nodiscard]] double expected_objective(
        const std::vector<std::vector<double>>& local_primals,
        const std::vector<core::NumericValues>& local_values
    ) const {
        const auto objectives = local_objectives(local_primals, local_values);
        const auto probabilities = tree_.probabilities();
        double expected{0.0};
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            expected += probabilities[scenario] * objectives[scenario];
        }
        return expected;
    }

    [[nodiscard]] double global_objective(
        const std::vector<double>& primal,
        const core::NumericValues& values
    ) const {
        if (primal.size() != static_cast<std::size_t>(structure_.variables())) {
            throw std::invalid_argument("global scenario primal has the wrong size");
        }
        values.validate(structure_);
        return objective(structure_, values, primal);
    }

  private:
    ScenarioTree tree_;
    core::FixedStructure local_structure_{};
    BlockArrowLayout layout_;
    core::FixedStructure structure_{};
    std::vector<std::vector<std::size_t>> quadratic_positions_{};
    std::vector<std::vector<std::size_t>> scalar_positions_{};
    std::vector<std::vector<std::size_t>> affine_positions_{};
    std::vector<std::size_t> nonanticipativity_positions_{};
    std::vector<double> nonanticipativity_values_{};

    [[nodiscard]] core::FixedStructure build_structure() const {
        std::set<std::pair<std::size_t, std::size_t>> quadratic_entries{};
        std::set<std::pair<std::size_t, std::size_t>> scalar_entries{};
        std::set<std::pair<std::size_t, std::size_t>> affine_entries{};
        const auto local_variables = static_cast<std::size_t>(local_structure_.variables());
        const auto local_scalar_rows = static_cast<std::size_t>(local_structure_.scalar_rows());
        const auto local_affine_rows = static_cast<std::size_t>(local_structure_.affine_rows());
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            append_pattern(
                quadratic_entries,
                local_structure_.quadratic,
                scenario * local_variables,
                scenario * local_variables
            );
            append_pattern(
                scalar_entries,
                local_structure_.scalar_constraint,
                scenario * local_scalar_rows,
                scenario * local_variables
            );
            if (local_structure_.affine_cone.has_value()) {
                append_pattern(
                    affine_entries,
                    *local_structure_.affine_cone,
                    scenario * local_affine_rows,
                    scenario * local_variables
                );
            }
        }
        const auto nonanticipativity_base = scenario_count() * local_scalar_rows;
        for (const auto& triplet : layout_.nonanticipativity_triplets()) {
            scalar_entries.insert(
                {nonanticipativity_base + triplet.row, triplet.column}
            );
        }

        const auto variables = layout_.total_variables();
        core::FixedStructure result{};
        result.quadratic = make_pattern(variables, variables, quadratic_entries);
        result.scalar_constraint = make_pattern(
            scenario_count() * local_scalar_rows + nonanticipativity_rows(),
            variables,
            scalar_entries
        );
        if (local_structure_.affine_cone.has_value()) {
            result.affine_cone = make_pattern(
                scenario_count() * local_affine_rows,
                variables,
                affine_entries
            );
            for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
                for (const auto& cone : local_structure_.affine_cones) {
                    auto shifted = cone;
                    shifted.start += static_cast<Index>(scenario * local_affine_rows);
                    result.affine_cones.push_back(shifted);
                }
            }
        }
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            for (const auto& cone : local_structure_.variable_cones) {
                auto shifted = cone;
                shifted.start += static_cast<Index>(scenario * local_variables);
                result.variable_cones.push_back(shifted);
            }
        }
        return result;
    }

    void build_position_maps() {
        const auto quadratic_lookup = position_lookup(structure_.quadratic);
        const auto scalar_lookup = position_lookup(structure_.scalar_constraint);
        const auto affine_lookup = structure_.affine_cone.has_value()
                                         ? position_lookup(*structure_.affine_cone)
                                         : PositionLookup{};
        const auto local_variables = static_cast<std::size_t>(local_structure_.variables());
        const auto local_scalar_rows = static_cast<std::size_t>(local_structure_.scalar_rows());
        const auto local_affine_rows = static_cast<std::size_t>(local_structure_.affine_rows());
        quadratic_positions_.reserve(scenario_count());
        scalar_positions_.reserve(scenario_count());
        affine_positions_.reserve(scenario_count());
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            quadratic_positions_.push_back(
                shifted_positions(
                    local_structure_.quadratic,
                    quadratic_lookup,
                    scenario * local_variables,
                    scenario * local_variables
                )
            );
            scalar_positions_.push_back(
                shifted_positions(
                    local_structure_.scalar_constraint,
                    scalar_lookup,
                    scenario * local_scalar_rows,
                    scenario * local_variables
                )
            );
            if (local_structure_.affine_cone.has_value()) {
                affine_positions_.push_back(
                    shifted_positions(
                        *local_structure_.affine_cone,
                        affine_lookup,
                        scenario * local_affine_rows,
                        scenario * local_variables
                    )
                );
            } else {
                affine_positions_.emplace_back();
            }
        }
        const auto nonanticipativity_base = scenario_count() * local_scalar_rows;
        for (const auto& triplet : layout_.nonanticipativity_triplets()) {
            const auto key = std::make_pair(
                nonanticipativity_base + triplet.row,
                triplet.column
            );
            const auto iterator = scalar_lookup.find(key);
            if (iterator == scalar_lookup.end()) {
                throw std::logic_error("non-anticipativity entry is absent from global pattern");
            }
            nonanticipativity_positions_.push_back(iterator->second);
            nonanticipativity_values_.push_back(triplet.value);
        }
    }

    [[nodiscard]] std::vector<core::NumericValues> validate_local_values(
        const std::vector<core::NumericValues>& local_values
    ) const {
        if (local_values.size() != scenario_count()) {
            throw std::invalid_argument("one local CQP value set is required per scenario");
        }
        auto validated = local_values;
        for (const auto& values : validated) {
            values.validate(local_structure_);
        }
        return validated;
    }

    using PositionLookup = std::map<std::pair<std::size_t, std::size_t>, std::size_t>;

    static PositionLookup position_lookup(const core::CscPattern& pattern) {
        PositionLookup result{};
        for (Index column = 0; column < pattern.columns; ++column) {
            const auto begin = static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column)]);
            const auto end = static_cast<std::size_t>(
                pattern.offsets[static_cast<std::size_t>(column) + 1U]
            );
            for (std::size_t position = begin; position < end; ++position) {
                result[
                    {static_cast<std::size_t>(pattern.indices[position]),
                     static_cast<std::size_t>(column)}
                ] = position;
            }
        }
        return result;
    }

    static std::vector<std::size_t> shifted_positions(
        const core::CscPattern& local,
        const PositionLookup& global_lookup,
        std::size_t row_offset,
        std::size_t column_offset
    ) {
        std::vector<std::size_t> positions(local.nonzeros(), 0U);
        for (Index column = 0; column < local.columns; ++column) {
            const auto begin = static_cast<std::size_t>(local.offsets[static_cast<std::size_t>(column)]);
            const auto end = static_cast<std::size_t>(
                local.offsets[static_cast<std::size_t>(column) + 1U]
            );
            for (std::size_t position = begin; position < end; ++position) {
                const auto key = std::make_pair(
                    row_offset + static_cast<std::size_t>(local.indices[position]),
                    column_offset + static_cast<std::size_t>(column)
                );
                const auto iterator = global_lookup.find(key);
                if (iterator == global_lookup.end()) {
                    throw std::logic_error("local sparse entry is absent from global pattern");
                }
                positions[position] = iterator->second;
            }
        }
        return positions;
    }

    static void append_pattern(
        std::set<std::pair<std::size_t, std::size_t>>& entries,
        const core::CscPattern& local,
        std::size_t row_offset,
        std::size_t column_offset
    ) {
        for (Index column = 0; column < local.columns; ++column) {
            const auto begin = static_cast<std::size_t>(local.offsets[static_cast<std::size_t>(column)]);
            const auto end = static_cast<std::size_t>(
                local.offsets[static_cast<std::size_t>(column) + 1U]
            );
            for (std::size_t position = begin; position < end; ++position) {
                entries.insert(
                    {row_offset + static_cast<std::size_t>(local.indices[position]),
                     column_offset + static_cast<std::size_t>(column)}
                );
            }
        }
    }

    static core::CscPattern make_pattern(
        std::size_t rows,
        std::size_t columns,
        const std::set<std::pair<std::size_t, std::size_t>>& entries
    ) {
        std::vector<std::pair<std::size_t, std::size_t>> ordered(entries.begin(), entries.end());
        std::sort(
            ordered.begin(),
            ordered.end(),
            [](const auto& left, const auto& right) {
                return std::tie(left.second, left.first) < std::tie(right.second, right.first);
            }
        );
        core::CscPattern pattern{};
        pattern.rows = static_cast<Index>(rows);
        pattern.columns = static_cast<Index>(columns);
        pattern.offsets.assign(columns + 1U, 0);
        pattern.indices.reserve(ordered.size());
        for (const auto& [row, column] : ordered) {
            ++pattern.offsets[column + 1U];
            pattern.indices.push_back(static_cast<Index>(row));
        }
        for (std::size_t column = 0; column < columns; ++column) {
            pattern.offsets[column + 1U] += pattern.offsets[column];
        }
        pattern.validate();
        return pattern;
    }

    static double objective(
        const core::FixedStructure& structure,
        const core::NumericValues& values,
        const std::vector<double>& primal
    ) {
        const auto product = matvec(structure.quadratic, values.quadratic, primal);
        double quadratic{0.0};
        double linear{0.0};
        for (std::size_t variable = 0; variable < primal.size(); ++variable) {
            quadratic += primal[variable] * product[variable];
            linear += values.linear_objective[variable] * primal[variable];
        }
        return 0.5 * quadratic + linear;
    }

    static std::vector<double> matvec(
        const core::CscPattern& pattern,
        const std::vector<double>& values,
        const std::vector<double>& vector
    ) {
        std::vector<double> result(static_cast<std::size_t>(pattern.rows), 0.0);
        for (Index column = 0; column < pattern.columns; ++column) {
            const auto begin = static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column)]);
            const auto end = static_cast<std::size_t>(
                pattern.offsets[static_cast<std::size_t>(column) + 1U]
            );
            for (std::size_t position = begin; position < end; ++position) {
                result[static_cast<std::size_t>(pattern.indices[position])] +=
                    values[position] * vector[static_cast<std::size_t>(column)];
            }
        }
        return result;
    }
};

}  // namespace spacepdhcg::distributed
