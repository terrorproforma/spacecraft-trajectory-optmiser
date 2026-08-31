#pragma once

#include "spacepdhcg/core/cqp.hpp"
#include "spacepdhcg/core/csc_operator.hpp"
#include "spacepdhcg/core/scenario.hpp"
#include "spacepdhcg/core/sparse_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spacepdhcg::core {

struct ScenarioCQPPrimal {
    std::vector<std::vector<double>> local;
    std::vector<std::vector<double>> consensus;
};

class ScenarioCQPBundle {
  public:
    ScenarioCQPBundle(
        ScenarioTree tree,
        CQPStructure local_structure,
        const std::size_t state_dimension,
        const std::size_t control_dimension,
        const std::size_t local_auxiliary_dimension = 0U
    )
        : tree_(std::move(tree)),
          local_structure_(std::move(local_structure)),
          layout_(
              tree_,
              state_dimension,
              control_dimension,
              local_auxiliary_dimension
          ) {
        if (layout_.local_variables_per_scenario() !=
            static_cast<std::size_t>(local_structure_.variables())) {
            throw std::invalid_argument(
                "scenario local layout does not match local CQP variable count"
            );
        }
        validate_global_dimensions();
        structure_.emplace(build_structure());
    }

    [[nodiscard]] const ScenarioTree& tree() const noexcept { return tree_; }
    [[nodiscard]] const BlockArrowLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] const CQPStructure& local_structure() const noexcept {
        return local_structure_;
    }
    [[nodiscard]] const CQPStructure& structure() const noexcept { return *structure_; }
    [[nodiscard]] std::size_t scenario_count() const noexcept {
        return tree_.scenario_count();
    }
    [[nodiscard]] std::size_t nonanticipativity_rows() const noexcept {
        return layout_.nonanticipativity_rows();
    }

    [[nodiscard]] CQPValues values(std::span<const CQPValues> local_values) const {
        validate_local_values(local_values);
        const auto matrices = assemble(local_values, false);
        if (matrices.quadratic.structure.fingerprint() !=
                structure().quadratic().fingerprint() ||
            matrices.scalar.structure.fingerprint() !=
                structure().scalar_constraint().fingerprint()) {
            throw std::logic_error("scenario numerical assembly changed fixed sparse topology");
        }
        if (structure().affine_cone().has_value()) {
            if (!matrices.affine.has_value() ||
                matrices.affine->structure.fingerprint() !=
                    structure().affine_cone()->fingerprint()) {
                throw std::logic_error("scenario affine-cone topology changed");
            }
        }

        CQPValues result;
        result.quadratic = matrices.quadratic.values;
        result.scalar_constraint = matrices.scalar.values;
        result.affine_cone = matrices.affine.has_value()
            ? matrices.affine->values
            : std::vector<double>{};
        result.linear_objective.assign(layout_.total_variables(), 0.0);
        result.scalar_lower.reserve(
            scenario_count() * static_cast<std::size_t>(local_structure_.scalar_rows()) +
            nonanticipativity_rows()
        );
        result.scalar_upper.reserve(result.scalar_lower.capacity());
        result.affine_offset.reserve(
            scenario_count() * static_cast<std::size_t>(local_structure_.affine_rows())
        );
        result.variable_lower.assign(
            layout_.total_variables(),
            -std::numeric_limits<double>::infinity()
        );
        result.variable_upper.assign(
            layout_.total_variables(),
            std::numeric_limits<double>::infinity()
        );

        const auto probabilities = tree_.probabilities();
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto local_range = layout_.scenario_range(scenario);
            const auto& local = local_values[scenario];
            for (std::size_t index = 0; index < local.linear_objective.size(); ++index) {
                result.linear_objective[local_range.first + index] =
                    probabilities[scenario] * local.linear_objective[index];
                result.variable_lower[local_range.first + index] =
                    local.variable_lower[index];
                result.variable_upper[local_range.first + index] =
                    local.variable_upper[index];
            }
            result.scalar_lower.insert(
                result.scalar_lower.end(),
                local.scalar_lower.begin(),
                local.scalar_lower.end()
            );
            result.scalar_upper.insert(
                result.scalar_upper.end(),
                local.scalar_upper.begin(),
                local.scalar_upper.end()
            );
            result.affine_offset.insert(
                result.affine_offset.end(),
                local.affine_offset.begin(),
                local.affine_offset.end()
            );
        }
        result.scalar_lower.insert(
            result.scalar_lower.end(),
            nonanticipativity_rows(),
            0.0
        );
        result.scalar_upper.insert(
            result.scalar_upper.end(),
            nonanticipativity_rows(),
            0.0
        );
        validate_values(structure(), result);
        return result;
    }

    [[nodiscard]] ScenarioCQPPrimal decode_primal(
        std::span<const double> primal
    ) const {
        if (primal.size() != layout_.total_variables()) {
            throw std::invalid_argument("scenario primal has an incompatible size");
        }
        ScenarioCQPPrimal result;
        result.local.reserve(scenario_count());
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto range = layout_.scenario_range(scenario);
            result.local.emplace_back(
                primal.begin() + static_cast<std::ptrdiff_t>(range.first),
                primal.begin() + static_cast<std::ptrdiff_t>(range.second)
            );
        }
        result.consensus.reserve(layout_.consensus_blocks().size());
        for (const auto& block : layout_.consensus_blocks()) {
            result.consensus.emplace_back(
                primal.begin() + static_cast<std::ptrdiff_t>(block.offset),
                primal.begin() +
                    static_cast<std::ptrdiff_t>(block.offset + block.dimension)
            );
        }
        return result;
    }

    [[nodiscard]] std::vector<double> lift_primal(
        std::span<const std::vector<double>> local_primals,
        const double equality_tolerance = 1.0e-12
    ) const {
        if (local_primals.size() != scenario_count()) {
            throw std::invalid_argument("one local primal is required per scenario");
        }
        if (!std::isfinite(equality_tolerance) || equality_tolerance < 0.0) {
            throw std::invalid_argument("scenario equality tolerance must be non-negative");
        }
        std::vector<double> global(layout_.total_variables(), 0.0);
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto& local = local_primals[scenario];
            if (local.size() !=
                static_cast<std::size_t>(local_structure_.variables())) {
                throw std::invalid_argument("scenario local primal has an incompatible size");
            }
            const auto range = layout_.scenario_range(scenario);
            std::copy(
                local.begin(),
                local.end(),
                global.begin() + static_cast<std::ptrdiff_t>(range.first)
            );
        }
        for (const auto& block : layout_.consensus_blocks()) {
            const auto anchor = block.node->scenario_indices.front();
            const auto anchor_range = layout_.control_range(anchor, block.node->stage);
            for (std::size_t component = 0; component < block.dimension; ++component) {
                const double value = global[anchor_range.first + component];
                global[block.offset + component] = value;
                for (const auto scenario : block.node->scenario_indices) {
                    const auto local_range = layout_.control_range(
                        scenario,
                        block.node->stage
                    );
                    if (std::abs(global[local_range.first + component] - value) >
                        equality_tolerance) {
                        throw std::invalid_argument(
                            "local primals violate requested non-anticipativity tolerance"
                        );
                    }
                }
            }
        }
        return global;
    }

    [[nodiscard]] double maximum_nonanticipativity_violation(
        std::span<const double> primal
    ) const {
        const auto decoded = decode_primal(primal);
        double maximum = 0.0;
        for (std::size_t block_index = 0;
             block_index < layout_.consensus_blocks().size();
             ++block_index) {
            const auto& block = layout_.consensus_blocks()[block_index];
            const auto& consensus = decoded.consensus[block_index];
            for (const auto scenario : block.node->scenario_indices) {
                const auto control = layout_.control_range(scenario, block.node->stage);
                const auto scenario_base = layout_.scenario_range(scenario).first;
                for (std::size_t component = 0; component < block.dimension; ++component) {
                    const auto local_index = control.first + component - scenario_base;
                    maximum = std::max(
                        maximum,
                        std::abs(decoded.local[scenario][local_index] - consensus[component])
                    );
                }
            }
        }
        return maximum;
    }

    [[nodiscard]] double expected_objective(
        std::span<const std::vector<double>> local_primals,
        std::span<const CQPValues> local_values
    ) const {
        if (local_primals.size() != scenario_count()) {
            throw std::invalid_argument("one local primal is required per scenario");
        }
        validate_local_values(local_values);
        const auto probabilities = tree_.probabilities();
        double objective = 0.0;
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto& primal = local_primals[scenario];
            if (primal.size() !=
                static_cast<std::size_t>(local_structure_.variables())) {
                throw std::invalid_argument("local primal has an incompatible size");
            }
            CscOperator quadratic{
                local_structure_.quadratic(),
                local_values[scenario].quadratic
            };
            const auto product = quadratic.multiply(primal);
            double local_objective = 0.0;
            for (std::size_t index = 0; index < primal.size(); ++index) {
                local_objective += 0.5 * primal[index] * product[index] +
                    local_values[scenario].linear_objective[index] * primal[index];
            }
            objective += probabilities[scenario] * local_objective;
        }
        return objective;
    }

    [[nodiscard]] double maximum_scalar_violation(
        std::span<const double> primal,
        const CQPValues& values
    ) const {
        validate_values(structure(), values);
        CscOperator scalar{structure().scalar_constraint(), values.scalar_constraint};
        const auto activity = scalar.multiply(primal);
        double maximum = 0.0;
        for (std::size_t row = 0; row < activity.size(); ++row) {
            maximum = std::max(
                maximum,
                std::max({
                    values.scalar_lower[row] - activity[row],
                    activity[row] - values.scalar_upper[row],
                    0.0,
                })
            );
        }
        return maximum;
    }

    [[nodiscard]] double maximum_affine_cone_violation(
        std::span<const double> primal,
        const CQPValues& values
    ) const {
        validate_values(structure(), values);
        if (!structure().affine_cone().has_value()) {
            return 0.0;
        }
        CscOperator affine{*structure().affine_cone(), values.affine_cone};
        auto activity = affine.multiply(primal);
        for (std::size_t row = 0; row < activity.size(); ++row) {
            activity[row] += values.affine_offset[row];
        }
        double maximum = 0.0;
        for (const auto& cone : structure().affine_cones()) {
            if (cone.kind != ConeKind::second_order) {
                throw std::logic_error(
                    "scenario cone diagnostic currently supports second-order cones only"
                );
            }
            double squared = 0.0;
            for (Index local = 0; local < cone.slot_count() - 1; ++local) {
                const double value = activity[static_cast<std::size_t>(cone.start + local)];
                squared += value * value;
            }
            maximum = std::max(
                maximum,
                std::sqrt(squared) -
                    activity[static_cast<std::size_t>(cone.stop() - 1)]
            );
        }
        return std::max(0.0, maximum);
    }

  private:
    struct MatrixBundle {
        CscMatrixData quadratic;
        CscMatrixData scalar;
        std::optional<CscMatrixData> affine;
    };

    void validate_global_dimensions() const {
        const auto maximum = static_cast<std::size_t>(
            std::numeric_limits<Index>::max()
        );
        const auto scalar_rows = scenario_count() *
            static_cast<std::size_t>(local_structure_.scalar_rows()) +
            nonanticipativity_rows();
        const auto affine_rows = scenario_count() *
            static_cast<std::size_t>(local_structure_.affine_rows());
        if (layout_.total_variables() > maximum || scalar_rows > maximum ||
            affine_rows > maximum) {
            throw std::invalid_argument("scenario CQP exceeds 32-bit index capacity");
        }
    }

    void validate_local_values(std::span<const CQPValues> local_values) const {
        if (local_values.size() != scenario_count()) {
            throw std::invalid_argument("one local CQP value set is required per scenario");
        }
        for (const auto& values : local_values) {
            validate_values(local_structure_, values);
        }
    }

    [[nodiscard]] CQPStructure build_structure() const {
        std::vector<CQPValues> symbolic_values;
        symbolic_values.reserve(scenario_count());
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            CQPValues values;
            values.quadratic.assign(local_structure_.quadratic().nonzeros(), 1.0);
            values.scalar_constraint.assign(
                local_structure_.scalar_constraint().nonzeros(),
                1.0
            );
            values.affine_cone.assign(
                local_structure_.affine_cone().has_value()
                    ? local_structure_.affine_cone()->nonzeros()
                    : 0U,
                1.0
            );
            values.linear_objective.assign(
                static_cast<std::size_t>(local_structure_.variables()),
                0.0
            );
            values.scalar_lower.assign(
                static_cast<std::size_t>(local_structure_.scalar_rows()),
                0.0
            );
            values.scalar_upper = values.scalar_lower;
            values.affine_offset.assign(
                static_cast<std::size_t>(local_structure_.affine_rows()),
                0.0
            );
            values.variable_lower.assign(
                static_cast<std::size_t>(local_structure_.variables()),
                -std::numeric_limits<double>::infinity()
            );
            values.variable_upper.assign(
                static_cast<std::size_t>(local_structure_.variables()),
                std::numeric_limits<double>::infinity()
            );
            symbolic_values.push_back(std::move(values));
        }
        const auto matrices = assemble(symbolic_values, true);
        std::vector<ConeBlock> affine_cones;
        affine_cones.reserve(
            scenario_count() * local_structure_.affine_cones().size()
        );
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto offset = static_cast<Index>(
                scenario * static_cast<std::size_t>(local_structure_.affine_rows())
            );
            for (const auto& cone : local_structure_.affine_cones()) {
                affine_cones.emplace_back(
                    cone.kind,
                    offset + cone.start,
                    cone.vector_dimension,
                    cone.power_alpha
                );
            }
        }
        std::vector<ConeBlock> variable_cones;
        variable_cones.reserve(
            scenario_count() * local_structure_.variable_cones().size()
        );
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto offset = static_cast<Index>(
                scenario * static_cast<std::size_t>(local_structure_.variables())
            );
            for (const auto& cone : local_structure_.variable_cones()) {
                variable_cones.emplace_back(
                    cone.kind,
                    offset + cone.start,
                    cone.vector_dimension,
                    cone.power_alpha
                );
            }
        }
        return CQPStructure{
            matrices.quadratic.structure,
            matrices.scalar.structure,
            matrices.affine.has_value()
                ? std::optional<CscStructure>{matrices.affine->structure}
                : std::nullopt,
            std::move(affine_cones),
            std::move(variable_cones),
        };
    }

    [[nodiscard]] MatrixBundle assemble(
        std::span<const CQPValues> local_values,
        const bool symbolic
    ) const {
        const auto scalar_rows = scenario_count() *
            static_cast<std::size_t>(local_structure_.scalar_rows()) +
            nonanticipativity_rows();
        const auto affine_rows = scenario_count() *
            static_cast<std::size_t>(local_structure_.affine_rows());
        CscBuilder quadratic{
            static_cast<Index>(layout_.total_variables()),
            static_cast<Index>(layout_.total_variables()),
        };
        CscBuilder scalar{
            static_cast<Index>(scalar_rows),
            static_cast<Index>(layout_.total_variables()),
        };
        std::optional<CscBuilder> affine;
        if (local_structure_.affine_cone().has_value()) {
            affine.emplace(
                static_cast<Index>(affine_rows),
                static_cast<Index>(layout_.total_variables())
            );
        }
        const auto probabilities = tree_.probabilities();
        for (std::size_t scenario = 0; scenario < scenario_count(); ++scenario) {
            const auto variable_offset = static_cast<Index>(
                scenario * static_cast<std::size_t>(local_structure_.variables())
            );
            const auto scalar_offset = static_cast<Index>(
                scenario * static_cast<std::size_t>(local_structure_.scalar_rows())
            );
            const auto affine_offset = static_cast<Index>(
                scenario * static_cast<std::size_t>(local_structure_.affine_rows())
            );
            append_matrix(
                quadratic,
                local_structure_.quadratic(),
                local_values[scenario].quadratic,
                0,
                variable_offset,
                symbolic ? 1.0 : probabilities[scenario]
            );
            append_matrix(
                scalar,
                local_structure_.scalar_constraint(),
                local_values[scenario].scalar_constraint,
                scalar_offset,
                variable_offset,
                1.0
            );
            if (affine.has_value()) {
                append_matrix(
                    *affine,
                    *local_structure_.affine_cone(),
                    local_values[scenario].affine_cone,
                    affine_offset,
                    variable_offset,
                    1.0
                );
            }
        }
        const auto row_offset = static_cast<Index>(
            scenario_count() * static_cast<std::size_t>(local_structure_.scalar_rows())
        );
        for (const auto& triplet : layout_.nonanticipativity_triplets()) {
            scalar.add(
                row_offset + triplet.row,
                triplet.column,
                triplet.value
            );
        }
        MatrixBundle result{
            quadratic.build(),
            scalar.build(),
            std::nullopt,
        };
        if (affine.has_value()) {
            result.affine.emplace(affine->build());
        }
        return result;
    }

    static void append_matrix(
        CscBuilder& builder,
        const CscStructure& structure,
        std::span<const double> values,
        const Index row_offset,
        const Index column_offset,
        const double scale
    ) {
        if (values.size() != structure.nonzeros()) {
            throw std::invalid_argument("scenario local matrix values have an incompatible size");
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
                    scale * values[position]
                );
            }
        }
    }

    ScenarioTree tree_;
    CQPStructure local_structure_;
    BlockArrowLayout layout_;
    std::optional<CQPStructure> structure_;
};

}  // namespace spacepdhcg::core
