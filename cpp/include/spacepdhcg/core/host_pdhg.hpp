#pragma once

#include "spacepdhcg/core/cqp.hpp"
#include "spacepdhcg/core/csc_operator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::core {

struct HostPDHGOptions {
    double tolerance{1.0e-6};
    std::size_t iteration_limit{100'000};
    std::size_t check_interval{25};
    std::size_t norm_iterations{30};
    double step_safety{0.95};
    double extrapolation{1.0};

    void validate() const {
        if (!std::isfinite(tolerance) || tolerance <= 0.0) {
            throw std::invalid_argument("host PDHG tolerance must be finite and positive");
        }
        if (iteration_limit == 0U || check_interval == 0U || norm_iterations == 0U) {
            throw std::invalid_argument("host PDHG iteration counts must be positive");
        }
        if (!std::isfinite(step_safety) || step_safety <= 0.0 || step_safety >= 1.0) {
            throw std::invalid_argument("host PDHG step safety must lie in (0,1)");
        }
        if (!std::isfinite(extrapolation) || extrapolation < 0.0 || extrapolation > 1.0) {
            throw std::invalid_argument("host PDHG extrapolation must lie in [0,1]");
        }
    }
};

struct HostPDHGSolution {
    std::string status{"iteration_limit"};
    std::vector<double> primal;
    std::vector<double> dual;
    double objective{std::numeric_limits<double>::quiet_NaN()};
    double primal_residual{std::numeric_limits<double>::infinity()};
    double dual_residual{std::numeric_limits<double>::infinity()};
    std::size_t iterations{0};
    double operator_norm{0.0};
    double primal_step{0.0};
    double dual_step{0.0};

    [[nodiscard]] bool solved() const noexcept { return status == "solved"; }
};

/// Persistent, dependency-free CPU PDHG reference for diagonal-Q CQP problems.
///
/// This is a correctness and lifecycle backend, not a performance substitute for
/// PDHCG-CQP.  It supports scalar interval rows and affine SOC/rotated-SOC rows.
/// Variable cone blocks and non-diagonal quadratic objectives deliberately fail fast.
class PersistentHostPDHG {
  public:
    PersistentHostPDHG(CQPStructure structure, CQPValues values)
        : structure_(std::move(structure)),
          scalar_operator_(structure_.scalar_constraint(), values.scalar_constraint),
          affine_operator_(make_affine_operator(structure_, values)),
          values_(std::move(values)) {
        validate_values(structure_, values_);
        if (!structure_.variable_cones().empty()) {
            throw std::invalid_argument(
                "host PDHG does not yet support native variable-cone blocks"
            );
        }
        extract_quadratic_diagonal();
        primal_.assign(static_cast<std::size_t>(structure_.variables()), 0.0);
        dual_.assign(static_cast<std::size_t>(structure_.duals()), 0.0);
        previous_primal_ = primal_;
        refresh_step_sizes();
    }

    [[nodiscard]] const CQPStructure& structure() const noexcept { return structure_; }
    [[nodiscard]] const CQPValues& values() const noexcept { return values_; }
    [[nodiscard]] std::size_t update_count() const noexcept { return update_count_; }
    [[nodiscard]] std::size_t warm_start_count() const noexcept { return warm_start_count_; }
    [[nodiscard]] std::size_t solve_count() const noexcept { return solve_count_; }

    void update_values(CQPValues values, const bool refresh_norm = false) {
        validate_values(structure_, values);
        values_ = std::move(values);
        scalar_operator_.update_values(values_.scalar_constraint);
        if (affine_operator_.has_value()) {
            affine_operator_->update_values(values_.affine_cone);
        }
        extract_quadratic_diagonal();
        if (refresh_norm) {
            refresh_step_sizes();
        }
        ++update_count_;
    }

    void warm_start(
        std::span<const double> primal,
        std::span<const double> dual = {}
    ) {
        if (primal.size() != primal_.size()) {
            throw std::invalid_argument("host PDHG primal warm start has an incompatible size");
        }
        if (!dual.empty() && dual.size() != dual_.size()) {
            throw std::invalid_argument("host PDHG dual warm start has an incompatible size");
        }
        require_finite(primal, "host PDHG primal warm start");
        std::copy(primal.begin(), primal.end(), primal_.begin());
        project_variable_bounds(primal_);
        previous_primal_ = primal_;
        if (!dual.empty()) {
            require_finite(dual, "host PDHG dual warm start");
            std::copy(dual.begin(), dual.end(), dual_.begin());
        }
        ++warm_start_count_;
    }

    void reset_iterates() noexcept {
        std::fill(primal_.begin(), primal_.end(), 0.0);
        std::fill(dual_.begin(), dual_.end(), 0.0);
        project_variable_bounds(primal_);
        previous_primal_ = primal_;
    }

    [[nodiscard]] HostPDHGSolution solve(HostPDHGOptions options = {}) {
        options.validate();
        if (!step_sizes_initialised_ || options.norm_iterations != norm_iterations_ ||
            options.step_safety != step_safety_) {
            refresh_step_sizes(options.norm_iterations, options.step_safety);
        }
        std::vector<double> extrapolated(primal_.size(), 0.0);
        std::vector<double> next_primal(primal_.size(), 0.0);
        std::vector<double> temporary_dual(dual_.size(), 0.0);
        double primal_residual = std::numeric_limits<double>::infinity();
        double dual_residual = std::numeric_limits<double>::infinity();
        std::size_t completed_iterations = 0U;

        for (std::size_t iteration = 0; iteration < options.iteration_limit; ++iteration) {
            for (std::size_t index = 0; index < primal_.size(); ++index) {
                extrapolated[index] = primal_[index] +
                    options.extrapolation * (primal_[index] - previous_primal_[index]);
            }

            dual_update(extrapolated, temporary_dual);
            const auto transpose = transpose_product(dual_);
            for (std::size_t index = 0; index < primal_.size(); ++index) {
                const double proximal_argument = primal_[index] - primal_step_ * transpose[index];
                const double denominator = 1.0 + primal_step_ * quadratic_diagonal_[index];
                next_primal[index] =
                    (proximal_argument - primal_step_ * values_.linear_objective[index]) /
                    denominator;
            }
            project_variable_bounds(next_primal);
            previous_primal_.swap(primal_);
            primal_.swap(next_primal);
            completed_iterations = iteration + 1U;

            if (completed_iterations % options.check_interval == 0U ||
                completed_iterations == options.iteration_limit) {
                primal_residual = maximum_primal_violation(primal_);
                dual_residual = stationarity_residual(primal_, dual_);
                if (std::max(primal_residual, dual_residual) <= options.tolerance) {
                    ++solve_count_;
                    return make_solution(
                        "solved",
                        completed_iterations,
                        primal_residual,
                        dual_residual
                    );
                }
            }
        }
        ++solve_count_;
        return make_solution(
            "iteration_limit",
            completed_iterations,
            primal_residual,
            dual_residual
        );
    }

  private:
    [[nodiscard]] static std::optional<CscOperator> make_affine_operator(
        const CQPStructure& structure,
        const CQPValues& values
    ) {
        if (!structure.affine_cone().has_value()) {
            return std::nullopt;
        }
        return CscOperator{*structure.affine_cone(), values.affine_cone};
    }

    static void require_finite(std::span<const double> values, const char* name) {
        if (!std::all_of(values.begin(), values.end(), [](const double value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument(std::string{name} + " must be finite");
        }
    }

    void extract_quadratic_diagonal() {
        quadratic_diagonal_.assign(static_cast<std::size_t>(structure_.variables()), 0.0);
        const auto offsets = structure_.quadratic().offsets();
        const auto indices = structure_.quadratic().indices();
        for (Index column = 0; column < structure_.variables(); ++column) {
            const auto column_index = static_cast<std::size_t>(column);
            const auto begin = static_cast<std::size_t>(offsets[column_index]);
            const auto end = static_cast<std::size_t>(offsets[column_index + 1U]);
            for (auto position = begin; position < end; ++position) {
                const double value = values_.quadratic[position];
                if (indices[position] == column) {
                    quadratic_diagonal_[column_index] += value;
                } else if (std::abs(value) > 1.0e-14) {
                    throw std::invalid_argument(
                        "host PDHG currently requires a diagonal quadratic objective"
                    );
                }
            }
            if (quadratic_diagonal_[column_index] < -1.0e-14) {
                throw std::invalid_argument("host PDHG quadratic diagonal must be non-negative");
            }
            quadratic_diagonal_[column_index] =
                std::max(0.0, quadratic_diagonal_[column_index]);
        }
    }

    void refresh_step_sizes(
        const std::size_t norm_iterations = 30U,
        const double step_safety = 0.95
    ) {
        if (norm_iterations == 0U || !std::isfinite(step_safety) || step_safety <= 0.0 ||
            step_safety >= 1.0) {
            throw std::invalid_argument("host PDHG norm-estimation parameters are invalid");
        }
        operator_norm_ = estimate_operator_norm(norm_iterations);
        const double effective_norm = std::max(operator_norm_, 1.0e-12);
        primal_step_ = step_safety / effective_norm;
        dual_step_ = step_safety / effective_norm;
        norm_iterations_ = norm_iterations;
        step_safety_ = step_safety;
        step_sizes_initialised_ = true;
    }

    [[nodiscard]] double estimate_operator_norm(const std::size_t iterations) const {
        const std::size_t variables = static_cast<std::size_t>(structure_.variables());
        if (structure_.duals() == 0 || variables == 0U) {
            return 0.0;
        }
        std::vector<double> vector(variables, 1.0 / std::sqrt(static_cast<double>(variables)));
        double eigenvalue = 0.0;
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            const auto forward = forward_product(vector);
            auto transpose = transpose_product(forward);
            const double norm = l2_norm(transpose);
            if (norm <= 1.0e-30) {
                return 0.0;
            }
            for (auto& value : transpose) {
                value /= norm;
            }
            vector.swap(transpose);
            const auto image = forward_product(vector);
            eigenvalue = dot(image, image);
        }
        return std::sqrt(std::max(0.0, eigenvalue));
    }

    [[nodiscard]] std::vector<double> forward_product(
        std::span<const double> primal
    ) const {
        auto scalar = scalar_operator_.multiply(primal);
        if (!affine_operator_.has_value()) {
            return scalar;
        }
        const auto affine = affine_operator_->multiply(primal);
        scalar.insert(scalar.end(), affine.begin(), affine.end());
        return scalar;
    }

    [[nodiscard]] std::vector<double> transpose_product(
        std::span<const double> dual
    ) const {
        const std::size_t scalar_rows = static_cast<std::size_t>(structure_.scalar_rows());
        if (dual.size() != static_cast<std::size_t>(structure_.duals())) {
            throw std::invalid_argument("host PDHG dual vector has an incompatible size");
        }
        auto result = scalar_operator_.transpose_multiply(dual.first(scalar_rows));
        if (affine_operator_.has_value()) {
            const auto affine = affine_operator_->transpose_multiply(dual.subspan(scalar_rows));
            for (std::size_t index = 0; index < result.size(); ++index) {
                result[index] += affine[index];
            }
        }
        return result;
    }

    void dual_update(
        std::span<const double> extrapolated_primal,
        std::vector<double>& temporary
    ) {
        const auto activity = forward_product(extrapolated_primal);
        const std::size_t scalar_rows = static_cast<std::size_t>(structure_.scalar_rows());
        for (std::size_t row = 0; row < scalar_rows; ++row) {
            const double candidate = dual_[row] + dual_step_ * activity[row];
            const double scaled = candidate / dual_step_;
            const double projection = project_interval(
                scaled,
                values_.scalar_lower[row],
                values_.scalar_upper[row]
            );
            temporary[row] = candidate - dual_step_ * projection;
        }

        if (affine_operator_.has_value()) {
            const std::size_t affine_offset = scalar_rows;
            for (const auto& cone : structure_.affine_cones()) {
                std::vector<double> shifted(static_cast<std::size_t>(cone.slot_count()), 0.0);
                for (Index local = 0; local < cone.slot_count(); ++local) {
                    const std::size_t affine_row = static_cast<std::size_t>(cone.start + local);
                    const std::size_t dual_row = affine_offset + affine_row;
                    const double candidate = dual_[dual_row] +
                        dual_step_ * activity[dual_row];
                    shifted[static_cast<std::size_t>(local)] =
                        candidate / dual_step_ + values_.affine_offset[affine_row];
                }
                const auto projection = project_cone(cone, shifted);
                for (Index local = 0; local < cone.slot_count(); ++local) {
                    const std::size_t affine_row = static_cast<std::size_t>(cone.start + local);
                    const std::size_t dual_row = affine_offset + affine_row;
                    const double candidate = dual_[dual_row] +
                        dual_step_ * activity[dual_row];
                    const double shifted_projection =
                        projection[static_cast<std::size_t>(local)] -
                        values_.affine_offset[affine_row];
                    temporary[dual_row] = candidate - dual_step_ * shifted_projection;
                }
            }
        }
        dual_.swap(temporary);
    }

    [[nodiscard]] static double project_interval(
        const double value,
        const double lower,
        const double upper
    ) noexcept {
        return std::min(std::max(value, lower), upper);
    }

    [[nodiscard]] static std::vector<double> project_cone(
        const ConeBlock& cone,
        std::span<const double> values
    ) {
        switch (cone.kind) {
            case ConeKind::second_order:
                return project_soc(values);
            case ConeKind::rotated_second_order:
                return project_rotated_soc(cone.vector_dimension, values);
            case ConeKind::exponential:
            case ConeKind::power:
            case ConeKind::positive_semidefinite:
                throw std::invalid_argument(
                    "host PDHG cone projector supports only SOC and rotated SOC"
                );
        }
        throw std::logic_error("unhandled host PDHG cone kind");
    }

    [[nodiscard]] static std::vector<double> project_soc(
        std::span<const double> values
    ) {
        if (values.size() < 2U) {
            throw std::invalid_argument("SOC projection requires at least two slots");
        }
        const std::size_t scalar_index = values.size() - 1U;
        const double vector_norm = l2_norm(values.first(scalar_index));
        const double scalar = values[scalar_index];
        std::vector<double> result(values.begin(), values.end());
        if (vector_norm <= scalar) {
            return result;
        }
        if (vector_norm <= -scalar) {
            std::fill(result.begin(), result.end(), 0.0);
            return result;
        }
        const double projected_scalar = 0.5 * (vector_norm + scalar);
        const double scale = projected_scalar / vector_norm;
        for (std::size_t index = 0; index < scalar_index; ++index) {
            result[index] = scale * values[index];
        }
        result[scalar_index] = projected_scalar;
        return result;
    }

    [[nodiscard]] static std::vector<double> project_rotated_soc(
        const Index vector_dimension,
        std::span<const double> values
    ) {
        const std::size_t dimension = static_cast<std::size_t>(vector_dimension);
        if (values.size() != dimension + 2U) {
            throw std::invalid_argument("rotated-SOC slot count is inconsistent");
        }
        std::vector<double> transformed(values.size(), 0.0);
        transformed[0] = values[dimension] + values[dimension + 1U];
        for (std::size_t index = 0; index < dimension; ++index) {
            transformed[index + 1U] = std::sqrt(2.0) * values[index];
        }
        transformed.back() = values[dimension] - values[dimension + 1U];

        // project_soc expects vector slots first and scalar last. Reorder the
        // equivalent standard SOC from [scalar, vector...] to [vector..., scalar].
        std::vector<double> native_soc(values.size(), 0.0);
        for (std::size_t index = 1U; index < transformed.size(); ++index) {
            native_soc[index - 1U] = transformed[index];
        }
        native_soc.back() = transformed[0];
        const auto projected_soc = project_soc(native_soc);

        std::vector<double> result(values.size(), 0.0);
        const double projected_sum = projected_soc.back();
        const double projected_difference = projected_soc[dimension];
        for (std::size_t index = 0; index < dimension; ++index) {
            result[index] = projected_soc[index] / std::sqrt(2.0);
        }
        result[dimension] = 0.5 * (projected_sum + projected_difference);
        result[dimension + 1U] = 0.5 * (projected_sum - projected_difference);
        return result;
    }

    void project_variable_bounds(std::vector<double>& primal) const noexcept {
        for (std::size_t index = 0; index < primal.size(); ++index) {
            primal[index] = project_interval(
                primal[index],
                values_.variable_lower[index],
                values_.variable_upper[index]
            );
        }
    }

    [[nodiscard]] double maximum_primal_violation(
        std::span<const double> primal
    ) const {
        double maximum = 0.0;
        const auto scalar = scalar_operator_.multiply(primal);
        for (std::size_t row = 0; row < scalar.size(); ++row) {
            maximum = std::max(
                maximum,
                std::max(
                    values_.scalar_lower[row] - scalar[row],
                    scalar[row] - values_.scalar_upper[row]
                )
            );
        }
        for (std::size_t index = 0; index < primal.size(); ++index) {
            maximum = std::max(
                maximum,
                std::max(
                    values_.variable_lower[index] - primal[index],
                    primal[index] - values_.variable_upper[index]
                )
            );
        }
        if (affine_operator_.has_value()) {
            const auto affine = affine_operator_->multiply(primal);
            for (const auto& cone : structure_.affine_cones()) {
                std::vector<double> shifted(static_cast<std::size_t>(cone.slot_count()), 0.0);
                for (Index local = 0; local < cone.slot_count(); ++local) {
                    const std::size_t row = static_cast<std::size_t>(cone.start + local);
                    shifted[static_cast<std::size_t>(local)] =
                        affine[row] + values_.affine_offset[row];
                }
                const auto projection = project_cone(cone, shifted);
                for (std::size_t index = 0; index < shifted.size(); ++index) {
                    maximum = std::max(maximum, std::abs(shifted[index] - projection[index]));
                }
            }
        }
        return std::max(0.0, maximum);
    }

    [[nodiscard]] double stationarity_residual(
        std::span<const double> primal,
        std::span<const double> dual
    ) const {
        auto gradient = transpose_product(dual);
        for (std::size_t index = 0; index < gradient.size(); ++index) {
            gradient[index] += quadratic_diagonal_[index] * primal[index] +
                values_.linear_objective[index];
        }
        double maximum = 0.0;
        for (std::size_t index = 0; index < gradient.size(); ++index) {
            const double projected = project_interval(
                primal[index] - gradient[index],
                values_.variable_lower[index],
                values_.variable_upper[index]
            );
            maximum = std::max(maximum, std::abs(primal[index] - projected));
        }
        return maximum;
    }

    [[nodiscard]] double objective(std::span<const double> primal) const {
        double value = 0.0;
        for (std::size_t index = 0; index < primal.size(); ++index) {
            value += 0.5 * quadratic_diagonal_[index] * primal[index] * primal[index] +
                values_.linear_objective[index] * primal[index];
        }
        return value;
    }

    [[nodiscard]] HostPDHGSolution make_solution(
        std::string status,
        const std::size_t iterations,
        const double primal_residual,
        const double dual_residual
    ) const {
        return HostPDHGSolution{
            std::move(status),
            primal_,
            dual_,
            objective(primal_),
            primal_residual,
            dual_residual,
            iterations,
            operator_norm_,
            primal_step_,
            dual_step_,
        };
    }

    [[nodiscard]] static double dot(
        std::span<const double> left,
        std::span<const double> right
    ) {
        if (left.size() != right.size()) {
            throw std::invalid_argument("dot-product dimensions do not match");
        }
        double result = 0.0;
        for (std::size_t index = 0; index < left.size(); ++index) {
            result += left[index] * right[index];
        }
        return result;
    }

    [[nodiscard]] static double l2_norm(std::span<const double> values) {
        return std::sqrt(dot(values, values));
    }

    CQPStructure structure_;
    CscOperator scalar_operator_;
    std::optional<CscOperator> affine_operator_;
    CQPValues values_;
    std::vector<double> quadratic_diagonal_;
    std::vector<double> primal_;
    std::vector<double> previous_primal_;
    std::vector<double> dual_;
    double operator_norm_{0.0};
    double primal_step_{0.0};
    double dual_step_{0.0};
    std::size_t norm_iterations_{0U};
    double step_safety_{0.0};
    bool step_sizes_initialised_{false};
    std::size_t update_count_{0U};
    std::size_t warm_start_count_{0U};
    std::size_t solve_count_{0U};
};

}  // namespace spacepdhcg::core
