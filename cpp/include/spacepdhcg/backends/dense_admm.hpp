#pragma once

#include "spacepdhcg/core/host_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::backends {

struct DenseAdmmConfig {
    double rho{1.0};
    double regularisation{1.0e-10};
    std::size_t maximum_variables{2'048U};

    void validate() const {
        if (!std::isfinite(rho) || rho <= 0.0) {
            throw std::invalid_argument("dense ADMM rho must be finite and positive");
        }
        if (!std::isfinite(regularisation) || regularisation <= 0.0) {
            throw std::invalid_argument(
                "dense ADMM regularisation must be finite and positive"
            );
        }
        if (maximum_variables == 0U) {
            throw std::invalid_argument("dense ADMM variable limit must be positive");
        }
    }
};

/// Small-problem CPU correctness backend for QP/SOCP development.
///
/// This backend intentionally forms and factorises a dense augmented matrix. It exists to run
/// the native C++ lifecycle before CUDA and to debug canonical CQP mappings. It is not a
/// performance baseline for large trajectories and must never be reported as PDHCG.
class DenseAdmmBackend final : public core::HostPersistentBackend {
  public:
    explicit DenseAdmmBackend(core::FixedCQP problem, DenseAdmmConfig config = {})
        : structure_(problem.structure()), values_(problem.values()), config_(config) {
        config_.validate();
        structure_.validate();
        values_.validate(structure_);
        if (static_cast<std::size_t>(structure_.variables()) > config_.maximum_variables) {
            throw std::invalid_argument("dense ADMM variable count exceeds its safety limit");
        }
        if (!structure_.variable_cones.empty()) {
            throw std::invalid_argument("dense ADMM does not support variable cone blocks");
        }
        for (const auto& cone : structure_.affine_cones) {
            if (cone.kind != ConeKind::second_order) {
                throw std::invalid_argument("dense ADMM currently supports affine SOC blocks only");
            }
        }
        const auto start = Clock::now();
        rebuild_numeric_operators();
        initialise_iterates();
        setup_seconds_ = seconds_since(start);
    }

    [[nodiscard]] const core::FixedStructure& structure() const noexcept override {
        return structure_;
    }

    [[nodiscard]] std::size_t update_count() const noexcept override { return update_count_; }

    void update(core::NumericValues values) override {
        values.validate(structure_);
        const auto start = Clock::now();
        values_ = std::move(values);
        rebuild_numeric_operators();
        reproject_iterates();
        last_update_seconds_ = seconds_since(start);
        ++update_count_;
    }

    void warm_start(const core::HostWarmStart& start) override {
        if (start.primal.empty() && start.dual.empty()) {
            throw std::invalid_argument("dense ADMM warm start requires a primal or dual vector");
        }
        if (!start.primal.empty()) {
            if (start.primal.size() != variables()) {
                throw std::invalid_argument("dense ADMM primal warm start has the wrong size");
            }
            x_ = start.primal;
        }
        if (!start.dual.empty()) {
            if (start.dual.size() != static_cast<std::size_t>(structure_.duals())) {
                throw std::invalid_argument("dense ADMM dual warm start has the wrong size");
            }
            for (std::size_t row = 0; row < scalar_rows(); ++row) {
                u_scalar_[row] = start.dual[row] / config_.rho;
            }
            for (std::size_t row = 0; row < affine_rows(); ++row) {
                u_affine_[row] = start.dual[scalar_rows() + row] / config_.rho;
            }
        }
        reproject_iterates();
        ++warm_start_count_;
    }

    [[nodiscard]] core::HostCqpSolution solve(
        double tolerance,
        std::size_t iteration_limit
    ) override {
        if (!std::isfinite(tolerance) || tolerance <= 0.0 || iteration_limit == 0U) {
            throw std::invalid_argument("dense ADMM solve request is invalid");
        }
        const auto solve_start = Clock::now();
        core::HostCqpSolution result{};
        result.setup_seconds = setup_seconds_;
        result.update_seconds = last_update_seconds_;
        result.status = SolveStatus::iteration_limit;

        double primal_residual = std::numeric_limits<double>::infinity();
        double dual_residual = std::numeric_limits<double>::infinity();
        std::size_t iteration{0U};
        for (; iteration < iteration_limit; ++iteration) {
            const auto previous_scalar = z_scalar_;
            const auto previous_affine = z_affine_;
            const auto previous_variable = z_variable_;
            x_update();
            const auto scalar_activity = multiply(a_, scalar_rows(), variables(), x_);
            const auto affine_activity = multiply(f_, affine_rows(), variables(), x_);

            for (std::size_t row = 0; row < scalar_rows(); ++row) {
                z_scalar_[row] = project_interval(
                    scalar_activity[row] + u_scalar_[row],
                    values_.scalar_lower[row],
                    values_.scalar_upper[row]
                );
            }
            for (std::size_t row = 0; row < affine_rows(); ++row) {
                affine_work_[row] = affine_activity[row] + values_.affine_offset[row]
                                    + u_affine_[row];
            }
            project_affine_cones(affine_work_, z_affine_);
            for (std::size_t variable = 0; variable < variables(); ++variable) {
                z_variable_[variable] = project_interval(
                    x_[variable] + u_variable_[variable],
                    values_.variable_lower[variable],
                    values_.variable_upper[variable]
                );
            }

            primal_residual = 0.0;
            for (std::size_t row = 0; row < scalar_rows(); ++row) {
                const auto residual = scalar_activity[row] - z_scalar_[row];
                u_scalar_[row] += residual;
                primal_residual = std::max(primal_residual, std::abs(residual));
            }
            for (std::size_t row = 0; row < affine_rows(); ++row) {
                const auto residual = affine_activity[row] + values_.affine_offset[row]
                                      - z_affine_[row];
                u_affine_[row] += residual;
                primal_residual = std::max(primal_residual, std::abs(residual));
            }
            for (std::size_t variable = 0; variable < variables(); ++variable) {
                const auto residual = x_[variable] - z_variable_[variable];
                u_variable_[variable] += residual;
                primal_residual = std::max(primal_residual, std::abs(residual));
            }

            const auto delta_scalar = difference(z_scalar_, previous_scalar);
            const auto delta_affine = difference(z_affine_, previous_affine);
            const auto delta_variable = difference(z_variable_, previous_variable);
            auto dual_vector = transpose_multiply(
                a_,
                scalar_rows(),
                variables(),
                delta_scalar
            );
            add_in_place(
                dual_vector,
                transpose_multiply(
                    f_,
                    affine_rows(),
                    variables(),
                    delta_affine
                )
            );
            add_in_place(dual_vector, delta_variable);
            dual_residual = config_.rho * infinity_norm(dual_vector);
            if (std::max(primal_residual, dual_residual) <= tolerance) {
                result.status = SolveStatus::optimal;
                ++iteration;
                break;
            }
        }

        result.primal = x_;
        result.dual.resize(static_cast<std::size_t>(structure_.duals()), 0.0);
        for (std::size_t row = 0; row < scalar_rows(); ++row) {
            result.dual[row] = config_.rho * u_scalar_[row];
        }
        for (std::size_t row = 0; row < affine_rows(); ++row) {
            result.dual[scalar_rows() + row] = config_.rho * u_affine_[row];
        }
        result.objective = objective(x_);
        result.primal_residual = primal_residual;
        result.dual_residual = dual_residual;
        result.outer_iterations = iteration;
        result.inner_iterations = 0U;
        result.solve_seconds = seconds_since(solve_start);
        ++solve_count_;
        return result;
    }

    [[nodiscard]] std::size_t solve_count() const noexcept { return solve_count_; }
    [[nodiscard]] std::size_t warm_start_count() const noexcept { return warm_start_count_; }

  private:
    using Clock = std::chrono::steady_clock;

    core::FixedStructure structure_{};
    core::NumericValues values_{};
    DenseAdmmConfig config_{};
    std::vector<double> q_{};
    std::vector<double> a_{};
    std::vector<double> f_{};
    std::vector<double> factor_{};
    std::vector<double> x_{};
    std::vector<double> z_scalar_{};
    std::vector<double> z_affine_{};
    std::vector<double> z_variable_{};
    std::vector<double> u_scalar_{};
    std::vector<double> u_affine_{};
    std::vector<double> u_variable_{};
    std::vector<double> affine_work_{};
    std::size_t update_count_{0U};
    std::size_t solve_count_{0U};
    std::size_t warm_start_count_{0U};
    double setup_seconds_{0.0};
    double last_update_seconds_{0.0};

    [[nodiscard]] std::size_t variables() const noexcept {
        return static_cast<std::size_t>(structure_.variables());
    }
    [[nodiscard]] std::size_t scalar_rows() const noexcept {
        return static_cast<std::size_t>(structure_.scalar_rows());
    }
    [[nodiscard]] std::size_t affine_rows() const noexcept {
        return static_cast<std::size_t>(structure_.affine_rows());
    }

    void rebuild_numeric_operators() {
        q_ = dense(structure_.quadratic, values_.quadratic);
        symmetrise(q_, variables());
        a_ = dense(structure_.scalar_constraint, values_.scalar_constraint);
        f_ = structure_.affine_cone.has_value()
                 ? dense(*structure_.affine_cone, values_.affine_cone)
                 : std::vector<double>{};
        build_factor();
    }

    void build_factor() {
        const auto n = variables();
        std::vector<double> matrix = q_;
        matrix.resize(n * n, 0.0);
        add_gram(matrix, a_, scalar_rows(), n, config_.rho);
        add_gram(matrix, f_, affine_rows(), n, config_.rho);
        for (std::size_t diagonal = 0; diagonal < n; ++diagonal) {
            matrix[diagonal * n + diagonal] += config_.rho + config_.regularisation;
        }
        factor_ = cholesky(matrix, n);
    }

    void initialise_iterates() {
        x_.assign(variables(), 0.0);
        z_scalar_.assign(scalar_rows(), 0.0);
        z_affine_.assign(affine_rows(), 0.0);
        z_variable_.assign(variables(), 0.0);
        u_scalar_.assign(scalar_rows(), 0.0);
        u_affine_.assign(affine_rows(), 0.0);
        u_variable_.assign(variables(), 0.0);
        affine_work_.assign(affine_rows(), 0.0);
        reproject_iterates();
    }

    void reproject_iterates() {
        const auto scalar_activity = multiply(a_, scalar_rows(), variables(), x_);
        const auto affine_activity = multiply(f_, affine_rows(), variables(), x_);
        for (std::size_t row = 0; row < scalar_rows(); ++row) {
            z_scalar_[row] = project_interval(
                scalar_activity[row] + u_scalar_[row],
                values_.scalar_lower[row],
                values_.scalar_upper[row]
            );
        }
        for (std::size_t row = 0; row < affine_rows(); ++row) {
            affine_work_[row] = affine_activity[row] + values_.affine_offset[row]
                                + u_affine_[row];
        }
        project_affine_cones(affine_work_, z_affine_);
        for (std::size_t variable = 0; variable < variables(); ++variable) {
            z_variable_[variable] = project_interval(
                x_[variable] + u_variable_[variable],
                values_.variable_lower[variable],
                values_.variable_upper[variable]
            );
        }
    }

    void x_update() {
        const auto scalar_term = difference(z_scalar_, u_scalar_);
        auto affine_term = difference(z_affine_, u_affine_);
        for (std::size_t row = 0; row < affine_rows(); ++row) {
            affine_term[row] -= values_.affine_offset[row];
        }
        const auto variable_term = difference(z_variable_, u_variable_);
        auto right_hand_side = transpose_multiply(
            a_,
            scalar_rows(),
            variables(),
            scalar_term
        );
        add_in_place(
            right_hand_side,
            transpose_multiply(f_, affine_rows(), variables(), affine_term)
        );
        add_in_place(right_hand_side, variable_term);
        for (std::size_t variable = 0; variable < variables(); ++variable) {
            right_hand_side[variable] = config_.rho * right_hand_side[variable]
                                        - values_.linear_objective[variable];
        }
        x_ = cholesky_solve(factor_, right_hand_side, variables());
    }

    void project_affine_cones(
        const std::vector<double>& source,
        std::vector<double>& destination
    ) const {
        if (source.size() != affine_rows()) {
            throw std::logic_error("affine cone projection received the wrong vector size");
        }
        destination = source;
        for (const auto& cone : structure_.affine_cones) {
            const auto start = static_cast<std::size_t>(cone.start);
            const auto slots = static_cast<std::size_t>(cone.vector_dimension) + 2U;
            double vector_norm_squared{0.0};
            for (std::size_t local = 0; local + 1U < slots; ++local) {
                const auto value = source[start + local];
                vector_norm_squared += value * value;
            }
            const auto vector_norm = std::sqrt(vector_norm_squared);
            const auto scalar = source[start + slots - 1U];
            if (vector_norm <= scalar) {
                continue;
            }
            if (vector_norm <= -scalar) {
                std::fill_n(destination.begin() + static_cast<std::ptrdiff_t>(start), slots, 0.0);
                continue;
            }
            const auto projected_scalar = 0.5 * (vector_norm + scalar);
            const auto scale = projected_scalar / vector_norm;
            for (std::size_t local = 0; local + 1U < slots; ++local) {
                destination[start + local] = scale * source[start + local];
            }
            destination[start + slots - 1U] = projected_scalar;
        }
    }

    [[nodiscard]] double objective(const std::vector<double>& primal) const {
        const auto product = multiply(q_, variables(), variables(), primal);
        double value{0.0};
        for (std::size_t variable = 0; variable < variables(); ++variable) {
            value += 0.5 * primal[variable] * product[variable]
                     + values_.linear_objective[variable] * primal[variable];
        }
        return value;
    }

    static double project_interval(double value, double lower, double upper) noexcept {
        if (std::isfinite(lower)) {
            value = std::max(value, lower);
        }
        if (std::isfinite(upper)) {
            value = std::min(value, upper);
        }
        return value;
    }

    static std::vector<double> dense(
        const core::CscPattern& pattern,
        const std::vector<double>& values
    ) {
        if (values.size() != pattern.nonzeros()) {
            throw std::invalid_argument("dense conversion value count is invalid");
        }
        const auto rows = static_cast<std::size_t>(pattern.rows);
        const auto columns = static_cast<std::size_t>(pattern.columns);
        std::vector<double> matrix(rows * columns, 0.0);
        for (Index column = 0; column < pattern.columns; ++column) {
            const auto begin = static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column)]);
            const auto end = static_cast<std::size_t>(
                pattern.offsets[static_cast<std::size_t>(column) + 1U]
            );
            for (std::size_t position = begin; position < end; ++position) {
                const auto row = static_cast<std::size_t>(pattern.indices[position]);
                matrix[row * columns + static_cast<std::size_t>(column)] += values[position];
            }
        }
        return matrix;
    }

    static void symmetrise(std::vector<double>& matrix, std::size_t dimension) {
        for (std::size_t row = 0; row < dimension; ++row) {
            for (std::size_t column = row + 1U; column < dimension; ++column) {
                const auto value = 0.5
                                   * (matrix[row * dimension + column]
                                      + matrix[column * dimension + row]);
                matrix[row * dimension + column] = value;
                matrix[column * dimension + row] = value;
            }
        }
    }

    static void add_gram(
        std::vector<double>& destination,
        const std::vector<double>& matrix,
        std::size_t rows,
        std::size_t columns,
        double scale
    ) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t left = 0; left < columns; ++left) {
                const auto left_value = matrix[row * columns + left];
                if (left_value == 0.0) {
                    continue;
                }
                for (std::size_t right = 0; right < columns; ++right) {
                    const auto right_value = matrix[row * columns + right];
                    if (right_value != 0.0) {
                        destination[left * columns + right] +=
                            scale * left_value * right_value;
                    }
                }
            }
        }
    }

    static std::vector<double> cholesky(
        const std::vector<double>& matrix,
        std::size_t dimension
    ) {
        std::vector<double> factor(dimension * dimension, 0.0);
        for (std::size_t row = 0; row < dimension; ++row) {
            for (std::size_t column = 0; column <= row; ++column) {
                auto value = matrix[row * dimension + column];
                for (std::size_t inner = 0; inner < column; ++inner) {
                    value -= factor[row * dimension + inner]
                             * factor[column * dimension + inner];
                }
                if (row == column) {
                    if (!std::isfinite(value) || value <= 0.0) {
                        throw std::runtime_error("dense ADMM Cholesky factorisation failed");
                    }
                    factor[row * dimension + column] = std::sqrt(value);
                } else {
                    factor[row * dimension + column] =
                        value / factor[column * dimension + column];
                }
            }
        }
        return factor;
    }

    static std::vector<double> cholesky_solve(
        const std::vector<double>& factor,
        const std::vector<double>& right_hand_side,
        std::size_t dimension
    ) {
        std::vector<double> intermediate(dimension, 0.0);
        for (std::size_t row = 0; row < dimension; ++row) {
            auto value = right_hand_side[row];
            for (std::size_t column = 0; column < row; ++column) {
                value -= factor[row * dimension + column] * intermediate[column];
            }
            intermediate[row] = value / factor[row * dimension + row];
        }
        std::vector<double> solution(dimension, 0.0);
        for (std::size_t reverse = 0; reverse < dimension; ++reverse) {
            const auto row = dimension - 1U - reverse;
            auto value = intermediate[row];
            for (std::size_t column = row + 1U; column < dimension; ++column) {
                value -= factor[column * dimension + row] * solution[column];
            }
            solution[row] = value / factor[row * dimension + row];
        }
        return solution;
    }

    static std::vector<double> multiply(
        const std::vector<double>& matrix,
        std::size_t rows,
        std::size_t columns,
        const std::vector<double>& vector
    ) {
        if (vector.size() != columns || matrix.size() != rows * columns) {
            throw std::invalid_argument("dense matrix-vector product has incompatible sizes");
        }
        std::vector<double> result(rows, 0.0);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                result[row] += matrix[row * columns + column] * vector[column];
            }
        }
        return result;
    }

    static std::vector<double> transpose_multiply(
        const std::vector<double>& matrix,
        std::size_t rows,
        std::size_t columns,
        const std::vector<double>& vector
    ) {
        if (vector.size() != rows || matrix.size() != rows * columns) {
            throw std::invalid_argument("dense transpose product has incompatible sizes");
        }
        std::vector<double> result(columns, 0.0);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                result[column] += matrix[row * columns + column] * vector[row];
            }
        }
        return result;
    }

    static std::vector<double> difference(
        const std::vector<double>& left,
        const std::vector<double>& right
    ) {
        if (left.size() != right.size()) {
            throw std::invalid_argument("dense ADMM vector sizes do not match");
        }
        std::vector<double> result(left.size(), 0.0);
        for (std::size_t index = 0; index < left.size(); ++index) {
            result[index] = left[index] - right[index];
        }
        return result;
    }

    static void add_in_place(
        std::vector<double>& destination,
        const std::vector<double>& source
    ) {
        if (destination.size() != source.size()) {
            throw std::invalid_argument("dense ADMM accumulation sizes do not match");
        }
        for (std::size_t index = 0; index < destination.size(); ++index) {
            destination[index] += source[index];
        }
    }

    static double infinity_norm(const std::vector<double>& values) noexcept {
        double maximum{0.0};
        for (const auto value : values) {
            maximum = std::max(maximum, std::abs(value));
        }
        return maximum;
    }

    static double seconds_since(const Clock::time_point& start) noexcept {
        return std::chrono::duration<double>(Clock::now() - start).count();
    }
};

}  // namespace spacepdhcg::backends
