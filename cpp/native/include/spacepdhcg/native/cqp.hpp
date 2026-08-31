#pragma once

#include "spacepdhcg/persistent_cqp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::native {

struct Triplet {
    Index row{0};
    Index column{0};
    double value{0.0};
};

class CscMatrix {
  public:
    Index rows{0};
    Index columns{0};
    std::vector<Index> offsets{};
    std::vector<Index> indices{};
    std::vector<double> values{};

    CscMatrix() = default;

    CscMatrix(
        Index row_count,
        Index column_count,
        std::vector<Index> column_offsets,
        std::vector<Index> row_indices,
        std::vector<double> numerical_values
    )
        : rows(row_count),
          columns(column_count),
          offsets(std::move(column_offsets)),
          indices(std::move(row_indices)),
          values(std::move(numerical_values)) {
        validate();
    }

    [[nodiscard]] std::size_t nonzeros() const noexcept { return values.size(); }

    void validate() const {
        if (rows < 0 || columns < 0) {
            throw std::invalid_argument("CSC dimensions must be non-negative");
        }
        if (offsets.size() != static_cast<std::size_t>(columns) + 1U) {
            throw std::invalid_argument("CSC offsets must contain columns + 1 entries");
        }
        if (offsets.empty() || offsets.front() != 0) {
            throw std::invalid_argument("CSC offsets must begin at zero");
        }
        if (indices.size() != values.size()) {
            throw std::invalid_argument("CSC row indices and values must have equal size");
        }
        if (offsets.back() != static_cast<Index>(indices.size())) {
            throw std::invalid_argument("CSC final offset must equal the number of entries");
        }
        for (std::size_t i = 1; i < offsets.size(); ++i) {
            if (offsets[i] < offsets[i - 1]) {
                throw std::invalid_argument("CSC offsets must be non-decreasing");
            }
        }
        for (Index column = 0; column < columns; ++column) {
            Index previous = -1;
            for (Index position = offsets[static_cast<std::size_t>(column)];
                 position < offsets[static_cast<std::size_t>(column) + 1U];
                 ++position) {
                const auto index = static_cast<std::size_t>(position);
                const Index row = indices[index];
                if (row < 0 || row >= rows) {
                    throw std::invalid_argument("CSC row index lies outside the matrix");
                }
                if (row <= previous) {
                    throw std::invalid_argument(
                        "CSC row indices must be strictly increasing in each column"
                    );
                }
                if (!std::isfinite(values[index])) {
                    throw std::invalid_argument("CSC numerical values must be finite");
                }
                previous = row;
            }
        }
    }

    [[nodiscard]] std::vector<double> multiply(std::span<const double> vector) const {
        if (vector.size() != static_cast<std::size_t>(columns)) {
            throw std::invalid_argument("matrix-vector input has the wrong dimension");
        }
        std::vector<double> result(static_cast<std::size_t>(rows), 0.0);
        for (Index column = 0; column < columns; ++column) {
            const double x = vector[static_cast<std::size_t>(column)];
            for (Index position = offsets[static_cast<std::size_t>(column)];
                 position < offsets[static_cast<std::size_t>(column) + 1U];
                 ++position) {
                const auto index = static_cast<std::size_t>(position);
                result[static_cast<std::size_t>(indices[index])] += values[index] * x;
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<double> transpose_multiply(
        std::span<const double> vector
    ) const {
        if (vector.size() != static_cast<std::size_t>(rows)) {
            throw std::invalid_argument("transpose-matrix input has the wrong dimension");
        }
        std::vector<double> result(static_cast<std::size_t>(columns), 0.0);
        for (Index column = 0; column < columns; ++column) {
            double value = 0.0;
            for (Index position = offsets[static_cast<std::size_t>(column)];
                 position < offsets[static_cast<std::size_t>(column) + 1U];
                 ++position) {
                const auto index = static_cast<std::size_t>(position);
                value += values[index] * vector[static_cast<std::size_t>(indices[index])];
            }
            result[static_cast<std::size_t>(column)] = value;
        }
        return result;
    }

    [[nodiscard]] SparsePatternView pattern_view() const noexcept {
        return SparsePatternView{
            SparseFormat::csc,
            rows,
            columns,
            HostConstSpan<Index>{offsets.data(), offsets.size()},
            HostConstSpan<Index>{indices.data(), indices.size()},
        };
    }
};

class CscBuilder {
  public:
    CscBuilder(Index rows, Index columns) : rows_(rows), columns_(columns) {
        if (rows < 0 || columns < 0) {
            throw std::invalid_argument("sparse builder dimensions must be non-negative");
        }
    }

    void add(Index row, Index column, double value) {
        if (row < 0 || row >= rows_ || column < 0 || column >= columns_) {
            throw std::out_of_range("sparse triplet lies outside the matrix");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("sparse triplet value must be finite");
        }
        triplets_.push_back(Triplet{row, column, value});
    }

    [[nodiscard]] CscMatrix build() const {
        auto ordered = triplets_;
        std::sort(
            ordered.begin(),
            ordered.end(),
            [](const Triplet& left, const Triplet& right) {
                return std::pair{left.column, left.row} < std::pair{right.column, right.row};
            }
        );

        std::vector<Triplet> combined;
        combined.reserve(ordered.size());
        for (const auto& entry : ordered) {
            if (!combined.empty() && combined.back().row == entry.row &&
                combined.back().column == entry.column) {
                combined.back().value += entry.value;
            } else {
                combined.push_back(entry);
            }
        }

        std::vector<Index> offsets(static_cast<std::size_t>(columns_) + 1U, 0);
        for (const auto& entry : combined) {
            ++offsets[static_cast<std::size_t>(entry.column) + 1U];
        }
        for (std::size_t i = 1; i < offsets.size(); ++i) {
            offsets[i] += offsets[i - 1];
        }

        std::vector<Index> indices;
        std::vector<double> values;
        indices.reserve(combined.size());
        values.reserve(combined.size());
        for (const auto& entry : combined) {
            indices.push_back(entry.row);
            values.push_back(entry.value);
        }
        return CscMatrix(rows_, columns_, std::move(offsets), std::move(indices), std::move(values));
    }

  private:
    Index rows_{0};
    Index columns_{0};
    std::vector<Triplet> triplets_{};
};

struct CqpDiagnostics {
    double scalar_violation{0.0};
    double variable_violation{0.0};
    double affine_cone_violation{0.0};

    [[nodiscard]] double maximum_violation() const noexcept {
        return std::max({scalar_violation, variable_violation, affine_cone_violation});
    }
};

struct OwnedCqp {
    CscMatrix quadratic{};
    CscMatrix scalar_constraint{};
    CscMatrix affine_cone{};
    std::vector<double> linear{};
    std::vector<double> scalar_lower{};
    std::vector<double> scalar_upper{};
    std::vector<double> affine_offset{};
    std::vector<double> variable_lower{};
    std::vector<double> variable_upper{};
    std::vector<ConeBlockDescriptor> affine_cones{};
    std::vector<ConeBlockDescriptor> variable_cones{};

    [[nodiscard]] Index variables() const noexcept { return quadratic.columns; }

    void validate() const {
        quadratic.validate();
        scalar_constraint.validate();
        affine_cone.validate();
        if (quadratic.rows != quadratic.columns) {
            throw std::invalid_argument("quadratic matrix must be square");
        }
        if (scalar_constraint.columns != variables() || affine_cone.columns != variables()) {
            throw std::invalid_argument("all CQP matrices must share the variable dimension");
        }
        validate_vector(linear, variables(), "linear objective", true);
        validate_vector(scalar_lower, scalar_constraint.rows, "scalar lower bound", false);
        validate_vector(scalar_upper, scalar_constraint.rows, "scalar upper bound", false);
        validate_vector(affine_offset, affine_cone.rows, "affine offset", true);
        validate_vector(variable_lower, variables(), "variable lower bound", false);
        validate_vector(variable_upper, variables(), "variable upper bound", false);
        validate_bounds(scalar_lower, scalar_upper, "scalar");
        validate_bounds(variable_lower, variable_upper, "variable");
        validate_cones(affine_cones, affine_cone.rows, true);
        validate_cones(variable_cones, variables(), false);
    }

    [[nodiscard]] double objective(std::span<const double> decision) const {
        require_decision(decision);
        const auto qx = quadratic.multiply(decision);
        double quadratic_term = 0.0;
        double linear_term = 0.0;
        for (std::size_t i = 0; i < decision.size(); ++i) {
            quadratic_term += decision[i] * qx[i];
            linear_term += linear[i] * decision[i];
        }
        return 0.5 * quadratic_term + linear_term;
    }

    [[nodiscard]] CqpDiagnostics diagnostics(std::span<const double> decision) const {
        require_decision(decision);
        return CqpDiagnostics{
            bound_violation(
                scalar_constraint.multiply(decision),
                scalar_lower,
                scalar_upper
            ),
            bound_violation(
                std::vector<double>(decision.begin(), decision.end()),
                variable_lower,
                variable_upper
            ),
            affine_violation(decision),
        };
    }

  private:
    static void validate_vector(
        const std::vector<double>& values,
        Index expected,
        const std::string& name,
        bool require_finite
    ) {
        if (values.size() != static_cast<std::size_t>(expected)) {
            throw std::invalid_argument(name + " has the wrong dimension");
        }
        for (double value : values) {
            if (std::isnan(value) || (require_finite && !std::isfinite(value))) {
                throw std::invalid_argument(name + " contains an invalid number");
            }
        }
    }

    static void validate_bounds(
        const std::vector<double>& lower,
        const std::vector<double>& upper,
        const std::string& name
    ) {
        for (std::size_t i = 0; i < lower.size(); ++i) {
            if (lower[i] > upper[i]) {
                throw std::invalid_argument(name + " lower bound exceeds upper bound");
            }
        }
    }

    [[nodiscard]] static Index cone_slots(const ConeBlockDescriptor& cone) {
        if (cone.vector_dimension <= 0 || cone.start < 0) {
            throw std::invalid_argument("cone metadata must be positive and non-negative");
        }
        switch (cone.kind) {
            case ConeKind::second_order:
            case ConeKind::rotated_second_order:
                return cone.vector_dimension + 2;
            case ConeKind::exponential:
            case ConeKind::power:
                return 3;
            case ConeKind::positive_semidefinite:
                return cone.vector_dimension * (cone.vector_dimension + 1) / 2;
        }
        throw std::invalid_argument("unknown cone kind");
    }

    static void validate_cones(
        const std::vector<ConeBlockDescriptor>& cones,
        Index ambient_dimension,
        bool require_cover
    ) {
        Index previous_stop = 0;
        for (const auto& cone : cones) {
            const Index stop = cone.start + cone_slots(cone);
            if (cone.start < previous_stop || stop > ambient_dimension) {
                throw std::invalid_argument("cone blocks overlap or exceed their ambient space");
            }
            if (require_cover && cone.start != previous_stop) {
                throw std::invalid_argument("affine cones must cover every affine row");
            }
            if (cone.kind == ConeKind::power &&
                !(cone.power_alpha > 0.0 && cone.power_alpha < 1.0)) {
                throw std::invalid_argument("power-cone alpha must lie in (0, 1)");
            }
            previous_stop = stop;
        }
        if (require_cover && previous_stop != ambient_dimension) {
            throw std::invalid_argument("affine cones do not cover every affine row");
        }
    }

    void require_decision(std::span<const double> decision) const {
        if (decision.size() != static_cast<std::size_t>(variables())) {
            throw std::invalid_argument("decision vector has the wrong dimension");
        }
        for (double value : decision) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("decision vector must be finite");
            }
        }
    }

    [[nodiscard]] static double bound_violation(
        const std::vector<double>& activity,
        const std::vector<double>& lower,
        const std::vector<double>& upper
    ) {
        double maximum = 0.0;
        for (std::size_t i = 0; i < activity.size(); ++i) {
            maximum = std::max(maximum, lower[i] - activity[i]);
            maximum = std::max(maximum, activity[i] - upper[i]);
        }
        return std::max(maximum, 0.0);
    }

    [[nodiscard]] double affine_violation(std::span<const double> decision) const {
        if (affine_cone.rows == 0) {
            return 0.0;
        }
        auto activity = affine_cone.multiply(decision);
        for (std::size_t i = 0; i < activity.size(); ++i) {
            activity[i] += affine_offset[i];
        }

        double maximum = 0.0;
        for (const auto& cone : affine_cones) {
            const auto start = static_cast<std::size_t>(cone.start);
            if (cone.kind == ConeKind::second_order) {
                const auto stop = start + static_cast<std::size_t>(cone_slots(cone));
                double norm_squared = 0.0;
                for (std::size_t i = start; i + 1U < stop; ++i) {
                    norm_squared += activity[i] * activity[i];
                }
                maximum = std::max(
                    maximum,
                    std::sqrt(norm_squared) - activity[stop - 1U]
                );
            } else if (cone.kind == ConeKind::rotated_second_order) {
                double norm_squared = 0.0;
                for (Index i = 0; i < cone.vector_dimension; ++i) {
                    const double value = activity[start + static_cast<std::size_t>(i)];
                    norm_squared += value * value;
                }
                const double first = activity[start + static_cast<std::size_t>(cone.vector_dimension)];
                const double second = activity[
                    start + static_cast<std::size_t>(cone.vector_dimension) + 1U
                ];
                maximum = std::max(maximum, -first);
                maximum = std::max(maximum, -second);
                if (first >= 0.0 && second >= 0.0) {
                    maximum = std::max(
                        maximum,
                        std::sqrt(norm_squared) - std::sqrt(2.0 * first * second)
                    );
                }
            } else {
                throw std::logic_error(
                    "host diagnostics currently implement SOC and rotated-SOC cones only"
                );
            }
        }
        return std::max(maximum, 0.0);
    }
};

}  // namespace spacepdhcg::native
