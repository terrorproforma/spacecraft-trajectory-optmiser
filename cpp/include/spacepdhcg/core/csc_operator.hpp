#pragma once

#include "spacepdhcg/core/cqp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace spacepdhcg::core {

/// CPU truth implementation of the sparse products used by PDHCG iterations.
class CscOperator {
  public:
    CscOperator(CscStructure structure, std::vector<double> values)
        : structure_(std::move(structure)), values_(std::move(values)) {
        if (values_.size() != structure_.nonzeros()) {
            throw std::invalid_argument("CSC numerical values do not match fixed topology");
        }
        if (!std::all_of(values_.begin(), values_.end(), [](const double value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument("CSC numerical values must be finite");
        }
    }

    [[nodiscard]] const CscStructure& structure() const noexcept { return structure_; }
    [[nodiscard]] std::span<const double> values() const noexcept { return values_; }

    void update_values(std::span<const double> values) {
        if (values.size() != values_.size()) {
            throw std::invalid_argument("CSC update changed numerical buffer length");
        }
        if (!std::all_of(values.begin(), values.end(), [](const double value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument("CSC numerical update must be finite");
        }
        std::copy(values.begin(), values.end(), values_.begin());
        ++update_count_;
    }

    [[nodiscard]] std::size_t update_count() const noexcept { return update_count_; }

    [[nodiscard]] std::vector<double> multiply(std::span<const double> vector) const {
        if (vector.size() != static_cast<std::size_t>(structure_.columns())) {
            throw std::invalid_argument("CSC forward product received an incompatible vector");
        }
        std::vector<double> result(static_cast<std::size_t>(structure_.rows()), 0.0);
        const auto offsets = structure_.offsets();
        const auto indices = structure_.indices();
        for (Index column = 0; column < structure_.columns(); ++column) {
            const auto column_index = static_cast<std::size_t>(column);
            const auto begin = static_cast<std::size_t>(offsets[column_index]);
            const auto end = static_cast<std::size_t>(offsets[column_index + 1U]);
            const double x = vector[column_index];
            for (auto position = begin; position < end; ++position) {
                result[static_cast<std::size_t>(indices[position])] += values_[position] * x;
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<double> transpose_multiply(
        std::span<const double> vector
    ) const {
        if (vector.size() != static_cast<std::size_t>(structure_.rows())) {
            throw std::invalid_argument("CSC transpose product received an incompatible vector");
        }
        std::vector<double> result(static_cast<std::size_t>(structure_.columns()), 0.0);
        const auto offsets = structure_.offsets();
        const auto indices = structure_.indices();
        for (Index column = 0; column < structure_.columns(); ++column) {
            const auto column_index = static_cast<std::size_t>(column);
            const auto begin = static_cast<std::size_t>(offsets[column_index]);
            const auto end = static_cast<std::size_t>(offsets[column_index + 1U]);
            double sum = 0.0;
            for (auto position = begin; position < end; ++position) {
                sum += values_[position] * vector[static_cast<std::size_t>(indices[position])];
            }
            result[column_index] = sum;
        }
        return result;
    }

    [[nodiscard]] static double infinity_norm(std::span<const double> vector) noexcept {
        double maximum = 0.0;
        for (const auto value : vector) {
            maximum = std::max(maximum, std::abs(value));
        }
        return maximum;
    }

  private:
    CscStructure structure_;
    std::vector<double> values_;
    std::size_t update_count_{0};
};

}  // namespace spacepdhcg::core
