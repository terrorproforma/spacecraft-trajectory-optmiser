#pragma once

// Shared helpers for the free-final-time (time-dilated) transcriptions.
//
// The fixed-final-time transcriptions (`powered_descent_3dof.hpp`, `powered_descent_6dof.hpp`)
// each carry private copies of these utilities.  The free-final-time variants are NEW
// topologies (`pd3_fft`, `pd6_fft`) and deliberately do not touch those headers so the frozen
// P1-C / P1-D fixtures and their CSC fingerprints stay byte-identical.

#include "spacepdhcg/core/fixed_cqp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace spacepdhcg::transcription::free_time {

struct Range {
    std::size_t start{0U};
    std::size_t size{0U};

    [[nodiscard]] std::size_t stop() const noexcept { return start + size; }
};

/// Row/column -> CSC slot lookup so numeric fills never disturb the frozen pattern.
class ValueIndex {
  public:
    ValueIndex() = default;
    explicit ValueIndex(const core::CscPattern& pattern) {
        for (Index column = 0; column < pattern.columns; ++column) {
            const auto begin =
                static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column)]);
            const auto end =
                static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column) + 1U]);
            for (std::size_t position = begin; position < end; ++position) {
                positions_[{pattern.indices[position], column}] = position;
            }
        }
    }

    void set(std::vector<double>& values, std::size_t row, std::size_t column, double value)
        const {
        const auto iterator =
            positions_.find({static_cast<Index>(row), static_cast<Index>(column)});
        if (iterator == positions_.end()) {
            throw std::logic_error(
                "coefficient (" + std::to_string(row) + ", " + std::to_string(column)
                + ") is absent from the fixed sparse pattern"
            );
        }
        values[iterator->second] = value;
    }

  private:
    std::map<std::pair<Index, Index>, std::size_t> positions_{};
};

using EntrySet = std::set<std::pair<std::size_t, std::size_t>>;

inline core::CscPattern make_pattern(std::size_t rows, std::size_t columns, const EntrySet& entries) {
    std::vector<std::pair<std::size_t, std::size_t>> ordered(entries.begin(), entries.end());
    std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
        return std::tie(left.second, left.first) < std::tie(right.second, right.first);
    });
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

inline std::vector<double> matvec(
    const core::CscPattern& pattern,
    const std::vector<double>& values,
    const std::vector<double>& vector
) {
    std::vector<double> result(static_cast<std::size_t>(pattern.rows), 0.0);
    for (Index column = 0; column < pattern.columns; ++column) {
        const auto begin =
            static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column)]);
        const auto end =
            static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column) + 1U]);
        for (std::size_t position = begin; position < end; ++position) {
            result[static_cast<std::size_t>(pattern.indices[position])] +=
                values[position] * vector[static_cast<std::size_t>(column)];
        }
    }
    return result;
}

inline void append_uniform_cones(
    std::vector<ConeBlockDescriptor>& blocks,
    std::size_t start,
    std::size_t count,
    std::size_t stride,
    Index vector_dimension
) {
    for (std::size_t item = 0; item < count; ++item) {
        blocks.push_back(ConeBlockDescriptor{
            ConeKind::second_order,
            static_cast<Index>(start + stride * item),
            vector_dimension,
            0.0,
        });
    }
}

/// Second-order-cone violation `max(||vector|| - scalar, 0)` for every affine cone block.
inline double cone_violation(
    const core::FixedStructure& structure,
    const std::vector<double>& affine,
    const std::vector<double>& affine_offset
) {
    double worst = 0.0;
    for (const auto& cone : structure.affine_cones) {
        const auto slots = core::detail::cone_slot_count(cone);
        const auto start = static_cast<std::size_t>(cone.start);
        double norm_squared = 0.0;
        for (std::size_t local = 0; local + 1U < slots; ++local) {
            const auto value = affine[start + local] + affine_offset[start + local];
            norm_squared += value * value;
        }
        const auto scalar = affine[start + slots - 1U] + affine_offset[start + slots - 1U];
        worst = std::max(worst, std::sqrt(norm_squared) - scalar);
    }
    return worst;
}

struct FreeTimeDiagnostics {
    double scalar_violation_inf{0.0};
    double variable_violation_inf{0.0};
    double cone_violation_inf{0.0};
    double linearised_dynamics_defect_inf{0.0};
    double virtual_control_inf{0.0};

    [[nodiscard]] double maximum_violation() const noexcept {
        return std::max({scalar_violation_inf, variable_violation_inf, cone_violation_inf});
    }
};

inline void require_positive(double value, const char* message) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(message);
    }
}

inline void require_nonnegative(double value, const char* message) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(message);
    }
}

}  // namespace spacepdhcg::transcription::free_time
