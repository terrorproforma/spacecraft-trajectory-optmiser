#pragma once

#include "spacepdhcg/persistent_cqp.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace spacepdhcg::core {

namespace detail {

inline void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}

inline bool finite(double value) noexcept { return std::isfinite(value); }
inline bool not_nan(double value) noexcept { return !std::isnan(value); }

class Fnv1a64 {
  public:
    void byte(std::uint8_t value) noexcept {
        value_ ^= value;
        value_ *= 1099511628211ULL;
    }

    template <typename UInt>
    void unsigned_scalar(UInt value) noexcept {
        static_assert(std::is_unsigned_v<UInt>);
        for (std::size_t byte_index = 0; byte_index < sizeof(UInt); ++byte_index) {
            byte(static_cast<std::uint8_t>((value >> (8U * byte_index)) & 0xffU));
        }
    }

    void index(Index value) noexcept {
        unsigned_scalar(static_cast<std::uint32_t>(value));
    }

    void floating(double value) noexcept {
        unsigned_scalar(std::bit_cast<std::uint64_t>(value));
    }

    [[nodiscard]] std::uint64_t value() const noexcept { return value_; }

  private:
    std::uint64_t value_{1469598103934665603ULL};
};

inline std::size_t cone_slot_count(const ConeBlockDescriptor& cone) {
    require(cone.start >= 0, "cone start must be non-negative");
    require(cone.vector_dimension > 0, "cone vector dimension must be positive");
    switch (cone.kind) {
        case ConeKind::second_order:
        case ConeKind::rotated_second_order:
            return static_cast<std::size_t>(cone.vector_dimension) + 2U;
        case ConeKind::exponential:
            require(
                cone.vector_dimension == 1,
                "exponential cones require vector_dimension == 1"
            );
            return 3U;
        case ConeKind::power:
            require(cone.vector_dimension == 1, "power cones require vector_dimension == 1");
            require(
                finite(cone.power_alpha) && cone.power_alpha > 0.0 && cone.power_alpha < 1.0,
                "power cone alpha must lie strictly between zero and one"
            );
            return 3U;
        case ConeKind::positive_semidefinite: {
            const auto order = static_cast<std::size_t>(cone.vector_dimension);
            return order * (order + 1U) / 2U;
        }
    }
    throw std::invalid_argument("unknown cone kind");
}

}  // namespace detail

/// Owning compressed-sparse-column topology used by CPU references and CUDA uploads.
struct CscPattern {
    Index rows{0};
    Index columns{0};
    std::vector<Index> offsets{};
    std::vector<Index> indices{};

    void validate() const {
        detail::require(rows >= 0 && columns >= 0, "CSC dimensions must be non-negative");
        detail::require(
            offsets.size() == static_cast<std::size_t>(columns) + 1U,
            "CSC offsets must contain columns + 1 entries"
        );
        detail::require(!offsets.empty() && offsets.front() == 0, "CSC offsets must start at zero");
        detail::require(
            offsets.back() == static_cast<Index>(indices.size()),
            "CSC final offset must equal the number of stored entries"
        );
        for (std::size_t index = 1; index < offsets.size(); ++index) {
            detail::require(offsets[index] >= offsets[index - 1], "CSC offsets must be ordered");
        }
        for (Index column = 0; column < columns; ++column) {
            const auto begin = static_cast<std::size_t>(offsets[static_cast<std::size_t>(column)]);
            const auto end = static_cast<std::size_t>(offsets[static_cast<std::size_t>(column) + 1U]);
            Index previous{-1};
            for (std::size_t position = begin; position < end; ++position) {
                const auto row = indices[position];
                detail::require(row >= 0 && row < rows, "CSC row index is outside the matrix");
                detail::require(row > previous, "CSC row indices must be strictly increasing");
                previous = row;
            }
        }
    }

    [[nodiscard]] std::size_t nonzeros() const noexcept { return indices.size(); }

    [[nodiscard]] SparsePatternView view() const noexcept {
        return SparsePatternView{
            SparseFormat::csc,
            rows,
            columns,
            HostConstSpan<Index>{offsets.data(), offsets.size()},
            HostConstSpan<Index>{indices.data(), indices.size()},
        };
    }
};

/// Immutable, owning topology for one conic quadratic programme.
struct FixedStructure {
    CscPattern quadratic{};
    CscPattern scalar_constraint{};
    std::optional<CscPattern> affine_cone{};
    std::vector<ConeBlockDescriptor> affine_cones{};
    std::vector<ConeBlockDescriptor> variable_cones{};

    void validate() const {
        quadratic.validate();
        scalar_constraint.validate();
        detail::require(quadratic.rows == quadratic.columns, "quadratic matrix must be square");
        detail::require(
            scalar_constraint.columns == quadratic.columns,
            "scalar constraint matrix has the wrong column count"
        );
        if (affine_cone.has_value()) {
            affine_cone->validate();
            detail::require(
                affine_cone->columns == quadratic.columns,
                "affine cone matrix has the wrong column count"
            );
            validate_cones(affine_cones, affine_cone->rows, true);
        } else {
            detail::require(affine_cones.empty(), "affine cones require an affine matrix");
        }
        validate_cones(variable_cones, quadratic.columns, false);
    }

    [[nodiscard]] Index variables() const noexcept { return quadratic.columns; }
    [[nodiscard]] Index scalar_rows() const noexcept { return scalar_constraint.rows; }
    [[nodiscard]] Index affine_rows() const noexcept {
        return affine_cone.has_value() ? affine_cone->rows : 0;
    }
    [[nodiscard]] Index duals() const noexcept { return scalar_rows() + affine_rows(); }

    [[nodiscard]] StructureDescriptor descriptor() const noexcept {
        return StructureDescriptor{
            variables(),
            quadratic.view(),
            scalar_constraint.view(),
            affine_cone.has_value() ? affine_cone->view() : SparsePatternView{},
            HostConstSpan<ConeBlockDescriptor>{affine_cones.data(), affine_cones.size()},
            HostConstSpan<ConeBlockDescriptor>{variable_cones.data(), variable_cones.size()},
        };
    }

    [[nodiscard]] std::uint64_t fingerprint() const noexcept {
        detail::Fnv1a64 hash;
        hash_pattern(hash, quadratic);
        hash_pattern(hash, scalar_constraint);
        hash.byte(affine_cone.has_value() ? 1U : 0U);
        if (affine_cone.has_value()) {
            hash_pattern(hash, *affine_cone);
        }
        hash_cones(hash, affine_cones);
        hash_cones(hash, variable_cones);
        return hash.value();
    }

  private:
    static void validate_cones(
        const std::vector<ConeBlockDescriptor>& cones,
        Index ambient_dimension,
        bool require_cover
    ) {
        std::size_t previous_stop{0};
        for (std::size_t index = 0; index < cones.size(); ++index) {
            const auto& cone = cones[index];
            const auto start = static_cast<std::size_t>(cone.start);
            const auto stop = start + detail::cone_slot_count(cone);
            if (index > 0U) {
                detail::require(start >= previous_stop, "cone blocks overlap or are unsorted");
            }
            if (require_cover) {
                detail::require(start == previous_stop, "affine cone blocks must cover every row");
            }
            detail::require(
                stop <= static_cast<std::size_t>(ambient_dimension),
                "cone block exceeds its ambient dimension"
            );
            previous_stop = stop;
        }
        if (require_cover) {
            detail::require(
                previous_stop == static_cast<std::size_t>(ambient_dimension),
                "affine cone blocks must cover every affine row"
            );
        }
    }

    static void hash_pattern(detail::Fnv1a64& hash, const CscPattern& pattern) noexcept {
        hash.index(pattern.rows);
        hash.index(pattern.columns);
        hash.unsigned_scalar(static_cast<std::uint64_t>(pattern.offsets.size()));
        for (const auto value : pattern.offsets) {
            hash.index(value);
        }
        hash.unsigned_scalar(static_cast<std::uint64_t>(pattern.indices.size()));
        for (const auto value : pattern.indices) {
            hash.index(value);
        }
    }

    static void hash_cones(
        detail::Fnv1a64& hash,
        const std::vector<ConeBlockDescriptor>& cones
    ) noexcept {
        hash.unsigned_scalar(static_cast<std::uint64_t>(cones.size()));
        for (const auto& cone : cones) {
            hash.byte(static_cast<std::uint8_t>(cone.kind));
            hash.index(cone.start);
            hash.index(cone.vector_dimension);
            hash.floating(cone.power_alpha);
        }
    }
};

/// Owning numerical buffers whose sizes are fixed by a FixedStructure.
struct NumericValues {
    std::vector<double> quadratic{};
    std::vector<double> scalar_constraint{};
    std::vector<double> affine_cone{};
    std::vector<double> linear_objective{};
    std::vector<double> scalar_lower{};
    std::vector<double> scalar_upper{};
    std::vector<double> affine_offset{};
    std::vector<double> variable_lower{};
    std::vector<double> variable_upper{};

    void validate(const FixedStructure& structure) const {
        require_size(quadratic, structure.quadratic.nonzeros(), "quadratic values");
        require_size(
            scalar_constraint,
            structure.scalar_constraint.nonzeros(),
            "scalar constraint values"
        );
        require_size(
            affine_cone,
            structure.affine_cone.has_value() ? structure.affine_cone->nonzeros() : 0U,
            "affine cone values"
        );
        require_size(
            linear_objective,
            static_cast<std::size_t>(structure.variables()),
            "linear objective"
        );
        require_size(scalar_lower, static_cast<std::size_t>(structure.scalar_rows()), "scalar lower");
        require_size(scalar_upper, static_cast<std::size_t>(structure.scalar_rows()), "scalar upper");
        require_size(
            affine_offset,
            static_cast<std::size_t>(structure.affine_rows()),
            "affine offset"
        );
        require_size(
            variable_lower,
            static_cast<std::size_t>(structure.variables()),
            "variable lower"
        );
        require_size(
            variable_upper,
            static_cast<std::size_t>(structure.variables()),
            "variable upper"
        );

        require_finite(quadratic, "quadratic values must be finite");
        require_finite(scalar_constraint, "scalar constraint values must be finite");
        require_finite(affine_cone, "affine cone values must be finite");
        require_finite(linear_objective, "linear objective must be finite");
        require_finite(affine_offset, "affine offsets must be finite");
        require_not_nan(scalar_lower, "scalar lower bounds may be infinite but not NaN");
        require_not_nan(scalar_upper, "scalar upper bounds may be infinite but not NaN");
        require_not_nan(variable_lower, "variable lower bounds may be infinite but not NaN");
        require_not_nan(variable_upper, "variable upper bounds may be infinite but not NaN");
        require_ordered(scalar_lower, scalar_upper, "scalar lower bound exceeds upper bound");
        require_ordered(variable_lower, variable_upper, "variable lower bound exceeds upper bound");
    }

  private:
    static void require_size(
        const std::vector<double>& values,
        std::size_t expected,
        std::string_view name
    ) {
        if (values.size() != expected) {
            throw std::invalid_argument(
                std::string(name) + " has size " + std::to_string(values.size())
                + "; expected " + std::to_string(expected)
            );
        }
    }

    static void require_finite(const std::vector<double>& values, std::string_view message) {
        detail::require(
            std::all_of(values.begin(), values.end(), detail::finite),
            message
        );
    }

    static void require_not_nan(const std::vector<double>& values, std::string_view message) {
        detail::require(
            std::all_of(values.begin(), values.end(), detail::not_nan),
            message
        );
    }

    static void require_ordered(
        const std::vector<double>& lower,
        const std::vector<double>& upper,
        std::string_view message
    ) {
        for (std::size_t index = 0; index < lower.size(); ++index) {
            detail::require(lower[index] <= upper[index], message);
        }
    }
};

/// Host-owned fixed CQP used to validate topology and stage uploads to a native workspace.
class FixedCQP {
  public:
    FixedCQP(FixedStructure structure, NumericValues values)
        : structure_(std::move(structure)), values_(std::move(values)) {
        structure_.validate();
        values_.validate(structure_);
    }

    [[nodiscard]] const FixedStructure& structure() const noexcept { return structure_; }
    [[nodiscard]] const NumericValues& values() const noexcept { return values_; }
    [[nodiscard]] std::uint64_t topology_fingerprint() const noexcept {
        return structure_.fingerprint();
    }
    [[nodiscard]] std::uint64_t update_count() const noexcept { return update_count_; }

    void update_values(NumericValues values) {
        values.validate(structure_);
        values_ = std::move(values);
        ++update_count_;
    }

  private:
    FixedStructure structure_{};
    NumericValues values_{};
    std::uint64_t update_count_{0};
};

}  // namespace spacepdhcg::core
