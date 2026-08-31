#pragma once

#include "spacepdhcg/persistent_cqp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::core {

/// Owning compressed-sparse-column topology used by the CPU reference and CUDA bridge.
class CscStructure {
  public:
    CscStructure(
        Index rows,
        Index columns,
        std::vector<Index> offsets,
        std::vector<Index> indices
    )
        : rows_(rows),
          columns_(columns),
          offsets_(std::move(offsets)),
          indices_(std::move(indices)) {
        validate();
    }

    [[nodiscard]] Index rows() const noexcept { return rows_; }
    [[nodiscard]] Index columns() const noexcept { return columns_; }
    [[nodiscard]] std::size_t nonzeros() const noexcept { return indices_.size(); }
    [[nodiscard]] std::span<const Index> offsets() const noexcept { return offsets_; }
    [[nodiscard]] std::span<const Index> indices() const noexcept { return indices_; }

    [[nodiscard]] SparsePatternView view() const noexcept {
        return SparsePatternView{
            SparseFormat::csc,
            rows_,
            columns_,
            HostConstSpan<Index>{offsets_.data(), offsets_.size()},
            HostConstSpan<Index>{indices_.data(), indices_.size()},
        };
    }

    [[nodiscard]] std::uint64_t fingerprint() const noexcept {
        std::uint64_t hash = fnv_offset;
        hash_value(hash, rows_);
        hash_value(hash, columns_);
        for (const auto value : offsets_) {
            hash_value(hash, value);
        }
        for (const auto value : indices_) {
            hash_value(hash, value);
        }
        return hash;
    }

  private:
    static constexpr std::uint64_t fnv_offset = 1469598103934665603ULL;
    static constexpr std::uint64_t fnv_prime = 1099511628211ULL;

    template <typename T>
    static void hash_value(std::uint64_t& hash, const T value) noexcept {
        const auto bytes = reinterpret_cast<const unsigned char*>(&value);
        for (std::size_t index = 0; index < sizeof(T); ++index) {
            hash ^= bytes[index];
            hash *= fnv_prime;
        }
    }

    void validate() const {
        if (rows_ < 0 || columns_ < 0) {
            throw std::invalid_argument("CSC dimensions must be non-negative");
        }
        const auto expected_offsets = static_cast<std::size_t>(columns_) + 1U;
        if (offsets_.size() != expected_offsets) {
            throw std::invalid_argument("CSC offset length must equal columns plus one");
        }
        if (offsets_.empty() || offsets_.front() != 0) {
            throw std::invalid_argument("CSC offsets must begin at zero");
        }
        if (offsets_.back() < 0 || static_cast<std::size_t>(offsets_.back()) != indices_.size()) {
            throw std::invalid_argument("CSC final offset must equal stored index count");
        }
        for (std::size_t index = 1; index < offsets_.size(); ++index) {
            if (offsets_[index] < offsets_[index - 1]) {
                throw std::invalid_argument("CSC offsets must be non-decreasing");
            }
        }
        for (const auto row : indices_) {
            if (row < 0 || row >= rows_) {
                throw std::invalid_argument("CSC row index is outside matrix dimensions");
            }
        }
        for (Index column = 0; column < columns_; ++column) {
            const auto begin = static_cast<std::size_t>(offsets_[static_cast<std::size_t>(column)]);
            const auto end = static_cast<std::size_t>(offsets_[static_cast<std::size_t>(column) + 1U]);
            for (auto position = begin + 1U; position < end; ++position) {
                if (indices_[position] <= indices_[position - 1U]) {
                    throw std::invalid_argument(
                        "CSC row indices must be strictly increasing within each column"
                    );
                }
            }
        }
    }

    Index rows_{0};
    Index columns_{0};
    std::vector<Index> offsets_{};
    std::vector<Index> indices_{};
};

/// Owning cone descriptor with the same slot convention as upstream PDHCG.
struct ConeBlock {
    ConeKind kind{ConeKind::second_order};
    Index start{0};
    Index vector_dimension{0};
    double power_alpha{0.0};

    ConeBlock() = default;

    ConeBlock(ConeKind kind_value, Index start_value, Index vector_dimension_value, double alpha = 0.0)
        : kind(kind_value),
          start(start_value),
          vector_dimension(vector_dimension_value),
          power_alpha(alpha) {
        validate();
    }

    [[nodiscard]] Index slot_count() const {
        switch (kind) {
            case ConeKind::second_order:
            case ConeKind::rotated_second_order:
                return vector_dimension + 2;
            case ConeKind::exponential:
            case ConeKind::power:
                return 3;
            case ConeKind::positive_semidefinite:
                return vector_dimension * (vector_dimension + 1) / 2;
        }
        throw std::logic_error("unhandled cone kind");
    }

    [[nodiscard]] Index stop() const { return start + slot_count(); }

    [[nodiscard]] ConeBlockDescriptor descriptor() const noexcept {
        return ConeBlockDescriptor{kind, start, vector_dimension, power_alpha};
    }

    void validate() const {
        if (start < 0) {
            throw std::invalid_argument("cone start must be non-negative");
        }
        if (vector_dimension <= 0) {
            throw std::invalid_argument("cone vector dimension must be positive");
        }
        if ((kind == ConeKind::exponential || kind == ConeKind::power) && vector_dimension != 1) {
            throw std::invalid_argument("exponential and power cones require vector dimension one");
        }
        if (kind == ConeKind::power) {
            if (!std::isfinite(power_alpha) || power_alpha <= 0.0 || power_alpha >= 1.0) {
                throw std::invalid_argument("power-cone alpha must lie strictly between zero and one");
            }
        } else if (power_alpha != 0.0) {
            throw std::invalid_argument("power alpha is valid only for a power cone");
        }
    }
};

/// Immutable symbolic CQP structure.
class CQPStructure {
  public:
    CQPStructure(
        CscStructure quadratic,
        CscStructure scalar_constraint,
        std::optional<CscStructure> affine_cone = std::nullopt,
        std::vector<ConeBlock> affine_cones = {},
        std::vector<ConeBlock> variable_cones = {}
    )
        : quadratic_(std::move(quadratic)),
          scalar_constraint_(std::move(scalar_constraint)),
          affine_cone_(std::move(affine_cone)),
          affine_cones_(std::move(affine_cones)),
          variable_cones_(std::move(variable_cones)) {
        validate();
    }

    [[nodiscard]] const CscStructure& quadratic() const noexcept { return quadratic_; }
    [[nodiscard]] const CscStructure& scalar_constraint() const noexcept {
        return scalar_constraint_;
    }
    [[nodiscard]] const std::optional<CscStructure>& affine_cone() const noexcept {
        return affine_cone_;
    }
    [[nodiscard]] std::span<const ConeBlock> affine_cones() const noexcept {
        return affine_cones_;
    }
    [[nodiscard]] std::span<const ConeBlock> variable_cones() const noexcept {
        return variable_cones_;
    }

    [[nodiscard]] Index variables() const noexcept { return quadratic_.rows(); }
    [[nodiscard]] Index scalar_rows() const noexcept { return scalar_constraint_.rows(); }
    [[nodiscard]] Index affine_rows() const noexcept {
        return affine_cone_.has_value() ? affine_cone_->rows() : 0;
    }
    [[nodiscard]] Index duals() const noexcept { return scalar_rows() + affine_rows(); }

    [[nodiscard]] std::uint64_t fingerprint() const noexcept {
        std::uint64_t hash = quadratic_.fingerprint();
        hash ^= scalar_constraint_.fingerprint() + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        if (affine_cone_.has_value()) {
            hash ^= affine_cone_->fingerprint() + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        }
        for (const auto& cone : affine_cones_) {
            hash ^= static_cast<std::uint64_t>(cone.start + 1);
            hash ^= static_cast<std::uint64_t>(cone.vector_dimension + 1) << 17U;
            hash ^= static_cast<std::uint64_t>(cone.kind) << 33U;
        }
        for (const auto& cone : variable_cones_) {
            hash ^= static_cast<std::uint64_t>(cone.start + 1) << 7U;
            hash ^= static_cast<std::uint64_t>(cone.vector_dimension + 1) << 23U;
            hash ^= static_cast<std::uint64_t>(cone.kind) << 39U;
        }
        return hash;
    }

    [[nodiscard]] StructureDescriptor descriptor() const noexcept {
        return StructureDescriptor{
            variables(),
            quadratic_.view(),
            scalar_constraint_.view(),
            affine_cone_.has_value() ? affine_cone_->view() : SparsePatternView{},
            HostConstSpan<ConeBlockDescriptor>{nullptr, 0},
            HostConstSpan<ConeBlockDescriptor>{nullptr, 0},
        };
    }

  private:
    static void validate_cones(
        std::span<const ConeBlock> cones,
        Index ambient_dimension,
        bool require_cover
    ) {
        Index previous_stop = 0;
        for (std::size_t index = 0; index < cones.size(); ++index) {
            cones[index].validate();
            if (index > 0U && cones[index].start < previous_stop) {
                throw std::invalid_argument("cone blocks must be ordered and non-overlapping");
            }
            if (require_cover && cones[index].start != previous_stop) {
                throw std::invalid_argument("affine cones must contiguously cover every affine row");
            }
            if (cones[index].stop() > ambient_dimension) {
                throw std::invalid_argument("cone block exceeds its ambient dimension");
            }
            previous_stop = cones[index].stop();
        }
        if (require_cover && previous_stop != ambient_dimension) {
            throw std::invalid_argument("affine cones must cover every affine row");
        }
    }

    void validate() const {
        if (quadratic_.rows() != quadratic_.columns()) {
            throw std::invalid_argument("quadratic topology must be square");
        }
        if (scalar_constraint_.columns() != variables()) {
            throw std::invalid_argument("scalar-constraint columns must equal variable count");
        }
        if (affine_cone_.has_value()) {
            if (affine_cone_->columns() != variables()) {
                throw std::invalid_argument("affine-cone columns must equal variable count");
            }
            validate_cones(affine_cones_, affine_cone_->rows(), true);
        } else if (!affine_cones_.empty()) {
            throw std::invalid_argument("affine cone metadata requires an affine cone matrix");
        }
        validate_cones(variable_cones_, variables(), false);
    }

    CscStructure quadratic_;
    CscStructure scalar_constraint_;
    std::optional<CscStructure> affine_cone_;
    std::vector<ConeBlock> affine_cones_;
    std::vector<ConeBlock> variable_cones_;
};

/// Numerical values associated with one immutable CQP topology.
struct CQPValues {
    std::vector<double> quadratic{};
    std::vector<double> scalar_constraint{};
    std::vector<double> affine_cone{};
    std::vector<double> linear_objective{};
    std::vector<double> scalar_lower{};
    std::vector<double> scalar_upper{};
    std::vector<double> affine_offset{};
    std::vector<double> variable_lower{};
    std::vector<double> variable_upper{};
};

inline void validate_values(const CQPStructure& structure, const CQPValues& values) {
    const auto require_size = [](std::span<const double> vector, std::size_t expected, const char* name) {
        if (vector.size() != expected) {
            throw std::invalid_argument(std::string{name} + " has an incompatible size");
        }
    };
    const auto require_finite = [](std::span<const double> vector, const char* name) {
        if (!std::all_of(vector.begin(), vector.end(), [](const double value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument(std::string{name} + " must contain only finite values");
        }
    };
    const auto require_not_nan = [](std::span<const double> vector, const char* name) {
        if (std::any_of(vector.begin(), vector.end(), [](const double value) {
                return std::isnan(value);
            })) {
            throw std::invalid_argument(std::string{name} + " may contain infinity but not NaN");
        }
    };

    require_size(values.quadratic, structure.quadratic().nonzeros(), "quadratic values");
    require_size(
        values.scalar_constraint,
        structure.scalar_constraint().nonzeros(),
        "scalar-constraint values"
    );
    require_size(
        values.affine_cone,
        structure.affine_cone().has_value() ? structure.affine_cone()->nonzeros() : 0U,
        "affine-cone values"
    );
    require_size(values.linear_objective, static_cast<std::size_t>(structure.variables()), "linear objective");
    require_size(values.scalar_lower, static_cast<std::size_t>(structure.scalar_rows()), "scalar lower bounds");
    require_size(values.scalar_upper, static_cast<std::size_t>(structure.scalar_rows()), "scalar upper bounds");
    require_size(values.affine_offset, static_cast<std::size_t>(structure.affine_rows()), "affine offset");
    require_size(values.variable_lower, static_cast<std::size_t>(structure.variables()), "variable lower bounds");
    require_size(values.variable_upper, static_cast<std::size_t>(structure.variables()), "variable upper bounds");

    require_finite(values.quadratic, "quadratic values");
    require_finite(values.scalar_constraint, "scalar-constraint values");
    require_finite(values.affine_cone, "affine-cone values");
    require_finite(values.linear_objective, "linear objective");
    require_finite(values.affine_offset, "affine offset");
    require_not_nan(values.scalar_lower, "scalar lower bounds");
    require_not_nan(values.scalar_upper, "scalar upper bounds");
    require_not_nan(values.variable_lower, "variable lower bounds");
    require_not_nan(values.variable_upper, "variable upper bounds");

    for (std::size_t index = 0; index < values.scalar_lower.size(); ++index) {
        if (values.scalar_lower[index] > values.scalar_upper[index]) {
            throw std::invalid_argument("scalar lower bound exceeds upper bound");
        }
    }
    for (std::size_t index = 0; index < values.variable_lower.size(); ++index) {
        if (values.variable_lower[index] > values.variable_upper[index]) {
            throw std::invalid_argument("variable lower bound exceeds upper bound");
        }
    }
}

}  // namespace spacepdhcg::core
