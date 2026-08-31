#pragma once

#include "spacepdhcg/core/fixed_cqp.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace spacepdhcg::core {

struct CheckpointState {
    std::uint64_t topology_fingerprint{0U};
    std::uint64_t numerical_update{0U};
    NumericValues values{};
    std::vector<double> primal{};
    std::vector<double> dual{};
};

namespace checkpoint_detail {

inline constexpr std::array<std::uint8_t, 8U> magic{
    'S',
    'P',
    'D',
    'H',
    'C',
    'G',
    '0',
    '1',
};
inline constexpr std::uint32_t format_version = 1U;

class Writer {
  public:
    void byte(std::uint8_t value) { bytes_.push_back(value); }

    void u32(std::uint32_t value) {
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            byte(static_cast<std::uint8_t>((value >> (8U * index)) & 0xffU));
        }
    }

    void u64(std::uint64_t value) {
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            byte(static_cast<std::uint8_t>((value >> (8U * index)) & 0xffU));
        }
    }

    void floating(double value) { u64(std::bit_cast<std::uint64_t>(value)); }

    void array(const std::vector<double>& values) {
        u64(static_cast<std::uint64_t>(values.size()));
        for (const auto value : values) {
            floating(value);
        }
    }

    [[nodiscard]] std::vector<std::uint8_t> finish() && { return std::move(bytes_); }

  private:
    std::vector<std::uint8_t> bytes_{};
};

class Reader {
  public:
    explicit Reader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

    [[nodiscard]] std::uint8_t byte() {
        require_available(1U);
        return bytes_[position_++];
    }

    [[nodiscard]] std::uint32_t u32() {
        std::uint32_t value{0U};
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            value |= static_cast<std::uint32_t>(byte()) << (8U * index);
        }
        return value;
    }

    [[nodiscard]] std::uint64_t u64() {
        std::uint64_t value{0U};
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            value |= static_cast<std::uint64_t>(byte()) << (8U * index);
        }
        return value;
    }

    [[nodiscard]] double floating() { return std::bit_cast<double>(u64()); }

    [[nodiscard]] std::vector<double> array(std::size_t expected, std::string_view name) {
        const auto count = u64();
        if (count != expected) {
            throw std::invalid_argument(
                std::string(name) + " checkpoint length does not match the CQP structure"
            );
        }
        if (count > remaining() / sizeof(double)) {
            throw std::invalid_argument("checkpoint array length exceeds the remaining payload");
        }
        std::vector<double> result(expected, 0.0);
        for (auto& value : result) {
            value = floating();
        }
        return result;
    }

    [[nodiscard]] std::size_t remaining() const noexcept { return bytes_.size() - position_; }

  private:
    const std::vector<std::uint8_t>& bytes_;
    std::size_t position_{0U};

    void require_available(std::size_t count) const {
        if (count > remaining()) {
            throw std::invalid_argument("checkpoint payload is truncated");
        }
    }
};

inline void require_finite(const std::vector<double>& values, const char* name) {
    for (const auto value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(name) + " checkpoint values must be finite");
        }
    }
}

}  // namespace checkpoint_detail

inline std::vector<std::uint8_t> encode_checkpoint(
    const FixedStructure& structure,
    std::uint64_t numerical_update,
    const NumericValues& values,
    const std::vector<double>& primal = {},
    const std::vector<double>& dual = {}
) {
    structure.validate();
    values.validate(structure);
    if (!primal.empty() && primal.size() != static_cast<std::size_t>(structure.variables())) {
        throw std::invalid_argument("checkpoint primal has the wrong size");
    }
    if (!dual.empty() && dual.size() != static_cast<std::size_t>(structure.duals())) {
        throw std::invalid_argument("checkpoint dual has the wrong size");
    }
    checkpoint_detail::require_finite(primal, "primal");
    checkpoint_detail::require_finite(dual, "dual");

    checkpoint_detail::Writer writer{};
    for (const auto value : checkpoint_detail::magic) {
        writer.byte(value);
    }
    writer.u32(checkpoint_detail::format_version);
    writer.u64(structure.fingerprint());
    writer.u64(numerical_update);
    writer.array(values.quadratic);
    writer.array(values.scalar_constraint);
    writer.array(values.affine_cone);
    writer.array(values.linear_objective);
    writer.array(values.scalar_lower);
    writer.array(values.scalar_upper);
    writer.array(values.affine_offset);
    writer.array(values.variable_lower);
    writer.array(values.variable_upper);
    writer.array(primal);
    writer.array(dual);
    return std::move(writer).finish();
}

inline CheckpointState decode_checkpoint(
    const std::vector<std::uint8_t>& payload,
    const FixedStructure& structure
) {
    structure.validate();
    checkpoint_detail::Reader reader(payload);
    for (const auto expected : checkpoint_detail::magic) {
        if (reader.byte() != expected) {
            throw std::invalid_argument("checkpoint magic does not identify SpacePDHCG data");
        }
    }
    if (reader.u32() != checkpoint_detail::format_version) {
        throw std::invalid_argument("checkpoint format version is unsupported");
    }
    const auto fingerprint = reader.u64();
    if (fingerprint != structure.fingerprint()) {
        throw std::invalid_argument("checkpoint topology fingerprint does not match the CQP");
    }
    CheckpointState result{};
    result.topology_fingerprint = fingerprint;
    result.numerical_update = reader.u64();
    result.values.quadratic = reader.array(structure.quadratic.nonzeros(), "quadratic");
    result.values.scalar_constraint = reader.array(
        structure.scalar_constraint.nonzeros(),
        "scalar constraint"
    );
    result.values.affine_cone = reader.array(
        structure.affine_cone.has_value() ? structure.affine_cone->nonzeros() : 0U,
        "affine cone"
    );
    result.values.linear_objective = reader.array(
        static_cast<std::size_t>(structure.variables()),
        "linear objective"
    );
    result.values.scalar_lower = reader.array(
        static_cast<std::size_t>(structure.scalar_rows()),
        "scalar lower"
    );
    result.values.scalar_upper = reader.array(
        static_cast<std::size_t>(structure.scalar_rows()),
        "scalar upper"
    );
    result.values.affine_offset = reader.array(
        static_cast<std::size_t>(structure.affine_rows()),
        "affine offset"
    );
    result.values.variable_lower = reader.array(
        static_cast<std::size_t>(structure.variables()),
        "variable lower"
    );
    result.values.variable_upper = reader.array(
        static_cast<std::size_t>(structure.variables()),
        "variable upper"
    );

    const auto primal_count = reader.u64();
    if (primal_count != 0U
        && primal_count != static_cast<std::uint64_t>(structure.variables())) {
        throw std::invalid_argument("checkpoint primal length is invalid");
    }
    result.primal.resize(static_cast<std::size_t>(primal_count));
    for (auto& value : result.primal) {
        value = reader.floating();
    }
    const auto dual_count = reader.u64();
    if (dual_count != 0U && dual_count != static_cast<std::uint64_t>(structure.duals())) {
        throw std::invalid_argument("checkpoint dual length is invalid");
    }
    result.dual.resize(static_cast<std::size_t>(dual_count));
    for (auto& value : result.dual) {
        value = reader.floating();
    }
    if (reader.remaining() != 0U) {
        throw std::invalid_argument("checkpoint contains trailing bytes");
    }
    checkpoint_detail::require_finite(result.primal, "primal");
    checkpoint_detail::require_finite(result.dual, "dual");
    result.values.validate(structure);
    return result;
}

}  // namespace spacepdhcg::core
