#pragma once

#include <cstdint>

namespace spacepdhcg::transcription {

enum class DiscretisationMethod : std::uint8_t {
    forward_euler,
    rk4_finite_difference,
    rk4_variational,
};

[[nodiscard]] constexpr bool uses_rk4(
    const DiscretisationMethod method
) noexcept {
    return method == DiscretisationMethod::rk4_finite_difference
           || method == DiscretisationMethod::rk4_variational;
}

}  // namespace spacepdhcg::transcription
