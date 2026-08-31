#pragma once

#include <cstdint>

namespace spacepdhcg::transcription {

/// Discrete dynamics and sensitivity strategy.
///
/// `rk4_finite_difference` is retained as a source-compatible alias for the production RK4
/// mode introduced before analytic sensitivities were available. It now selects variational
/// sensitivities. Use `rk4_finite_difference_reference` only for correctness comparisons and
/// boundary-domain diagnostics; it is intentionally not the production hot path.
enum class DiscretisationMethod : std::uint8_t {
    forward_euler = 0,
    rk4_variational = 1,
    rk4_finite_difference = rk4_variational,
    rk4_finite_difference_reference = 2,
};

[[nodiscard]] constexpr bool uses_rk4(
    const DiscretisationMethod method
) noexcept {
    return method != DiscretisationMethod::forward_euler;
}

[[nodiscard]] constexpr bool uses_variational_sensitivities(
    const DiscretisationMethod method
) noexcept {
    return method == DiscretisationMethod::rk4_variational;
}

}  // namespace spacepdhcg::transcription
