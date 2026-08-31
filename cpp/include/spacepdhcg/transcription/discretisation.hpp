#pragma once

#include <cstdint>

namespace spacepdhcg::transcription {

enum class DiscretisationMethod : std::uint8_t {
    forward_euler,
    rk4_finite_difference,
};

}  // namespace spacepdhcg::transcription
