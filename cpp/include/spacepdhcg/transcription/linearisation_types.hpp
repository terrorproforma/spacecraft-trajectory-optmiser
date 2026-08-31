#pragma once

#include <array>
#include <cstddef>

namespace spacepdhcg::transcription {

template <std::size_t StateDimension, std::size_t ControlDimension>
struct DiscreteAffineLinearisation {
    std::array<double, StateDimension * StateDimension> state{};
    std::array<double, StateDimension * ControlDimension> control{};
    std::array<double, StateDimension> offset{};
};

}  // namespace spacepdhcg::transcription
