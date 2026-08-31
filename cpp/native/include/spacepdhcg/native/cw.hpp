#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <utility>

namespace spacepdhcg::native {

inline constexpr std::size_t cw_state_dimension = 6;
inline constexpr std::size_t cw_control_dimension = 3;

using CwState = std::array<double, cw_state_dimension>;
using CwControl = std::array<double, cw_control_dimension>;
using CwStateMatrix = std::array<double, cw_state_dimension * cw_state_dimension>;
using CwControlMatrix = std::array<double, cw_state_dimension * cw_control_dimension>;

struct CwDiscreteDynamics {
    CwStateMatrix state{};
    CwControlMatrix control{};
};

[[nodiscard]] CwDiscreteDynamics discretise_cw(double mean_motion, double step_seconds);

[[nodiscard]] CwState propagate_cw(
    const CwDiscreteDynamics& dynamics,
    std::span<const double, cw_state_dimension> state,
    std::span<const double, cw_control_dimension> control
);

[[nodiscard]] CwStateMatrix multiply_state_matrices(
    const CwStateMatrix& left,
    const CwStateMatrix& right
);

[[nodiscard]] CwControlMatrix compose_control_matrices(
    const CwStateMatrix& later_state,
    const CwControlMatrix& earlier_control,
    const CwControlMatrix& later_control
);

[[nodiscard]] double maximum_absolute_difference(
    std::span<const double> left,
    std::span<const double> right
);

}  // namespace spacepdhcg::native
