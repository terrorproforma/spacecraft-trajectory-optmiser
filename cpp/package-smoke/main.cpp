#include "spacepdhcg/c_api.h"
#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"

#include <array>
#include <cmath>

int main() {
    if (spacepdhcg_c_api_version() != 1U) {
        return 1;
    }

    const spacepdhcg::dynamics::PoweredDescent3DofModel model{};
    const spacepdhcg::dynamics::PoweredDescentState state{
        0.0,
        0.0,
        100.0,
        0.0,
        0.0,
        -5.0,
        2'000.0,
    };
    const spacepdhcg::dynamics::PoweredDescentControl control{
        0.0,
        0.0,
        8'000.0,
        8'000.0,
    };
    const auto derivative = model.dynamics(state, control);
    return std::isfinite(derivative[5]) ? 0 : 1;
}
