#include "spacepdhcg/dynamics/low_thrust_two_body.hpp"
#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
#include "spacepdhcg/transcription/discrete_flow_linearisation.hpp"
#include "spacepdhcg/transcription/variational_rk4.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {

template <typename Array>
double maximum_difference(const Array& left, const Array& right) {
    double maximum{0.0};
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

}  // namespace

int main() {
    using spacepdhcg::dynamics::LowThrustControl;
    using spacepdhcg::dynamics::LowThrustState;
    using spacepdhcg::dynamics::LowThrustTwoBodyModel;
    using spacepdhcg::dynamics::PoweredDescent3DofModel;
    using spacepdhcg::dynamics::PoweredDescentControl;
    using spacepdhcg::dynamics::PoweredDescentState;
    using spacepdhcg::transcription::DiscretisationMethod;
    using spacepdhcg::transcription::linearise_discrete_flow;
    using spacepdhcg::transcription::linearise_rk4_variational;

    const PoweredDescent3DofModel descent_model{};
    const PoweredDescentState descent_state{
        20.0,
        -10.0,
        120.0,
        1.0,
        -0.5,
        -7.0,
        2'000.0,
    };
    const PoweredDescentControl descent_control{500.0, -250.0, 8'000.0, 8'020.0};
    const auto descent_fd = linearise_discrete_flow<7U, 4U>(
        descent_model,
        descent_state,
        descent_control,
        0.5,
        DiscretisationMethod::rk4_finite_difference,
        1.0e-6
    );
    const auto descent_variational = linearise_rk4_variational<7U, 4U>(
        descent_model,
        descent_state,
        descent_control,
        0.5
    );
    if (maximum_difference(descent_fd.state, descent_variational.state) > 2.0e-6) {
        return 1;
    }
    if (maximum_difference(descent_fd.control, descent_variational.control) > 2.0e-6) {
        return 2;
    }
    if (maximum_difference(descent_fd.offset, descent_variational.offset) > 2.0e-5) {
        return 3;
    }

    const LowThrustTwoBodyModel low_thrust_model{};
    const LowThrustState low_thrust_state{
        7.0e6,
        1.0e5,
        -2.0e5,
        -100.0,
        7'500.0,
        50.0,
        500.0,
    };
    const LowThrustControl low_thrust_control{0.1, -0.05, 0.02, 0.12};
    const auto low_thrust_fd = linearise_discrete_flow<7U, 4U>(
        low_thrust_model,
        low_thrust_state,
        low_thrust_control,
        2.0,
        DiscretisationMethod::rk4_finite_difference,
        1.0e-6
    );
    const auto low_thrust_variational = linearise_rk4_variational<7U, 4U>(
        low_thrust_model,
        low_thrust_state,
        low_thrust_control,
        2.0
    );
    if (maximum_difference(low_thrust_fd.state, low_thrust_variational.state) > 5.0e-6) {
        return 4;
    }
    if (maximum_difference(low_thrust_fd.control, low_thrust_variational.control) > 5.0e-6) {
        return 5;
    }
    if (maximum_difference(low_thrust_fd.offset, low_thrust_variational.offset) > 5.0e-4) {
        return 6;
    }
    return 0;
}
