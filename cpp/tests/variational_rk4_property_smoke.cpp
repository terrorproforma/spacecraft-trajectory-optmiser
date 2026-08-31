#include "spacepdhcg/dynamics/low_thrust_two_body.hpp"
#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"
#include "spacepdhcg/transcription/discrete_flow_linearisation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

using spacepdhcg::transcription::DiscreteAffineLinearisation;
using spacepdhcg::transcription::DiscretisationMethod;
using spacepdhcg::transcription::linearise_discrete_flow;

class DeterministicGenerator {
  public:
    explicit DeterministicGenerator(std::uint64_t seed) : state_(seed) {}

    [[nodiscard]] double uniform(const double lower, const double upper) {
        state_ = state_ * 6'364'136'223'846'793'005ULL + 1'442'695'040'888'963'407ULL;
        const auto fraction = static_cast<double>(state_ >> 11U)
                              / static_cast<double>(std::uint64_t{1} << 53U);
        return lower + (upper - lower) * fraction;
    }

  private:
    std::uint64_t state_;
};

template <std::size_t StateDimension, std::size_t ControlDimension>
[[nodiscard]] double maximum_matrix_difference(
    const DiscreteAffineLinearisation<StateDimension, ControlDimension>& left,
    const DiscreteAffineLinearisation<StateDimension, ControlDimension>& right
) {
    double maximum{0.0};
    for (std::size_t index = 0; index < left.state.size(); ++index) {
        maximum = std::max(maximum, std::abs(left.state[index] - right.state[index]));
    }
    for (std::size_t index = 0; index < left.control.size(); ++index) {
        maximum = std::max(maximum, std::abs(left.control[index] - right.control[index]));
    }
    return maximum;
}

template <std::size_t StateDimension, std::size_t ControlDimension>
[[nodiscard]] double maximum_affine_reference_error(
    const DiscreteAffineLinearisation<StateDimension, ControlDimension>& linearisation,
    const std::array<double, StateDimension>& state,
    const std::array<double, ControlDimension>& control,
    const std::array<double, StateDimension>& reference
) {
    double maximum{0.0};
    for (std::size_t row = 0; row < StateDimension; ++row) {
        auto value = linearisation.offset[row];
        for (std::size_t column = 0; column < StateDimension; ++column) {
            value += linearisation.state[row * StateDimension + column] * state[column];
        }
        for (std::size_t column = 0; column < ControlDimension; ++column) {
            value += linearisation.control[row * ControlDimension + column] * control[column];
        }
        maximum = std::max(maximum, std::abs(value - reference[row]));
    }
    return maximum;
}

void test_powered_descent_3dof() {
    using namespace spacepdhcg::dynamics;
    const PoweredDescent3DofModel model{};
    DeterministicGenerator random{0xC0FFEEULL};
    for (std::size_t trial = 0; trial < 64U; ++trial) {
        const PoweredDescentState state{
            random.uniform(-50.0, 50.0),
            random.uniform(-50.0, 50.0),
            random.uniform(20.0, 300.0),
            random.uniform(-5.0, 5.0),
            random.uniform(-5.0, 5.0),
            random.uniform(-25.0, -0.1),
            random.uniform(1'200.0, 3'000.0),
        };
        const PoweredDescentControl control{
            random.uniform(-2'000.0, 2'000.0),
            random.uniform(-2'000.0, 2'000.0),
            random.uniform(2'000.0, 12'000.0),
            random.uniform(2'000.0, 12'500.0),
        };
        const auto step = random.uniform(0.01, 1.0);
        const auto analytic = linearise_discrete_flow<7U, 4U>(
            model,
            state,
            control,
            step,
            DiscretisationMethod::rk4_variational
        );
        const auto reference = linearise_discrete_flow<7U, 4U>(
            model,
            state,
            control,
            step,
            DiscretisationMethod::rk4_finite_difference_reference,
            2.0e-6
        );
        if (maximum_matrix_difference(analytic, reference) > 8.0e-7) {
            throw std::runtime_error(
                "3-DoF variational RK4 differs from finite-difference reference"
            );
        }
        if (maximum_affine_reference_error(
                analytic,
                state,
                control,
                model.rk4_step(state, control, step)
            ) > 2.0e-11) {
            throw std::runtime_error(
                "3-DoF variational affine model misses its reference step"
            );
        }
    }
}

void test_low_thrust_two_body() {
    using namespace spacepdhcg::dynamics;
    const LowThrustTwoBodyModel model{};
    DeterministicGenerator random{0x5EED1234ULL};
    for (std::size_t trial = 0; trial < 64U; ++trial) {
        const auto radius = random.uniform(6'800.0, 20'000.0);
        const auto angle = random.uniform(-3.0, 3.0);
        const LowThrustState state{
            radius * std::cos(angle),
            radius * std::sin(angle),
            random.uniform(-1'000.0, 1'000.0),
            random.uniform(-2.0, 2.0),
            random.uniform(3.0, 9.0),
            random.uniform(-1.0, 1.0),
            random.uniform(250.0, 900.0),
        };
        const LowThrustControl control{
            random.uniform(-0.4, 0.4),
            random.uniform(-0.4, 0.4),
            random.uniform(-0.4, 0.4),
            random.uniform(0.05, 0.9),
        };
        const auto step = random.uniform(0.05, 15.0);
        const auto analytic = linearise_discrete_flow<7U, 4U>(
            model,
            state,
            control,
            step,
            DiscretisationMethod::rk4_variational
        );
        const auto reference = linearise_discrete_flow<7U, 4U>(
            model,
            state,
            control,
            step,
            DiscretisationMethod::rk4_finite_difference_reference,
            2.0e-6
        );
        if (maximum_matrix_difference(analytic, reference) > 3.0e-6) {
            throw std::runtime_error(
                "low-thrust variational RK4 differs from finite-difference reference"
            );
        }
        if (maximum_affine_reference_error(
                analytic,
                state,
                control,
                model.rk4_step(state, control, step)
            ) > 3.0e-10) {
            throw std::runtime_error(
                "low-thrust variational affine model misses its reference step"
            );
        }
    }

    // The analytic path remains valid exactly at the non-negative sigma boundary. The retained
    // finite-difference oracle must fall back to its valid one-sided column and agree.
    const LowThrustState boundary_state{
        7'000.0,
        0.0,
        0.0,
        0.0,
        7.5,
        0.0,
        500.0,
    };
    const LowThrustControl boundary_control{0.0, 0.0, 0.0, 0.0};
    const auto analytic = linearise_discrete_flow<7U, 4U>(
        model,
        boundary_state,
        boundary_control,
        1.0,
        DiscretisationMethod::rk4_variational
    );
    const auto reference = linearise_discrete_flow<7U, 4U>(
        model,
        boundary_state,
        boundary_control,
        1.0,
        DiscretisationMethod::rk4_finite_difference_reference,
        1.0e-6
    );
    if (maximum_matrix_difference(analytic, reference) > 2.0e-6) {
        throw std::runtime_error(
            "low-thrust boundary sensitivity disagrees with one-sided reference"
        );
    }
}

void test_powered_descent_6dof_projection() {
    using namespace spacepdhcg::dynamics;
    const PoweredDescent6DofModel model{};
    DeterministicGenerator random{0x6D0FCAFEULL};
    for (std::size_t trial = 0; trial < 32U; ++trial) {
        std::array<double, 4U> quaternion{
            random.uniform(0.5, 1.5),
            random.uniform(-0.4, 0.4),
            random.uniform(-0.4, 0.4),
            random.uniform(-0.4, 0.4),
        };
        const auto quaternion_norm = std::sqrt(
            quaternion[0U] * quaternion[0U]
            + quaternion[1U] * quaternion[1U]
            + quaternion[2U] * quaternion[2U]
            + quaternion[3U] * quaternion[3U]
        );
        for (auto& value : quaternion) {
            value /= quaternion_norm;
        }
        const PoweredDescent6DofState state{
            random.uniform(-20.0, 20.0),
            random.uniform(-20.0, 20.0),
            random.uniform(30.0, 250.0),
            random.uniform(-3.0, 3.0),
            random.uniform(-3.0, 3.0),
            random.uniform(-15.0, -0.1),
            quaternion[0U],
            quaternion[1U],
            quaternion[2U],
            quaternion[3U],
            random.uniform(-0.15, 0.15),
            random.uniform(-0.15, 0.15),
            random.uniform(-0.15, 0.15),
            random.uniform(1'300.0, 3'000.0),
        };
        const PoweredDescent6DofControl control{
            random.uniform(-1'500.0, 1'500.0),
            random.uniform(-1'500.0, 1'500.0),
            random.uniform(3'000.0, 11'000.0),
            random.uniform(-200.0, 200.0),
            random.uniform(-200.0, 200.0),
            random.uniform(-200.0, 200.0),
            random.uniform(3'000.0, 12'000.0),
        };
        const auto step = random.uniform(0.005, 0.2);
        const auto analytic = linearise_discrete_flow<14U, 7U>(
            model,
            state,
            control,
            step,
            DiscretisationMethod::rk4_variational
        );
        const auto reference = linearise_discrete_flow<14U, 7U>(
            model,
            state,
            control,
            step,
            DiscretisationMethod::rk4_finite_difference_reference,
            3.0e-6
        );
        if (maximum_matrix_difference(analytic, reference) > 2.5e-5) {
            throw std::runtime_error(
                "6-DoF projected variational RK4 differs from finite differences"
            );
        }
        const auto output = model.rk4_step(state, control, step);
        if (maximum_affine_reference_error(
                analytic,
                state,
                control,
                output
            ) > 2.0e-9) {
            throw std::runtime_error(
                "6-DoF projected affine model misses its reference step"
            );
        }
        const std::array<double, 4U> output_quaternion{
            output[6U], output[7U], output[8U], output[9U]
        };
        for (std::size_t column = 0; column < 14U; ++column) {
            double radial_component{0.0};
            for (std::size_t component = 0; component < 4U; ++component) {
                radial_component += output_quaternion[component]
                                    * analytic.state[
                                        (6U + component) * 14U + column
                                    ];
            }
            if (std::abs(radial_component) > 2.0e-10) {
                throw std::runtime_error(
                    "6-DoF state sensitivity is not tangent to the unit quaternion"
                );
            }
        }
        for (std::size_t column = 0; column < 7U; ++column) {
            double radial_component{0.0};
            for (std::size_t component = 0; component < 4U; ++component) {
                radial_component += output_quaternion[component]
                                    * analytic.control[
                                        (6U + component) * 7U + column
                                    ];
            }
            if (std::abs(radial_component) > 2.0e-10) {
                throw std::runtime_error(
                    "6-DoF control sensitivity is not tangent to the unit quaternion"
                );
            }
        }
    }
}

}  // namespace

int main() {
    test_powered_descent_3dof();
    test_low_thrust_two_body();
    test_powered_descent_6dof_projection();
    return 0;
}
