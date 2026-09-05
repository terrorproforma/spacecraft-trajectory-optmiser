// Regression: the time-dilated linearisation must project the 6-DoF quaternion rows in every
// translation unit, not only in those that happen to include
// `spacepdhcg/dynamics/powered_descent_6dof_variational.hpp` first.
//
// `device_time_dilated_test` (cpp/cuda/tests) included only the model and the linearisation
// header; the `requires`-expression that detects `project_rk4_variational` then failed by ADL and
// the host reference was linearised *without* the normalisation Jacobian.  Against the
// (projected) device kernel that showed up as a 0.98 gap in the quaternion row of the affine
// offset - the raw quaternion component itself - while the kernel agreed with the
// finite-difference oracle.  This test deliberately includes nothing but the model and the
// linearisation header and asserts the properties the projected linearisation must have.
#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"
#include "spacepdhcg/transcription/time_dilated_flow_linearisation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace {

using spacepdhcg::dynamics::PoweredDescent6DofControl;
using spacepdhcg::dynamics::PoweredDescent6DofModel;
using spacepdhcg::dynamics::PoweredDescent6DofState;
using spacepdhcg::transcription::linearise_time_dilated_flow;
using spacepdhcg::transcription::time_dilated_step;

constexpr std::size_t N = 14U;
constexpr std::size_t M = 7U;

// The hook must be visible through the linearisation header alone.
static_assert(
    spacepdhcg::transcription::time_dilated_detail::
        has_quaternion_projection<N, M, PoweredDescent6DofModel>,
    "the linearisation header must make the 6-DoF quaternion projection visible"
);

int failures = 0;

void check(const bool condition, const char* label, const double value, const double bound) {
    std::printf("%-60s %.3e (bound %.1e) %s\n", label, value, bound, condition ? "ok" : "FAIL");
    if (!condition) {
        ++failures;
    }
}

void exercise(const std::size_t substeps) {
    // Same fixture as the CUDA parity test (deliberately non-unit input quaternion, |q| = 1.0012).
    const PoweredDescent6DofModel model{};
    const PoweredDescent6DofState state{
        5.0, -3.0, 100.0, 0.2, -0.4, -7.0,
        0.9805806756909202, 0.09805806756909202, -0.147087101353638,
        0.09805806756909202, 0.05, -0.04, 0.02, 2'000.0,
    };
    const PoweredDescent6DofControl control{400.0, -100.0, 8'000.0, 10.0, -20.0, 15.0, 8'020.0};
    const double sigma = 3.4;
    const double d_tau = 1.0 / 49.0;
    std::printf("== pd6_fft include-order regression (substeps=%zu)\n", substeps);
    const auto lin = linearise_time_dilated_flow<N, M>(model, state, control, sigma, d_tau, substeps);

    // 1. Tangent rule on A, B and S: the quaternion block of every column is orthogonal to the
    //    normalised output quaternion.  Without the projection this is O(1).
    std::array<double, 4U> q{};
    double norm{0.0};
    for (std::size_t component = 0; component < 4U; ++component) {
        q[component] = lin.propagated[6U + component];
        norm += q[component] * q[component];
    }
    check(std::abs(std::sqrt(norm) - 1.0) <= 1.0e-12, "propagated quaternion is unit", std::abs(std::sqrt(norm) - 1.0), 1.0e-12);
    double tangent{0.0};
    for (std::size_t column = 0; column < N; ++column) {
        double dot{0.0};
        for (std::size_t component = 0; component < 4U; ++component) {
            dot += q[component] * lin.state[(6U + component) * N + column];
        }
        tangent = std::max(tangent, std::abs(dot));
    }
    check(tangent <= 1.0e-12, "A quaternion rows are tangent", tangent, 1.0e-12);
    tangent = 0.0;
    for (std::size_t column = 0; column < M; ++column) {
        double dot{0.0};
        for (std::size_t component = 0; component < 4U; ++component) {
            dot += q[component] * lin.control[(6U + component) * M + column];
        }
        tangent = std::max(tangent, std::abs(dot));
    }
    check(tangent <= 1.0e-12, "B quaternion rows are tangent", tangent, 1.0e-12);
    double sigma_dot{0.0};
    for (std::size_t component = 0; component < 4U; ++component) {
        sigma_dot += q[component] * lin.sigma[6U + component];
    }
    check(std::abs(sigma_dot) <= 1.0e-12, "S quaternion rows are tangent", std::abs(sigma_dot), 1.0e-12);

    // 2. Affine reconstruction through the projected blocks reproduces the propagated state.  The
    //    projected quaternion rows annihilate the radial direction, so their offset carries the
    //    output quaternion itself (z_q ~ q_out ~ 0.98); an unprojected reference puts that 0.98
    //    into A_q x instead and its offset is near zero - which is the gap the CUDA parity test saw.
    double reconstruction{0.0};
    double offset_gap{0.0};
    for (std::size_t row = 0; row < N; ++row) {
        double value = lin.offset[row] + lin.sigma[row] * sigma;
        for (std::size_t column = 0; column < N; ++column) {
            value += lin.state[row * N + column] * state[column];
        }
        for (std::size_t column = 0; column < M; ++column) {
            value += lin.control[row * M + column] * control[column];
        }
        reconstruction = std::max(reconstruction, std::abs(value - lin.propagated[row]));
        if (row >= 6U && row < 10U) {
            offset_gap = std::max(offset_gap, std::abs(lin.offset[row] - lin.propagated[row]));
        }
    }
    check(reconstruction <= 1.0e-9, "affine reconstruction", reconstruction, 1.0e-9);
    check(offset_gap <= 0.05, "offset quaternion rows carry the output quaternion", offset_gap, 0.05);

    // 3. A, B and S differentiate the normalising reference map (central differences).
    const auto finite_difference = [](auto&& evaluate, const double delta) {
        return (evaluate(delta) - evaluate(-delta)) / (2.0 * delta);
    };
    double state_error{0.0};
    for (std::size_t column = 0; column < N; ++column) {
        const auto delta = 1.0e-6 * std::max(1.0, std::abs(state[column]));
        for (std::size_t row = 0; row < N; ++row) {
            const auto fd = finite_difference(
                [&](const double d) {
                    auto perturbed = state;
                    perturbed[column] += d;
                    return time_dilated_step<N, M>(model, perturbed, control, sigma, d_tau, substeps)[row];
                },
                delta
            );
            state_error = std::max(state_error, std::abs(fd - lin.state[row * N + column]));
        }
    }
    check(state_error <= 2.0e-6, "A vs central differences of the normalising step", state_error, 2.0e-6);
    double control_error{0.0};
    for (std::size_t column = 0; column < M; ++column) {
        const auto delta = 1.0e-6 * std::max(1.0, std::abs(control[column]));
        for (std::size_t row = 0; row < N; ++row) {
            const auto fd = finite_difference(
                [&](const double d) {
                    auto perturbed = control;
                    perturbed[column] += d;
                    return time_dilated_step<N, M>(model, state, perturbed, sigma, d_tau, substeps)[row];
                },
                delta
            );
            control_error = std::max(control_error, std::abs(fd - lin.control[row * M + column]));
        }
    }
    check(control_error <= 2.0e-6, "B vs central differences of the normalising step", control_error, 2.0e-6);
    double sigma_error{0.0};
    for (std::size_t row = 0; row < N; ++row) {
        const auto fd = finite_difference(
            [&](const double d) {
                return time_dilated_step<N, M>(model, state, control, sigma + d, d_tau, substeps)[row];
            },
            1.0e-6 * sigma
        );
        sigma_error = std::max(sigma_error, std::abs(fd - lin.sigma[row]));
    }
    check(sigma_error <= 1.0e-6, "S vs central differences of the normalising step", sigma_error, 1.0e-6);
}

}  // namespace

int main() {
    exercise(1U);
    exercise(4U);
    std::printf("%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
