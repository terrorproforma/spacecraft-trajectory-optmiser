// Free-final-time (sigma time-dilation) RK4 linearisation: affine reconstruction identity,
// finite-difference oracle for A, B and the sigma column, and the quaternion tangent rule.
#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"
#include "spacepdhcg/dynamics/powered_descent_6dof_variational.hpp"
#include "spacepdhcg/transcription/time_dilated_flow_linearisation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace {

template <typename Array>
double maximum_abs(const Array& values) {
    double maximum{0.0};
    for (const auto value : values) {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

int failures = 0;

void check(const bool condition, const char* label, const double value, const double bound) {
    std::printf("%-56s %.3e (bound %.1e) %s\n", label, value, bound, condition ? "ok" : "FAIL");
    if (!condition) {
        ++failures;
    }
}

template <std::size_t N, std::size_t M, typename Model>
void exercise(
    const char* name,
    const Model& model,
    const std::array<double, N>& state,
    const std::array<double, M>& control,
    const double sigma,
    const double d_tau,
    const std::size_t substeps
) {
    using spacepdhcg::transcription::linearise_time_dilated_flow;
    using spacepdhcg::transcription::time_dilated_step;
    std::printf("== %s (sigma=%.3f, d_tau=%.4f, substeps=%zu)\n", name, sigma, d_tau, substeps);
    const auto lin = linearise_time_dilated_flow<N, M>(model, state, control, sigma, d_tau, substeps);

    // Affine reconstruction: A x + B u + S sigma + z == propagated (to roundoff).
    std::array<double, N> reconstructed{};
    for (std::size_t row = 0; row < N; ++row) {
        double value = lin.offset[row] + lin.sigma[row] * sigma;
        for (std::size_t column = 0; column < N; ++column) {
            value += lin.state[row * N + column] * state[column];
        }
        for (std::size_t column = 0; column < M; ++column) {
            value += lin.control[row * M + column] * control[column];
        }
        reconstructed[row] = value - lin.propagated[row];
    }
    const auto scale = std::max(1.0, maximum_abs(lin.propagated));
    check(maximum_abs(reconstructed) <= 1.0e-9 * scale, "affine reconstruction", maximum_abs(reconstructed), 1.0e-9 * scale);

    const auto reference = time_dilated_step<N, M>(model, state, control, sigma, d_tau, substeps);
    std::array<double, N> reference_gap{};
    for (std::size_t row = 0; row < N; ++row) {
        reference_gap[row] = reference[row] - lin.propagated[row];
    }
    check(maximum_abs(reference_gap) == 0.0, "propagated equals reference step", maximum_abs(reference_gap), 0.0);

    // Central finite differences of the reference map (oracle).
    const auto finite_difference = [](auto&& evaluate, const double delta) {
        return (evaluate(delta) - evaluate(-delta)) / (2.0 * delta);
    };
    double sigma_error{0.0};
    double sigma_scale{0.0};
    {
        const auto delta = 1.0e-6 * std::max(1.0, sigma);
        for (std::size_t row = 0; row < N; ++row) {
            const auto fd = finite_difference(
                [&](const double d) {
                    return time_dilated_step<N, M>(model, state, control, sigma + d, d_tau, substeps)[row];
                },
                delta
            );
            sigma_error = std::max(sigma_error, std::abs(fd - lin.sigma[row]));
            sigma_scale = std::max(sigma_scale, std::abs(lin.sigma[row]));
        }
    }
    check(sigma_error <= 1.0e-6 * std::max(1.0, sigma_scale), "dS/dsigma vs central differences", sigma_error, 1.0e-6 * std::max(1.0, sigma_scale));

    double state_error{0.0};
    double state_scale{0.0};
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
            state_scale = std::max(state_scale, std::abs(lin.state[row * N + column]));
        }
    }
    check(state_error <= 2.0e-6 * std::max(1.0, state_scale), "A vs central differences", state_error, 2.0e-6 * std::max(1.0, state_scale));

    double control_error{0.0};
    double control_scale{0.0};
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
            control_scale = std::max(control_scale, std::abs(lin.control[row * M + column]));
        }
    }
    check(control_error <= 2.0e-6 * std::max(1.0, control_scale), "B vs central differences", control_error, 2.0e-6 * std::max(1.0, control_scale));

    if constexpr (N == 14U) {
        // Quaternion tangent rule: every sensitivity column's quaternion block is orthogonal to
        // the normalised output quaternion (the columns live in T_q S^3).
        std::array<double, 4U> q{};
        for (std::size_t component = 0; component < 4U; ++component) {
            q[component] = lin.propagated[6U + component];
        }
        double tangent_error{0.0};
        for (std::size_t column = 0; column < N; ++column) {
            double dot{0.0};
            for (std::size_t component = 0; component < 4U; ++component) {
                dot += q[component] * lin.state[(6U + component) * N + column];
            }
            tangent_error = std::max(tangent_error, std::abs(dot));
        }
        for (std::size_t column = 0; column < M; ++column) {
            double dot{0.0};
            for (std::size_t component = 0; component < 4U; ++component) {
                dot += q[component] * lin.control[(6U + component) * M + column];
            }
            tangent_error = std::max(tangent_error, std::abs(dot));
        }
        double sigma_dot{0.0};
        for (std::size_t component = 0; component < 4U; ++component) {
            sigma_dot += q[component] * lin.sigma[6U + component];
        }
        tangent_error = std::max(tangent_error, std::abs(sigma_dot));
        check(tangent_error <= 1.0e-12, "quaternion tangent rule (A, B, S columns)", tangent_error, 1.0e-12);
        double norm{0.0};
        for (const auto component : q) {
            norm += component * component;
        }
        check(std::abs(std::sqrt(norm) - 1.0) <= 1.0e-12, "propagated quaternion is unit", std::abs(std::sqrt(norm) - 1.0), 1.0e-12);
    }
}

}  // namespace

int main() {
    using spacepdhcg::dynamics::PoweredDescent3DofModel;
    using spacepdhcg::dynamics::PoweredDescent6DofConfig;
    using spacepdhcg::dynamics::PoweredDescent6DofModel;
    using spacepdhcg::dynamics::PoweredDescentControl;
    using spacepdhcg::dynamics::PoweredDescentState;
    using spacepdhcg::dynamics::PoweredDescent6DofControl;
    using spacepdhcg::dynamics::PoweredDescent6DofState;

    const PoweredDescent3DofModel pd3{};
    const PoweredDescentState pd3_state{20.0, -10.0, 120.0, 1.0, -0.5, -7.0, 2'000.0};
    const PoweredDescentControl pd3_control{500.0, -250.0, 8'000.0, 8'020.0};
    exercise<7U, 4U>("pd3_fft", pd3, pd3_state, pd3_control, 45.0, 1.0 / 20.0, 1U);
    exercise<7U, 4U>("pd3_fft substeps", pd3, pd3_state, pd3_control, 45.0, 1.0 / 20.0, 4U);

    // Szmuk 2018 non-dimensional configuration in the native (z-up, body-z thrust) frame.
    PoweredDescent6DofConfig szmuk{};
    szmuk.gravity = {0.0, 0.0, -1.0};
    szmuk.principal_inertia = {1.0e-2, 1.0e-2, 1.0e-2};
    szmuk.mass_flow_coefficient = 1.0e-2;
    szmuk.minimum_mass = 1.0;
    szmuk.maximum_thrust = 5.0;
    szmuk.maximum_torque = 1.0;
    szmuk.maximum_angular_rate = 1.0471975511965976;
    const PoweredDescent6DofModel pd6{szmuk};
    PoweredDescent6DofState pd6_state{
        3.0, 0.2, 3.5, -3.0, 0.1, -0.4,
        0.98, 0.05, -0.15, 0.1, 0.05, -0.2, 0.1, 1.8,
    };
    {
        double norm{0.0};
        for (std::size_t component = 6U; component < 10U; ++component) {
            norm += pd6_state[component] * pd6_state[component];
        }
        norm = std::sqrt(norm);
        for (std::size_t component = 6U; component < 10U; ++component) {
            pd6_state[component] /= norm;
        }
    }
    const PoweredDescent6DofControl pd6_control{0.3, -0.2, 2.5, 3.0e-3, -2.0e-3, 1.0e-3, 2.53};
    exercise<14U, 7U>("pd6_fft", pd6, pd6_state, pd6_control, 3.4, 1.0 / 49.0, 1U);
    exercise<14U, 7U>("pd6_fft substeps", pd6, pd6_state, pd6_control, 3.4, 1.0 / 49.0, 4U);

    std::printf("%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
