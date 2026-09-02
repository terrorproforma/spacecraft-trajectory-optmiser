// CPU/GPU coefficient parity for the free-final-time (time-dilated) variational RK4 kernels
// behind the pd3_fft / pd6_fft topologies.  Truth is the host
// spacepdhcg::transcription::linearise_time_dilated_flow (same RK4 stages, same per-substep
// quaternion normalisation); the device result must agree to roundoff for A, B, S, the affine
// offset and the propagated state, satisfy the quaternion tangent rule on all three blocks, and
// match a central-difference oracle for the sigma column.
#include "cuda_test_support.hpp"
#include "spacepdhcg/cuda/device_scvx_c_api.h"
#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"
#include "spacepdhcg/transcription/time_dilated_flow_linearisation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace test = spacepdhcg::cuda::test;
namespace dynamics = spacepdhcg::dynamics;
namespace transcription = spacepdhcg::transcription;

namespace {

template <std::size_t Size>
std::vector<double> vector_of(const std::array<double, Size>& values) {
    return {values.begin(), values.end()};
}

spacepdhcg_cuda_dynamics_config base_config(spacepdhcg_cuda_dynamics_model model) {
    return spacepdhcg_cuda_dynamics_config{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        model,
        1.0,  // step_seconds is ignored by the time-dilated entry point
        1.13e-3,
        {0.0, 0.0, -3.711},
        398'600.4418,
        1.0e-3,
        4.6e-4,
        {2'500.0, 2'200.0, 1'800.0},
    };
}

template <std::size_t StateDimension, std::size_t ControlDimension>
struct DeviceResult {
    std::vector<double> propagated;
    std::vector<double> transition;
    std::vector<double> sensitivity;
    std::vector<double> sigma;
    std::vector<double> offset;
};

template <std::size_t StateDimension, std::size_t ControlDimension>
DeviceResult<StateDimension, ControlDimension> run_device(
    const spacepdhcg_cuda_dynamics_config& config,
    const std::array<double, StateDimension>& state,
    const std::array<double, ControlDimension>& control,
    const double sigma,
    const double d_tau,
    const std::size_t substeps
) {
    cudaStream_t native_stream{};
    test::cuda_require(cudaStreamCreateWithFlags(&native_stream, cudaStreamNonBlocking), "stream");
    test::CudaBuffer<double> states(StateDimension, false);
    test::CudaBuffer<double> controls(ControlDimension, false);
    test::CudaBuffer<double> propagated(StateDimension, false);
    test::CudaBuffer<double> transition(StateDimension * StateDimension, false);
    test::CudaBuffer<double> sensitivity(StateDimension * ControlDimension, false);
    test::CudaBuffer<double> sigma_sensitivity(StateDimension, false);
    test::CudaBuffer<double> offset(StateDimension, false);
    states.upload(vector_of(state), native_stream);
    controls.upload(vector_of(control), native_stream);
    const auto rw = [](auto& buffer) {
        return test::view(
            buffer.get(), buffer.size(), false, SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_WRITE
        );
    };
    const auto ro = [](auto& buffer) {
        return test::view(
            buffer.get(), buffer.size(), false, SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_ONLY
        );
    };
    const spacepdhcg_cuda_time_dilated_request request{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        1U,
        substeps,
        sigma,
        d_tau,
        ro(states),
        ro(controls),
        rw(propagated),
        rw(transition),
        rw(sensitivity),
        rw(sigma_sensitivity),
        rw(offset),
    };
    const spacepdhcg_accelerator_stream stream{
        {SPACEPDHCG_DEVICE_CUDA, 0},
        reinterpret_cast<std::uintptr_t>(native_stream),
    };
    test::status_require(
        spacepdhcg_cuda_time_dilated_variational_rk4_async(&config, &request, stream),
        "device time-dilated variational RK4"
    );
    DeviceResult<StateDimension, ControlDimension> result{
        propagated.download(native_stream),
        transition.download(native_stream),
        sensitivity.download(native_stream),
        sigma_sensitivity.download(native_stream),
        offset.download(native_stream),
    };
    test::cuda_require(cudaStreamDestroy(native_stream), "destroy stream");
    return result;
}

struct Parity {
    double coefficients{0.0};
    double sigma_column{0.0};
    double sigma_finite_difference{0.0};
    double quaternion_radial{0.0};
    double reconstruction{0.0};
};

template <std::size_t StateDimension, std::size_t ControlDimension, typename Model>
Parity compare(
    const Model& model,
    const spacepdhcg_cuda_dynamics_config& config,
    const std::array<double, StateDimension>& state,
    const std::array<double, ControlDimension>& control,
    const double sigma,
    const double d_tau,
    const std::size_t substeps
) {
    const auto cpu = transcription::linearise_time_dilated_flow<StateDimension, ControlDimension>(
        model, state, control, sigma, d_tau, substeps
    );
    const auto device = run_device(config, state, control, sigma, d_tau, substeps);
    Parity parity{};
    for (std::size_t index = 0; index < StateDimension; ++index) {
        parity.coefficients = std::max(
            parity.coefficients, std::abs(device.propagated[index] - cpu.propagated[index])
        );
        parity.coefficients =
            std::max(parity.coefficients, std::abs(device.offset[index] - cpu.offset[index]));
        parity.sigma_column =
            std::max(parity.sigma_column, std::abs(device.sigma[index] - cpu.sigma[index]));
    }
    for (std::size_t index = 0; index < cpu.state.size(); ++index) {
        parity.coefficients =
            std::max(parity.coefficients, std::abs(device.transition[index] - cpu.state[index]));
    }
    for (std::size_t index = 0; index < cpu.control.size(); ++index) {
        parity.coefficients = std::max(
            parity.coefficients, std::abs(device.sensitivity[index] - cpu.control[index])
        );
    }
    // Affine reconstruction through the device blocks.
    for (std::size_t row = 0; row < StateDimension; ++row) {
        double value = device.offset[row] + device.sigma[row] * sigma;
        for (std::size_t column = 0; column < StateDimension; ++column) {
            value += device.transition[row * StateDimension + column] * state[column];
        }
        for (std::size_t column = 0; column < ControlDimension; ++column) {
            value += device.sensitivity[row * ControlDimension + column] * control[column];
        }
        parity.reconstruction =
            std::max(parity.reconstruction, std::abs(value - device.propagated[row]));
    }
    // Central-difference oracle for the sigma column on the host reference map.
    const double step = 1.0e-5 * sigma;
    const auto plus = transcription::time_dilated_step<StateDimension, ControlDimension>(
        model, state, control, sigma + step, d_tau, substeps
    );
    const auto minus = transcription::time_dilated_step<StateDimension, ControlDimension>(
        model, state, control, sigma - step, d_tau, substeps
    );
    for (std::size_t row = 0; row < StateDimension; ++row) {
        const double oracle = (plus[row] - minus[row]) / (2.0 * step);
        parity.sigma_finite_difference = std::max(
            parity.sigma_finite_difference,
            std::abs(device.sigma[row] - oracle) / std::max(1.0, std::abs(oracle))
        );
    }
    if constexpr (StateDimension == 14U) {
        for (std::size_t column = 0; column < StateDimension; ++column) {
            double radial = 0.0;
            for (std::size_t row = 0; row < 4U; ++row) {
                radial += device.propagated[6U + row]
                          * device.transition[(6U + row) * StateDimension + column];
            }
            parity.quaternion_radial = std::max(parity.quaternion_radial, std::abs(radial));
        }
        for (std::size_t column = 0; column < ControlDimension; ++column) {
            double radial = 0.0;
            for (std::size_t row = 0; row < 4U; ++row) {
                radial += device.propagated[6U + row]
                          * device.sensitivity[(6U + row) * ControlDimension + column];
            }
            parity.quaternion_radial = std::max(parity.quaternion_radial, std::abs(radial));
        }
        double radial = 0.0;
        for (std::size_t row = 0; row < 4U; ++row) {
            radial += device.propagated[6U + row] * device.sigma[6U + row];
        }
        parity.quaternion_radial = std::max(parity.quaternion_radial, std::abs(radial));
    }
    return parity;
}

Parity test_pd3(const std::size_t substeps) {
    const dynamics::PoweredDescentState state{10.0, -4.0, 120.0, 1.0, -0.5, -8.0, 2'000.0};
    const dynamics::PoweredDescentControl control{500.0, -200.0, 8'000.0, 8'050.0};
    const dynamics::PoweredDescent3DofModel model{};
    return compare(
        model, base_config(SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF), state, control, 48.0,
        1.0 / 20.0, substeps
    );
}

Parity test_pd6(const std::size_t substeps) {
    const dynamics::PoweredDescent6DofState state{
        5.0, -3.0, 100.0, 0.2, -0.4, -7.0,
        0.9805806756909202, 0.09805806756909202, -0.147087101353638,
        0.09805806756909202, 0.05, -0.04, 0.02, 2'000.0,
    };
    const dynamics::PoweredDescent6DofControl control{
        400.0, -100.0, 8'000.0, 10.0, -20.0, 15.0, 8'020.0
    };
    const dynamics::PoweredDescent6DofModel model{};
    return compare(
        model, base_config(SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF), state, control, 3.4,
        1.0 / 49.0, substeps
    );
}

void test_sigma_csc_fill() {
    constexpr std::size_t state_dimension = 7U;
    constexpr std::size_t control_dimension = 4U;
    constexpr std::size_t state_entries = state_dimension * state_dimension;
    constexpr std::size_t control_entries = state_dimension * control_dimension;
    constexpr std::size_t nonzeros = state_entries + control_entries + 3U * state_dimension;
    const dynamics::PoweredDescentState state{10.0, -4.0, 120.0, 1.0, -0.5, -8.0, 2'000.0};
    const dynamics::PoweredDescentControl control{500.0, -200.0, 8'000.0, 8'050.0};
    const auto linearisation = run_device(
        base_config(SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF), state, control, 48.0,
        1.0 / 20.0, 2U
    );
    std::vector<int> state_positions(state_entries);
    std::vector<int> control_positions(control_entries);
    std::vector<int> next_positions(state_dimension);
    std::vector<int> virtual_positions(state_dimension);
    std::vector<int> sigma_positions(state_dimension);
    for (std::size_t index = 0; index < state_entries; ++index) {
        state_positions[index] = static_cast<int>(index);
    }
    for (std::size_t index = 0; index < control_entries; ++index) {
        control_positions[index] = static_cast<int>(state_entries + index);
    }
    for (std::size_t index = 0; index < state_dimension; ++index) {
        next_positions[index] = static_cast<int>(state_entries + control_entries + index);
        virtual_positions[index] =
            static_cast<int>(state_entries + control_entries + state_dimension + index);
        sigma_positions[index] =
            static_cast<int>(state_entries + control_entries + 2U * state_dimension + index);
    }
    cudaStream_t native_stream{};
    test::cuda_require(
        cudaStreamCreateWithFlags(&native_stream, cudaStreamNonBlocking), "CSC stream"
    );
    test::CudaBuffer<double> transition(state_entries, false);
    test::CudaBuffer<double> sensitivity(control_entries, false);
    test::CudaBuffer<double> sigma_sensitivity(state_dimension, false);
    test::CudaBuffer<double> offset(state_dimension, false);
    test::CudaBuffer<int> d_state_positions(state_entries, false);
    test::CudaBuffer<int> d_control_positions(control_entries, false);
    test::CudaBuffer<int> d_next_positions(state_dimension, false);
    test::CudaBuffer<int> d_virtual_positions(state_dimension, false);
    test::CudaBuffer<int> d_sigma_positions(state_dimension, false);
    test::CudaBuffer<double> scalar_values(nonzeros, false);
    test::CudaBuffer<double> scalar_lower(10U, false);
    test::CudaBuffer<double> scalar_upper(10U, false);
    transition.upload(linearisation.transition, native_stream);
    sensitivity.upload(linearisation.sensitivity, native_stream);
    sigma_sensitivity.upload(linearisation.sigma, native_stream);
    offset.upload(linearisation.offset, native_stream);
    d_state_positions.upload(state_positions, native_stream);
    d_control_positions.upload(control_positions, native_stream);
    d_next_positions.upload(next_positions, native_stream);
    d_virtual_positions.upload(virtual_positions, native_stream);
    d_sigma_positions.upload(sigma_positions, native_stream);
    scalar_values.upload(std::vector<double>(nonzeros, 0.0), native_stream);
    scalar_lower.upload(std::vector<double>(10U, 0.0), native_stream);
    scalar_upper.upload(std::vector<double>(10U, 0.0), native_stream);
    const auto view64 = [](auto& buffer) {
        return test::view(
            buffer.get(), buffer.size(), false, SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_WRITE
        );
    };
    const auto view32 = [](auto& buffer) {
        return test::view(
            buffer.get(), buffer.size(), false, SPACEPDHCG_SCALAR_INT32,
            SPACEPDHCG_ACCESS_READ_ONLY
        );
    };
    const spacepdhcg_cuda_csc_time_dilated_fill request{
        spacepdhcg_cuda_csc_dynamics_fill{
            SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
            1U,
            state_dimension,
            control_dimension,
            view64(transition),
            view64(sensitivity),
            view64(offset),
            view32(d_state_positions),
            view32(d_control_positions),
            view32(d_next_positions),
            view32(d_virtual_positions),
            view64(scalar_values),
            view64(scalar_lower),
            view64(scalar_upper),
            3U,
        },
        view64(sigma_sensitivity),
        view32(d_sigma_positions),
    };
    const spacepdhcg_accelerator_stream stream{
        {SPACEPDHCG_DEVICE_CUDA, 0},
        reinterpret_cast<std::uintptr_t>(native_stream),
    };
    test::status_require(
        spacepdhcg_cuda_fill_time_dilated_csc_async(&request, stream), "time-dilated CSC fill"
    );
    const auto values = scalar_values.download(native_stream);
    const auto lower = scalar_lower.download(native_stream);
    for (std::size_t index = 0; index < state_entries; ++index) {
        test::require(
            values[index] == -linearisation.transition[index],
            "state coefficient was not written directly"
        );
    }
    for (std::size_t row = 0; row < state_dimension; ++row) {
        test::require(
            values[sigma_positions[row]] == -linearisation.sigma[row],
            "sigma coefficient was not written directly"
        );
        test::require(
            lower[3U + row] == linearisation.offset[row], "affine offset was not reconstructed"
        );
    }
    test::cuda_require(cudaStreamDestroy(native_stream), "destroy CSC stream");
}

}  // namespace

int main() {
    const auto pd3_one = test_pd3(1U);
    const auto pd3_four = test_pd3(4U);
    const auto pd6_one = test_pd6(1U);
    const auto pd6_four = test_pd6(4U);
    for (const auto* parity : {&pd3_one, &pd3_four}) {
        test::require(parity->coefficients < 5.0e-11, "pd3_fft device A/B/z differ from CPU");
        test::require(parity->sigma_column < 5.0e-11, "pd3_fft device S differs from CPU");
        test::require(parity->reconstruction < 1.0e-8, "pd3_fft affine reconstruction fails");
        test::require(
            parity->sigma_finite_difference < 1.0e-6, "pd3_fft S disagrees with finite differences"
        );
    }
    for (const auto* parity : {&pd6_one, &pd6_four}) {
        test::require(parity->coefficients < 2.0e-9, "pd6_fft device A/B/z differ from CPU");
        test::require(parity->sigma_column < 2.0e-9, "pd6_fft device S differs from CPU");
        test::require(parity->reconstruction < 1.0e-8, "pd6_fft affine reconstruction fails");
        test::require(
            parity->sigma_finite_difference < 1.0e-6, "pd6_fft S disagrees with finite differences"
        );
        test::require(
            parity->quaternion_radial < 2.0e-10, "pd6_fft quaternion sensitivities not tangent"
        );
    }
    test_sigma_csc_fill();
    std::printf(
        "{\"case\":\"device_time_dilated\",\"pd3_fft\":{\"coefficients\":%.3e,\"sigma\":%.3e,"
        "\"sigma_fd\":%.3e,\"reconstruction\":%.3e},\"pd6_fft\":{\"coefficients\":%.3e,"
        "\"sigma\":%.3e,\"sigma_fd\":%.3e,\"reconstruction\":%.3e,\"quaternion_radial\":%.3e},"
        "\"substeps\":[1,4],\"sigma_csc_fill\":true}\n",
        std::max(pd3_one.coefficients, pd3_four.coefficients),
        std::max(pd3_one.sigma_column, pd3_four.sigma_column),
        std::max(pd3_one.sigma_finite_difference, pd3_four.sigma_finite_difference),
        std::max(pd3_one.reconstruction, pd3_four.reconstruction),
        std::max(pd6_one.coefficients, pd6_four.coefficients),
        std::max(pd6_one.sigma_column, pd6_four.sigma_column),
        std::max(pd6_one.sigma_finite_difference, pd6_four.sigma_finite_difference),
        std::max(pd6_one.reconstruction, pd6_four.reconstruction),
        std::max(pd6_one.quaternion_radial, pd6_four.quaternion_radial)
    );
    return 0;
}
