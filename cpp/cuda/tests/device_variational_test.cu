#include "cuda_test_support.hpp"
#include "spacepdhcg/cuda/device_scvx_c_api.h"
#include "spacepdhcg/dynamics/hcw.hpp"
#include "spacepdhcg/dynamics/low_thrust_two_body.hpp"
#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"
#include "spacepdhcg/transcription/discrete_flow_linearisation.hpp"

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

spacepdhcg_cuda_dynamics_config base_config(
    spacepdhcg_cuda_dynamics_model model,
    double step
) {
    return spacepdhcg_cuda_dynamics_config{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        model,
        step,
        1.13e-3,
        {0.0, 0.0, -3.711},
        398'600.4418,
        1.0e-3,
        model == SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST ? 3.4e-5 : 4.6e-4,
        {2'500.0, 2'200.0, 1'800.0},
    };
}

template <std::size_t StateDimension, std::size_t ControlDimension>
struct DeviceResult {
    std::vector<double> propagated;
    std::vector<double> transition;
    std::vector<double> sensitivity;
    std::vector<double> offset;
};

template <std::size_t StateDimension, std::size_t ControlDimension>
DeviceResult<StateDimension, ControlDimension> run_device(
    const spacepdhcg_cuda_dynamics_config& config,
    const std::array<double, StateDimension>& state,
    const std::array<double, ControlDimension>& control
) {
    cudaStream_t native_stream{};
    test::cuda_require(cudaStreamCreateWithFlags(&native_stream, cudaStreamNonBlocking), "stream");
    test::CudaBuffer<double> states(StateDimension, false);
    test::CudaBuffer<double> controls(ControlDimension, false);
    test::CudaBuffer<double> propagated(StateDimension, false);
    test::CudaBuffer<double> transition(StateDimension * StateDimension, false);
    test::CudaBuffer<double> sensitivity(StateDimension * ControlDimension, false);
    test::CudaBuffer<double> offset(StateDimension, false);
    states.upload(vector_of(state), native_stream);
    controls.upload(vector_of(control), native_stream);
    const spacepdhcg_cuda_variational_request request{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        1U,
        test::view(
            states.get(), states.size(), false, SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_ONLY
        ),
        test::view(
            controls.get(), controls.size(), false, SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_ONLY
        ),
        test::view(
            propagated.get(), propagated.size(), false, SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_WRITE
        ),
        test::view(
            transition.get(), transition.size(), false, SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_WRITE
        ),
        test::view(
            sensitivity.get(), sensitivity.size(), false, SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_WRITE
        ),
        test::view(
            offset.get(), offset.size(), false, SPACEPDHCG_SCALAR_FLOAT64,
            SPACEPDHCG_ACCESS_READ_WRITE
        ),
    };
    const spacepdhcg_accelerator_stream stream{
        {SPACEPDHCG_DEVICE_CUDA, 0},
        reinterpret_cast<std::uintptr_t>(native_stream),
    };
    test::status_require(
        spacepdhcg_cuda_variational_rk4_async(&config, &request, stream),
        "device variational RK4"
    );
    DeviceResult<StateDimension, ControlDimension> result{
        propagated.download(native_stream),
        transition.download(native_stream),
        sensitivity.download(native_stream),
        offset.download(native_stream),
    };
    test::cuda_require(cudaStreamDestroy(native_stream), "destroy stream");
    return result;
}

template <std::size_t StateDimension, std::size_t ControlDimension>
double compare(
    const DeviceResult<StateDimension, ControlDimension>& device,
    const transcription::DiscreteAffineLinearisation<StateDimension, ControlDimension>& cpu,
    const std::array<double, StateDimension>& propagated
) {
    double maximum = 0.0;
    for (std::size_t index = 0; index < StateDimension; ++index) {
        maximum = std::max(maximum, std::abs(device.propagated[index] - propagated[index]));
        maximum = std::max(maximum, std::abs(device.offset[index] - cpu.offset[index]));
    }
    for (std::size_t index = 0; index < cpu.state.size(); ++index) {
        maximum = std::max(maximum, std::abs(device.transition[index] - cpu.state[index]));
    }
    for (std::size_t index = 0; index < cpu.control.size(); ++index) {
        maximum = std::max(maximum, std::abs(device.sensitivity[index] - cpu.control[index]));
    }
    return maximum;
}

double test_hcw() {
    const dynamics::HcwState state{100.0, -20.0, 5.0, 0.3, -0.2, 0.05};
    const dynamics::HcwControl control{1.0e-3, -2.0e-3, 5.0e-4};
    const double step = 20.0;
    const auto matrices = dynamics::discretise_hcw(1.13e-3, step);
    transcription::DiscreteAffineLinearisation<6U, 3U> cpu{};
    cpu.state = matrices.state;
    cpu.control = matrices.control;
    const auto propagated = dynamics::hcw_step(matrices, state, control);
    return compare(
        run_device(base_config(SPACEPDHCG_CUDA_DYNAMICS_HCW, step), state, control),
        cpu,
        propagated
    );
}

double test_pd3() {
    const dynamics::PoweredDescentState state{
        10.0, -4.0, 120.0, 1.0, -0.5, -8.0, 2'000.0
    };
    const dynamics::PoweredDescentControl control{500.0, -200.0, 8'000.0, 8'050.0};
    const double step = 0.25;
    const dynamics::PoweredDescent3DofModel model{};
    const auto cpu = transcription::linearise_discrete_flow<7U, 4U>(
        model, state, control, step, transcription::DiscretisationMethod::rk4_variational
    );
    return compare(
        run_device(
            base_config(SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF, step),
            state,
            control
        ),
        cpu,
        model.rk4_step(state, control, step)
    );
}

double test_low_thrust() {
    const dynamics::LowThrustState state{
        7'000.0, 100.0, -50.0, -0.1, 7.5, 0.2, 500.0
    };
    const dynamics::LowThrustControl control{0.2, -0.1, 0.05, 0.25};
    const double step = 5.0;
    const dynamics::LowThrustTwoBodyModel model{};
    const auto cpu = transcription::linearise_discrete_flow<7U, 4U>(
        model, state, control, step, transcription::DiscretisationMethod::rk4_variational
    );
    return compare(
        run_device(
            base_config(SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST, step), state, control
        ),
        cpu,
        model.rk4_step(state, control, step)
    );
}

double test_pd6(double& radial_tangency) {
    const dynamics::PoweredDescent6DofState state{
        5.0, -3.0, 100.0, 0.2, -0.4, -7.0,
        0.9805806756909202, 0.09805806756909202, -0.147087101353638,
        0.09805806756909202, 0.05, -0.04, 0.02, 2'000.0,
    };
    const dynamics::PoweredDescent6DofControl control{
        400.0, -100.0, 8'000.0, 10.0, -20.0, 15.0, 8'020.0
    };
    const double step = 0.05;
    const dynamics::PoweredDescent6DofModel model{};
    const auto cpu = transcription::linearise_discrete_flow<14U, 7U>(
        model, state, control, step, transcription::DiscretisationMethod::rk4_variational
    );
    const auto device = run_device(
        base_config(SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF, step),
        state,
        control
    );
    for (std::size_t column = 0; column < 14U; ++column) {
        double radial = 0.0;
        for (std::size_t row = 0; row < 4U; ++row) {
            radial += device.propagated[6U + row]
                      * device.transition[(6U + row) * 14U + column];
        }
        radial_tangency = std::max(radial_tangency, std::abs(radial));
    }
    for (std::size_t column = 0; column < 7U; ++column) {
        double radial = 0.0;
        for (std::size_t row = 0; row < 4U; ++row) {
            radial += device.propagated[6U + row]
                      * device.sensitivity[(6U + row) * 7U + column];
        }
        radial_tangency = std::max(radial_tangency, std::abs(radial));
    }
    return compare(device, cpu, model.rk4_step(state, control, step));
}

void test_direct_csc_fill() {
    constexpr std::size_t state_dimension = 7U;
    constexpr std::size_t control_dimension = 4U;
    constexpr std::size_t state_entries = state_dimension * state_dimension;
    constexpr std::size_t control_entries = state_dimension * control_dimension;
    constexpr std::size_t row_entries = state_dimension;
    constexpr std::size_t nonzeros =
        state_entries + control_entries + 2U * row_entries;
    const dynamics::PoweredDescentState state{
        10.0, -4.0, 120.0, 1.0, -0.5, -8.0, 2'000.0
    };
    const dynamics::PoweredDescentControl control{500.0, -200.0, 8'000.0, 8'050.0};
    const auto linearisation = run_device(
        base_config(SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF, 0.25),
        state,
        control
    );
    std::vector<int> state_positions(state_entries);
    std::vector<int> control_positions(control_entries);
    std::vector<int> next_positions(row_entries);
    std::vector<int> virtual_positions(row_entries);
    for (std::size_t index = 0; index < state_entries; ++index) {
        state_positions[index] = static_cast<int>(index);
    }
    for (std::size_t index = 0; index < control_entries; ++index) {
        control_positions[index] = static_cast<int>(state_entries + index);
    }
    for (std::size_t index = 0; index < row_entries; ++index) {
        next_positions[index] =
            static_cast<int>(state_entries + control_entries + index);
        virtual_positions[index] =
            static_cast<int>(state_entries + control_entries + row_entries + index);
    }
    cudaStream_t native_stream{};
    test::cuda_require(
        cudaStreamCreateWithFlags(&native_stream, cudaStreamNonBlocking),
        "CSC stream"
    );
    test::CudaBuffer<double> transition(state_entries, false);
    test::CudaBuffer<double> sensitivity(control_entries, false);
    test::CudaBuffer<double> offset(row_entries, false);
    test::CudaBuffer<int> d_state_positions(state_entries, false);
    test::CudaBuffer<int> d_control_positions(control_entries, false);
    test::CudaBuffer<int> d_next_positions(row_entries, false);
    test::CudaBuffer<int> d_virtual_positions(row_entries, false);
    test::CudaBuffer<double> scalar_values(nonzeros, false);
    test::CudaBuffer<double> scalar_lower(10U, false);
    test::CudaBuffer<double> scalar_upper(10U, false);
    transition.upload(linearisation.transition, native_stream);
    sensitivity.upload(linearisation.sensitivity, native_stream);
    offset.upload(linearisation.offset, native_stream);
    d_state_positions.upload(state_positions, native_stream);
    d_control_positions.upload(control_positions, native_stream);
    d_next_positions.upload(next_positions, native_stream);
    d_virtual_positions.upload(virtual_positions, native_stream);
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
    const spacepdhcg_cuda_csc_dynamics_fill request{
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
    };
    const spacepdhcg_accelerator_stream stream{
        {SPACEPDHCG_DEVICE_CUDA, 0},
        reinterpret_cast<std::uintptr_t>(native_stream),
    };
    const auto* stable_pointer = scalar_values.get();
    test::status_require(
        spacepdhcg_cuda_fill_dynamics_csc_async(&request, stream),
        "direct CSC fill"
    );
    test::status_require(
        spacepdhcg_cuda_fill_dynamics_csc_async(&request, stream),
        "repeated direct CSC fill"
    );
    test::require(stable_pointer == scalar_values.get(), "CSC values pointer changed");
    const auto values = scalar_values.download(native_stream);
    const auto lower = scalar_lower.download(native_stream);
    const auto upper = scalar_upper.download(native_stream);
    for (std::size_t index = 0; index < state_entries; ++index) {
        test::require(
            values[index] == -linearisation.transition[index],
            "state coefficient was not written directly"
        );
    }
    for (std::size_t index = 0; index < control_entries; ++index) {
        test::require(
            values[state_entries + index] == -linearisation.sensitivity[index],
            "control coefficient was not written directly"
        );
    }
    for (std::size_t row = 0; row < row_entries; ++row) {
        test::require(
            values[state_entries + control_entries + row] == 1.0,
            "next-state coefficient is not one"
        );
        test::require(
            values[state_entries + control_entries + row_entries + row] == -1.0,
            "virtual-control coefficient is not minus one"
        );
        test::require(
            lower[3U + row] == linearisation.offset[row]
                && upper[3U + row] == linearisation.offset[row],
            "affine dynamics bound was not reconstructed"
        );
    }
    test::cuda_require(cudaStreamDestroy(native_stream), "destroy CSC stream");
}

}  // namespace

int main() {
    const double hcw_error = test_hcw();
    const double pd3_error = test_pd3();
    const double low_thrust_error = test_low_thrust();
    double radial_tangency = 0.0;
    const double pd6_error = test_pd6(radial_tangency);
    test::require(hcw_error < 5.0e-11, "HCW device coefficients differ from CPU truth");
    test::require(pd3_error < 5.0e-11, "3-DoF device variational RK4 differs from CPU");
    test::require(
        low_thrust_error < 5.0e-10,
        "low-thrust device variational RK4 differs from CPU"
    );
    test::require(pd6_error < 2.0e-9, "6-DoF device variational RK4 differs from CPU");
    test::require(radial_tangency < 2.0e-10, "quaternion sensitivities are not tangent");
    test_direct_csc_fill();
    std::printf(
        "{\"case\":\"device_variational\",\"hcw\":%.3e,\"pd3\":%.3e,"
        "\"low_thrust\":%.3e,\"pd6\":%.3e,\"quaternion_radial\":%.3e,"
        "\"production_finite_difference\":false,\"direct_csc_fill\":true,"
        "\"pointer_stable\":true}\n",
        hcw_error,
        pd3_error,
        low_thrust_error,
        pd6_error,
        radial_tangency
    );
    return 0;
}
