/*
 * Analytic float64 device dynamics and variational RK4.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "spacepdhcg/cuda/device_scvx_driver_c_api.h"

#include "native_qoco_adapter.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace {

template <int StateDimension, int ControlDimension>
struct Augmented {
    double state[StateDimension];
    double transition[StateDimension * StateDimension];
    double sensitivity[StateDimension * ControlDimension];
};

template <typename T>
__host__ __device__
T* view_pointer(const spacepdhcg_accelerator_buffer_view& view) {
    return reinterpret_cast<T*>(
        reinterpret_cast<std::uintptr_t>(view.data) + view.byte_offset
    );
}

spacepdhcg_cuda_status validate_device_view(
    const spacepdhcg_accelerator_buffer_view& view,
    const size_t elements,
    const spacepdhcg_accelerator_scalar_type scalar_type,
    const int device
) {
    if (view.elements != elements || view.scalar_type != scalar_type
        || view.element_stride != 1 || view.device.type != SPACEPDHCG_DEVICE_CUDA
        || view.device.id != device || (elements > 0U && view.data == nullptr)) {
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    if (elements == 0U) {
        return view.data == nullptr && view.byte_offset == 0U
            ? SPACEPDHCG_CUDA_SUCCESS
            : SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    cudaPointerAttributes attributes{};
    const auto status = cudaPointerGetAttributes(&attributes, view_pointer<void>(view));
    if (status != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        return SPACEPDHCG_CUDA_POINTER_CONTRACT;
    }
    return attributes.type == cudaMemoryTypeDevice && attributes.device == device
        ? SPACEPDHCG_CUDA_SUCCESS
        : SPACEPDHCG_CUDA_POINTER_CONTRACT;
}

__device__ void evaluate_hcw(
    const double* state,
    const double* control,
    const spacepdhcg_cuda_dynamics_config& config,
    double* derivative,
    double* state_jacobian,
    double* control_jacobian
) {
    const double n = config.mean_motion;
    derivative[0] = state[3];
    derivative[1] = state[4];
    derivative[2] = state[5];
    derivative[3] = 3.0 * n * n * state[0] + 2.0 * n * state[4] + control[0];
    derivative[4] = -2.0 * n * state[3] + control[1];
    derivative[5] = -n * n * state[2] + control[2];
    for (int axis = 0; axis < 3; ++axis) {
        state_jacobian[axis * 6 + 3 + axis] = 1.0;
        control_jacobian[(3 + axis) * 3 + axis] = 1.0;
    }
    state_jacobian[3 * 6] = 3.0 * n * n;
    state_jacobian[3 * 6 + 4] = 2.0 * n;
    state_jacobian[4 * 6 + 3] = -2.0 * n;
    state_jacobian[5 * 6 + 2] = -n * n;
}

__device__ void evaluate_pd3(
    const double* state,
    const double* control,
    const spacepdhcg_cuda_dynamics_config& config,
    double* derivative,
    double* state_jacobian,
    double* control_jacobian
) {
    const double mass = state[6];
    for (int axis = 0; axis < 3; ++axis) {
        derivative[axis] = state[3 + axis];
        derivative[3 + axis] = control[axis] / mass + config.gravity[axis];
        state_jacobian[axis * 7 + 3 + axis] = 1.0;
        state_jacobian[(3 + axis) * 7 + 6] =
            -control[axis] / (mass * mass);
        control_jacobian[(3 + axis) * 4 + axis] = 1.0 / mass;
    }
    derivative[6] = -config.mass_flow_coefficient * control[3];
    control_jacobian[6 * 4 + 3] = -config.mass_flow_coefficient;
}

__device__ void evaluate_low_thrust(
    const double* state,
    const double* control,
    const spacepdhcg_cuda_dynamics_config& config,
    double* derivative,
    double* state_jacobian,
    double* control_jacobian
) {
    const double radius_squared =
        state[0] * state[0] + state[1] * state[1] + state[2] * state[2];
    const double radius = sqrt(radius_squared);
    const double inverse_radius_cubed = 1.0 / (radius_squared * radius);
    const double inverse_radius_fifth = inverse_radius_cubed / radius_squared;
    const double mass = state[6];
    for (int axis = 0; axis < 3; ++axis) {
        derivative[axis] = state[3 + axis];
        derivative[3 + axis] =
            -config.gravitational_parameter * state[axis] * inverse_radius_cubed
            + config.thrust_to_acceleration * control[axis] / mass;
        state_jacobian[axis * 7 + 3 + axis] = 1.0;
        for (int column = 0; column < 3; ++column) {
            state_jacobian[(3 + axis) * 7 + column] =
                config.gravitational_parameter
                * (3.0 * state[axis] * state[column] * inverse_radius_fifth
                   - (axis == column ? inverse_radius_cubed : 0.0));
        }
        state_jacobian[(3 + axis) * 7 + 6] =
            -config.thrust_to_acceleration * control[axis] / (mass * mass);
        control_jacobian[(3 + axis) * 4 + axis] =
            config.thrust_to_acceleration / mass;
    }
    derivative[6] = -config.mass_flow_coefficient * control[3];
    control_jacobian[6 * 4 + 3] = -config.mass_flow_coefficient;
}

__device__ void rotation_and_derivatives(
    const double* quaternion,
    double* rotation,
    double derivatives[4][9]
) {
    const double w = quaternion[0];
    const double x = quaternion[1];
    const double y = quaternion[2];
    const double z = quaternion[3];
    const double r[9] = {
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
        2.0 * (x * z + w * y), 2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    };
    const double d[4][9] = {
        {0.0, -2.0 * z, 2.0 * y, 2.0 * z, 0.0, -2.0 * x,
         -2.0 * y, 2.0 * x, 0.0},
        {0.0, 2.0 * y, 2.0 * z, 2.0 * y, -4.0 * x, -2.0 * w,
         2.0 * z, 2.0 * w, -4.0 * x},
        {-4.0 * y, 2.0 * x, 2.0 * w, 2.0 * x, 0.0, 2.0 * z,
         -2.0 * w, 2.0 * z, -4.0 * y},
        {-4.0 * z, -2.0 * w, 2.0 * x, 2.0 * w, -4.0 * z, 2.0 * y,
         2.0 * x, 2.0 * y, 0.0},
    };
    for (int index = 0; index < 9; ++index) {
        rotation[index] = r[index];
        for (int component = 0; component < 4; ++component) {
            derivatives[component][index] = d[component][index];
        }
    }
}

__device__ void evaluate_pd6(
    const double* state,
    const double* control,
    const spacepdhcg_cuda_dynamics_config& config,
    double* derivative,
    double* state_jacobian,
    double* control_jacobian
) {
    double rotation[9]{};
    double rotation_derivatives[4][9]{};
    rotation_and_derivatives(state + 6, rotation, rotation_derivatives);
    double inertial_thrust[3]{};
    for (int axis = 0; axis < 3; ++axis) {
        for (int column = 0; column < 3; ++column) {
            inertial_thrust[axis] += rotation[axis * 3 + column] * control[column];
        }
    }
    const double mass = state[13];
    for (int axis = 0; axis < 3; ++axis) {
        derivative[axis] = state[3 + axis];
        derivative[3 + axis] = inertial_thrust[axis] / mass + config.gravity[axis];
        state_jacobian[axis * 14 + 3 + axis] = 1.0;
        state_jacobian[(3 + axis) * 14 + 13] =
            -inertial_thrust[axis] / (mass * mass);
        for (int body = 0; body < 3; ++body) {
            control_jacobian[(3 + axis) * 7 + body] =
                rotation[axis * 3 + body] / mass;
        }
        for (int q = 0; q < 4; ++q) {
            double value = 0.0;
            for (int body = 0; body < 3; ++body) {
                value += rotation_derivatives[q][axis * 3 + body] * control[body];
            }
            state_jacobian[(3 + axis) * 14 + 6 + q] = value / mass;
        }
    }
    const double w = state[6], x = state[7], y = state[8], z = state[9];
    const double wx = state[10], wy = state[11], wz = state[12];
    derivative[6] = -0.5 * (x * wx + y * wy + z * wz);
    derivative[7] = 0.5 * (w * wx + y * wz - z * wy);
    derivative[8] = 0.5 * (w * wy + z * wx - x * wz);
    derivative[9] = 0.5 * (w * wz + x * wy - y * wx);
    const double dq_dq[16] = {
        0.0, -0.5 * wx, -0.5 * wy, -0.5 * wz,
        0.5 * wx, 0.0, 0.5 * wz, -0.5 * wy,
        0.5 * wy, -0.5 * wz, 0.0, 0.5 * wx,
        0.5 * wz, 0.5 * wy, -0.5 * wx, 0.0,
    };
    const double dq_dw[12] = {
        -0.5 * x, -0.5 * y, -0.5 * z,
        0.5 * w, -0.5 * z, 0.5 * y,
        0.5 * z, 0.5 * w, -0.5 * x,
        -0.5 * y, 0.5 * x, 0.5 * w,
    };
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            state_jacobian[(6 + row) * 14 + 6 + column] =
                dq_dq[row * 4 + column];
        }
        for (int column = 0; column < 3; ++column) {
            state_jacobian[(6 + row) * 14 + 10 + column] =
                dq_dw[row * 3 + column];
        }
    }
    const double ix = config.principal_inertia[0];
    const double iy = config.principal_inertia[1];
    const double iz = config.principal_inertia[2];
    derivative[10] = (control[3] - (iz - iy) * wy * wz) / ix;
    derivative[11] = (control[4] - (ix - iz) * wx * wz) / iy;
    derivative[12] = (control[5] - (iy - ix) * wx * wy) / iz;
    state_jacobian[10 * 14 + 11] = -(iz - iy) * wz / ix;
    state_jacobian[10 * 14 + 12] = -(iz - iy) * wy / ix;
    state_jacobian[11 * 14 + 10] = -(ix - iz) * wz / iy;
    state_jacobian[11 * 14 + 12] = -(ix - iz) * wx / iy;
    state_jacobian[12 * 14 + 10] = -(iy - ix) * wy / iz;
    state_jacobian[12 * 14 + 11] = -(iy - ix) * wx / iz;
    control_jacobian[10 * 7 + 3] = 1.0 / ix;
    control_jacobian[11 * 7 + 4] = 1.0 / iy;
    control_jacobian[12 * 7 + 5] = 1.0 / iz;
    derivative[13] = -config.mass_flow_coefficient * control[6];
    control_jacobian[13 * 7 + 6] = -config.mass_flow_coefficient;
}

template <int Model, int StateDimension, int ControlDimension>
__device__ void evaluate(
    const double* state,
    const double* control,
    const spacepdhcg_cuda_dynamics_config& config,
    double* derivative,
    double* state_jacobian,
    double* control_jacobian
) {
    if constexpr (Model == SPACEPDHCG_CUDA_DYNAMICS_HCW) {
        evaluate_hcw(state, control, config, derivative, state_jacobian, control_jacobian);
    } else if constexpr (Model == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF) {
        evaluate_pd3(state, control, config, derivative, state_jacobian, control_jacobian);
    } else if constexpr (Model == SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST) {
        evaluate_low_thrust(
            state, control, config, derivative, state_jacobian, control_jacobian
        );
    } else {
        evaluate_pd6(state, control, config, derivative, state_jacobian, control_jacobian);
    }
}

template <int Model, int StateDimension, int ControlDimension>
__device__ Augmented<StateDimension, ControlDimension> augmented_derivative(
    const Augmented<StateDimension, ControlDimension>& input,
    const double* control,
    const spacepdhcg_cuda_dynamics_config& config
) {
    Augmented<StateDimension, ControlDimension> result{};
    double state_jacobian[StateDimension * StateDimension]{};
    double control_jacobian[StateDimension * ControlDimension]{};
    evaluate<Model, StateDimension, ControlDimension>(
        input.state,
        control,
        config,
        result.state,
        state_jacobian,
        control_jacobian
    );
    for (int row = 0; row < StateDimension; ++row) {
        for (int column = 0; column < StateDimension; ++column) {
            for (int inner = 0; inner < StateDimension; ++inner) {
                result.transition[row * StateDimension + column] +=
                    state_jacobian[row * StateDimension + inner]
                    * input.transition[inner * StateDimension + column];
            }
        }
        for (int column = 0; column < ControlDimension; ++column) {
            result.sensitivity[row * ControlDimension + column] =
                control_jacobian[row * ControlDimension + column];
            for (int inner = 0; inner < StateDimension; ++inner) {
                result.sensitivity[row * ControlDimension + column] +=
                    state_jacobian[row * StateDimension + inner]
                    * input.sensitivity[inner * ControlDimension + column];
            }
        }
    }
    return result;
}

template <int StateDimension, int ControlDimension>
__device__ Augmented<StateDimension, ControlDimension> add_scaled(
    const Augmented<StateDimension, ControlDimension>& base,
    const Augmented<StateDimension, ControlDimension>& increment,
    const double scale
) {
    Augmented<StateDimension, ControlDimension> result = base;
    for (int index = 0; index < StateDimension; ++index) {
        result.state[index] += scale * increment.state[index];
    }
    for (int index = 0; index < StateDimension * StateDimension; ++index) {
        result.transition[index] += scale * increment.transition[index];
    }
    for (int index = 0; index < StateDimension * ControlDimension; ++index) {
        result.sensitivity[index] += scale * increment.sensitivity[index];
    }
    return result;
}

template <int Model, int StateDimension, int ControlDimension>
__global__ void variational_kernel(
    const double* states,
    const double* controls,
    double* propagated,
    double* transition,
    double* sensitivity,
    double* offset,
    const size_t intervals,
    const spacepdhcg_cuda_dynamics_config config
) {
    const size_t interval = blockIdx.x;
    if (threadIdx.x != 0 || interval >= intervals) {
        return;
    }
    const double* state = states + interval * StateDimension;
    const double* control = controls + interval * ControlDimension;
    Augmented<StateDimension, ControlDimension> initial{};
    for (int index = 0; index < StateDimension; ++index) {
        initial.state[index] = state[index];
        initial.transition[index * StateDimension + index] = 1.0;
    }
    const double step = config.step_seconds;
    const auto k1 =
        augmented_derivative<Model, StateDimension, ControlDimension>(initial, control, config);
    const auto k2 = augmented_derivative<Model, StateDimension, ControlDimension>(
        add_scaled(initial, k1, 0.5 * step), control, config
    );
    const auto k3 = augmented_derivative<Model, StateDimension, ControlDimension>(
        add_scaled(initial, k2, 0.5 * step), control, config
    );
    const auto k4 = augmented_derivative<Model, StateDimension, ControlDimension>(
        add_scaled(initial, k3, step), control, config
    );
    Augmented<StateDimension, ControlDimension> result = initial;
    for (int index = 0; index < StateDimension; ++index) {
        result.state[index] += step
            * (k1.state[index] + 2.0 * k2.state[index] + 2.0 * k3.state[index]
               + k4.state[index])
            / 6.0;
    }
    for (int index = 0; index < StateDimension * StateDimension; ++index) {
        result.transition[index] += step
            * (k1.transition[index] + 2.0 * k2.transition[index]
               + 2.0 * k3.transition[index] + k4.transition[index])
            / 6.0;
    }
    for (int index = 0; index < StateDimension * ControlDimension; ++index) {
        result.sensitivity[index] += step
            * (k1.sensitivity[index] + 2.0 * k2.sensitivity[index]
               + 2.0 * k3.sensitivity[index] + k4.sensitivity[index])
            / 6.0;
    }
    if constexpr (Model == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF) {
        double norm = 0.0;
        for (int component = 0; component < 4; ++component) {
            norm += result.state[6 + component] * result.state[6 + component];
        }
        norm = sqrt(norm);
        double unit[4]{};
        for (int component = 0; component < 4; ++component) {
            unit[component] = result.state[6 + component] / norm;
            result.state[6 + component] = unit[component];
        }
        double projected_transition[4 * StateDimension]{};
        double projected_sensitivity[4 * ControlDimension]{};
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < StateDimension; ++column) {
                for (int inner = 0; inner < 4; ++inner) {
                    projected_transition[row * StateDimension + column] +=
                        (((row == inner ? 1.0 : 0.0) - unit[row] * unit[inner]) / norm)
                        * result.transition[(6 + inner) * StateDimension + column];
                }
            }
            for (int column = 0; column < ControlDimension; ++column) {
                for (int inner = 0; inner < 4; ++inner) {
                    projected_sensitivity[row * ControlDimension + column] +=
                        (((row == inner ? 1.0 : 0.0) - unit[row] * unit[inner]) / norm)
                        * result.sensitivity[(6 + inner) * ControlDimension + column];
                }
            }
        }
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < StateDimension; ++column) {
                result.transition[(6 + row) * StateDimension + column] =
                    projected_transition[row * StateDimension + column];
            }
            for (int column = 0; column < ControlDimension; ++column) {
                result.sensitivity[(6 + row) * ControlDimension + column] =
                    projected_sensitivity[row * ControlDimension + column];
            }
        }
    }
    double* output_state = propagated + interval * StateDimension;
    double* output_transition =
        transition + interval * StateDimension * StateDimension;
    double* output_sensitivity =
        sensitivity + interval * StateDimension * ControlDimension;
    double* output_offset = offset + interval * StateDimension;
    for (int row = 0; row < StateDimension; ++row) {
        output_state[row] = result.state[row];
        double affine = result.state[row];
        for (int column = 0; column < StateDimension; ++column) {
            const double value = result.transition[row * StateDimension + column];
            output_transition[row * StateDimension + column] = value;
            affine -= value * state[column];
        }
        for (int column = 0; column < ControlDimension; ++column) {
            const double value = result.sensitivity[row * ControlDimension + column];
            output_sensitivity[row * ControlDimension + column] = value;
            affine -= value * control[column];
        }
        output_offset[row] = affine;
    }
}

__global__ void hcw_exact_kernel(
    const double* states,
    const double* controls,
    double* propagated,
    double* transition,
    double* sensitivity,
    double* offset,
    const size_t intervals,
    const spacepdhcg_cuda_dynamics_config config
) {
    const size_t interval = blockIdx.x;
    if (threadIdx.x != 0 || interval >= intervals) {
        return;
    }
    const double n = config.mean_motion;
    const double t = config.step_seconds;
    const double angle = n * t;
    const double c = cos(angle);
    const double s = sin(angle);
    const double inverse_n = 1.0 / n;
    const double inverse_n_squared = inverse_n * inverse_n;
    double state_matrix[36]{};
    double control_matrix[18]{};
    state_matrix[0 * 6 + 0] = 4.0 - 3.0 * c;
    state_matrix[0 * 6 + 3] = s * inverse_n;
    state_matrix[0 * 6 + 4] = 2.0 * (1.0 - c) * inverse_n;
    state_matrix[1 * 6 + 0] = 6.0 * (s - angle);
    state_matrix[1 * 6 + 1] = 1.0;
    state_matrix[1 * 6 + 3] = -2.0 * (1.0 - c) * inverse_n;
    state_matrix[1 * 6 + 4] = (4.0 * s - 3.0 * angle) * inverse_n;
    state_matrix[2 * 6 + 2] = c;
    state_matrix[2 * 6 + 5] = s * inverse_n;
    state_matrix[3 * 6 + 0] = 3.0 * n * s;
    state_matrix[3 * 6 + 3] = c;
    state_matrix[3 * 6 + 4] = 2.0 * s;
    state_matrix[4 * 6 + 0] = -6.0 * n * (1.0 - c);
    state_matrix[4 * 6 + 3] = -2.0 * s;
    state_matrix[4 * 6 + 4] = 4.0 * c - 3.0;
    state_matrix[5 * 6 + 2] = -n * s;
    state_matrix[5 * 6 + 5] = c;
    control_matrix[0 * 3 + 0] = (1.0 - c) * inverse_n_squared;
    control_matrix[0 * 3 + 1] = 2.0 * (angle - s) * inverse_n_squared;
    control_matrix[1 * 3 + 0] = 2.0 * (s - angle) * inverse_n_squared;
    control_matrix[1 * 3 + 1] =
        4.0 * (1.0 - c) * inverse_n_squared - 1.5 * t * t;
    control_matrix[2 * 3 + 2] = (1.0 - c) * inverse_n_squared;
    control_matrix[3 * 3 + 0] = s * inverse_n;
    control_matrix[3 * 3 + 1] = 2.0 * (1.0 - c) * inverse_n;
    control_matrix[4 * 3 + 0] = -2.0 * (1.0 - c) * inverse_n;
    control_matrix[4 * 3 + 1] = 4.0 * s * inverse_n - 3.0 * t;
    control_matrix[5 * 3 + 2] = s * inverse_n;
    const double* state = states + interval * 6U;
    const double* control = controls + interval * 3U;
    double* output = propagated + interval * 6U;
    for (int row = 0; row < 6; ++row) {
        output[row] = 0.0;
        for (int column = 0; column < 6; ++column) {
            const double value = state_matrix[row * 6 + column];
            transition[interval * 36U + row * 6 + column] = value;
            output[row] += value * state[column];
        }
        for (int column = 0; column < 3; ++column) {
            const double value = control_matrix[row * 3 + column];
            sensitivity[interval * 18U + row * 3 + column] = value;
            output[row] += value * control[column];
        }
        offset[interval * 6U + row] = 0.0;
    }
}

__global__ void fill_dynamics_csc_kernel(
    const double* transition,
    const double* sensitivity,
    const double* offset,
    const int* state_positions,
    const int* control_positions,
    const int* next_positions,
    const int* virtual_positions,
    double* scalar_values,
    double* scalar_lower,
    double* scalar_upper,
    const size_t intervals,
    const size_t state_dimension,
    const size_t control_dimension,
    const size_t row_start
) {
    const size_t interval = blockIdx.x;
    if (interval >= intervals) {
        return;
    }
    for (size_t flat = threadIdx.x;
         flat < state_dimension * state_dimension;
         flat += blockDim.x) {
        scalar_values[state_positions[
            interval * state_dimension * state_dimension + flat
        ]] = -transition[interval * state_dimension * state_dimension + flat];
    }
    for (size_t flat = threadIdx.x;
         flat < state_dimension * control_dimension;
         flat += blockDim.x) {
        scalar_values[control_positions[
            interval * state_dimension * control_dimension + flat
        ]] = -sensitivity[interval * state_dimension * control_dimension + flat];
    }
    for (size_t row = threadIdx.x; row < state_dimension; row += blockDim.x) {
        scalar_values[next_positions[interval * state_dimension + row]] = 1.0;
        if (virtual_positions != nullptr) {
            scalar_values[virtual_positions[interval * state_dimension + row]] = -1.0;
        }
        const size_t target_row = row_start + interval * state_dimension + row;
        const double value = offset[interval * state_dimension + row];
        scalar_lower[target_row] = value;
        scalar_upper[target_row] = value;
    }
}

template <int Model, int StateDimension, int ControlDimension>
spacepdhcg_cuda_status launch_variational(
    const spacepdhcg_cuda_dynamics_config& config,
    const spacepdhcg_cuda_variational_request& request,
    const cudaStream_t stream
) {
    if constexpr (Model == SPACEPDHCG_CUDA_DYNAMICS_HCW) {
        hcw_exact_kernel<<<request.intervals, 32, 0, stream>>>(
            view_pointer<const double>(request.reference_states),
            view_pointer<const double>(request.reference_controls),
            view_pointer<double>(request.propagated_states),
            view_pointer<double>(request.state_transition),
            view_pointer<double>(request.control_sensitivity),
            view_pointer<double>(request.affine_offset),
            request.intervals,
            config
        );
    } else {
        variational_kernel<Model, StateDimension, ControlDimension>
            <<<request.intervals, 32, 0, stream>>>(
                view_pointer<const double>(request.reference_states),
                view_pointer<const double>(request.reference_controls),
                view_pointer<double>(request.propagated_states),
                view_pointer<double>(request.state_transition),
                view_pointer<double>(request.control_sensitivity),
                view_pointer<double>(request.affine_offset),
                request.intervals,
                config
            );
    }
    return cudaGetLastError() == cudaSuccess
        ? SPACEPDHCG_CUDA_SUCCESS
        : SPACEPDHCG_CUDA_RUNTIME_ERROR;
}

}  // namespace

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_variational_rk4_async(
    const spacepdhcg_cuda_dynamics_config* config,
    const spacepdhcg_cuda_variational_request* request,
    const spacepdhcg_accelerator_stream stream
) {
    if (config == nullptr || request == nullptr
        || config->abi_version != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        || request->abi_version != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        || request->intervals == 0U || !std::isfinite(config->step_seconds)
        || config->step_seconds <= 0.0 || stream.device.type != SPACEPDHCG_DEVICE_CUDA
        || stream.device.id < 0) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    size_t state_dimension = 0U;
    size_t control_dimension = 0U;
    switch (config->model) {
        case SPACEPDHCG_CUDA_DYNAMICS_HCW:
            state_dimension = 6U;
            control_dimension = 3U;
            break;
        case SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF:
        case SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST:
            state_dimension = 7U;
            control_dimension = 4U;
            break;
        case SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF:
            state_dimension = 14U;
            control_dimension = 7U;
            break;
        default:
            return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    const size_t expected[] = {
        request->intervals * state_dimension,
        request->intervals * control_dimension,
        request->intervals * state_dimension,
        request->intervals * state_dimension * state_dimension,
        request->intervals * state_dimension * control_dimension,
        request->intervals * state_dimension,
    };
    const spacepdhcg_accelerator_buffer_view views[] = {
        request->reference_states,
        request->reference_controls,
        request->propagated_states,
        request->state_transition,
        request->control_sensitivity,
        request->affine_offset,
    };
    for (size_t index = 0; index < 6U; ++index) {
        const auto status = validate_device_view(
            views[index], expected[index], SPACEPDHCG_SCALAR_FLOAT64, stream.device.id
        );
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            return status;
        }
    }
    const auto native_stream = reinterpret_cast<cudaStream_t>(stream.native_handle);
    switch (config->model) {
        case SPACEPDHCG_CUDA_DYNAMICS_HCW:
            return launch_variational<SPACEPDHCG_CUDA_DYNAMICS_HCW, 6, 3>(
                *config, *request, native_stream
            );
        case SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF:
            return launch_variational<
                SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF, 7, 4
            >(*config, *request, native_stream);
        case SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST:
            return launch_variational<SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST, 7, 4>(
                *config, *request, native_stream
            );
        case SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF:
            return launch_variational<
                SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF, 14, 7
            >(*config, *request, native_stream);
        default:
            return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_fill_dynamics_csc_async(
    const spacepdhcg_cuda_csc_dynamics_fill* request,
    const spacepdhcg_accelerator_stream stream
) {
    if (request == nullptr
        || request->abi_version != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        || request->intervals == 0U || request->state_dimension == 0U
        || request->control_dimension == 0U
        || stream.device.type != SPACEPDHCG_DEVICE_CUDA || stream.device.id < 0) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    const size_t state_matrix =
        request->intervals * request->state_dimension * request->state_dimension;
    const size_t control_matrix =
        request->intervals * request->state_dimension * request->control_dimension;
    const size_t state_rows = request->intervals * request->state_dimension;
    const spacepdhcg_accelerator_buffer_view views[] = {
        request->state_transition,
        request->control_sensitivity,
        request->affine_offset,
        request->state_positions,
        request->control_positions,
        request->next_state_positions,
        request->scalar_values,
        request->scalar_lower,
        request->scalar_upper,
    };
    const size_t minimum_elements[] = {
        state_matrix,
        control_matrix,
        state_rows,
        state_matrix,
        control_matrix,
        state_rows,
        request->scalar_values.elements,
        request->scalar_lower.elements,
        request->scalar_upper.elements,
    };
    const spacepdhcg_accelerator_scalar_type types[] = {
        SPACEPDHCG_SCALAR_FLOAT64,
        SPACEPDHCG_SCALAR_FLOAT64,
        SPACEPDHCG_SCALAR_FLOAT64,
        SPACEPDHCG_SCALAR_INT32,
        SPACEPDHCG_SCALAR_INT32,
        SPACEPDHCG_SCALAR_INT32,
        SPACEPDHCG_SCALAR_FLOAT64,
        SPACEPDHCG_SCALAR_FLOAT64,
        SPACEPDHCG_SCALAR_FLOAT64,
    };
    for (size_t index = 0; index < 9U; ++index) {
        const auto status = validate_device_view(
            views[index], minimum_elements[index], types[index], stream.device.id
        );
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            return status;
        }
    }
    const int* virtual_positions = nullptr;
    if (request->virtual_positions.elements > 0U) {
        const auto status = validate_device_view(
            request->virtual_positions,
            state_rows,
            SPACEPDHCG_SCALAR_INT32,
            stream.device.id
        );
        if (status != SPACEPDHCG_CUDA_SUCCESS) {
            return status;
        }
        virtual_positions = view_pointer<const int>(request->virtual_positions);
    }
    fill_dynamics_csc_kernel<<<
        request->intervals,
        128,
        0,
        reinterpret_cast<cudaStream_t>(stream.native_handle)
    >>>(
        view_pointer<const double>(request->state_transition),
        view_pointer<const double>(request->control_sensitivity),
        view_pointer<const double>(request->affine_offset),
        view_pointer<const int>(request->state_positions),
        view_pointer<const int>(request->control_positions),
        view_pointer<const int>(request->next_state_positions),
        virtual_positions,
        view_pointer<double>(request->scalar_values),
        view_pointer<double>(request->scalar_lower),
        view_pointer<double>(request->scalar_upper),
        request->intervals,
        request->state_dimension,
        request->control_dimension,
        request->dynamics_row_start
    );
    return cudaGetLastError() == cudaSuccess
        ? SPACEPDHCG_CUDA_SUCCESS
        : SPACEPDHCG_CUDA_RUNTIME_ERROR;
}

namespace {

struct ScvxMetrics {
    double objective;
    double merit;
    double model_merit;
    double dynamics;
    double path;
    double path_thrust;
    double path_mass;
    double path_altitude;
    double terminal;
    double virtual_control;
    double step;
    double thrust;
    double torque;
    double pointing;
    double mass;
    double altitude;
    double glide_slope;
    double angular_rate;
    double quaternion;
    double maximum_stage_trust_distance;
    double terminal_trust_distance;
};

template <int Model, int StateDimension, int ControlDimension>
__device__ void state_rk4_step(
    const double* state,
    const double* control,
    const spacepdhcg_cuda_dynamics_config& config,
    double* output
) {
    double state_jacobian[StateDimension * StateDimension]{};
    double control_jacobian[StateDimension * ControlDimension]{};
    double k1[StateDimension]{};
    double k2[StateDimension]{};
    double k3[StateDimension]{};
    double k4[StateDimension]{};
    double stage[StateDimension]{};
    evaluate<Model, StateDimension, ControlDimension>(
        state, control, config, k1, state_jacobian, control_jacobian
    );
    for (int index = 0; index < StateDimension; ++index) {
        stage[index] = state[index] + 0.5 * config.step_seconds * k1[index];
    }
    for (int index = 0;
         index < StateDimension * StateDimension;
         ++index) {
        state_jacobian[index] = 0.0;
    }
    for (int index = 0;
         index < StateDimension * ControlDimension;
         ++index) {
        control_jacobian[index] = 0.0;
    }
    evaluate<Model, StateDimension, ControlDimension>(
        stage, control, config, k2, state_jacobian, control_jacobian
    );
    for (int index = 0; index < StateDimension; ++index) {
        stage[index] = state[index] + 0.5 * config.step_seconds * k2[index];
    }
    for (int index = 0;
         index < StateDimension * StateDimension;
         ++index) {
        state_jacobian[index] = 0.0;
    }
    for (int index = 0;
         index < StateDimension * ControlDimension;
         ++index) {
        control_jacobian[index] = 0.0;
    }
    evaluate<Model, StateDimension, ControlDimension>(
        stage, control, config, k3, state_jacobian, control_jacobian
    );
    for (int index = 0; index < StateDimension; ++index) {
        stage[index] = state[index] + config.step_seconds * k3[index];
    }
    for (int index = 0;
         index < StateDimension * StateDimension;
         ++index) {
        state_jacobian[index] = 0.0;
    }
    for (int index = 0;
         index < StateDimension * ControlDimension;
         ++index) {
        control_jacobian[index] = 0.0;
    }
    evaluate<Model, StateDimension, ControlDimension>(
        stage, control, config, k4, state_jacobian, control_jacobian
    );
    for (int index = 0; index < StateDimension; ++index) {
        output[index] = state[index]
            + config.step_seconds
                * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index])
                / 6.0;
    }
    if constexpr (Model == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF) {
        double norm_squared = 0.0;
        for (int component = 0; component < 4; ++component) {
            norm_squared += output[6 + component] * output[6 + component];
        }
        const double inverse_norm = 1.0 / sqrt(norm_squared);
        for (int component = 0; component < 4; ++component) {
            output[6 + component] *= inverse_norm;
        }
    }
}

__global__ void gather_scvx_candidate_kernel(
    const double* primal,
    const int* state_indices,
    const int* control_indices,
    double* states,
    double* controls,
    const size_t state_elements,
    const size_t control_elements
) {
    for (size_t index = threadIdx.x;
         index < state_elements;
         index += blockDim.x) {
        states[index] = primal[state_indices[index]];
    }
    for (size_t index = threadIdx.x;
         index < control_elements;
         index += blockDim.x) {
        controls[index] = primal[control_indices[index]];
    }
}

__global__ void update_scvx_numeric_kernel(
    const spacepdhcg_cuda_scvx_problem problem,
    const double trust_radius,
    const double virtual_penalty
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const auto& update = problem.numeric_update;
    const auto* states = view_pointer<const double>(problem.reference_states);
    const auto* controls = view_pointer<const double>(problem.reference_controls);
    const auto* q_positions =
        view_pointer<const int>(update.quadratic_diagonal_positions);
    const auto* q = view_pointer<const double>(problem.numeric.quadratic);
    auto* c = view_pointer<double>(problem.numeric.linear_objective);
    auto* scalar = view_pointer<double>(problem.numeric.scalar_constraint);
    auto* scalar_lower = view_pointer<double>(problem.numeric.scalar_lower);
    auto* scalar_upper = view_pointer<double>(problem.numeric.scalar_upper);
    auto* affine_offset = view_pointer<double>(problem.numeric.affine_offset);
    const auto* initial = view_pointer<const double>(problem.initial_state);
    const auto* target = view_pointer<const double>(problem.target_state);
    const size_t state_elements =
        (problem.intervals + 1U) * problem.state_dimension;
    const size_t control_elements =
        problem.intervals * problem.control_dimension;
    for (size_t index = 0U; index < state_elements; ++index) {
        const int q_position = q_positions[index];
        const size_t component = index % problem.state_dimension;
        const size_t node = index / problem.state_dimension;
        double reference = states[index];
        if (problem.dynamics.model == SPACEPDHCG_CUDA_DYNAMICS_HCW) {
            const double fraction =
                static_cast<double>(node)
                / static_cast<double>(problem.intervals);
            reference = (1.0 - fraction) * initial[component]
                + fraction * target[component];
        }
        c[view_pointer<const int>(problem.state_variable_indices)[index]] =
            -q[q_position] * reference;
    }
    const size_t terminal_dimension =
        problem.dynamics.model
                == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF
            ? 13U
            : (problem.dynamics.model == SPACEPDHCG_CUDA_DYNAMICS_HCW
                ? problem.state_dimension
                : 6U);
    for (size_t component = 0U;
         component < problem.state_dimension;
         ++component) {
        scalar_lower[component] = initial[component];
        scalar_upper[component] = initial[component];
    }
    for (size_t component = 0U; component < terminal_dimension; ++component) {
        scalar_lower[update.terminal_row_start + component] = target[component];
        scalar_upper[update.terminal_row_start + component] = target[component];
    }
    for (size_t index = 0U; index < control_elements; ++index) {
        const int variable =
            view_pointer<const int>(problem.control_variable_indices)[index];
        const int q_position = q_positions[state_elements + index];
        c[variable] = -q[q_position] * controls[index];
    }
    if (problem.dynamics.model != SPACEPDHCG_CUDA_DYNAMICS_HCW) {
        const size_t sigma =
            problem.dynamics.model
                    == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF
                ? 6U
                : 3U;
        for (size_t interval = 0U; interval < problem.intervals; ++interval) {
            const int variable = view_pointer<const int>(
                problem.control_variable_indices
            )[interval * problem.control_dimension + sigma];
            c[variable] += update.fuel_weight * problem.dynamics.step_seconds;
        }
        const size_t virtual_elements =
            problem.intervals * problem.state_dimension;
        for (size_t index = 0U; index < virtual_elements; ++index) {
            c[update.epigraph_variable_offset + index] = virtual_penalty;
        }
        for (size_t interval = 0U; interval < problem.intervals; ++interval) {
            const size_t row =
                update.stage_trust_row_start
                + interval * update.stage_trust_stride;
            for (size_t component = 0U;
                 component < problem.state_dimension;
                 ++component) {
                affine_offset[row + component] =
                    -update.state_trust_scales[component]
                    * states[interval * problem.state_dimension + component];
            }
            for (size_t component = 0U;
                 component < problem.control_dimension;
                 ++component) {
                affine_offset[row + problem.state_dimension + component] =
                    -update.control_trust_scales[component]
                    * controls[interval * problem.control_dimension + component];
            }
            affine_offset[row + update.stage_trust_stride - 1U] = trust_radius;
        }
        for (size_t component = 0U;
             component < problem.state_dimension;
             ++component) {
            affine_offset[update.terminal_trust_row_start + component] =
                -update.state_trust_scales[component]
                * states[problem.intervals * problem.state_dimension + component];
        }
        affine_offset[
            update.terminal_trust_row_start + problem.state_dimension
        ] = trust_radius;
    }
    if (problem.dynamics.model == SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST) {
        const auto* positions = view_pointer<const int>(update.radial_positions);
        for (size_t node = 0U; node <= problem.intervals; ++node) {
            const double* state = states + node * problem.state_dimension;
            const double radius = sqrt(
                state[0U] * state[0U]
                + state[1U] * state[1U]
                + state[2U] * state[2U]
            );
            for (size_t component = 0U; component < 3U; ++component) {
                scalar[positions[3U * node + component]] =
                    state[component] / radius;
            }
        }
    }
    if (problem.dynamics.model
        == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF) {
        const auto* positions =
            view_pointer<const int>(update.quaternion_positions);
        for (size_t node = 0U; node <= problem.intervals; ++node) {
            const size_t row = update.quaternion_row_start + node;
            if (node == problem.intervals) {
                for (size_t component = 0U; component < 4U; ++component) {
                    scalar[positions[4U * node + component]] = 0.0;
                }
                scalar_lower[row] = 0.0;
                scalar_upper[row] = 0.0;
                continue;
            }
            const double* quaternion =
                states + node * problem.state_dimension + 6U;
            double norm_squared = 0.0;
            for (size_t component = 0U; component < 4U; ++component) {
                scalar[positions[4U * node + component]] =
                    2.0 * quaternion[component];
                norm_squared += quaternion[component] * quaternion[component];
            }
            scalar_lower[row] = 1.0 + norm_squared;
            scalar_upper[row] = 1.0 + norm_squared;
        }
    }
}

__device__ double conditioning_factor(
    const size_t row,
    const size_t rows,
    const double log10_span
) {
    if (log10_span == 0.0 || rows <= 1U) {
        return 1.0;
    }
    const double fraction =
        static_cast<double>(row) / static_cast<double>(rows - 1U);
    return pow(10.0, log10_span * (fraction - 0.5));
}

__global__ void condition_dynamics_rows_kernel(
    const spacepdhcg_cuda_scvx_problem problem
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row_start = problem.dynamics_fill.dynamics_row_start;
    const size_t row_count = problem.intervals * problem.state_dimension;
    const size_t row_end = row_start + row_count;
    const double span = problem.numeric_update.conditioning_log10_span;
    auto* scalar = view_pointer<double>(problem.numeric.scalar_constraint);
    const auto* scalar_indices =
        view_pointer<const int>(problem.canonical_topology.scalar_indices);
    if (index < problem.numeric.scalar_constraint.elements) {
        const auto row = static_cast<size_t>(scalar_indices[index]);
        if (row >= row_start && row < row_end) {
            scalar[index] *= conditioning_factor(row - row_start, row_count, span);
        }
    }
    if (index < row_count) {
        const double factor = conditioning_factor(index, row_count, span);
        auto* lower = view_pointer<double>(problem.numeric.scalar_lower);
        auto* upper = view_pointer<double>(problem.numeric.scalar_upper);
        lower[row_start + index] *= factor;
        upper[row_start + index] *= factor;
    }
}

template <int Model, int StateDimension, int ControlDimension>
__global__ void replay_scvx_kernel(
    const double* initial_state,
    const double* controls,
    double* replay,
    const size_t intervals,
    const spacepdhcg_cuda_dynamics_config config
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (int state = 0; state < StateDimension; ++state) {
        replay[state] = initial_state[state];
    }
    for (size_t interval = 0; interval < intervals; ++interval) {
        state_rk4_step<Model, StateDimension, ControlDimension>(
            replay + interval * StateDimension,
            controls + interval * ControlDimension,
            config,
            replay + (interval + 1U) * StateDimension
        );
    }
}

__global__ void scvx_metrics_kernel(
    const double* states,
    const double* controls,
    const double* replay,
    const double* reference_states,
    const double* reference_controls,
    const double* target,
    const double* primal,
    const int* virtual_indices,
    const size_t virtual_elements,
    const size_t intervals,
    const size_t state_dimension,
    const size_t control_dimension,
    const int model,
    const double feasibility_penalty,
    const double virtual_penalty,
    const double trust_radius,
    const spacepdhcg_cuda_scvx_numeric_update update,
    const double* scalar_lower,
    const double* variable_lower,
    const double* variable_upper,
    const int* state_variable_indices,
    const int* control_variable_indices,
    ScvxMetrics* metrics
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    ScvxMetrics result{};
    double actual_terminal_sum = 0.0;
    double model_terminal_sum = 0.0;
    double actual_path_sum = 0.0;
    double model_path_sum = 0.0;
    for (size_t interval = 0U; interval < intervals; ++interval) {
        double step_squared = 0.0;
        for (size_t component = 0U; component < state_dimension; ++component) {
            const double scale = model == SPACEPDHCG_CUDA_DYNAMICS_HCW
                ? 1.0
                : update.state_trust_scales[component];
            const double delta =
                (states[interval * state_dimension + component]
                 - reference_states[interval * state_dimension + component])
                * scale;
            step_squared += delta * delta;
        }
        for (size_t component = 0U; component < control_dimension; ++component) {
            const double value =
                controls[interval * control_dimension + component];
            const double scale = model == SPACEPDHCG_CUDA_DYNAMICS_HCW
                ? 1.0
                : update.control_trust_scales[component];
            const double delta =
                (value
                 - reference_controls[
                     interval * control_dimension + component
                 ]) * scale;
            step_squared += delta * delta;
            if (model == SPACEPDHCG_CUDA_DYNAMICS_HCW) {
                result.objective += 0.5 * value * value;
            }
        }
        const double stage_step = sqrt(step_squared);
        result.step = fmax(result.step, stage_step);
        result.maximum_stage_trust_distance = fmax(
            result.maximum_stage_trust_distance,
            stage_step - trust_radius
        );
    }
    double terminal_step_squared = 0.0;
    for (size_t component = 0U; component < state_dimension; ++component) {
        const double scale = model == SPACEPDHCG_CUDA_DYNAMICS_HCW
            ? 1.0
            : update.state_trust_scales[component];
        const double delta =
            (states[intervals * state_dimension + component]
             - reference_states[intervals * state_dimension + component])
            * scale;
        terminal_step_squared += delta * delta;
    }
    const double terminal_step = sqrt(terminal_step_squared);
    result.step = fmax(result.step, terminal_step);
    result.terminal_trust_distance = fmax(
        0.0,
        terminal_step - trust_radius
    );
    for (size_t node = 1; node <= intervals; ++node) {
        for (size_t state = 0; state < state_dimension; ++state) {
            result.dynamics = fmax(
                result.dynamics,
                fabs(
                    states[node * state_dimension + state]
                    - replay[node * state_dimension + state]
                ) * (model == SPACEPDHCG_CUDA_DYNAMICS_HCW
                    ? 1.0
                    : update.state_trust_scales[state])
            );
        }
    }
    const size_t terminal_dimension =
        model == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF
        ? 13U
        : (model == SPACEPDHCG_CUDA_DYNAMICS_HCW ? state_dimension : 6U);
    for (size_t state = 0; state < terminal_dimension; ++state) {
        const double scale = model == SPACEPDHCG_CUDA_DYNAMICS_HCW
            ? 1.0
            : update.state_trust_scales[state];
        const double model_error = fabs(
            states[intervals * state_dimension + state] - target[state]
        ) * scale;
        const double actual_error = fabs(
            replay[intervals * state_dimension + state] - target[state]
        ) * scale;
        model_terminal_sum += model_error;
        actual_terminal_sum += actual_error;
        result.terminal = fmax(
            result.terminal,
            actual_error
        );
    }
    if (model != SPACEPDHCG_CUDA_DYNAMICS_HCW) {
        const size_t sigma_index =
            model == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF
            ? 6U
            : 3U;
        for (size_t interval = 0; interval < intervals; ++interval) {
            const double* control = controls + interval * control_dimension;
            const double thrust_norm = sqrt(
                control[0] * control[0]
                + control[1] * control[1]
                + control[2] * control[2]
            );
            const double thrust_violation =
                fmax(0.0, thrust_norm - control[sigma_index]);
            const int sigma_variable =
                control_variable_indices[
                    interval * control_dimension + sigma_index
                ];
            const double maximum_thrust = variable_upper[sigma_variable];
            const double normalised_thrust_violation =
                thrust_violation / maximum_thrust;
            const double throttle_violation =
                fmax(0.0, control[sigma_index] - maximum_thrust)
                / maximum_thrust;
            const double thrust_path =
                fmax(normalised_thrust_violation, throttle_violation);
            result.thrust = fmax(result.thrust, normalised_thrust_violation);
            result.path = fmax(result.path, normalised_thrust_violation);
            result.path = fmax(result.path, throttle_violation);
            result.path_thrust = fmax(result.path_thrust, thrust_path);
            actual_path_sum += normalised_thrust_violation + throttle_violation;
            model_path_sum += normalised_thrust_violation + throttle_violation;
            if (model
                == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF) {
                const double torque_norm = sqrt(
                    control[3] * control[3]
                    + control[4] * control[4]
                    + control[5] * control[5]
                );
                const double torque_violation = fmax(
                    0.0,
                    torque_norm - update.maximum_torque
                ) / update.maximum_torque;
                const double pointing_violation = fmax(
                    0.0,
                    update.tilt_cosine * control[6] - control[2]
                ) / update.maximum_thrust;
                result.torque = fmax(result.torque, torque_violation);
                result.pointing = fmax(result.pointing, pointing_violation);
                result.path = fmax(
                    result.path,
                    fmax(torque_violation, pointing_violation)
                );
                actual_path_sum += torque_violation + pointing_violation;
                model_path_sum += torque_violation + pointing_violation;
            }
            result.objective += control[sigma_index]
                / (static_cast<double>(intervals) * maximum_thrust);
        }
        const size_t mass_index =
            model == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF
            ? 13U
            : 6U;
        for (size_t node = 0; node <= intervals; ++node) {
            const int mass_variable =
                state_variable_indices[node * state_dimension + mass_index];
            const double mass_scale = update.state_trust_scales[mass_index];
            const double actual_mass_violation = fmax(
                0.0,
                variable_lower[mass_variable]
                    - replay[node * state_dimension + mass_index]
            ) * mass_scale;
            const double model_mass_violation = fmax(
                0.0,
                variable_lower[mass_variable]
                    - states[node * state_dimension + mass_index]
            ) * mass_scale;
            result.path = fmax(
                result.path,
                actual_mass_violation
            );
            result.path_mass = fmax(result.path_mass, actual_mass_violation);
            result.mass = fmax(result.mass, actual_mass_violation);
            actual_path_sum += actual_mass_violation;
            model_path_sum += model_mass_violation;
            if (model
                == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF) {
                const double* actual_state =
                    replay + node * state_dimension;
                const double* model_state =
                    states + node * state_dimension;
                const double position_scale = fmax(
                    update.state_trust_scales[0],
                    fmax(
                        update.state_trust_scales[1],
                        update.state_trust_scales[2]
                    )
                );
                const double actual_altitude =
                    fmax(0.0, -actual_state[2]) * position_scale;
                const double model_altitude =
                    fmax(0.0, -model_state[2]) * position_scale;
                const double actual_glide = fmax(
                    0.0,
                    hypot(actual_state[0], actual_state[1])
                        - update.glide_slope_tangent * actual_state[2]
                ) * position_scale;
                const double model_glide = fmax(
                    0.0,
                    hypot(model_state[0], model_state[1])
                        - update.glide_slope_tangent * model_state[2]
                ) * position_scale;
                result.altitude = fmax(result.altitude, actual_altitude);
                result.glide_slope = fmax(result.glide_slope, actual_glide);
                result.path = fmax(
                    result.path,
                    fmax(actual_altitude, actual_glide)
                );
                actual_path_sum += actual_altitude + actual_glide;
                model_path_sum += model_altitude + model_glide;
            } else if (model
                       == SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST) {
                const double* actual_state =
                    replay + node * state_dimension;
                const double* model_state =
                    states + node * state_dimension;
                const double position_scale = fmax(
                    update.state_trust_scales[0],
                    fmax(
                        update.state_trust_scales[1],
                        update.state_trust_scales[2]
                    )
                );
                const double actual_radius = sqrt(
                    actual_state[0] * actual_state[0]
                    + actual_state[1] * actual_state[1]
                    + actual_state[2] * actual_state[2]
                );
                const double model_radius = sqrt(
                    model_state[0] * model_state[0]
                    + model_state[1] * model_state[1]
                    + model_state[2] * model_state[2]
                );
                const double actual_altitude = fmax(
                    0.0,
                    update.minimum_radius - actual_radius
                ) * position_scale;
                const double model_altitude = fmax(
                    0.0,
                    update.minimum_radius - model_radius
                ) * position_scale;
                result.altitude = fmax(result.altitude, actual_altitude);
                result.path = fmax(result.path, actual_altitude);
                actual_path_sum += actual_altitude;
                model_path_sum += model_altitude;
            }
            if (model
                == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF) {
                const double* actual_state =
                    replay + node * state_dimension;
                const double* model_state =
                    states + node * state_dimension;
                const double position_scale = fmax(
                    update.state_trust_scales[0],
                    fmax(
                        update.state_trust_scales[1],
                        update.state_trust_scales[2]
                    )
                );
                const double angular_rate_scale = fmax(
                    update.state_trust_scales[10],
                    fmax(
                        update.state_trust_scales[11],
                        update.state_trust_scales[12]
                    )
                );
                const double actual_altitude = fmax(
                    0.0,
                    -actual_state[2]
                ) * position_scale;
                const double model_altitude = fmax(
                    0.0,
                    -model_state[2]
                ) * position_scale;
                const double actual_glide = fmax(
                    0.0,
                    hypot(actual_state[0], actual_state[1])
                        - update.glide_slope_tangent * actual_state[2]
                ) * position_scale;
                const double model_glide = fmax(
                    0.0,
                    hypot(model_state[0], model_state[1])
                        - update.glide_slope_tangent * model_state[2]
                ) * position_scale;
                const double actual_rate = fmax(
                    0.0,
                    sqrt(
                        actual_state[10] * actual_state[10]
                        + actual_state[11] * actual_state[11]
                        + actual_state[12] * actual_state[12]
                    ) - update.maximum_angular_rate
                ) * angular_rate_scale;
                const double model_rate = fmax(
                    0.0,
                    sqrt(
                        model_state[10] * model_state[10]
                        + model_state[11] * model_state[11]
                        + model_state[12] * model_state[12]
                    ) - update.maximum_angular_rate
                ) * angular_rate_scale;
                result.path = fmax(
                    result.path,
                    fmax(
                        actual_altitude,
                        fmax(actual_glide, actual_rate)
                    )
                );
                result.altitude = fmax(result.altitude, actual_altitude);
                result.glide_slope = fmax(result.glide_slope, actual_glide);
                result.angular_rate = fmax(result.angular_rate, actual_rate);
                actual_path_sum +=
                    actual_altitude + actual_glide + actual_rate;
                model_path_sum += model_altitude + model_glide + model_rate;
            }
        }
    }
    if (model == SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST) {
        const double minimum_radius = scalar_lower[update.radial_row_start];
        const double position_scale = fmax(
            update.state_trust_scales[0U],
            fmax(update.state_trust_scales[1U], update.state_trust_scales[2U])
        );
        for (size_t node = 0U; node <= intervals; ++node) {
            const double* actual_state = replay + node * state_dimension;
            const double* model_state = states + node * state_dimension;
            const double actual_radius = sqrt(
                actual_state[0U] * actual_state[0U]
                + actual_state[1U] * actual_state[1U]
                + actual_state[2U] * actual_state[2U]
            );
            const double model_radius = sqrt(
                model_state[0U] * model_state[0U]
                + model_state[1U] * model_state[1U]
                + model_state[2U] * model_state[2U]
            );
            const double actual_altitude_violation =
                fmax(0.0, minimum_radius - actual_radius) * position_scale;
            const double model_altitude_violation =
                fmax(0.0, minimum_radius - model_radius) * position_scale;
            result.path = fmax(result.path, actual_altitude_violation);
            result.path_altitude =
                fmax(result.path_altitude, actual_altitude_violation);
            actual_path_sum += actual_altitude_violation;
            model_path_sum += model_altitude_violation;
        }
    }
    if (model == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF) {
        for (size_t node = 0; node <= intervals; ++node) {
            double norm_squared = 0.0;
            for (size_t component = 0; component < 4U; ++component) {
                const double value =
                    replay[node * state_dimension + 6U + component];
                norm_squared += value * value;
            }
            result.path = fmax(
                result.path,
                fabs(sqrt(norm_squared) - 1.0)
            );
            result.quaternion = fmax(
                result.quaternion,
                fabs(sqrt(norm_squared) - 1.0)
            );
            double model_norm_squared = 0.0;
            for (size_t component = 0; component < 4U; ++component) {
                const double value =
                    states[node * state_dimension + 6U + component];
                model_norm_squared += value * value;
            }
            actual_path_sum += fabs(sqrt(norm_squared) - 1.0);
            model_path_sum += fabs(sqrt(model_norm_squared) - 1.0);
        }
    }
    double virtual_sum = 0.0;
    for (size_t index = 0; index < virtual_elements; ++index) {
        const double scaled = fabs(primal[virtual_indices[index]])
            * update.state_trust_scales[index % state_dimension];
        virtual_sum += scaled;
        result.virtual_control = fmax(
            result.virtual_control,
            scaled
        );
    }
    const double virtual_measure = virtual_elements == 0U
        ? 0.0
        : virtual_sum / static_cast<double>(virtual_elements);
    result.merit = result.objective
        + feasibility_penalty
            * (actual_path_sum + actual_terminal_sum);
    result.model_merit = result.objective
        + feasibility_penalty * (model_path_sum + model_terminal_sum)
        + virtual_penalty * virtual_measure;
    *metrics = result;
}

spacepdhcg_cuda_status launch_replay(
    const spacepdhcg_cuda_dynamics_config& config,
    const size_t intervals,
    const double* initial_state,
    const double* controls,
    double* replay,
    const cudaStream_t stream
) {
    switch (config.model) {
        case SPACEPDHCG_CUDA_DYNAMICS_HCW:
            replay_scvx_kernel<SPACEPDHCG_CUDA_DYNAMICS_HCW, 6, 3>
                <<<1, 1, 0, stream>>>(
                    initial_state, controls, replay, intervals, config
                );
            break;
        case SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF:
            replay_scvx_kernel<
                SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF, 7, 4
            ><<<1, 1, 0, stream>>>(
                initial_state, controls, replay, intervals, config
            );
            break;
        case SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST:
            replay_scvx_kernel<SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST, 7, 4>
                <<<1, 1, 0, stream>>>(
                    initial_state, controls, replay, intervals, config
                );
            break;
        case SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF:
            replay_scvx_kernel<
                SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF, 14, 7
            ><<<1, 1, 0, stream>>>(
                initial_state, controls, replay, intervals, config
            );
            break;
        default:
            return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    return cudaGetLastError() == cudaSuccess
        ? SPACEPDHCG_CUDA_SUCCESS
        : SPACEPDHCG_CUDA_RUNTIME_ERROR;
}

__device__ unsigned long long mix_numeric_word(
    unsigned long long value,
    const unsigned long long index
) {
    value ^= index + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

__global__ void hash_numeric_kernel(
    const double* values,
    const size_t elements,
    const unsigned long long tag,
    unsigned long long* fingerprint
) {
    unsigned long long local = 0ULL;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements;
         index += blockDim.x * gridDim.x) {
        local ^= mix_numeric_word(
            static_cast<unsigned long long>(__double_as_longlong(values[index])),
            tag ^ static_cast<unsigned long long>(index)
        );
    }
    atomicXor(fingerprint, local);
}

spacepdhcg_cuda_scvx_phase forcing_phase(
    const double residual,
    const uint32_t outer_iteration,
    const spacepdhcg_cuda_scvx_options& policy,
    spacepdhcg_cuda_solve_options* options
) {
    options->abi_version = SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;
    options->residual_check_frequency = 25U;
    const double residual_target = policy.adaptive_coefficient
        * std::pow(std::max(0.0, residual), 1.0 + policy.adaptive_alpha);
    const double geometric_target = policy.adaptive_epsilon_0
        * std::pow(policy.adaptive_gamma, static_cast<double>(outer_iteration));
    const double requested = std::max(
        policy.adaptive_epsilon_floor,
        std::min({
            policy.adaptive_epsilon_max,
            residual_target,
            geometric_target,
        })
    );
    if (!std::isfinite(residual) || residual > 2.5e-1) {
        options->optimality_tolerance =
            std::min(policy.repair_tolerance_ceiling, requested);
        options->feasibility_tolerance = options->optimality_tolerance;
        options->iteration_limit = policy.repair_iteration_limit;
        return SPACEPDHCG_CUDA_SCVX_REPAIR;
    }
    if (residual > 2.0e-2) {
        options->optimality_tolerance =
            std::min(policy.progress_tolerance_ceiling, requested);
        options->feasibility_tolerance = options->optimality_tolerance;
        options->iteration_limit = policy.progress_iteration_limit;
        return SPACEPDHCG_CUDA_SCVX_PROGRESS;
    }
    if (residual > 5.0e-4) {
        options->optimality_tolerance =
            std::min(policy.refinement_tolerance_ceiling, requested);
        options->feasibility_tolerance = options->optimality_tolerance;
        options->iteration_limit = policy.refinement_iteration_limit;
        return SPACEPDHCG_CUDA_SCVX_REFINEMENT;
    }
    options->optimality_tolerance =
        std::min(policy.polish_tolerance_ceiling, requested);
    options->feasibility_tolerance = options->optimality_tolerance;
    options->iteration_limit = policy.polish_iteration_limit;
    return SPACEPDHCG_CUDA_SCVX_POLISH;
}

double maximum_outer_residual(const ScvxMetrics& metrics) {
    return std::max({
        metrics.dynamics,
        metrics.path,
        metrics.terminal,
        metrics.virtual_control,
    });
}

}  // namespace

struct spacepdhcg_cuda_scvx_driver {
    spacepdhcg_cuda_scvx_problem problem{};
    spacepdhcg_cuda_scvx_options options{};
    double* candidate_states{nullptr};
    double* candidate_controls{nullptr};
    double* replay_states{nullptr};
    double* checkpoint{nullptr};
    ScvxMetrics* device_metrics{nullptr};
    ScvxMetrics* host_metrics{nullptr};
    unsigned long long* device_numeric_fingerprint{nullptr};
    unsigned long long* host_numeric_fingerprint{nullptr};
    spacepdhcg_native_qoco* qoco{nullptr};
    spacepdhcg_native_qoco_report qoco_report{};
    double* primal{nullptr};
    spacepdhcg_cuda_scvx_path_inventory path_inventory{};
    double* dual{nullptr};
    size_t checkpoint_elements{0U};
    cudaEvent_t timer_start{nullptr};
    cudaEvent_t timer_stop{nullptr};
    std::atomic_bool cancelled{false};
    uint64_t allocation_count{0U};
    uint64_t allocation_bytes{0U};
    uint64_t d2h_copy_count{0U};
    uint64_t d2h_bytes{0U};
    uint64_t device_copy_count{0U};
    uint64_t device_copy_bytes{0U};
};

namespace {

void destroy_driver_storage(spacepdhcg_cuda_scvx_driver* driver) {
    if (driver == nullptr) {
        return;
    }
    spacepdhcg_native_qoco_destroy(driver->qoco);
    driver->qoco = nullptr;
    static_cast<void>(cudaEventDestroy(driver->timer_start));
    static_cast<void>(cudaEventDestroy(driver->timer_stop));
    static_cast<void>(cudaFreeHost(driver->host_metrics));
    static_cast<void>(cudaFreeHost(driver->host_numeric_fingerprint));
    static_cast<void>(cudaFree(driver->candidate_states));
    static_cast<void>(cudaFree(driver->candidate_controls));
    static_cast<void>(cudaFree(driver->replay_states));
    static_cast<void>(cudaFree(driver->checkpoint));
    static_cast<void>(cudaFree(driver->device_metrics));
    static_cast<void>(cudaFree(driver->device_numeric_fingerprint));
}

spacepdhcg_cuda_status allocate_driver(
    spacepdhcg_cuda_scvx_driver* driver,
    void** pointer,
    const size_t bytes
) {
    if (bytes == 0U) {
        *pointer = nullptr;
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    const auto status = cudaMalloc(pointer, bytes);
    if (status != cudaSuccess) {
        return status == cudaErrorMemoryAllocation
            ? SPACEPDHCG_CUDA_OUT_OF_MEMORY
            : SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    ++driver->allocation_count;
    driver->allocation_bytes += bytes;
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status time_stop(
    spacepdhcg_cuda_scvx_driver* driver,
    const cudaStream_t stream,
    double* seconds
) {
    auto status = cudaEventRecord(driver->timer_stop, stream);
    if (status == cudaSuccess) {
        status = cudaEventSynchronize(driver->timer_stop);
    }
    float milliseconds = 0.0F;
    if (status == cudaSuccess) {
        status = cudaEventElapsedTime(
            &milliseconds,
            driver->timer_start,
            driver->timer_stop
        );
    }
    if (status != cudaSuccess) {
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    *seconds += static_cast<double>(milliseconds) * 1.0e-3;
    return SPACEPDHCG_CUDA_SUCCESS;
}

spacepdhcg_cuda_status collect_metrics(
    spacepdhcg_cuda_scvx_driver* driver,
    const double* states,
    const double* controls,
    const bool include_virtual,
    const double trust_radius,
    const cudaStream_t stream,
    double* replay_seconds,
    double* d2h_seconds
) {
    auto status = cudaEventRecord(driver->timer_start, stream);
    if (status != cudaSuccess) {
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    auto api_status = launch_replay(
        driver->problem.dynamics,
        driver->problem.intervals,
        view_pointer<const double>(driver->problem.initial_state),
        controls,
        driver->replay_states,
        stream
    );
    if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
        return api_status;
    }
    const auto* virtual_indices = include_virtual
        && driver->problem.virtual_variable_indices.elements > 0U
        ? view_pointer<const int>(driver->problem.virtual_variable_indices)
        : nullptr;
    scvx_metrics_kernel<<<1, 1, 0, stream>>>(
        states,
        controls,
        driver->replay_states,
        view_pointer<const double>(driver->problem.reference_states),
        view_pointer<const double>(driver->problem.reference_controls),
        view_pointer<const double>(driver->problem.target_state),
        driver->primal,
        virtual_indices,
        include_virtual
            ? driver->problem.virtual_variable_indices.elements
            : 0U,
        driver->problem.intervals,
        driver->problem.state_dimension,
        driver->problem.control_dimension,
        static_cast<int>(driver->problem.dynamics.model),
        driver->options.feasibility_penalty,
        driver->options.virtual_penalty,
        trust_radius,
        driver->problem.numeric_update,
        view_pointer<const double>(driver->problem.numeric.scalar_lower),
        view_pointer<const double>(driver->problem.numeric.variable_lower),
        view_pointer<const double>(driver->problem.numeric.variable_upper),
        view_pointer<const int>(driver->problem.state_variable_indices),
        view_pointer<const int>(driver->problem.control_variable_indices),
        driver->device_metrics
    );
    api_status = time_stop(driver, stream, replay_seconds);
    if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
        return api_status;
    }
    status = cudaEventRecord(driver->timer_start, stream);
    if (status != cudaSuccess) {
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    status = cudaMemcpyAsync(
        driver->host_metrics,
        driver->device_metrics,
        sizeof(ScvxMetrics),
        cudaMemcpyDeviceToHost,
        stream
    );
    if (status != cudaSuccess) {
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    ++driver->d2h_copy_count;
    driver->d2h_bytes += sizeof(ScvxMetrics);
    return time_stop(driver, stream, d2h_seconds);
}

spacepdhcg_cuda_status collect_numeric_fingerprint(
    spacepdhcg_cuda_scvx_driver* driver,
    const cudaStream_t stream,
    uint64_t* fingerprint
) {
    if (cudaMemsetAsync(
            driver->device_numeric_fingerprint,
            0,
            sizeof(unsigned long long),
            stream
        ) != cudaSuccess) {
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    const spacepdhcg_accelerator_buffer_view views[] = {
        driver->problem.numeric.quadratic,
        driver->problem.numeric.scalar_constraint,
        driver->problem.numeric.affine_cone,
        driver->problem.numeric.linear_objective,
        driver->problem.numeric.scalar_lower,
        driver->problem.numeric.scalar_upper,
        driver->problem.numeric.affine_offset,
        driver->problem.numeric.variable_lower,
        driver->problem.numeric.variable_upper,
    };
    for (size_t view_index = 0U; view_index < std::size(views); ++view_index) {
        if (views[view_index].elements == 0U) {
            continue;
        }
        const auto blocks = static_cast<unsigned int>(std::min<size_t>(
            256U,
            (views[view_index].elements + 255U) / 256U
        ));
        hash_numeric_kernel<<<blocks, 256, 0, stream>>>(
            view_pointer<const double>(views[view_index]),
            views[view_index].elements,
            0x100000001b3ULL * static_cast<unsigned long long>(view_index + 1U),
            driver->device_numeric_fingerprint
        );
    }
    if (cudaGetLastError() != cudaSuccess
        || cudaMemcpyAsync(
            driver->host_numeric_fingerprint,
            driver->device_numeric_fingerprint,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost,
            stream
        ) != cudaSuccess
        || cudaStreamSynchronize(stream) != cudaSuccess) {
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    ++driver->d2h_copy_count;
    driver->d2h_bytes += sizeof(unsigned long long);
    *fingerprint = static_cast<uint64_t>(*driver->host_numeric_fingerprint);
    return SPACEPDHCG_CUDA_SUCCESS;
}

}  // namespace

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_scvx_update_numeric_async(
    const spacepdhcg_cuda_scvx_problem* problem,
    const double trust_radius,
    const double virtual_penalty,
    const spacepdhcg_accelerator_stream stream
) {
    if (problem == nullptr
        || problem->abi_version != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        || problem->numeric_update.abi_version
               != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        || !(trust_radius > 0.0) || !std::isfinite(trust_radius)
        || !(virtual_penalty >= 0.0) || !std::isfinite(virtual_penalty)
        || !(problem->numeric_update.conditioning_log10_span >= 0.0)
        || !std::isfinite(problem->numeric_update.conditioning_log10_span)
        || stream.device.type != SPACEPDHCG_DEVICE_CUDA
        || stream.device.id < 0) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    const size_t tracked_variables =
        (problem->intervals + 1U) * problem->state_dimension
        + problem->intervals * problem->control_dimension;
    if (problem->numeric_update.quadratic_diagonal_positions.elements
            != tracked_variables) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    update_scvx_numeric_kernel<<<
        1, 1, 0, reinterpret_cast<cudaStream_t>(stream.native_handle)
    >>>(*problem, trust_radius, virtual_penalty);
    if (cudaGetLastError() != cudaSuccess) {
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    if (problem->numeric_update.conditioning_log10_span == 0.0) {
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    const size_t row_count = problem->intervals * problem->state_dimension;
    const size_t work = std::max(
        problem->numeric.scalar_constraint.elements,
        row_count
    );
    constexpr size_t threads = 256U;
    condition_dynamics_rows_kernel<<<
        static_cast<unsigned int>((work + threads - 1U) / threads),
        threads,
        0,
        reinterpret_cast<cudaStream_t>(stream.native_handle)
    >>>(*problem);
    return cudaGetLastError() == cudaSuccess
        ? SPACEPDHCG_CUDA_SUCCESS
        : SPACEPDHCG_CUDA_RUNTIME_ERROR;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_scvx_driver_create(
    const spacepdhcg_cuda_scvx_problem* problem,
    const spacepdhcg_cuda_scvx_options* options,
    spacepdhcg_cuda_scvx_driver** driver
) {
    if (problem == nullptr || options == nullptr || driver == nullptr
        || problem->abi_version != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        || problem->numeric_update.abi_version
               != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        || options->abi_version != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        || problem->workspace == nullptr || problem->intervals == 0U
        || problem->state_dimension == 0U
        || problem->control_dimension == 0U
        || options->maximum_outer_iterations == 0U
        || options->minimum_outer_iterations > options->maximum_outer_iterations
        || !(options->convergence_tolerance > 0.0)
        || !(options->step_tolerance > 0.0)
        || !(options->initial_trust_radius > 0.0)
        || !(options->minimum_trust_radius > 0.0)
        || options->minimum_trust_radius > options->initial_trust_radius
        || options->initial_trust_radius > options->maximum_trust_radius
        || (options->fixed_inner_tolerance > 0.0
            && options->fixed_inner_iteration_limit == 0U)
        || ((options->policy == SPACEPDHCG_CUDA_SCVX_PURE_QOCO
             || options->policy == SPACEPDHCG_CUDA_SCVX_HYBRID_QOCO)
            && (problem->canonical_structure.abi_version
                    != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
                || problem->canonical_structure.topology_fingerprint
                    != problem->topology_fingerprint))
        || (problem->dynamics.model
                == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF
            && (!(problem->numeric_update.maximum_thrust > 0.0)
                || !(problem->numeric_update.maximum_torque > 0.0)
                || !(problem->numeric_update.maximum_angular_rate > 0.0)
                || !(problem->numeric_update.tilt_cosine > 0.0)
                || !(problem->numeric_update.glide_slope_tangent > 0.0)))) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    *driver = nullptr;
    auto* result = new (std::nothrow) spacepdhcg_cuda_scvx_driver{};
    if (result == nullptr) {
        return SPACEPDHCG_CUDA_OUT_OF_MEMORY;
    }
    result->problem = *problem;
    result->options = *options;
    if (!(result->options.adaptive_epsilon_max > 0.0)) {
        result->options.adaptive_epsilon_max = 1.0e-3;
    }
    if (!(result->options.adaptive_epsilon_floor > 0.0)) {
        result->options.adaptive_epsilon_floor = 1.0e-8;
    }
    if (!(result->options.adaptive_epsilon_0 > 0.0)) {
        result->options.adaptive_epsilon_0 = 1.0e-3;
    }
    if (!(result->options.adaptive_coefficient > 0.0)) {
        result->options.adaptive_coefficient = 0.2;
    }
    if (!(result->options.adaptive_alpha > 0.0)) {
        result->options.adaptive_alpha = 0.5;
    }
    if (!(result->options.adaptive_gamma > 0.0)
        || result->options.adaptive_gamma >= 1.0) {
        result->options.adaptive_gamma = 0.6;
    }
    if (!(result->options.repair_tolerance_ceiling > 0.0)) {
        result->options.repair_tolerance_ceiling = 1.0e-2;
    }
    if (!(result->options.progress_tolerance_ceiling > 0.0)) {
        result->options.progress_tolerance_ceiling = 2.0e-3;
    }
    if (!(result->options.refinement_tolerance_ceiling > 0.0)) {
        result->options.refinement_tolerance_ceiling = 1.0e-5;
    }
    if (!(result->options.polish_tolerance_ceiling > 0.0)) {
        result->options.polish_tolerance_ceiling = 1.0e-8;
    }
    if (result->options.repair_iteration_limit == 0U) {
        result->options.repair_iteration_limit = 5'000U;
    }
    if (result->options.progress_iteration_limit == 0U) {
        result->options.progress_iteration_limit = 25'000U;
    }
    if (result->options.refinement_iteration_limit == 0U) {
        result->options.refinement_iteration_limit = 100'000U;
    }
    if (result->options.polish_iteration_limit == 0U) {
        result->options.polish_iteration_limit = 1'000'000U;
    }
    if (!(result->options.resolve_trigger_multiple > 0.0)) {
        result->options.resolve_trigger_multiple = 5.0;
    }
    if (!(result->options.resolve_refinement_factor > 0.0)
        || result->options.resolve_refinement_factor >= 1.0) {
        result->options.resolve_refinement_factor = 0.1;
    }
    if (!(result->options.resolve_minimum_tolerance > 0.0)) {
        result->options.resolve_minimum_tolerance = 1.0e-8;
    }
    if (!(result->options.strong_agreement_threshold > 0.0)) {
        result->options.strong_agreement_threshold = 0.75;
    }
    if (!(result->options.near_boundary_fraction > 0.0)) {
        result->options.near_boundary_fraction = 0.8;
    }
    if (!(result->options.final_polish_tolerance > 0.0)) {
        result->options.final_polish_tolerance = 1.0e-8;
    }
    if (result->options.final_polish_iteration_limit == 0U) {
        result->options.final_polish_iteration_limit = 1'000'000U;
    }
    spacepdhcg_cuda_pointer_snapshot pointers{};
    auto api_status = spacepdhcg_cuda_workspace_pointer_snapshot(
        problem->workspace,
        &pointers
    );
    if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
        delete result;
        return api_status;
    }
    result->primal = reinterpret_cast<double*>(pointers.primal);
    result->dual = reinterpret_cast<double*>(pointers.dual);
    size_t checkpoint_bytes = 0U;
    api_status = spacepdhcg_cuda_workspace_checkpoint_bytes(
        problem->workspace,
        &checkpoint_bytes
    );
    if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
        delete result;
        return api_status;
    }
    result->checkpoint_elements = checkpoint_bytes / sizeof(double);
    const size_t state_elements =
        (problem->intervals + 1U) * problem->state_dimension;
    const size_t control_elements =
        problem->intervals * problem->control_dimension;
    const struct AllocationRequest {
        void** pointer;
        size_t bytes;
    } requests[] = {
        {reinterpret_cast<void**>(&result->candidate_states),
         state_elements * sizeof(double)},
        {reinterpret_cast<void**>(&result->candidate_controls),
         control_elements * sizeof(double)},
        {reinterpret_cast<void**>(&result->replay_states),
         state_elements * sizeof(double)},
        {reinterpret_cast<void**>(&result->checkpoint), checkpoint_bytes},
        {reinterpret_cast<void**>(&result->device_metrics), sizeof(ScvxMetrics)},
        {reinterpret_cast<void**>(&result->device_numeric_fingerprint),
         sizeof(unsigned long long)},
    };
    for (const auto& request : requests) {
        api_status = allocate_driver(
            result,
            request.pointer,
            request.bytes
        );
        if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
            destroy_driver_storage(result);
            delete result;
            return api_status;
        }
    }
    auto cuda_status = cudaHostAlloc(
        reinterpret_cast<void**>(&result->host_metrics),
        sizeof(ScvxMetrics),
        cudaHostAllocPortable
    );
    if (cuda_status != cudaSuccess) {
        destroy_driver_storage(result);
        delete result;
        return cuda_status == cudaErrorMemoryAllocation
            ? SPACEPDHCG_CUDA_OUT_OF_MEMORY
            : SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    ++result->allocation_count;
    result->allocation_bytes += sizeof(ScvxMetrics);
    cuda_status = cudaHostAlloc(
        reinterpret_cast<void**>(&result->host_numeric_fingerprint),
        sizeof(unsigned long long),
        cudaHostAllocPortable
    );
    if (cuda_status != cudaSuccess) {
        destroy_driver_storage(result);
        delete result;
        return cuda_status == cudaErrorMemoryAllocation
            ? SPACEPDHCG_CUDA_OUT_OF_MEMORY
            : SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    ++result->allocation_count;
    result->allocation_bytes += sizeof(unsigned long long);
    cuda_status = cudaEventCreate(&result->timer_start);
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaEventCreate(&result->timer_stop);
    }
    if (cuda_status != cudaSuccess) {
        destroy_driver_storage(result);
        delete result;
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    *driver = result;
    return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_scvx_driver_solve(
    spacepdhcg_cuda_scvx_driver* driver,
    const spacepdhcg_accelerator_stream stream,
    spacepdhcg_cuda_scvx_iteration* iterations,
    const size_t iteration_capacity,
    spacepdhcg_cuda_scvx_result* result
) {
    if (driver == nullptr || result == nullptr
        || (iterations == nullptr && iteration_capacity > 0U)
        || stream.device.type != SPACEPDHCG_DEVICE_CUDA
        || stream.device.id != driver->problem.reference_states.device.id
        || iteration_capacity < driver->options.maximum_outer_iterations) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::memset(result, 0, sizeof(*result));
    result->abi_version = SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;
    result->status = SPACEPDHCG_CUDA_SCVX_MAXIMUM_ITERATIONS;
    result->final_trust_radius = driver->options.initial_trust_radius;
    result->used_declared_stream = 1;
    spacepdhcg_cuda_diagnostics transfer_before{};
    auto api_status = spacepdhcg_cuda_workspace_diagnostics(
        driver->problem.workspace,
        &transfer_before
    );
    if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
        return api_status;
    }
    const uint64_t driver_d2h_count_before = driver->d2h_copy_count;
    const uint64_t driver_d2h_bytes_before = driver->d2h_bytes;
    const uint64_t driver_device_count_before = driver->device_copy_count;
    const uint64_t driver_device_bytes_before = driver->device_copy_bytes;
    const auto native = reinterpret_cast<cudaStream_t>(stream.native_handle);
    const size_t state_elements =
        (driver->problem.intervals + 1U)
        * driver->problem.state_dimension;
    const size_t control_elements =
        driver->problem.intervals * driver->problem.control_dimension;
    const auto started = std::chrono::steady_clock::now();

    auto cuda_status = cudaMemcpyAsync(
        driver->candidate_states,
        view_pointer<const double>(driver->problem.reference_states),
        state_elements * sizeof(double),
        cudaMemcpyDeviceToDevice,
        native
    );
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaMemcpyAsync(
            driver->candidate_controls,
            view_pointer<const double>(driver->problem.reference_controls),
            control_elements * sizeof(double),
            cudaMemcpyDeviceToDevice,
            native
        );
    }
    if (cuda_status != cudaSuccess) {
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    driver->device_copy_count += 2U;
    driver->device_copy_bytes +=
        (state_elements + control_elements) * sizeof(double);
    api_status = collect_metrics(
        driver,
        driver->candidate_states,
        driver->candidate_controls,
        false,
        driver->options.initial_trust_radius,
        native,
        &result->replay_seconds,
        &result->d2h_seconds
    );
    if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
        return api_status;
    }
    ScvxMetrics current = *driver->host_metrics;
    const double initial_outer_residual = maximum_outer_residual(current);
    double trust_radius = driver->options.initial_trust_radius;
    spacepdhcg_cuda_diagnostics last_diagnostics{};

    for (uint32_t outer = 0;
         outer < driver->options.maximum_outer_iterations;
         ++outer) {
        if (driver->cancelled.load(std::memory_order_acquire)) {
            result->status = SPACEPDHCG_CUDA_SCVX_CANCELLED;
            break;
        }
        const spacepdhcg_accelerator_buffer_view checkpoint_view{
            driver->checkpoint,
            stream.device,
            SPACEPDHCG_SCALAR_FLOAT64,
            driver->checkpoint_elements,
            0U,
            1,
            SPACEPDHCG_ACCESS_READ_WRITE,
        };
        api_status = spacepdhcg_cuda_workspace_checkpoint_async(
            driver->problem.workspace,
            checkpoint_view,
            stream
        );
        if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
            result->status = SPACEPDHCG_CUDA_SCVX_INNER_FAILURE;
            return api_status;
        }
        api_status = spacepdhcg_cuda_workspace_wait(
            driver->problem.workspace
        );
        if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
            result->status = SPACEPDHCG_CUDA_SCVX_INNER_FAILURE;
            return api_status;
        }

        cuda_status = cudaEventRecord(driver->timer_start, native);
        if (cuda_status != cudaSuccess) {
            return SPACEPDHCG_CUDA_RUNTIME_ERROR;
        }
        api_status = spacepdhcg_cuda_variational_rk4_async(
            &driver->problem.dynamics,
            &driver->problem.variational,
            stream
        );
        if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
            api_status = spacepdhcg_cuda_fill_dynamics_csc_async(
                &driver->problem.dynamics_fill,
                stream
            );
        }
        if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
            api_status = spacepdhcg_cuda_scvx_update_numeric_async(
                &driver->problem,
                trust_radius,
                driver->options.virtual_penalty,
                stream
            );
        }
        if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
            result->status = SPACEPDHCG_CUDA_SCVX_INNER_FAILURE;
            return api_status;
        }
        api_status = time_stop(
            driver,
            native,
            &result->coefficient_seconds
        );
        if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
            return api_status;
        }

        api_status = spacepdhcg_cuda_workspace_update_async(
            driver->problem.workspace,
            driver->problem.topology_fingerprint,
            &driver->problem.numeric,
            stream
        );
        if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
            result->status = SPACEPDHCG_CUDA_SCVX_INNER_FAILURE;
            return api_status;
        }
        api_status = spacepdhcg_cuda_workspace_wait(
            driver->problem.workspace
        );
        if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
            result->status = SPACEPDHCG_CUDA_SCVX_INNER_FAILURE;
            return api_status;
        }
        uint64_t numeric_fingerprint = 0U;
        api_status = collect_numeric_fingerprint(
            driver,
            native,
            &numeric_fingerprint
        );
        if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
            result->status = SPACEPDHCG_CUDA_SCVX_INNER_FAILURE;
            return api_status;
        }
        if (outer > 0U) {
            const auto requested_warm_start = driver->options.warm_start_mode;
            if (requested_warm_start == SPACEPDHCG_CUDA_WARM_START_PRIMAL) {
                const size_t dual_elements =
                    driver->problem.numeric.scalar_lower.elements
                    + driver->problem.numeric.affine_offset.elements;
                if (cudaMemsetAsync(
                        driver->dual,
                        0,
                        dual_elements * sizeof(double),
                        native
                    ) != cudaSuccess) {
                    result->status = SPACEPDHCG_CUDA_SCVX_INNER_FAILURE;
                    return SPACEPDHCG_CUDA_RUNTIME_ERROR;
                }
            }
            const auto resident_warm_start =
                requested_warm_start == SPACEPDHCG_CUDA_WARM_START_NONE
                ? SPACEPDHCG_CUDA_WARM_START_NONE
                : SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED;
            api_status = spacepdhcg_cuda_workspace_warm_start_async(
                driver->problem.workspace,
                resident_warm_start,
                nullptr,
                stream
            );
            if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                api_status = spacepdhcg_cuda_workspace_wait(
                    driver->problem.workspace
                );
            }
            if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
                result->status = SPACEPDHCG_CUDA_SCVX_INNER_FAILURE;
                return api_status;
            }
        }
        spacepdhcg_cuda_solve_options solve_options{};
        const auto phase = forcing_phase(
            maximum_outer_residual(current),
            outer,
            driver->options,
            &solve_options
        );
        if (driver->options.policy == SPACEPDHCG_CUDA_SCVX_FIXED_TIGHT) {
            solve_options.optimality_tolerance =
                driver->options.fixed_inner_tolerance;
            solve_options.feasibility_tolerance =
                driver->options.fixed_inner_tolerance;
            solve_options.iteration_limit =
                driver->options.fixed_inner_iteration_limit;
        } else if (driver->options.policy
                   == SPACEPDHCG_CUDA_SCVX_FIXED_LOOSE) {
            solve_options.optimality_tolerance =
                driver->options.fixed_inner_tolerance;
            solve_options.feasibility_tolerance =
                driver->options.fixed_inner_tolerance;
            solve_options.iteration_limit =
                driver->options.fixed_inner_iteration_limit;
        } else if (driver->options.fixed_inner_tolerance > 0.0) {
            solve_options.optimality_tolerance =
                driver->options.fixed_inner_tolerance;
            solve_options.feasibility_tolerance =
                driver->options.fixed_inner_tolerance;
            solve_options.iteration_limit =
                driver->options.fixed_inner_iteration_limit;
        }
        const bool pure_qoco =
            driver->options.policy == SPACEPDHCG_CUDA_SCVX_PURE_QOCO;
        const bool hybrid_qoco =
            driver->options.policy == SPACEPDHCG_CUDA_SCVX_HYBRID_QOCO;
        bool qoco_used = pure_qoco;
        if (pure_qoco) {
            if (driver->qoco == nullptr) {
                api_status = spacepdhcg_native_qoco_create(
                    &driver->problem,
                    native,
                    &driver->qoco
                );
            }
            if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                api_status = spacepdhcg_native_qoco_update_solve(
                    driver->qoco,
                    &driver->problem,
                    native,
                    driver->options.warm_start_mode,
                    driver->primal,
                    driver->dual,
                    &driver->qoco_report
                );
            }
            if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                last_diagnostics = {};
                last_diagnostics.abi_version =
                    SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;
                last_diagnostics.termination =
                    SPACEPDHCG_CUDA_TERMINATION_OPTIMAL;
                last_diagnostics.natural_residual_inf = std::max(
                    driver->qoco_report.primal_residual,
                    driver->qoco_report.dual_residual
                );
                last_diagnostics.stationarity_inf =
                    driver->qoco_report.dual_residual;
                last_diagnostics.relative_primal_residual =
                    driver->qoco_report.primal_residual;
                last_diagnostics.relative_dual_residual =
                    driver->qoco_report.dual_residual;
                last_diagnostics.iterations =
                    static_cast<uint64_t>(driver->qoco_report.iterations);
            }
        } else {
            api_status = spacepdhcg_cuda_workspace_solve_async(
                driver->problem.workspace,
                &solve_options,
                stream
            );
            if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                api_status = spacepdhcg_cuda_workspace_wait(
                    driver->problem.workspace
                );
            }
            if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                api_status = spacepdhcg_cuda_workspace_residuals_async(
                    driver->problem.workspace,
                    stream
                );
            }
            if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                api_status = spacepdhcg_cuda_workspace_wait(
                    driver->problem.workspace
                );
            }
            if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                api_status = spacepdhcg_cuda_workspace_diagnostics(
                    driver->problem.workspace,
                    &last_diagnostics
                );
            }
            if (api_status == SPACEPDHCG_CUDA_SUCCESS && hybrid_qoco) {
                result->hybrid_handoff_eligible =
                    last_diagnostics.natural_residual_inf <= 1.0e-6 ? 1 : 0;
                if (result->hybrid_handoff_eligible != 0) {
                    if (driver->qoco == nullptr) {
                        api_status = spacepdhcg_native_qoco_create(
                            &driver->problem,
                            native,
                            &driver->qoco
                        );
                    }
                    if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                        api_status = spacepdhcg_native_qoco_update_solve(
                            driver->qoco,
                            &driver->problem,
                            native,
                            driver->options.warm_start_mode,
                            driver->primal,
                            driver->dual,
                            &driver->qoco_report
                        );
                    }
                    if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                        qoco_used = true;
                        last_diagnostics.termination =
                            SPACEPDHCG_CUDA_TERMINATION_OPTIMAL;
                        last_diagnostics.natural_residual_inf = std::max(
                            driver->qoco_report.primal_residual,
                            driver->qoco_report.dual_residual
                        );
                        last_diagnostics.stationarity_inf =
                            driver->qoco_report.dual_residual;
                        last_diagnostics.relative_primal_residual =
                            driver->qoco_report.primal_residual;
                        last_diagnostics.relative_dual_residual =
                            driver->qoco_report.dual_residual;
                        last_diagnostics.iterations += static_cast<uint64_t>(
                            driver->qoco_report.iterations
                        );
                    }
                }
            }
        }
        if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
            static_cast<void>(spacepdhcg_cuda_workspace_restore_async(
                driver->problem.workspace,
                driver->problem.topology_fingerprint,
                checkpoint_view,
                stream
            ));
            static_cast<void>(spacepdhcg_cuda_workspace_wait(
                driver->problem.workspace
            ));
            result->status = SPACEPDHCG_CUDA_SCVX_INNER_FAILURE;
            if (qoco_used || hybrid_qoco) {
                result->qoco_failure =
                    driver->qoco == nullptr
                    ? (api_status == SPACEPDHCG_CUDA_OUT_OF_MEMORY
                        ? SPACEPDHCG_CUDA_QOCO_FAILURE_OUT_OF_MEMORY
                        : SPACEPDHCG_CUDA_QOCO_FAILURE_UNAVAILABLE)
                    : driver->qoco_report.failure;
                result->qoco_conversion_seconds =
                    driver->qoco_report.conversion_seconds;
                result->qoco_setup_seconds =
                    driver->qoco_report.setup_seconds;
                result->qoco_update_seconds =
                    driver->qoco_report.update_seconds;
                result->qoco_solve_seconds =
                    driver->qoco_report.solve_seconds;
            }
            return api_status;
        }
        if (pure_qoco) {
            result->update_seconds = driver->qoco_report.update_seconds;
            result->solve_seconds = driver->qoco_report.solve_seconds;
            result->qoco_conversion_seconds =
                driver->qoco_report.conversion_seconds;
            result->qoco_setup_seconds = driver->qoco_report.setup_seconds;
            result->qoco_update_seconds = driver->qoco_report.update_seconds;
            result->qoco_solve_seconds = driver->qoco_report.solve_seconds;
            result->qoco_workspace_creations =
                driver->qoco_report.workspace_creations;
            result->qoco_numeric_updates =
                driver->qoco_report.numeric_updates;
            result->qoco_dual_discarded =
                driver->qoco_report.dual_discarded;
            result->qoco_failure = driver->qoco_report.failure;
        } else {
            result->update_seconds += last_diagnostics.update_seconds;
            result->scaling_seconds += last_diagnostics.scaling_seconds;
            result->solve_seconds += std::max(
                0.0,
                last_diagnostics.solve_seconds
                    - last_diagnostics.recovery_seconds
            );
            result->recovery_seconds += last_diagnostics.recovery_seconds;
            result->residual_seconds += last_diagnostics.residual_seconds;
            if (qoco_used) {
                result->qoco_conversion_seconds +=
                    driver->qoco_report.conversion_seconds;
                result->qoco_setup_seconds += driver->qoco_report.setup_seconds;
                result->qoco_update_seconds += driver->qoco_report.update_seconds;
                result->qoco_solve_seconds += driver->qoco_report.solve_seconds;
                result->qoco_workspace_creations =
                    driver->qoco_report.workspace_creations;
                result->qoco_numeric_updates =
                    driver->qoco_report.numeric_updates;
                result->qoco_dual_discarded =
                    driver->qoco_report.dual_discarded;
                result->qoco_failure = driver->qoco_report.failure;
            }
        }
        result->inner_iterations += last_diagnostics.iterations;
        result->recovery_iterations +=
            last_diagnostics.recovery_iterations;

        cuda_status = cudaEventRecord(driver->timer_start, native);
        if (cuda_status != cudaSuccess) {
            return SPACEPDHCG_CUDA_RUNTIME_ERROR;
        }
        gather_scvx_candidate_kernel<<<1, 256, 0, native>>>(
            driver->primal,
            view_pointer<const int>(driver->problem.state_variable_indices),
            view_pointer<const int>(driver->problem.control_variable_indices),
            driver->candidate_states,
            driver->candidate_controls,
            state_elements,
            control_elements
        );
        if (cudaGetLastError() != cudaSuccess) {
            return SPACEPDHCG_CUDA_RUNTIME_ERROR;
        }
        api_status = collect_metrics(
            driver,
            driver->candidate_states,
            driver->candidate_controls,
            true,
            trust_radius,
            native,
            &result->replay_seconds,
            &result->d2h_seconds
        );
        if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
            return api_status;
        }
        ScvxMetrics candidate = *driver->host_metrics;
        const auto acceptance_started = std::chrono::steady_clock::now();
        double predicted = current.merit - candidate.model_merit;
        double actual = current.merit - candidate.merit;
        double ratio = predicted > 1.0e-12
            ? actual / predicted
            : -std::numeric_limits<double>::infinity();
        const double current_outer = maximum_outer_residual(current);
        const double candidate_outer = maximum_outer_residual(candidate);
        bool restoration = candidate_outer
            <= driver->options.restoration_reduction * current_outer;
        bool accepted =
            last_diagnostics.termination
                == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL
            && (!pure_qoco
                || last_diagnostics.natural_residual_inf
                    <= solve_options.optimality_tolerance)
            && ((actual > 1.0e-10
                 && std::isfinite(ratio)
                 && ratio >= driver->options.acceptance_threshold)
                || restoration);
        bool re_solved = false;
        uint64_t resolve_numeric_fingerprint = numeric_fingerprint;
        bool resolve_fingerprint_match = true;
        if (!pure_qoco
            && !accepted
            && last_diagnostics.natural_residual_inf
                > driver->options.resolve_trigger_multiple
                    * solve_options.optimality_tolerance) {
            for (uint32_t resolve = 0;
                 resolve < driver->options.maximum_resolves_per_iteration;
                 ++resolve) {
                api_status = collect_numeric_fingerprint(
                    driver,
                    native,
                    &resolve_numeric_fingerprint
                );
                if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
                    result->status = SPACEPDHCG_CUDA_SCVX_INNER_FAILURE;
                    return api_status;
                }
                resolve_fingerprint_match =
                    resolve_fingerprint_match
                    && resolve_numeric_fingerprint == numeric_fingerprint;
                if (!resolve_fingerprint_match) {
                    result->status = SPACEPDHCG_CUDA_SCVX_INNER_FAILURE;
                    return SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH;
                }
                solve_options.optimality_tolerance = std::max(
                    driver->options.resolve_minimum_tolerance,
                    driver->options.resolve_refinement_factor
                        * solve_options.optimality_tolerance
                );
                solve_options.feasibility_tolerance =
                    solve_options.optimality_tolerance;
                solve_options.iteration_limit = std::max<uint64_t>(
                    solve_options.iteration_limit,
                    350'000U
                );
                api_status = spacepdhcg_cuda_workspace_warm_start_async(
                    driver->problem.workspace,
                    SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED,
                    nullptr,
                    stream
                );
                if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                    api_status = spacepdhcg_cuda_workspace_wait(
                        driver->problem.workspace
                    );
                }
                if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                    api_status = spacepdhcg_cuda_workspace_solve_async(
                        driver->problem.workspace,
                        &solve_options,
                        stream
                    );
                }
                if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                    api_status = spacepdhcg_cuda_workspace_wait(
                        driver->problem.workspace
                    );
                }
                if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                    api_status = spacepdhcg_cuda_workspace_residuals_async(
                        driver->problem.workspace,
                        stream
                    );
                }
                if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                    api_status = spacepdhcg_cuda_workspace_wait(
                        driver->problem.workspace
                    );
                }
                if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                    api_status = spacepdhcg_cuda_workspace_diagnostics(
                        driver->problem.workspace,
                        &last_diagnostics
                    );
                }
                if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
                    result->status = SPACEPDHCG_CUDA_SCVX_INNER_FAILURE;
                    return api_status;
                }
                re_solved = true;
                ++result->resolved_steps;
                result->solve_seconds += last_diagnostics.solve_seconds;
                result->recovery_seconds += last_diagnostics.recovery_seconds;
                result->residual_seconds += last_diagnostics.residual_seconds;
                result->inner_iterations += last_diagnostics.iterations;
                result->recovery_iterations +=
                    last_diagnostics.recovery_iterations;
                if (last_diagnostics.natural_residual_inf
                    <= driver->options.resolve_trigger_multiple
                        * solve_options.optimality_tolerance) {
                    break;
                }
            }
        }
        if (re_solved) {
            gather_scvx_candidate_kernel<<<1, 256, 0, native>>>(
                driver->primal,
                view_pointer<const int>(driver->problem.state_variable_indices),
                view_pointer<const int>(driver->problem.control_variable_indices),
                driver->candidate_states,
                driver->candidate_controls,
                state_elements,
                control_elements
            );
            if (cudaGetLastError() != cudaSuccess) {
                return SPACEPDHCG_CUDA_RUNTIME_ERROR;
            }
            api_status = collect_metrics(
                driver,
                driver->candidate_states,
                driver->candidate_controls,
                true,
                trust_radius,
                native,
                &result->replay_seconds,
                &result->d2h_seconds
            );
            if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
                return api_status;
            }
            candidate = *driver->host_metrics;
            predicted = current.merit - candidate.model_merit;
            actual = current.merit - candidate.merit;
            ratio = predicted > 1.0e-12
                ? actual / predicted
                : -std::numeric_limits<double>::infinity();
            restoration = maximum_outer_residual(candidate)
                <= driver->options.restoration_reduction * current_outer;
            accepted =
                last_diagnostics.termination
                    == SPACEPDHCG_CUDA_TERMINATION_OPTIMAL
                && ((actual > 1.0e-10
                     && std::isfinite(ratio)
                     && ratio >= driver->options.acceptance_threshold)
                    || restoration);
        }
        const bool retained_converged =
            maximum_outer_residual(current)
                <= driver->options.convergence_tolerance
            && current.step <= driver->options.step_tolerance;
        if (retained_converged) {
            accepted = false;
            restoration = false;
        }
        const double radius_before = trust_radius;
        const ScvxMetrics retained_before = current;
        auto trust_action = SPACEPDHCG_CUDA_SCVX_TRUST_RETAIN;
        if (accepted) {
            cuda_status = cudaMemcpyAsync(
                view_pointer<double>(driver->problem.reference_states),
                driver->replay_states,
                state_elements * sizeof(double),
                cudaMemcpyDeviceToDevice,
                native
            );
            if (cuda_status == cudaSuccess) {
                cuda_status = cudaMemcpyAsync(
                    view_pointer<double>(driver->problem.reference_controls),
                    driver->candidate_controls,
                    control_elements * sizeof(double),
                    cudaMemcpyDeviceToDevice,
                    native
                );
            }
            if (cuda_status != cudaSuccess) {
                return SPACEPDHCG_CUDA_RUNTIME_ERROR;
            }
            driver->device_copy_count += 2U;
            driver->device_copy_bytes +=
                (state_elements + control_elements) * sizeof(double);
            current = candidate;
            ++result->accepted_steps;
            if (pure_qoco) {
                spacepdhcg_native_qoco_accept(driver->qoco);
            }
            if (ratio >= driver->options.strong_agreement_threshold
                && candidate.step
                    >= driver->options.near_boundary_fraction
                        * std::max(1.0e-12, trust_radius)) {
                trust_radius = std::min(
                    driver->options.maximum_trust_radius,
                    driver->options.expansion_factor * trust_radius
                );
                trust_action = SPACEPDHCG_CUDA_SCVX_TRUST_EXPAND;
            }
        } else {
            if (!retained_converged) {
                ++result->rejected_steps;
                api_status = spacepdhcg_cuda_workspace_restore_async(
                    driver->problem.workspace,
                    driver->problem.topology_fingerprint,
                    checkpoint_view,
                    stream
                );
                if (api_status == SPACEPDHCG_CUDA_SUCCESS) {
                    api_status = spacepdhcg_cuda_workspace_wait(
                        driver->problem.workspace
                    );
                }
                if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
                    result->status = SPACEPDHCG_CUDA_SCVX_INNER_FAILURE;
                    return api_status;
                }
                trust_radius = std::max(
                    driver->options.minimum_trust_radius,
                    driver->options.shrink_factor * trust_radius
                );
                trust_action = SPACEPDHCG_CUDA_SCVX_TRUST_SHRINK;
            }
        }
        result->acceptance_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - acceptance_started
        ).count();
        iterations[outer] = spacepdhcg_cuda_scvx_iteration{
            outer,
            phase,
            solve_options.optimality_tolerance,
            last_diagnostics.natural_residual_inf,
            last_diagnostics.iterations,
            radius_before,
            trust_radius,
            predicted,
            actual,
            ratio,
            candidate.step / std::max(1.0e-12, radius_before),
            candidate.model_merit,
            candidate.virtual_control,
            candidate.dynamics,
            candidate.path,
            candidate.terminal,
            accepted ? 1 : 0,
            accepted && restoration ? 1 : 0,
            re_solved ? 1 : 0,
            last_diagnostics.scaling_refreshed,
            last_diagnostics.recovery_count > 0U ? 1 : 0,
            last_diagnostics.relative_primal_residual,
            last_diagnostics.relative_dual_residual,
            last_diagnostics.complementarity_inf,
            last_diagnostics.scaling_min,
            last_diagnostics.scaling_max,
            6U * (
                last_diagnostics.iterations
                + last_diagnostics.recovery_iterations
            ),
            2U * last_diagnostics.iterations
                + last_diagnostics.recovery_iterations,
            numeric_fingerprint,
            resolve_numeric_fingerprint,
            resolve_fingerprint_match ? 1 : 0,
            trust_action,
            pure_qoco
                ? (driver->qoco_report.warm_primal_accepted != 0
                    ? SPACEPDHCG_CUDA_WARM_START_PRIMAL
                    : SPACEPDHCG_CUDA_WARM_START_NONE)
                : (outer == 0U
                    ? SPACEPDHCG_CUDA_WARM_START_NONE
                    : driver->options.warm_start_mode),
            last_diagnostics.recovery_outcome_reason,
            last_diagnostics.natural_residual_inf
                    <= driver->options.resolve_trigger_multiple
                        * solve_options.optimality_tolerance
                ? 1
                : 0,
            driver->options.policy == SPACEPDHCG_CUDA_SCVX_ADAPTIVE_POLISH
                    && phase == SPACEPDHCG_CUDA_SCVX_POLISH
                ? 1
                : 0,
            retained_before.merit,
            candidate.merit,
            candidate.model_merit,
            retained_before.dynamics,
            retained_before.path,
            retained_before.terminal,
            last_diagnostics.scalar_primal_violation_inf,
            last_diagnostics.box_violation_inf,
            last_diagnostics.affine_cone_distance_inf,
            last_diagnostics.stationarity_inf,
            last_diagnostics.natural_residual_inf,
            last_diagnostics.recovery_attempt_count,
            last_diagnostics.recovery_count,
            last_diagnostics.recovery_rejected_count,
            last_diagnostics.recovery_seconds,
            last_diagnostics.recovery_iterations,
            last_diagnostics.recovery_initial_residual,
            last_diagnostics.recovery_final_residual,
            last_diagnostics.recovery_final_primal_residual,
            last_diagnostics.recovery_final_stationarity,
            last_diagnostics.recovery_final_complementarity,
            candidate.maximum_stage_trust_distance,
            candidate.terminal_trust_distance,
        };
        result->outer_iterations = outer + 1U;
        if (outer + 1U >= driver->options.minimum_outer_iterations
            && maximum_outer_residual(current)
                <= driver->options.convergence_tolerance
            && current.step <= driver->options.step_tolerance
            && (result->accepted_steps > 0U
                || initial_outer_residual
                    <= driver->options.convergence_tolerance)) {
            result->status = SPACEPDHCG_CUDA_SCVX_CONVERGED;
            break;
        }
        if (!accepted
            && trust_radius <= driver->options.minimum_trust_radius
                * (1.0 + 1.0e-12)) {
            result->status = SPACEPDHCG_CUDA_SCVX_TRUST_REGION_EXHAUSTED;
            break;
        }
    }
    api_status = collect_metrics(
        driver,
        view_pointer<const double>(driver->problem.reference_states),
        view_pointer<const double>(driver->problem.reference_controls),
        false,
        trust_radius,
        native,
        &result->replay_seconds,
        &result->d2h_seconds
    );
    if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
        return api_status;
    }
    current = *driver->host_metrics;
    if (maximum_outer_residual(current) <= driver->options.convergence_tolerance
        && current.step <= driver->options.step_tolerance) {
        result->status = SPACEPDHCG_CUDA_SCVX_CONVERGED;
    }
    result->objective = current.objective;
    result->canonical_residual =
        result->accepted_steps == 0U
            && initial_outer_residual
                <= driver->options.convergence_tolerance
        ? 0.0
        : last_diagnostics.natural_residual_inf;
    result->dynamics_defect = current.dynamics;
    result->path_violation = current.path;
    driver->path_inventory = spacepdhcg_cuda_scvx_path_inventory{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        current.path_thrust,
        current.path_mass,
        current.path_altitude,
    };
    result->terminal_residual = current.terminal;
    result->virtual_control = current.virtual_control;
    result->trajectory_step = current.step;
    result->final_trust_radius = trust_radius;
    result->cqp_total_seconds =
        result->update_seconds + result->scaling_seconds
        + result->solve_seconds + result->residual_seconds
        + result->qoco_conversion_seconds + result->qoco_setup_seconds;
    result->scvx_total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    result->allocation_count = driver->allocation_count;
    result->allocation_bytes = driver->allocation_bytes;
    const bool pure_qoco =
        driver->options.policy == SPACEPDHCG_CUDA_SCVX_PURE_QOCO;
    result->h2d_copy_count = pure_qoco
        ? 2U * driver->qoco_report.solves
        : last_diagnostics.h2d_copy_count - transfer_before.h2d_copy_count;
    result->h2d_bytes = pure_qoco
        ? driver->qoco_report.solves
            * (driver->problem.canonical_structure.variables
               + driver->problem.canonical_structure.scalar_rows
               + driver->problem.canonical_structure.affine_rows)
            * sizeof(double)
        : last_diagnostics.h2d_copy_bytes - transfer_before.h2d_copy_bytes;
    result->d2h_copy_count =
        (pure_qoco
            ? driver->qoco_report.d2h_copy_count
            : last_diagnostics.d2h_copy_count
                - transfer_before.d2h_copy_count)
        + driver->d2h_copy_count - driver_d2h_count_before;
    result->d2h_bytes =
        (pure_qoco
            ? driver->qoco_report.d2h_bytes
            : last_diagnostics.d2h_copy_bytes
                - transfer_before.d2h_copy_bytes)
        + driver->d2h_bytes - driver_d2h_bytes_before;
    result->device_copy_count =
        (pure_qoco ? 0U
                   : last_diagnostics.d2d_copy_count
                       - transfer_before.d2d_copy_count)
        + driver->device_copy_count - driver_device_count_before;
    result->device_copy_bytes =
        (pure_qoco ? 0U
                   : last_diagnostics.d2d_copy_bytes
                       - transfer_before.d2d_copy_bytes)
        + driver->device_copy_bytes - driver_device_bytes_before;
    result->topology_allocation_count_after_create =
        pure_qoco ? 0U
                  : last_diagnostics.topology_allocation_delta_last_update;
    result->topology_index_copy_count_after_create =
        pure_qoco ? 0U
                  : last_diagnostics.topology_index_copy_delta_last_update;
    result->hidden_cpu_fallback =
        pure_qoco ? 0 : last_diagnostics.hidden_cpu_fallback;
    return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_scvx_driver_path_inventory(
    const spacepdhcg_cuda_scvx_driver* driver,
    spacepdhcg_cuda_scvx_path_inventory* inventory
) {
    if (driver == nullptr || inventory == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    *inventory = driver->path_inventory;
    return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status
spacepdhcg_cuda_scvx_driver_handback_qoco(
    spacepdhcg_cuda_scvx_driver* driver,
    const spacepdhcg_cuda_qoco_candidate* candidate,
    const spacepdhcg_accelerator_stream stream,
    spacepdhcg_cuda_qoco_handback_result* result
) {
    if (driver == nullptr || candidate == nullptr || result == nullptr
        || candidate->abi_version != SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        || stream.device.type != SPACEPDHCG_DEVICE_CUDA
        || stream.device.id != driver->problem.reference_states.device.id) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    std::memset(result, 0, sizeof(*result));
    result->abi_version = SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;
    result->mode = candidate->mode;
    result->dual_discarded = candidate->dual_discarded;
    result->trust_radius_before = candidate->trust_radius;
    result->trust_radius_after = candidate->trust_radius;
    result->trust_action = SPACEPDHCG_CUDA_SCVX_TRUST_RETAIN;
    result->conversion_seconds = candidate->conversion_seconds;
    result->setup_seconds = candidate->setup_seconds;
    result->polish_seconds = candidate->polish_seconds;
    result->hidden_cpu_fallback = 0;

    if (candidate->mode == SPACEPDHCG_CUDA_QOCO_HYBRID_PDHCG_IPM
        && candidate->hybrid_handoff_eligible == 0) {
        result->disposition =
            SPACEPDHCG_CUDA_QOCO_HANDBACK_HYBRID_INELIGIBLE;
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    if (candidate->qoco_solved == 0
        || !std::isfinite(candidate->canonical_primal_residual)
        || !std::isfinite(candidate->canonical_dual_residual)
        || !(candidate->quality_tolerance > 0.0)
        || candidate->canonical_primal_residual > candidate->quality_tolerance
        || candidate->canonical_dual_residual > candidate->quality_tolerance) {
        result->disposition =
            SPACEPDHCG_CUDA_QOCO_HANDBACK_CQP_UNQUALIFIED;
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    const size_t primal_elements =
        driver->problem.numeric.linear_objective.elements;
    const size_t dual_elements =
        driver->problem.numeric.scalar_lower.elements
        + driver->problem.numeric.affine_offset.elements;
    if (candidate->canonical_primal_host == nullptr
        || candidate->canonical_dual_host == nullptr
        || candidate->primal_elements != primal_elements
        || candidate->dual_elements != dual_elements
        || candidate->topology_fingerprint
            != driver->problem.topology_fingerprint) {
        result->disposition =
            SPACEPDHCG_CUDA_QOCO_HANDBACK_PERMUTATION_MISMATCH;
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    result->permutation_match = 1;
    const auto native =
        reinterpret_cast<cudaStream_t>(stream.native_handle);
    uint64_t numeric_fingerprint = 0U;
    auto api_status = collect_numeric_fingerprint(
        driver,
        native,
        &numeric_fingerprint
    );
    if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
        return api_status;
    }
    result->cqp_numeric_fingerprint = numeric_fingerprint;
    result->fingerprint_match =
        numeric_fingerprint == candidate->cqp_numeric_fingerprint ? 1 : 0;
    if (result->fingerprint_match == 0) {
        result->disposition =
            SPACEPDHCG_CUDA_QOCO_HANDBACK_STALE_CQP;
        return SPACEPDHCG_CUDA_SUCCESS;
    }

    auto cuda_status = cudaEventRecord(driver->timer_start, native);
    if (cuda_status != cudaSuccess) {
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    cuda_status = cudaMemcpyAsync(
        driver->primal,
        candidate->canonical_primal_host,
        primal_elements * sizeof(double),
        cudaMemcpyHostToDevice,
        native
    );
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaMemcpyAsync(
            driver->dual,
            candidate->canonical_dual_host,
            dual_elements * sizeof(double),
            cudaMemcpyHostToDevice,
            native
        );
    }
    if (cuda_status != cudaSuccess) {
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    api_status = time_stop(
        driver,
        native,
        &result->transfer_seconds
    );
    if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
        return api_status;
    }
    result->h2d_bytes =
        (primal_elements + dual_elements) * sizeof(double);
    const size_t state_elements =
        (driver->problem.intervals + 1U) * driver->problem.state_dimension;
    const size_t control_elements =
        driver->problem.intervals * driver->problem.control_dimension;
    gather_scvx_candidate_kernel<<<1, 256, 0, native>>>(
        driver->primal,
        view_pointer<const int>(driver->problem.state_variable_indices),
        view_pointer<const int>(driver->problem.control_variable_indices),
        driver->candidate_states,
        driver->candidate_controls,
        state_elements,
        control_elements
    );
    if (cudaGetLastError() != cudaSuccess) {
        return SPACEPDHCG_CUDA_RUNTIME_ERROR;
    }
    double ignored_d2h_seconds = 0.0;
    api_status = collect_metrics(
        driver,
        driver->candidate_states,
        driver->candidate_controls,
        true,
        candidate->trust_radius,
        native,
        &result->replay_seconds,
        &ignored_d2h_seconds
    );
    if (api_status != SPACEPDHCG_CUDA_SUCCESS) {
        return api_status;
    }
    result->device_replay = 1;
    const ScvxMetrics metrics = *driver->host_metrics;
    const auto acceptance_started = std::chrono::steady_clock::now();
    result->objective = metrics.objective;
    result->dynamics_residual = metrics.dynamics;
    result->path_residual = metrics.path;
    result->terminal_residual = metrics.terminal;
    result->virtual_control_residual = metrics.virtual_control;
    result->trajectory_step = metrics.step;
    result->thrust_violation = metrics.thrust;
    result->torque_violation = metrics.torque;
    result->pointing_violation = metrics.pointing;
    result->mass_violation = metrics.mass;
    result->altitude_violation = metrics.altitude;
    result->glide_slope_violation = metrics.glide_slope;
    result->angular_rate_violation = metrics.angular_rate;
    result->quaternion_violation = metrics.quaternion;
    result->predicted_reduction = std::max(
        1.0e-12,
        candidate->current_merit - metrics.model_merit
    );
    result->actual_reduction =
        candidate->current_merit - metrics.merit;
    result->reduction_ratio =
        result->actual_reduction / result->predicted_reduction;
    const bool restoration = maximum_outer_residual(metrics)
        <= driver->options.restoration_reduction
            * candidate->current_outer_residual;
    const bool accepted =
        (result->actual_reduction > 1.0e-10
         && std::isfinite(result->reduction_ratio)
         && result->reduction_ratio
            >= driver->options.acceptance_threshold)
        || restoration;
    result->accepted = accepted ? 1 : 0;
    result->restoration_accepted = accepted && restoration ? 1 : 0;
    if (accepted) {
        cuda_status = cudaMemcpyAsync(
            view_pointer<double>(driver->problem.reference_states),
            driver->replay_states,
            state_elements * sizeof(double),
            cudaMemcpyDeviceToDevice,
            native
        );
        if (cuda_status == cudaSuccess) {
            cuda_status = cudaMemcpyAsync(
                view_pointer<double>(driver->problem.reference_controls),
                driver->candidate_controls,
                control_elements * sizeof(double),
                cudaMemcpyDeviceToDevice,
                native
            );
        }
        if (cuda_status != cudaSuccess
            || cudaStreamSynchronize(native) != cudaSuccess) {
            return SPACEPDHCG_CUDA_RUNTIME_ERROR;
        }
        result->disposition =
            SPACEPDHCG_CUDA_QOCO_HANDBACK_ACCEPTED;
        if (result->reduction_ratio
                >= driver->options.strong_agreement_threshold
            && metrics.step
                >= driver->options.near_boundary_fraction
                    * std::max(1.0e-12, candidate->trust_radius)) {
            result->trust_radius_after = std::min(
                driver->options.maximum_trust_radius,
                driver->options.expansion_factor * candidate->trust_radius
            );
            result->trust_action =
                SPACEPDHCG_CUDA_SCVX_TRUST_EXPAND;
        }
    } else {
        result->disposition =
            SPACEPDHCG_CUDA_QOCO_HANDBACK_NONLINEAR_REJECTED;
        result->trust_radius_after = std::max(
            driver->options.minimum_trust_radius,
            driver->options.shrink_factor * candidate->trust_radius
        );
        result->trust_action = SPACEPDHCG_CUDA_SCVX_TRUST_SHRINK;
    }
    result->acceptance_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - acceptance_started
    ).count();
    return SPACEPDHCG_CUDA_SUCCESS;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_scvx_driver_cancel(
    spacepdhcg_cuda_scvx_driver* driver
) {
    if (driver == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    driver->cancelled.store(true, std::memory_order_release);
    const auto status = spacepdhcg_cuda_workspace_cancel(
        driver->problem.workspace
    );
    return status == SPACEPDHCG_CUDA_INVALID_STATE
        ? SPACEPDHCG_CUDA_SUCCESS
        : status;
}

extern "C" spacepdhcg_cuda_status
spacepdhcg_cuda_scvx_driver_reset_attempt(
    spacepdhcg_cuda_scvx_driver* driver,
    const spacepdhcg_cuda_warm_start_mode mode,
    const spacepdhcg_accelerator_stream stream
) {
    if (driver == nullptr
        || mode < SPACEPDHCG_CUDA_WARM_START_NONE
        || mode > SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED
        || stream.device.type != SPACEPDHCG_DEVICE_CUDA
        || stream.device.id != driver->problem.reference_states.device.id) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    driver->cancelled.store(false, std::memory_order_release);
    if (mode == SPACEPDHCG_CUDA_WARM_START_NONE) {
        spacepdhcg_native_qoco_reset_warm_state(driver->qoco, false);
        auto status = spacepdhcg_cuda_workspace_reset_async(
            driver->problem.workspace,
            SPACEPDHCG_CUDA_RESET_ITERATES,
            stream
        );
        if (status == SPACEPDHCG_CUDA_SUCCESS) {
            status = spacepdhcg_cuda_workspace_wait(driver->problem.workspace);
        }
        return status;
    }

    const auto native = reinterpret_cast<cudaStream_t>(stream.native_handle);
    if (mode == SPACEPDHCG_CUDA_WARM_START_PRIMAL) {
        const size_t dual_elements =
            driver->problem.numeric.scalar_lower.elements
            + driver->problem.numeric.affine_offset.elements;
        if (cudaMemsetAsync(
                driver->dual,
                0,
                dual_elements * sizeof(double),
                native
            ) != cudaSuccess) {
            return SPACEPDHCG_CUDA_RUNTIME_ERROR;
        }
    }
    spacepdhcg_native_qoco_reset_warm_state(driver->qoco, true);
    auto status = spacepdhcg_cuda_workspace_warm_start_async(
        driver->problem.workspace,
        SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED,
        nullptr,
        stream
    );
    if (status == SPACEPDHCG_CUDA_SUCCESS) {
        status = spacepdhcg_cuda_workspace_wait(driver->problem.workspace);
    }
    return status;
}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_scvx_driver_destroy(
    spacepdhcg_cuda_scvx_driver** driver
) {
    if (driver == nullptr) {
        return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    }
    if (*driver == nullptr) {
        return SPACEPDHCG_CUDA_SUCCESS;
    }
    destroy_driver_storage(*driver);
    delete *driver;
    *driver = nullptr;
    return SPACEPDHCG_CUDA_SUCCESS;
}
