/*
 * Analytic float64 device dynamics and variational RK4.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "spacepdhcg/cuda/device_scvx_c_api.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

template <int StateDimension, int ControlDimension>
struct Augmented {
    double state[StateDimension];
    double transition[StateDimension * StateDimension];
    double sensitivity[StateDimension * ControlDimension];
};

template <typename T>
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
