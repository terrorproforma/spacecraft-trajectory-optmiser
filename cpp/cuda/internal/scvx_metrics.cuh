#pragma once

#include "spacepdhcg/cuda/device_scvx_driver_c_api.h"
#include <cuda_runtime.h>

namespace spacepdhcg::cuda::detail {

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

// All calls use exactly 256 threads. Each field has explicit sum/max semantics;
// no packed-struct aliasing, global floating-point atomics, or host reduction.
__device__ inline ScvxMetrics combine_metrics(ScvxMetrics a, const ScvxMetrics& b) {
    a.objective = a.objective + b.objective;
    a.merit = a.merit + b.merit;
    a.model_merit = a.model_merit + b.model_merit;
    a.dynamics = fmax(a.dynamics, b.dynamics);
    a.path = fmax(a.path, b.path);
    a.path_thrust = fmax(a.path_thrust, b.path_thrust);
    a.path_mass = fmax(a.path_mass, b.path_mass);
    a.path_altitude = fmax(a.path_altitude, b.path_altitude);
    a.terminal = fmax(a.terminal, b.terminal);
    a.virtual_control = fmax(a.virtual_control, b.virtual_control);
    a.step = fmax(a.step, b.step);
    a.thrust = fmax(a.thrust, b.thrust);
    a.torque = fmax(a.torque, b.torque);
    a.pointing = fmax(a.pointing, b.pointing);
    a.mass = fmax(a.mass, b.mass);
    a.altitude = fmax(a.altitude, b.altitude);
    a.glide_slope = fmax(a.glide_slope, b.glide_slope);
    a.angular_rate = fmax(a.angular_rate, b.angular_rate);
    a.quaternion = fmax(a.quaternion, b.quaternion);
    a.maximum_stage_trust_distance = fmax(a.maximum_stage_trust_distance, b.maximum_stage_trust_distance);
    a.terminal_trust_distance = fmax(a.terminal_trust_distance, b.terminal_trust_distance);
    return a;
}

__device__ inline ScvxMetrics reduce_metrics_warp(ScvxMetrics value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        ScvxMetrics other{};
        other.objective = __shfl_down_sync(0xffffffffU, value.objective, offset);
        other.merit = __shfl_down_sync(0xffffffffU, value.merit, offset);
        other.model_merit = __shfl_down_sync(0xffffffffU, value.model_merit, offset);
        other.dynamics = __shfl_down_sync(0xffffffffU, value.dynamics, offset);
        other.path = __shfl_down_sync(0xffffffffU, value.path, offset);
        other.path_thrust = __shfl_down_sync(0xffffffffU, value.path_thrust, offset);
        other.path_mass = __shfl_down_sync(0xffffffffU, value.path_mass, offset);
        other.path_altitude = __shfl_down_sync(0xffffffffU, value.path_altitude, offset);
        other.terminal = __shfl_down_sync(0xffffffffU, value.terminal, offset);
        other.virtual_control = __shfl_down_sync(0xffffffffU, value.virtual_control, offset);
        other.step = __shfl_down_sync(0xffffffffU, value.step, offset);
        other.thrust = __shfl_down_sync(0xffffffffU, value.thrust, offset);
        other.torque = __shfl_down_sync(0xffffffffU, value.torque, offset);
        other.pointing = __shfl_down_sync(0xffffffffU, value.pointing, offset);
        other.mass = __shfl_down_sync(0xffffffffU, value.mass, offset);
        other.altitude = __shfl_down_sync(0xffffffffU, value.altitude, offset);
        other.glide_slope = __shfl_down_sync(0xffffffffU, value.glide_slope, offset);
        other.angular_rate = __shfl_down_sync(0xffffffffU, value.angular_rate, offset);
        other.quaternion = __shfl_down_sync(0xffffffffU, value.quaternion, offset);
        other.maximum_stage_trust_distance = __shfl_down_sync(0xffffffffU, value.maximum_stage_trust_distance, offset);
        other.terminal_trust_distance = __shfl_down_sync(0xffffffffU, value.terminal_trust_distance, offset);
        value = combine_metrics(value, other);
    }
    return value;
}

__device__ inline ScvxMetrics reduce_metrics_block(ScvxMetrics value) {
    __shared__ ScvxMetrics warps[8];
    value = reduce_metrics_warp(value);
    if ((threadIdx.x & 31U) == 0U) warps[threadIdx.x / 32U] = value;
    __syncthreads();
    if (threadIdx.x < 32U) {
        value = threadIdx.x < 8U ? warps[threadIdx.x] : ScvxMetrics{};
        value = reduce_metrics_warp(value);
    }
    return value;
}

// Parallel=false is the original serial oracle and must launch with <<<1, 1>>>.
// Partition only independent nodes/intervals; terminal sums belong to rank zero.
template <bool Parallel>
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
    const size_t rank = Parallel ? blockIdx.x * blockDim.x + threadIdx.x : 0U;
    const size_t stride = Parallel ? blockDim.x * gridDim.x : 1U;
    ScvxMetrics result{};
    double actual_terminal_sum = 0.0;
    double model_terminal_sum = 0.0;
    double actual_path_sum = 0.0;
    double model_path_sum = 0.0;
    for (size_t interval = rank; interval < intervals; interval += stride) {
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
    if (rank == 0U) {
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
    }
    for (size_t node = rank + 1U; node <= intervals; node += stride) {
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
    if (rank == 0U) {
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
    }
    if (model != SPACEPDHCG_CUDA_DYNAMICS_HCW) {
        const size_t sigma_index =
            model == SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF
            ? 6U
            : 3U;
        for (size_t interval = rank; interval < intervals; interval += stride) {
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
        for (size_t node = rank; node <= intervals; node += stride) {
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
        for (size_t node = rank; node <= intervals; node += stride) {
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
        for (size_t node = rank; node <= intervals; node += stride) {
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
    for (size_t index = rank; index < virtual_elements; index += stride) {
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
    if constexpr (Parallel) {
        result = reduce_metrics_block(result);
        if (threadIdx.x == 0U) metrics[blockIdx.x] = result;
    } else {
        *metrics = result;
    }
}

static __global__ void finish_scvx_metrics_kernel(
    const ScvxMetrics* partials, size_t count, ScvxMetrics* metrics
) {
    ScvxMetrics value{};
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        value = combine_metrics(value, partials[i]);
    }
    value = reduce_metrics_block(value);
    if (threadIdx.x == 0U) *metrics = value;
}

}  // namespace spacepdhcg::cuda::detail
