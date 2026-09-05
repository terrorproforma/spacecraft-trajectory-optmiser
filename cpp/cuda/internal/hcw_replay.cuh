#pragma once

#include "spacepdhcg/cuda/device_scvx_c_api.h"
#include <cuda_runtime.h>

namespace spacepdhcg::cuda::detail {

/// Exact zero-order-hold HCW discretisation over `config.step_seconds`: the same closed-form
/// state-transition and control matrices as the host `dynamics::discretise_hcw`.  Shared by the
/// coefficient kernel and the trajectory replay so the device never mixes two HCW integrators
/// (an RK4 replay of the continuous model drifted ~1e-6 from these exact rows per horizon,
/// which the planner's 1e-9 host/device replay-parity certificate rejected and which made the
/// SCvx merit ratio of an exactly-linear problem negative).
__device__ inline void hcw_exact_matrices(
    const spacepdhcg_cuda_dynamics_config& config,
    double* state_matrix,
    double* control_matrix
) {
    const double n = config.mean_motion;
    const double t = config.step_seconds;
    const double angle = n * t;
    const double c = cos(angle);
    const double s = sin(angle);
    const double inverse_n = 1.0 / n;
    const double inverse_n_squared = inverse_n * inverse_n;
    for (int index = 0; index < 36; ++index) {
        state_matrix[index] = 0.0;
    }
    for (int index = 0; index < 18; ++index) {
        control_matrix[index] = 0.0;
    }
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
}

/// One exact HCW step x_{k+1} = Phi x_k + Gamma u_k (the replay counterpart of `hcw_exact_kernel`).
__device__ inline void hcw_exact_step(
    const double* state,
    const double* control,
    const spacepdhcg_cuda_dynamics_config& config,
    double* output
) {
    double state_matrix[36];
    double control_matrix[18];
    hcw_exact_matrices(config, state_matrix, control_matrix);
    for (int row = 0; row < 6; ++row) {
        double value = 0.0;
        for (int column = 0; column < 6; ++column) {
            value += state_matrix[row * 6 + column] * state[column];
        }
        for (int column = 0; column < 3; ++column) {
            value += control_matrix[row * 3 + column] * control[column];
        }
        output[row] = value;
    }
}

// Time steps remain ordered. Six lanes own the six state rows, and the
// previous state is exchanged in registers. Retain every product in its original
// order, including structural zeros; this is the same exact ZOH recurrence.
// Parallel launches require exactly one full warp; false is the serial oracle.
template <bool Parallel>
__global__ void hcw_replay_kernel(
    const double* initial_state, const double* controls, double* replay,
    const size_t intervals, const spacepdhcg_cuda_dynamics_config config
) {
    if constexpr (!Parallel) {
        for (int row = 0; row < 6; ++row) replay[row] = initial_state[row];
        for (size_t interval = 0; interval < intervals; ++interval) {
            hcw_exact_step(replay + interval * 6, controls + interval * 3,
                           config, replay + (interval + 1) * 6);
        }
    } else {
        __shared__ double phi[36];
        __shared__ double gamma[18];
        if (threadIdx.x == 0U) hcw_exact_matrices(config, phi, gamma);
        __syncwarp();
        const unsigned int row = threadIdx.x;
        double a[6]{};
        double b[3]{};
        double state = 0.0;
        if (row < 6U) {
            state = initial_state[row];
            replay[row] = state;
            for (int column = 0; column < 6; ++column) a[column] = phi[row * 6 + column];
            for (int column = 0; column < 3; ++column) b[column] = gamma[row * 3 + column];
        }
        for (size_t interval = 0; interval < intervals; ++interval) {
            double value = 0.0;
            #pragma unroll
            for (int column = 0; column < 6; ++column) {
                const double previous = __shfl_sync(0xffffffffU, state, column);
                value += a[column] * previous;
            }
            #pragma unroll
            for (int column = 0; column < 3; ++column) {
                value += b[column] * controls[interval * 3 + column];
            }
            if (row < 6U) replay[(interval + 1) * 6 + row] = value;
            state = value;
        }
    }
}

}  // namespace spacepdhcg::cuda::detail
