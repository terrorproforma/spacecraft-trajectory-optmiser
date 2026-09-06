// SPDX-License-Identifier: Apache-2.0
#include "qoco_gpu_replay.h"
static_assert(sizeof(QocoGpuCompletion) == 64);
namespace qoco_device_control {
__global__ void completion_packet(const State* state, const int* iterations,
    const int* step, const int* total, QocoGpuCompletion* result) {
    *result = {1, state->status, *iterations, *total, *step, state->restored,
        state->metrics[0], state->metrics[1], state->metrics[2], state->metrics[6],
        state->dynamic_reg};
}
}
extern "C" void qoco_gpu_ipm_completion(QOCOSolver* solver, const int* iterations,
    const int* step, const int* total, QocoGpuCompletion* result) {
    qoco_device_control::completion_packet<<<1, 1>>>(
        static_cast<qoco_device_control::State*>(solver->work->gpu_control),
        iterations, step, total, result);
    CUDA_CHECK(cudaGetLastError());
}
