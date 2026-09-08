// SPDX-License-Identifier: Apache-2.0
// Optional GPU-controlled refinement. Included after the cuDSS function table.
#include <vector>
#include <cstring>
#include <cstdint>

struct QocoIrState {
    double best;
    int accepted, attempted, restore;
};
struct QocoIrRuntime {
    cudaStream_t stream{};
    cudaEvent_t ready{}, done{};
    double* norm_scratch{};
    QocoIrState* state{};
    void* range_inputs[2]{};
};

__global__ void qoco_ir_initialize_ranges(uint64_t* first, uint64_t* second, int count) {
    *first = *second = uint64_t(count - 1) << 32;
}
static QocoIrRuntime* qoco_ir_create(cudssHandle_t handle, int count) {
    auto* runtime = new QocoIrRuntime;
    CUDA_CHECK(cudaStreamCreateWithFlags(&runtime->stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&runtime->ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&runtime->done, cudaEventDisableTiming));
    CUDA_CHECK(cudaMalloc(&runtime->norm_scratch, 257 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&runtime->state, sizeof(QocoIrState)));
    CUDA_CHECK(cudaMemset(runtime->state, 0, sizeof(QocoIrState)));
    for (auto& input : runtime->range_inputs) CUDA_CHECK(cudaMalloc(&input, 8));
    qoco_ir_initialize_ranges<<<1, 1, 0, runtime->stream>>>(
        static_cast<uint64_t*>(runtime->range_inputs[0]),
        static_cast<uint64_t*>(runtime->range_inputs[1]), count);
    CUDA_CHECK(cudaGetLastError());
    auto set_stream = reinterpret_cast<decltype(&::cudssSetStream)>(dlsym(g_cudss_handle, "cudssSetStream"));
    if (!set_stream) { fprintf(stderr, "cuDSS stream API unavailable\n"); exit(1); }
    CUDSS_CHECK(set_stream(handle, runtime->stream));
    return runtime;
}
static void qoco_ir_destroy(QocoIrRuntime* runtime) {
    CUDA_CHECK(cudaStreamSynchronize(runtime->stream));
    for (auto input : runtime->range_inputs) CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(runtime->norm_scratch));
    CUDA_CHECK(cudaFree(runtime->state));
    CUDA_CHECK(cudaEventDestroy(runtime->ready));
    CUDA_CHECK(cudaEventDestroy(runtime->done));
    CUDA_CHECK(cudaStreamDestroy(runtime->stream));
    delete runtime;
}
static void qoco_ir_wait_default(QocoIrRuntime* runtime) {
    CUDA_CHECK(cudaEventRecord(runtime->ready, nullptr));
    CUDA_CHECK(cudaStreamWaitEvent(runtime->stream, runtime->ready));
}
static cudssStatus_t qoco_ir_execute(QocoIrRuntime* runtime, cudssHandle_t handle, int phase,
        const cudssConfig_t config, cudssData_t data, const cudssMatrix_t matrix,
        cudssMatrix_t solution, const cudssMatrix_t rhs) {
    qoco_ir_wait_default(runtime);
    const auto result = g_cuda_funcs.cudssExecute(handle, phase, config, data, matrix, solution, rhs);
    CUDA_CHECK(cudaEventRecord(runtime->done, runtime->stream));
    CUDA_CHECK(cudaStreamWaitEvent(nullptr, runtime->done));
    return result;
}

extern "C" cudaStream_t qoco_ir_exchange_stream(cudaStream_t);
extern "C" void qoco_ir_device_norm(const double*, int, double*, cudaStream_t);
extern "C" void qoco_ir_prepare_sparse(QOCOProblemData*);

__global__ void qoco_ir_initial(QocoIrState* state, const double* norm,
        cudaGraphConditionalHandle handle, double tolerance, int maximum) {
    state->best = *norm;
    state->accepted = state->attempted = state->restore = 0;
    cudaGraphSetConditional(handle, maximum > 0 && isfinite(*norm) &&
        isfinite(tolerance) && !(*norm < tolerance));
}
__global__ void qoco_ir_decide(QocoIrState* state, const double* norm,
        cudaGraphConditionalHandle handle, double tolerance, int maximum) {
    ++state->attempted;
    // NaN comparisons are false. Without this guard a nonfinite correction
    // replaces the finite backup and is incorrectly counted as an improvement.
    if (!isfinite(*norm) || *norm >= state->best) {
        state->restore = 1;
        cudaGraphSetConditional(handle, 0);
    } else {
        ++state->accepted;
        state->best = *norm;
        state->restore = 0;
        cudaGraphSetConditional(handle, state->attempted < maximum && !(*norm < tolerance));
    }
}
__global__ void qoco_ir_save_or_restore(const QocoIrState* state, double* x, double* best, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        if (state->restore) x[i] = best[i];
        else best[i] = x[i];
    }
}
