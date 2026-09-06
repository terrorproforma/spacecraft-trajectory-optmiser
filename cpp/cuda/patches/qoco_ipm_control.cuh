// SPDX-License-Identifier: Apache-2.0
// Appended after device stopping/control kernels in cuda_linalg.cu.
namespace qoco_device_control {
__global__ void ipm_check(State* state, const int* iteration,
    cudaGraphConditionalHandle loop, cudaGraphConditionalHandle step,
    double absolute, double relative, double inaccurate_absolute, double inaccurate_relative) {
    decide_impl(state, absolute, relative, inaccurate_absolute, inaccurate_relative, *iteration);
    if (state->stop) cudaGraphSetConditional(loop, 0);
    cudaGraphSetConditional(step, !state->stop);
}
__global__ void ipm_advance(State* state, int* iteration, int maximum,
                           cudaGraphConditionalHandle loop) {
    ++*iteration;
    if (*iteration >= maximum) {
        state->status = QOCO_MAX_ITER;
        cudaGraphSetConditional(loop, 0);
    }
}
}
extern "C" void qoco_gpu_ipm_prepare(QOCOSolver* solver) {
    auto* d = solver->work->data;
    double* scratch = nullptr;
    int temporary = 0;
    const size_t slots = std::max(size_t(512),
        size_t(std::max(d->l, d->nsoc) + 255LL) / 256 + 7);
    CUDA_CHECK(qoco_gpu_acquire_scalar_workspace(slots, &scratch, &temporary));
    if (temporary) { fprintf(stderr, "IPM requires a retained reduction scope\n"); exit(1); }
    bool temporary_handle = false;
    const auto handle = qoco_acquire_reduction_handle(&temporary_handle);
    if (!qoco_ipm_blas_workspace) {
        constexpr size_t bytes = 32 * 1024 * 1024;
        CUDA_CHECK(cudaMalloc(&qoco_ipm_blas_workspace, bytes));
        qoco_batched_stopping::blas(get_cuda_funcs()->cublasSetStream(handle, cudaStreamPerThread));
        qoco_batched_stopping::blas(get_cuda_funcs()->cublasSetWorkspace(
            handle, qoco_ipm_blas_workspace, bytes));
    }
    // Prime library kernels and metrics before any enclosing graph capture.
    auto* state = static_cast<qoco_device_control::State*>(solver->work->gpu_control);
    qoco_gpu_metrics<true>(solver, state->metrics, true);
}
extern "C" void qoco_gpu_ipm_check(QOCOSolver* solver, const int* iteration,
    cudaGraphConditionalHandle loop, cudaGraphConditionalHandle step) {
    auto* state = static_cast<qoco_device_control::State*>(solver->work->gpu_control);
    auto* settings = solver->settings;
    qoco_gpu_metrics<true>(solver, state->metrics, true);
    qoco_device_control::ipm_check<<<1, 1>>>(state, iteration, loop, step,
        settings->abstol, settings->reltol, settings->abstol_inacc, settings->reltol_inacc);
    qoco_device_control::copy<false>(solver->work);
    CUDA_CHECK(cudaGetLastError());
}
extern "C" void qoco_gpu_ipm_advance(QOCOSolver* solver, int* iteration,
                                     cudaGraphConditionalHandle loop) {
    qoco_device_control::ipm_advance<<<1, 1>>>(
        static_cast<qoco_device_control::State*>(solver->work->gpu_control),
        iteration, solver->settings->max_iters, loop);
    CUDA_CHECK(cudaGetLastError());
}
