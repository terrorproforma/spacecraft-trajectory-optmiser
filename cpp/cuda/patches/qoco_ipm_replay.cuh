// SPDX-License-Identifier: Apache-2.0
#include "input_validation.h"
extern "C" int qoco_gpu_ipm_resources_compatible(void*);
extern "C" int qoco_gpu_ipm_finish_device(QOCOSolver* solver) {
    if (!solver || !solver->work || !solver->linsys_data || !solver->linsys_data->ir) return 1;
    auto& cache = solver->linsys_data->ir->ipm;
    if (!cache.replay_pending) return 0;
    if (!qoco_gpu_ipm_resources_compatible(cache.resources)) return 2;
    if (cudaStreamSynchronize(cache.replay_stream) != cudaSuccess) return 4;
    cache.replay_pending = false;
    return 0;
}
extern "C" int qoco_gpu_ipm_replay_device(QOCOSolver* solver, void* stream_pointer, QocoGpuOutput* output) {
    if (!output) return 1;
    *output = {};
    if (!solver || !solver->work || !solver->settings || !solver->sol ||
        !solver->linsys_data || !solver->linsys_data->ir || qoco_validate_settings(solver->settings)) return 1;
    auto* s = solver->linsys_data;
    auto* work = solver->work;
    auto* settings = solver->settings;
    auto& cache = s->ir->ipm;
    if (!cache.executable || !cache.initialization || !cache.terminal ||
        !qoco_gpu_ipm_initialization_enabled() || !qoco_ir_counts_enabled() ||
        getenv("SPACEPDHCG_TEST_QOCO_IPM_TERMINAL_DISABLE") ||
        getenv("SPACEPDHCG_TEST_QOCO_IPM_CACHE_DISABLE") || settings->verbose ||
        getenv("SPACEPDHCG_TEST_QOCO_HOST_INITIAL_CONE") ||
        getenv("SPACEPDHCG_TEST_QOCO_DEVICE_STEPS_COMPARE") ||
        getenv("SPACEPDHCG_TEST_QOCO_COMBINED_RHS_COMPARE") ||
        getenv("SPACEPDHCG_TEST_QOCO_DEVICE_CONTROL_COMPARE") ||
        !qoco_gpu_ipm_resources_compatible(cache.resources) ||
        cache.work != work || cache.control != work->gpu_control ||
        cache.static_p != settings->kkt_static_reg_P || cache.static_a != settings->kkt_static_reg_A ||
        cache.static_g != settings->kkt_static_reg_G || cache.linsys_static_p != s->kkt_static_reg_P) return 2;
    const auto stream = static_cast<cudaStream_t>(stream_pointer);
    if (cache.replay_pending && cache.replay_stream != stream) return 3;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) return 4;
    if (capture != cudaStreamCaptureStatusNone) return 2;
    const QocoIpmParameters parameters{work->scaling->k, work->scaling->kinv,
        settings->abstol, settings->reltol, settings->abstol_inacc, settings->reltol_inacc,
        settings->ir_tol, settings->max_iters, settings->max_ir_iters,
        settings->kkt_dynamic_reg, int(work->use_x0)};
    // Mark ownership before queuing anything, so partial CUDA failures still
    // require completion before a caller updates or releases operand storage.
    cache.replay_pending = true;
    cache.replay_stream = stream;
    solver->sol->status = QOCO_UNSOLVED;
    qoco_ipm_parameters_set<<<1, 1, 0, stream>>>(cache.parameters, parameters);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemsetAsync(cache.iterations, 0, sizeof(int), stream) != cudaSuccess ||
        cudaGraphLaunch(cache.executable, stream) != cudaSuccess) return 4;
    ++cache.launches;
    *output = {cache.completion, work->x->d_data, work->y->d_data, work->s->d_data,
        work->z->d_data, work->data->n, work->data->p, work->data->m};
    return 0;
}
