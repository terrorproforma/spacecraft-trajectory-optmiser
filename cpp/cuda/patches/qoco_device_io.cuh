// SPDX-License-Identifier: Apache-2.0
// Per-workspace opt-in. Legacy QOCO callers keep completed host solutions.
extern "C" void qoco_reference_copy_solution(QOCOSolver*);
extern "C" int qoco_gpu_set_device_io(QOCOSolver* solver, int enabled) {
    if (!solver || !solver->work || (enabled != 0 && enabled != 1)) return 1;
    solver->work->gpu_device_io = enabled;
    return 0;
}
extern "C" int qoco_gpu_finish_device_solution(QOCOSolver* solver) {
    if (!solver->work->gpu_device_io) return 0;
    // Preserve qoco_solve's completion contract even inside a queued scope.
    return cudaStreamSynchronize(nullptr) == cudaSuccess ? 1 : -1;
}
extern "C" int qoco_gpu_download_solution(QOCOSolver* solver) {
    if (!solver || !solver->work || !solver->sol) return 1;
    qoco_reference_copy_solution(solver);
    return 0;
}
// 0: disable, retaining the saved start; 1: enable the saved start;
// 2: accept the completed unscaled primal into existing x0 device storage.
extern "C" int qoco_gpu_primal_start(QOCOSolver* solver, int action) {
    if (!solver || !solver->work || !solver->sol || action < 0 || action > 2) return 1;
    auto* work = solver->work;
    if (action == 2) {
        if ((solver->sol->status != 1 && solver->sol->status != 2)
            || !work->x || !work->x0 || !work->x->d_data || !work->x0->d_data) return 1;
        const auto copied = cudaMemcpyAsync(work->x0->d_data, work->x->d_data,
            static_cast<size_t>(work->data->n) * sizeof(QOCOFloat), cudaMemcpyDeviceToDevice);
        if (copied != cudaSuccess || cudaStreamSynchronize(nullptr) != cudaSuccess) return 1;
        work->gpu_primal_saved = 1;
        return 0;
    }
    if (action == 1 && !work->gpu_primal_saved) return 1;
    work->use_x0 = action;
    solver->sol->status = QOCO_UNSOLVED;
    return 0;
}
