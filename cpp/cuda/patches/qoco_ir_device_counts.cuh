// SPDX-License-Identifier: Apache-2.0
// Included after LinSysData and the retained IR runtime declarations.
static bool qoco_ir_counts_enabled() {
    return !getenv("SPACEPDHCG_TEST_QOCO_DEVICE_IR_DISABLE") &&
           !getenv("SPACEPDHCG_TEST_QOCO_IR_COUNTS_DISABLE");
}
__global__ void qoco_ir_reset_count_kernel(QocoIrCounts* counts, bool total) {
    counts->step = 0;
    if (total) counts->total = 0;
}
__global__ void qoco_ir_add_count_kernel(QocoIrCounts* counts, const QocoIrState* state) {
    counts->step += state->accepted;
}
__global__ void qoco_ir_finish_count_kernel(QocoIrCounts* counts) {
    counts->total += counts->step;
}
extern "C" void qoco_gpu_ir_reset_counts(QOCOSolver* solver, int total) {
    if (!qoco_ir_counts_enabled()) return;
    // Default-stream launches are ordered before every solver-stream replay by
    // the existing readiness event. No host download or barrier is required.
    qoco_ir_reset_count_kernel<<<1, 1>>>(solver->linsys_data->ir->counts, total != 0);
    CUDA_CHECK(cudaGetLastError());
}
extern "C" void qoco_gpu_ir_report_counts(QOCOSolver* solver) {
    if (!qoco_ir_counts_enabled()) return;
    QocoIrCounts host{};
    CUDA_CHECK(cudaMemcpy(&host, solver->linsys_data->ir->counts, sizeof(host), cudaMemcpyDeviceToHost));
    solver->work->ir_iters = host.step;
    solver->sol->ir_iters = host.total;
}
extern "C" void qoco_gpu_ir_finish_step(QOCOSolver* solver) {
    if (!qoco_ir_counts_enabled()) {
        solver->sol->ir_iters += solver->work->ir_iters;
        return;
    }
    // Initialization refinement is deliberately excluded from the solution's
    // accumulated count, exactly as in the original QOCO API.
    qoco_ir_finish_count_kernel<<<1, 1>>>(solver->linsys_data->ir->counts);
    CUDA_CHECK(cudaGetLastError());
    if (solver->settings->verbose) qoco_gpu_ir_report_counts(solver);
}
