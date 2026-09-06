// SPDX-License-Identifier: Apache-2.0
// Experimental whole-IPM graph. Requires --default-stream per-thread and the
// retained factor/IR/control/count extensions. Setup and terminal reports are host work.
struct QocoIpmCapture {
    cudaGraph_t graph{};
    cudaGraph_t linear{};
    std::vector<cudaGraphNode_t> dependencies;
};
static thread_local QocoIpmCapture* qoco_ipm_current = nullptr;
extern "C" int qoco_gpu_ipm_capturing() { return qoco_ipm_current != nullptr; }
extern "C" void qoco_gpu_ipm_prepare(QOCOSolver*);
extern "C" void qoco_gpu_ipm_check(QOCOSolver*, const int*, cudaGraphConditionalHandle,
                                    cudaGraphConditionalHandle);
extern "C" void qoco_gpu_ipm_advance(QOCOSolver*, int*, cudaGraphConditionalHandle);
extern "C" void qoco_gpu_sync_control(QOCOSolver*);

static void qoco_ipm_pause() {
    auto& c = *qoco_ipm_current;
    cudaStreamCaptureStatus status;
    const cudaGraphNode_t* dependencies = nullptr;
    size_t count = 0;
    CUDA_CHECK(cudaStreamGetCaptureInfo(cudaStreamPerThread, &status, nullptr, &c.graph,
                                       &dependencies, &count));
    c.dependencies.assign(dependencies, dependencies + count);
    cudaGraph_t ended;
    CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &ended));
    if (ended != c.graph) { fprintf(stderr, "IPM capture graph mismatch\n"); exit(1); }
}
static void qoco_ipm_resume() {
    auto& c = *qoco_ipm_current;
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(cudaStreamPerThread, c.graph,
        c.dependencies.data(), nullptr, c.dependencies.size(), cudaStreamCaptureModeThreadLocal));
}
static void qoco_ipm_child(cudaGraph_t child) {
    qoco_ipm_pause();
    auto& c = *qoco_ipm_current;
    cudaGraphNode_t node;
    CUDA_CHECK(cudaGraphAddChildGraphNode(&node, c.graph, c.dependencies.data(),
                                          c.dependencies.size(), child));
    c.dependencies = {node};
    qoco_ipm_resume();
}
static void qoco_ipm_factor(LinSysData* s) {
    if (!s->ir->factor_memory->factor_graph) {
        fprintf(stderr, "IPM requires primed retained factorization\n"); exit(1);
    }
    qoco_ipm_child(s->ir->factor_memory->factor_graph);
}
static void qoco_ipm_linear(LinSysData* s, QOCOWorkspace* work, const double* b,
                            double* x, double tolerance, int maximum) {
    auto& c = *qoco_ipm_current;
    auto* runtime = s->ir;
    const int count = s->Kn;
    const size_t bytes = count * sizeof(double);
    auto* norm = runtime->norm_scratch + 256;
    auto* best = get_data_vectorf(work->xyzbuff1);
    CUDA_CHECK(cudaMemcpyAsync(s->d_rhs_matrix_data, b, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemsetAsync(s->d_xyz_matrix_data, 0, bytes));
    qoco_ipm_child(c.linear);
    CUDA_CHECK(cudaMemcpyAsync(x, s->d_xyz_matrix_data, bytes, cudaMemcpyDeviceToDevice));

    // Emit the refinement conditional into the parent; conditional graphs cannot
    // be cloned as ordinary child graphs. All arithmetic matches the retained IR path.
    qoco_ipm_pause();
    cudaGraphConditionalHandle handle;
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, c.graph, 0, cudaGraphCondAssignDefault));
    qoco_ipm_resume();
    qoco_ir_set_parameters<<<1, 1>>>(runtime->cache.parameters, tolerance, maximum);
    compute_linsys_residual(s, work, b, x, s->d_xyz_matrix_data, false);
    qoco_ir_device_norm(s->d_xyz_matrix_data, count, runtime->norm_scratch, cudaStreamPerThread);
    CUDA_CHECK(cudaMemcpyAsync(best, x, bytes, cudaMemcpyDeviceToDevice));
    qoco_ir_initial<<<1, 1>>>(runtime->state, norm, handle, runtime->cache.parameters);
    CUDA_CHECK(cudaGetLastError());
    qoco_ipm_pause();
    const auto parent = c.graph;
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeWhile;
    params.conditional.size = 1;
    cudaGraphNode_t conditional;
    CUDA_CHECK(cudaGraphAddNode(&conditional, parent, c.dependencies.data(),
                                 c.dependencies.size(), &params));
    c.graph = params.conditional.phGraph_out[0];
    c.dependencies.clear();
    qoco_ipm_resume();
    CUDA_CHECK(cudaMemcpyAsync(s->d_rhs_matrix_data, s->d_xyz_matrix_data, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemsetAsync(s->d_xyz_matrix_data, 0, bytes));
    qoco_ipm_child(c.linear);
    qoco_axpy(s->d_xyz_matrix_data, x, x, 1.0, count);
    compute_linsys_residual(s, work, b, x, s->d_xyz_matrix_data, false);
    qoco_ir_device_norm(s->d_xyz_matrix_data, count, runtime->norm_scratch, cudaStreamPerThread);
    qoco_ir_decide<<<1, 1>>>(runtime->state, norm, handle, runtime->cache.parameters);
    qoco_ir_save_or_restore<<<(count + 255) / 256, 256>>>(runtime->state, x, best, count);
    CUDA_CHECK(cudaGetLastError());
    qoco_ipm_pause();
    c.graph = parent;
    c.dependencies = {conditional};
    qoco_ipm_resume();
    qoco_ir_add_count_kernel<<<1, 1>>>(runtime->counts, runtime->state);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" int qoco_gpu_ipm_loop(QOCOSolver* solver) {
    const char* enabled = getenv("SPACEPDHCG_TEST_QOCO_IPM_GRAPH");
    if (!enabled || enabled[0] != '1') return 0;
    if (!solver->work->gpu_control || !qoco_ir_counts_enabled() || solver->settings->verbose ||
        getenv("SPACEPDHCG_TEST_QOCO_DEVICE_STEPS_COMPARE") ||
        getenv("SPACEPDHCG_TEST_QOCO_COMBINED_RHS_COMPARE") ||
        getenv("SPACEPDHCG_TEST_QOCO_DEVICE_CONTROL_COMPARE")) {
        fprintf(stderr, "IPM graph requires GPU controls/counting and non-verbose, unaudited execution\n"); exit(1);
    }
    auto* s = solver->linsys_data;
    auto* work = solver->work;
    auto* data = work->data;
    qoco_ir_prepare_sparse(data);
    compute_kkt_residual(data, work->x, work->y, work->s, work->z, work->kktres,
        solver->settings->kkt_static_reg_P, work->xyzbuff1, work->xbuff, work->ubuff1);
    qoco_gpu_ipm_prepare(solver);
    CUDA_CHECK(cudaDeviceSynchronize());
    QocoIpmCapture capture;
    const auto stream = s->ir->stream;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CUDSS_CHECK(g_cuda_funcs.cudssExecute(s->handle, CUDSS_PHASE_SOLVE, s->config, s->data,
                                         s->K_csr, s->d_xyz_matrix, s->d_rhs_matrix));
    CUDA_CHECK(cudaStreamEndCapture(stream, &capture.linear));
    qoco_ir_snapshot_ranges(capture.linear, s->ir, s->Kn);
    CUDA_CHECK(cudaGraphCreate(&capture.graph, 0));
    const auto root = capture.graph;
    int* iterations = nullptr;
    CUDA_CHECK(cudaMalloc(&iterations, sizeof(int)));
    CUDA_CHECK(cudaMemset(iterations, 0, sizeof(int)));
    cudaGraphConditionalHandle loop, step;
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&loop, root, 1, cudaGraphCondAssignDefault));
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = loop;
    params.conditional.type = cudaGraphCondTypeWhile;
    params.conditional.size = 1;
    cudaGraphNode_t node;
    CUDA_CHECK(cudaGraphAddNode(&node, root, nullptr, 0, &params));
    const auto body = params.conditional.phGraph_out[0];
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&step, body, 0, cudaGraphCondAssignDefault));
    capture.graph = body;
    qoco_ipm_current = &capture;
    qoco_ipm_resume();
    compute_kkt_residual(data, work->x, work->y, work->s, work->z, work->kktres,
        solver->settings->kkt_static_reg_P, work->xyzbuff1, work->xbuff, work->ubuff1);
    qoco_gpu_ipm_check(solver, iterations, loop, step);
    qoco_ipm_pause();
    params = {};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = step;
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 1;
    CUDA_CHECK(cudaGraphAddNode(&node, body, capture.dependencies.data(),
                                 capture.dependencies.size(), &params));
    capture.graph = params.conditional.phGraph_out[0];
    capture.dependencies.clear();
    qoco_ipm_resume();
    compute_nt_scaling(work);
    solver->linsys->linsys_update_nt(s, work->WtW, solver->settings->kkt_static_reg_G, data->m);
    qoco_gpu_ir_reset_counts(solver, 0);
    predictor_corrector(solver);
    qoco_gpu_ir_finish_step(solver);
    qoco_gpu_ipm_advance(solver, iterations, loop);
    qoco_ipm_pause();
    qoco_ipm_current = nullptr;
    cudaGraphExec_t executable;
    cudaGraphNode_t error_node = nullptr;
    char error_log[8192] = {};
    const auto instantiate_status = cudaGraphInstantiate(&executable, root, &error_node,
                                                         error_log, sizeof(error_log));
    if (instantiate_status != cudaSuccess) {
        cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
        if (error_node) cudaGraphNodeGetType(error_node, &type);
        fprintf(stderr, "IPM instantiate error node=%p type=%d log=%s\n",
                static_cast<void*>(error_node), int(type), error_log);
        const char* dump = getenv("SPACEPDHCG_TEST_QOCO_IPM_DOT");
        if (dump) cudaGraphDebugDotPrint(root, dump, cudaGraphDebugDotFlagsVerbose);
    }
    CUDA_CHECK(instantiate_status);
    CUDA_CHECK(cudaGraphLaunch(executable, cudaStreamPerThread));
    CUDA_CHECK(cudaMemcpy(&solver->sol->iters, iterations, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    qoco_gpu_sync_control(solver);
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(root));
    CUDA_CHECK(cudaGraphDestroy(capture.linear));
    CUDA_CHECK(cudaFree(iterations));
    return 1;
}
