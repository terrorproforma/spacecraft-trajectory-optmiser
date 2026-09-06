// SPDX-License-Identifier: Apache-2.0
// Experimental GPU-controlled cuDSS refinement; original residual arithmetic.
// Graphs are deliberately scoped to a single solve until refactorization reuse
// can be proved. Runtime scratch and its stream live with the cuDSS handle.
static void qoco_ir_snapshot_ranges(cudaGraph_t graph, QocoIrRuntime* runtime, int count) {
    size_t size = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &size));
    std::vector<cudaGraphNode_t> nodes(size);
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &size));
    int found = 0;
    for (auto node : nodes) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeMemcpy) continue;
        cudaMemcpy3DParms copy = {};
        CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &copy));
        if (copy.kind != cudaMemcpyHostToDevice) continue;
        if (found == 2 || copy.srcArray || copy.dstArray || copy.extent.width != 8 ||
            copy.extent.height != 1 || copy.extent.depth != 1 ||
            copy.srcPos.x || copy.srcPos.y || copy.srcPos.z ||
            copy.dstPos.x || copy.dstPos.y || copy.dstPos.z) {
            fprintf(stderr, "Unsupported cuDSS host input in device refinement graph\n"); exit(1);
        }
        uint64_t bits = 0;
        std::memcpy(&bits, copy.srcPtr.ptr, 8);
        // Explicit version-specific guard: this is not a general transformation
        // of arbitrary vendor host inputs into frozen constants.
        if (bits != (uint64_t(count - 1) << 32)) {
            fprintf(stderr, "cuDSS refinement range input changed\n"); exit(1);
        }
        // Constants were initialized on the solver stream at workspace creation.
        // No cross-stream host upload is needed during capture/replay.
        copy.srcPtr.ptr = runtime->range_inputs[found++];
        copy.kind = cudaMemcpyDeviceToDevice;
        CUDA_CHECK(cudaGraphMemcpyNodeSetParams(node, &copy));
    }
    if (found != 2) { fprintf(stderr, "cuDSS refinement capture layout changed\n"); exit(1); }
}

static void qoco_ir_solve(LinSysData* s, QOCOWorkspace* work, const double* b,
                          double* x, double tolerance, int maximum) {
    auto* runtime = s->ir;
    const auto stream = runtime->stream;
    const auto previous_stream = qoco_ir_exchange_stream(nullptr);
    // Materialize only pending sparse views on their established default stream.
    // The readiness event also orders the initial solution copy before replay.
    qoco_ir_prepare_sparse(work->data);
    qoco_ir_wait_default(runtime);
    qoco_ir_exchange_stream(stream);
    const int count = s->Kn;
    const size_t bytes = count * sizeof(double);
    double* best = get_data_vectorf(work->xyzbuff1);
    double* norm = runtime->norm_scratch + 256;
    cudaGraph_t graph, ended;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle handle;
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault));
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0,
                                            cudaStreamCaptureModeThreadLocal));
    compute_linsys_residual(s, work, b, x, s->d_xyz_matrix_data, false);
    qoco_ir_device_norm(s->d_xyz_matrix_data, count, runtime->norm_scratch, stream);
    CUDA_CHECK(cudaMemcpyAsync(best, x, bytes, cudaMemcpyDeviceToDevice, stream));
    qoco_ir_initial<<<1, 1, 0, stream>>>(runtime->state, norm, handle, tolerance, maximum);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(stream, &ended));
    size_t size = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &size));
    std::vector<cudaGraphNode_t> initial(size);
    CUDA_CHECK(cudaGraphGetNodes(graph, initial.data(), &size));
    cudaGraphNodeParams params = {};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeWhile;
    params.conditional.size = 1;
    cudaGraphNode_t conditional;
    CUDA_CHECK(cudaGraphAddNode(&conditional, graph, initial.data(), size, &params));
    const auto body = params.conditional.phGraph_out[0];
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(stream, body, nullptr, nullptr, 0,
                                            cudaStreamCaptureModeThreadLocal));
    CUDA_CHECK(cudaMemcpyAsync(s->d_rhs_matrix_data, s->d_xyz_matrix_data, bytes,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(s->d_xyz_matrix_data, 0, bytes, stream));
    CUDSS_CHECK(g_cuda_funcs.cudssExecute(s->handle, CUDSS_PHASE_SOLVE, s->config, s->data,
                                         s->K_csr, s->d_xyz_matrix, s->d_rhs_matrix));
    qoco_axpy(s->d_xyz_matrix_data, x, x, 1.0, count);
    compute_linsys_residual(s, work, b, x, s->d_xyz_matrix_data, false);
    qoco_ir_device_norm(s->d_xyz_matrix_data, count, runtime->norm_scratch, stream);
    qoco_ir_decide<<<1, 1, 0, stream>>>(runtime->state, norm, handle, tolerance, maximum);
    qoco_ir_save_or_restore<<<(count + 255) / 256, 256, 0, stream>>>(runtime->state, x, best, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(stream, &ended));
    qoco_ir_exchange_stream(previous_stream);
    qoco_ir_snapshot_ranges(body, runtime, count);
    cudaGraphExec_t executable;
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    QocoIrState host;
    CUDA_CHECK(cudaMemcpyAsync(&host, runtime->state, sizeof(host), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    work->ir_iters += host.accepted;
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
}
