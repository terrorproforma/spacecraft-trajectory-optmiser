// SPDX-License-Identifier: Apache-2.0
static __global__ void qoco_ipm_warm_condition(cudaGraphConditionalHandle handle,
    const QocoIpmParameters* parameters) {
    cudaGraphSetConditional(handle, parameters->warm_start != 0);
}
// Emit the warm-start IF directly into the enclosing graph; its condition is
// current device configuration rather than the value observed during capture.
extern "C" int qoco_gpu_ipm_warmstart(QOCOSolver* solver) {
    if (!qoco_ipm_current) return 0;
    auto& capture = *qoco_ipm_current;
    qoco_ipm_pause();
    const auto parent = capture.graph;
    cudaGraphConditionalHandle handle;
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, parent, 0, cudaGraphCondAssignDefault));
    qoco_ipm_resume();
    qoco_ipm_warm_condition<<<1, 1>>>(handle, capture.parameters);
    CUDA_CHECK(cudaGetLastError());
    qoco_ipm_pause();
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 1;
    cudaGraphNode_t node;
    CUDA_CHECK(cudaGraphAddNode(&node, parent, capture.dependencies.data(),
        capture.dependencies.size(), &params));
    capture.graph = params.conditional.phGraph_out[0];
    capture.dependencies.clear();
    qoco_ipm_resume();
    auto* work = solver->work;
    auto* data = work->data;
    auto* x = get_data_vectorf(work->x);
    ew_product(get_data_vectorf(work->x0), get_data_vectorf(work->scaling->Dinvruiz), x, data->n);
    if (data->m) {
        auto* s = get_data_vectorf(work->s);
        SpMv(data->G, x, s);
        qoco_axpy(s, get_data_vectorf(data->h), s, -1.0, data->m);
        bring2cone(s, get_data_vectori(work->soc_idx), data);
    }
    qoco_ipm_pause();
    capture.graph = parent;
    capture.dependencies = {node};
    qoco_ipm_resume();
    return 1;
}
