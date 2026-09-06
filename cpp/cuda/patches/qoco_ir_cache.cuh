// SPDX-License-Identifier: Apache-2.0
// One retained refinement graph per fixed-topology cuDSS workspace.
struct QocoIrParameters { double tolerance; int maximum; };
struct QocoIrCache {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    QocoIrParameters* parameters{};
    const void* work{};
    const double* rhs{};
    double* solution{};
    int count{};
    size_t builds{}, launches{};
};
__global__ void qoco_ir_set_parameters(QocoIrParameters* parameters, double tolerance, int maximum) {
    parameters->tolerance = tolerance;
    parameters->maximum = maximum;
}
// The caller orders completion before invalidation or workspace destruction.
static void qoco_ir_cache_clear(QocoIrCache& cache) {
    if (cache.executable) CUDA_CHECK(cudaGraphExecDestroy(cache.executable));
    if (cache.graph) CUDA_CHECK(cudaGraphDestroy(cache.graph));
    cache.executable = nullptr;
    cache.graph = nullptr;
}
