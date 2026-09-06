// SPDX-License-Identifier: Apache-2.0
#include "qoco_ipm_parameters.cuh"
extern "C" void* qoco_gpu_ipm_resources_enter(void*);
extern "C" void qoco_gpu_ipm_resources_leave(void*);
extern "C" void qoco_gpu_ipm_resources_destroy(void*);
struct QocoIpmCache {
    cudaGraph_t graph{}, linear{};
    cudaGraphExec_t executable{};
    QocoIpmParameters* parameters{};
    int* iterations{};
    void* resources{};
    const void *work{}, *control{};
    double static_p{}, static_a{}, static_g{}, linsys_static_p{};
    size_t builds{}, launches{};
};
__global__ void qoco_ipm_ir_parameters(QocoIrParameters* output, const QocoIpmParameters* p) {
    output->tolerance = p->ir_tolerance;
    output->maximum = p->ir_maximum;
}
static void qoco_ipm_cache_clear(QocoIpmCache& cache) {
    if (cache.executable) CUDA_CHECK(cudaGraphExecDestroy(cache.executable));
    if (cache.graph) CUDA_CHECK(cudaGraphDestroy(cache.graph));
    if (cache.linear) CUDA_CHECK(cudaGraphDestroy(cache.linear));
    cache.executable = nullptr;
    cache.graph = cache.linear = nullptr;
}
static void qoco_ipm_cache_destroy(QocoIpmCache& cache) {
    if (!cache.resources) return;
    CUDA_CHECK(cudaDeviceSynchronize());
    if (getenv("SPACEPDHCG_TEST_QOCO_IPM_CACHE_TRACE"))
        fprintf(stderr, "IPM_CACHE builds=%zu launches=%zu\n", cache.builds, cache.launches);
    qoco_ipm_cache_clear(cache);
    CUDA_CHECK(cudaFree(cache.parameters));
    CUDA_CHECK(cudaFree(cache.iterations));
    qoco_gpu_ipm_resources_destroy(cache.resources);
    cache = {};
}
