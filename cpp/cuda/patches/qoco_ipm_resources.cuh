// SPDX-License-Identifier: Apache-2.0
// Borrow the queued API's thread-local slots during IPM only. Initialisation and
// terminal reporting retain their original scope and resources.
#include <thread>
#include <utility>
struct QocoIpmResources {
    cublasHandle_t handle{};
    void* blas_workspace{};
    double* scalar_workspace{};
    size_t capacity{};
    std::thread::id owner = std::this_thread::get_id();
    int device = -1;
    bool active = false;
    void exchange() {
        std::swap(handle, qoco_scoped_reduction_handle);
        std::swap(blas_workspace, qoco_ipm_blas_workspace);
        std::swap(scalar_workspace, qoco_scalar_workspace);
        std::swap(capacity, qoco_scalar_workspace_capacity);
    }
};
extern "C" void* qoco_gpu_ipm_resources_enter(void* pointer) {
    if (!qoco_reduction_scope_depth) {
        fprintf(stderr, "Retained IPM resources require a queued reduction scope\n"); exit(1);
    }
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    auto* resources = static_cast<QocoIpmResources*>(pointer);
    if (!resources) {
        resources = new QocoIpmResources;
        resources->device = device;
        qoco_batched_stopping::blas(get_cuda_funcs()->cublasCreate(&resources->handle));
        constexpr size_t bytes = 32 * 1024 * 1024;
        CUDA_CHECK(cudaMalloc(&resources->blas_workspace, bytes));
        qoco_batched_stopping::blas(get_cuda_funcs()->cublasSetStream(
            resources->handle, cudaStreamPerThread));
        qoco_batched_stopping::blas(get_cuda_funcs()->cublasSetWorkspace(
            resources->handle, resources->blas_workspace, bytes));
    }
    if (resources->active || resources->device != device ||
        resources->owner != std::this_thread::get_id()) {
        fprintf(stderr, "Retained IPM workspace moved thread/device or entered recursively\n"); exit(1);
    }
    resources->exchange();
    resources->active = true;
    return resources;
}
extern "C" void qoco_gpu_ipm_resources_leave(void* pointer) {
    auto* resources = static_cast<QocoIpmResources*>(pointer);
    if (!resources || !resources->active) {
        fprintf(stderr, "Unbalanced retained IPM resources\n"); exit(1);
    }
    resources->exchange();
    resources->active = false;
}
extern "C" void qoco_gpu_ipm_resources_destroy(void* pointer) {
    auto* resources = static_cast<QocoIpmResources*>(pointer);
    if (!resources) return;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    if (resources->active || resources->device != device) {
        fprintf(stderr, "Invalid retained IPM resource destruction\n"); exit(1);
    }
    qoco_batched_stopping::blas(get_cuda_funcs()->cublasDestroy(resources->handle));
    CUDA_CHECK(cudaFree(resources->blas_workspace));
    CUDA_CHECK(cudaFree(resources->scalar_workspace));
    delete resources;
}
