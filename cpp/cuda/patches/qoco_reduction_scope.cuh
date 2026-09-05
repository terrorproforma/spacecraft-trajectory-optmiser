// SPDX-License-Identifier: Apache-2.0
// Optional QOCO-GPU extension. Included after cudss_backend.h by prepare_qoco_gpu.py.
// A scope belongs to its calling host thread and CUDA device. It never survives
// the enclosing solve, so workspace/library destruction leaves no cached GPU handle.

static thread_local cublasHandle_t qoco_scoped_reduction_handle = nullptr;
static thread_local unsigned int qoco_reduction_scope_depth = 0;
static thread_local int qoco_reduction_scope_device = -1;

#ifdef SPACEPDHCG_QOCO_DEVICE_CONE_REDUCTIONS
static thread_local double* qoco_scalar_workspace = nullptr;
static thread_local size_t qoco_scalar_workspace_capacity = 0;
extern "C" cudaError_t qoco_gpu_acquire_scalar_workspace(size_t count, double** storage,
                                                         int* temporary)
{
    *temporary = qoco_reduction_scope_depth == 0;
    if (*temporary) return cudaMalloc(storage, count * sizeof(double));
    if (count > qoco_scalar_workspace_capacity) {
        auto status = cudaFree(qoco_scalar_workspace);
        qoco_scalar_workspace = nullptr; qoco_scalar_workspace_capacity = 0;
        if (status != cudaSuccess) return status;
        status = cudaMalloc(&qoco_scalar_workspace, count * sizeof(double));
        if (status != cudaSuccess) return status;
        qoco_scalar_workspace_capacity = count;
    }
    *storage = qoco_scalar_workspace;
    return cudaSuccess;
}
#endif

// Only the explicitly prepared queued-operator build changes completion behavior.
// All affected kernels and scoped cuBLAS calls use the default stream. Host
// scalar reads remain synchronous; the outermost scope also checks completion.
static cudaError_t qoco_complete_vector_operation()
{
#ifdef SPACEPDHCG_QOCO_QUEUED_OPERATORS
    if (qoco_reduction_scope_depth != 0) return cudaSuccess;
#endif
    return cudaDeviceSynchronize();
}

extern "C" int qoco_gpu_begin_reduction_scope()
{
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || !load_cuda_libraries()) return 1;
    if (qoco_reduction_scope_depth != 0) {
        if (device != qoco_reduction_scope_device) return 1;
        ++qoco_reduction_scope_depth;
        return 0;
    }
    cublasHandle_t handle = nullptr;
    if (get_cuda_funcs()->cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 1;
    qoco_scoped_reduction_handle = handle;
    qoco_reduction_scope_device = device;
    qoco_reduction_scope_depth = 1;
    return 0;
}

extern "C" void qoco_gpu_end_reduction_scope()
{
    if (qoco_reduction_scope_depth == 0 || --qoco_reduction_scope_depth != 0) return;
#ifdef SPACEPDHCG_QOCO_QUEUED_OPERATORS
    const auto status = cudaStreamSynchronize(nullptr);
    if (status != cudaSuccess) {
        fprintf(stderr, "QOCO queued operators failed: %s\n", cudaGetErrorString(status));
        exit(1);
    }
#endif
#ifdef SPACEPDHCG_QOCO_DEVICE_CONE_REDUCTIONS
    const auto released = cudaFree(qoco_scalar_workspace);
    qoco_scalar_workspace = nullptr;
    qoco_scalar_workspace_capacity = 0;
    if (released != cudaSuccess) {
        fprintf(stderr, "QOCO scalar workspace release failed: %s\n", cudaGetErrorString(released));
        exit(1);
    }
#endif
    get_cuda_funcs()->cublasDestroy(qoco_scoped_reduction_handle);
    qoco_scoped_reduction_handle = nullptr;
    qoco_reduction_scope_device = -1;
}

static cublasHandle_t qoco_acquire_reduction_handle(bool* temporary)
{
    *temporary = qoco_reduction_scope_depth == 0;
    if (!*temporary) return qoco_scoped_reduction_handle;
    cublasHandle_t handle = nullptr;
    if (!load_cuda_libraries()
        || get_cuda_funcs()->cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "QOCO GPU reduction handle creation failed\n");
        exit(1);
    }
    return handle;
}
