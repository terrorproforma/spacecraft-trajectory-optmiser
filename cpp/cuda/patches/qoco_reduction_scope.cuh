// SPDX-License-Identifier: Apache-2.0
// Optional QOCO-GPU extension. Included after cudss_backend.h by prepare_qoco_gpu.py.
// A scope belongs to its calling host thread and CUDA device. It never survives
// the enclosing solve, so workspace/library destruction leaves no cached GPU handle.

static thread_local cublasHandle_t qoco_scoped_reduction_handle = nullptr;
static thread_local unsigned int qoco_reduction_scope_depth = 0;
static thread_local int qoco_reduction_scope_device = -1;

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
