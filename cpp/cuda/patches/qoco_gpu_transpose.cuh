// SPDX-License-Identifier: Apache-2.0
// Included after qoco_gather.cuh in cuda_linalg.cu. The source's stable GPU
// row ordering is already the CSC order of its transpose. Each result owns
// its storage independently; no source pointers survive construction.

__global__ void qoco_transpose_entries(const QOCOCscMatrix source,
                                       const int* entries, const int* columns,
                                       int* rows, double* values, int* inverse)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= source.nnz) return;
    const int original = entries[k];
    rows[k] = columns[original];
    values[k] = source.x[original];
    if (inverse) inverse[original] = k;
}

static QOCOMatrix* qoco_make_gpu_transpose(const QOCOMatrix* source,
                                           QOCOInt* source_to_transpose, bool lazy)
{
#ifdef SPACEPDHCG_QOCO_DEFERRED_TRANSPOSES
    qoco_materialize_device_transpose(source);
#endif
    const auto* input = source->d_csc_host;
    const size_t count = static_cast<size_t>(input->nnz);
    auto* result = static_cast<QOCOMatrix*>(qoco_malloc(sizeof(QOCOMatrix)));
    result->gather = nullptr;
#ifdef SPACEPDHCG_QOCO_LAZY_HOST_MIRRORS
    result->lazy_host_mirror = lazy;
    result->host_values_pending = lazy;
#endif
#ifdef SPACEPDHCG_QOCO_DEFERRED_TRANSPOSES
    result->reference_count = 1;
    result->transpose_source = nullptr;
    result->transpose_map = nullptr;
    result->transpose_values_pending = 0;
#endif
    auto* host = static_cast<QOCOCscMatrix*>(qoco_malloc(sizeof(QOCOCscMatrix)));
    auto* device = static_cast<QOCOCscMatrix*>(qoco_malloc(sizeof(QOCOCscMatrix)));
    *host = {}; *device = {};
    host->m = device->m = input->n;
    host->n = device->n = input->m;
    host->nnz = device->nnz = input->nnz;
    const size_t offset_bytes = (static_cast<size_t>(input->m) + 1) * sizeof(int);
    if (!lazy) host->p = static_cast<int*>(qoco_malloc(offset_bytes));
    if (count) {
        if (!lazy) {
            host->i = static_cast<int*>(qoco_malloc(count * sizeof(int)));
            host->x = static_cast<double*>(qoco_malloc(count * sizeof(double)));
        }
        CUDA_CHECK(cudaMalloc(&device->i, count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&device->x, count * sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc(&device->p, offset_bytes));
    CUDA_CHECK(cudaMemcpyAsync(device->p, source->gather->offsets, offset_bytes,
                               cudaMemcpyDeviceToDevice));
    int* inverse = nullptr;
    if (count) {
        if (source_to_transpose) CUDA_CHECK(cudaMalloc(&inverse, count * sizeof(int)));
        qoco_transpose_entries<<<(input->nnz + 255) / 256, 256>>>(
            *input, source->gather->entries, source->gather->columns,
            device->i, device->x, inverse);
        CUDA_CHECK(cudaGetLastError());
    }
    result->csc = host;
    result->d_csc_host = device;
    CUDA_CHECK(cudaMalloc(&result->d_csc, sizeof(QOCOCscMatrix)));
    CUDA_CHECK(cudaMemcpy(result->d_csc, device, sizeof(QOCOCscMatrix), cudaMemcpyHostToDevice));
    // Compatibility mirrors for host Ruiz, inspection and legacy numeric
    // updates. No host transpose, sort, counting or inverse-map loop remains.
    if (!lazy) CUDA_CHECK(cudaMemcpy(host->p, device->p, offset_bytes, cudaMemcpyDeviceToHost));
    if (count) {
        if (!lazy) {
            CUDA_CHECK(cudaMemcpy(host->i, device->i, count * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(host->x, device->x, count * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (source_to_transpose) {
            CUDA_CHECK(cudaMemcpy(source_to_transpose, inverse, count * sizeof(int),
                                   cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(inverse));
        }
    }
    qoco_gpu_create_gather(result);
    return result;
}

extern "C" QOCOMatrix* qoco_gpu_transpose(const QOCOMatrix* source, QOCOInt* map)
{
    return qoco_make_gpu_transpose(source, map, false);
}
#ifdef SPACEPDHCG_QOCO_LAZY_HOST_MIRRORS
extern "C" QOCOMatrix* qoco_gpu_transpose_lazy(const QOCOMatrix* source, QOCOInt* map)
{
    return qoco_make_gpu_transpose(source, map, true);
}
#endif
