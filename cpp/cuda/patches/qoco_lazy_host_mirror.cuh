// SPDX-License-Identifier: Apache-2.0
// Only opt-in GPU transpose matrices use this cache. Callers must use the
// existing CPU-mode accessor before reading/writing the host representation.
static void qoco_materialize_host_mirror(const QOCOMatrix* matrix)
{
#ifdef SPACEPDHCG_QOCO_DEFERRED_TRANSPOSES
    qoco_materialize_device_transpose(matrix);
#endif
    if (!matrix->lazy_host_mirror || !matrix->host_values_pending) return;
    auto* host = matrix->csc;
    const auto* device = matrix->d_csc_host;
    const size_t count = static_cast<size_t>(device->nnz);
    if (!host->p) {
        const size_t offset_bytes = (static_cast<size_t>(device->n) + 1) * sizeof(int);
        host->p = static_cast<int*>(qoco_malloc(offset_bytes));
        CUDA_CHECK(cudaMemcpy(host->p, device->p, offset_bytes, cudaMemcpyDeviceToHost));
        if (count) {
            host->i = static_cast<int*>(qoco_malloc(count * sizeof(int)));
            host->x = static_cast<double*>(qoco_malloc(count * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(host->i, device->i, count * sizeof(int), cudaMemcpyDeviceToHost));
        }
    }
    if (count)
        CUDA_CHECK(cudaMemcpy(host->x, device->x, count * sizeof(double), cudaMemcpyDeviceToHost));
    matrix->host_values_pending = 0;
}

static bool qoco_device_mirror_current(const QOCOMatrix* matrix)
{
    return matrix && matrix->lazy_host_mirror && matrix->host_values_pending;
}
