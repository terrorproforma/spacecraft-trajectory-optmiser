// SPDX-License-Identifier: Apache-2.0
// Included after CUDA_CHECK. Topology is sorted on the GPU; numerical updates
// read the original CSC values, so scaling never requires duplicating values.
#include <cub/device/device_radix_sort.cuh>

struct QocoGpuGather {
    int* offsets{};
    int* columns{};
    int* entries{};
    int* keys_in{};
    int* keys_out{};
    int* entries_in{};
    void* scratch{};
    size_t scratch_bytes{};
};

__global__ void qoco_gather_keys(const QOCOCscMatrix* matrix, int* rows,
                                int* entries, int* columns)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= matrix->n) return;
    for (int j = matrix->p[col]; j < matrix->p[col + 1]; ++j) {
        rows[j] = matrix->i[j];
        entries[j] = j;
        columns[j] = col;
    }
}

__global__ void qoco_gather_offsets(const int* sorted_rows, int nnz, int rows, int* offsets)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > rows) return;
    int lo = 0, hi = nnz;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (sorted_rows[mid] < row) lo = mid + 1;
        else hi = mid;
    }
    offsets[row] = lo;
}

static void qoco_gpu_refresh_gather(QOCOMatrix* matrix)
{
    auto* map = matrix->gather;
    if (!map) return;
    const auto* csc = matrix->d_csc_host;
    if (csc->nnz > 0) {
        qoco_gather_keys<<<(csc->n + 255) / 256, 256>>>(
            matrix->d_csc, map->keys_in, map->entries_in, map->columns);
        CUDA_CHECK(cudaGetLastError());
        // Stable sorting preserves CSC entry order within each row.
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            map->scratch, map->scratch_bytes, map->keys_in, map->keys_out,
            map->entries_in, map->entries, csc->nnz));
    }
    qoco_gather_offsets<<<(csc->m + 256) / 256, 256>>>(
        map->keys_out, csc->nnz, csc->m, map->offsets);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void qoco_gpu_create_gather(QOCOMatrix* matrix)
{
    auto* map = new QocoGpuGather{};
    matrix->gather = map;
    const auto* csc = matrix->d_csc_host;
    CUDA_CHECK(cudaMalloc(&map->offsets, (static_cast<size_t>(csc->m) + 1) * sizeof(int)));
    if (csc->nnz > 0) {
        const size_t bytes = static_cast<size_t>(csc->nnz) * sizeof(int);
        CUDA_CHECK(cudaMalloc(&map->columns, bytes));
        CUDA_CHECK(cudaMalloc(&map->entries, bytes));
        CUDA_CHECK(cudaMalloc(&map->keys_in, bytes));
        CUDA_CHECK(cudaMalloc(&map->keys_out, bytes));
        CUDA_CHECK(cudaMalloc(&map->entries_in, bytes));
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, map->scratch_bytes, map->keys_in, map->keys_out,
            map->entries_in, map->entries, csc->nnz));
        CUDA_CHECK(cudaMalloc(&map->scratch, map->scratch_bytes));
    }
    qoco_gpu_refresh_gather(matrix);
}

static void qoco_gpu_free_gather(QOCOMatrix* matrix)
{
    auto* map = matrix->gather;
    if (!map) return;
    CUDA_CHECK(cudaFree(map->offsets));
    CUDA_CHECK(cudaFree(map->columns));
    CUDA_CHECK(cudaFree(map->entries));
    CUDA_CHECK(cudaFree(map->keys_in));
    CUDA_CHECK(cudaFree(map->keys_out));
    CUDA_CHECK(cudaFree(map->entries_in));
    CUDA_CHECK(cudaFree(map->scratch));
    delete map;
    matrix->gather = nullptr;
}

template<bool Symmetric>
__global__ void qoco_gather_product(const QOCOCscMatrix* matrix, const int* offsets,
                                   const int* entries, const int* columns,
                                   const QOCOFloat* vector, QOCOFloat* result)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= matrix->m) return;
    QOCOFloat sum = 0.0;
    for (int k = offsets[row]; k < offsets[row + 1]; ++k) {
        const int entry = entries[k];
        sum += matrix->x[entry] * vector[columns[entry]];
    }
    if (Symmetric) {
        // The input stores one triangle. Gather its mirrored off-diagonal part.
        for (int k = matrix->p[row]; k < matrix->p[row + 1]; ++k) {
            if (matrix->i[k] != row) sum += matrix->x[k] * vector[matrix->i[k]];
        }
    }
    result[row] = sum;
}

template<bool Symmetric>
static void qoco_gpu_product(const QOCOMatrix* matrix, const QOCOFloat* vector, QOCOFloat* result)
{
    if (matrix->d_csc_host->m == 0) return;
    const auto* map = matrix->gather;
    qoco_gather_product<Symmetric><<<(matrix->d_csc_host->m + 255) / 256, 256>>>(
        matrix->d_csc, map->offsets, map->entries, map->columns, vector, result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
