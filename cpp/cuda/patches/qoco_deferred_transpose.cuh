// SPDX-License-Identifier: Apache-2.0
// Solver compatibility objects. Sources are retained until the last dependent
// view is freed. The map belongs to QOCOProblemData and outlives its views.
// Numerical updates explicitly invalidate views; no update-time transpose work
// is needed until a caller requests a physical CSC matrix or a matrix product.
static void qoco_materialize_device_transpose(const QOCOMatrix* matrix)
{
    if (!matrix || !matrix->transpose_source || !matrix->transpose_values_pending) return;
    auto* result = const_cast<QOCOMatrix*>(matrix);
    auto* source = result->transpose_source;
    qoco_materialize_device_transpose(source);
    if (!result->d_csc) {
        auto* storage = qoco_make_gpu_transpose(source, result->transpose_map, true);
        free_qoco_csc_matrix(result->csc);
        result->csc = storage->csc;
        result->d_csc = storage->d_csc;
        result->d_csc_host = storage->d_csc_host;
        result->gather = storage->gather;
        qoco_free(storage); // Payload ownership moved; no source retained here.
    } else {
        const auto* input = source->d_csc_host;
        if (input->nnz) {
            qoco_transpose_entries<<<(input->nnz + 255) / 256, 256>>>(
                *input, source->gather->entries, source->gather->columns,
                result->d_csc_host->i, result->d_csc_host->x, nullptr);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    result->transpose_values_pending = 0;
    result->host_values_pending = 1;
}

extern "C" QOCOMatrix* qoco_gpu_transpose_deferred(QOCOMatrix* source, QOCOInt* map)
{
    auto* result = static_cast<QOCOMatrix*>(qoco_calloc(1, sizeof(QOCOMatrix)));
    result->reference_count = 1;
    result->lazy_host_mirror = 1;
    result->host_values_pending = 1;
    result->transpose_source = source;
    result->transpose_map = map;
    result->transpose_values_pending = 1;
    ++source->reference_count;
    result->csc = static_cast<QOCOCscMatrix*>(qoco_calloc(1, sizeof(QOCOCscMatrix)));
    result->csc->m = source->csc->n;
    result->csc->n = source->csc->m;
    result->csc->nnz = source->csc->nnz;
    return result;
}
