// SPDX-License-Identifier: Apache-2.0
// Queue the existing cone-boundary calculation into caller-owned device scratch.
// Requires max(ceil(l/256), ceil(nsoc/256)) + 1 doubles, plus a separate output.
extern "C" void qoco_gpu_enqueue_linesearch(QOCOFloat* u, QOCOFloat* direction,
    QOCOFloat factor, QOCOSolver* solver, QOCOFloat* storage, QOCOFloat* output) {
    const auto* w = solver->work;
    const auto* d = w->data;
    const int lp_blocks = (d->l + 255) / 256, soc_blocks = (d->nsoc + 255) / 256;
    auto* lp_step = storage;
    auto* partial = storage + 1;
    if (lp_blocks) {
        lp_step_length_kernel<<<lp_blocks, 256, 256 * sizeof(QOCOFloat)>>>(u, direction, d->l, partial);
        CUDA_CHECK(cudaGetLastError());
    }
    qoco_finish_cone_reduction<false><<<1, 256>>>(partial, lp_blocks, lp_step, nullptr, 1.0, false);
    CUDA_CHECK(cudaGetLastError());
    if (soc_blocks) {
        qoco_soc_steps<<<soc_blocks, 256>>>(u, direction, get_data_vectori(d->q),
            get_data_vectori(w->soc_idx), d->nsoc, lp_step, partial);
        CUDA_CHECK(cudaGetLastError());
    }
    qoco_finish_cone_reduction<false><<<1, 256>>>(partial, soc_blocks, output, lp_step, factor, true);
    CUDA_CHECK(cudaGetLastError());
}
