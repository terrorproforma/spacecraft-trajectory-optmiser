// SPDX-License-Identifier: Apache-2.0
// Keep centering's device scalar results alive until the combined RHS consumes
// them. Only the legacy host sigma metadata is downloaded after the chain.
#include <vector>
extern "C" void qoco_gpu_enqueue_combined_cones(QOCOWorkspace*, const double*, const double*);
extern "C" void construct_kkt_comb_rhs(QOCOWorkspace*);
#ifdef SPACEPDHCG_QOCO_QUEUED_CENTERING_METADATA
// Keep the existing workspace layout, but pin its storage for nonblocking
// metadata downloads. Its lifetime already encloses every IPM iteration.
extern "C" void* qoco_gpu_allocate_workspace(size_t bytes) {
    void* result{};
    CUDA_CHECK(cudaHostAlloc(&result, bytes, cudaHostAllocDefault));
    return result;
}
extern "C" void qoco_gpu_free_workspace(void* pointer) {
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaFreeHost(pointer));
}
#endif

extern "C" void qoco_gpu_center_and_combine(QOCOSolver* solver) {
    using namespace qoco_device_step_control;
    begin();
    auto* w = solver->work; const auto* d = w->data;
    const int n = d->n, p = d->p, m = d->m, total = n + p + m;
    const char* mode = std::getenv("SPACEPDHCG_TEST_QOCO_COMBINED_RHS_COMPARE");
    const bool audit = mode && mode[0] == '1';
    std::vector<double> affine, expected_ds, expected_rhs;
    if (audit) {
        affine.resize(m); expected_ds.resize(m); expected_rhs.resize(total);
        if (m) CUDA_CHECK(cudaMemcpy(affine.data(), w->Ds->d_data, m * sizeof(double), cudaMemcpyDeviceToHost));
        qoco_reference_compute_centering(solver);
        construct_kkt_comb_rhs(w);
        if (m) CUDA_CHECK(cudaMemcpy(expected_ds.data(), w->Ds->d_data, m * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(expected_rhs.data(), w->rhs->d_data, total * sizeof(double), cudaMemcpyDeviceToHost));
        if (m) CUDA_CHECK(cudaMemcpy(w->Ds->d_data, affine.data(), m * sizeof(double), cudaMemcpyHostToDevice));
    }
    qoco_gpu_compute_centering_impl(solver, false);
    // The enclosing scope retains this allocation. None of the following cone
    // or vector operations acquire scalar scratch or mutate s/z.
    double* scalar = workspace(solver);
    auto* nt = w->nt_scaling->d_data;
    auto* ntidx = w->nt_scaling_soc_idx->d_data;
    auto* starts = w->soc_idx->d_data; auto* q = d->q->d_data;
    auto* u1 = w->ubuff1->d_data; auto* u2 = w->ubuff2->d_data;
    auto* rhs = w->rhs->d_data;
    copy_and_negate_arrayf(w->kktres->d_data, rhs, total);
    nt_multiply_inv(nt, ntidx, starts, w->Ds->d_data, u1, d->l, m, d->nsoc, q);
    nt_multiply(nt, ntidx, starts, w->xyz->d_data + n + p, u2, d->l, m, d->nsoc, q);
    qoco_gpu_enqueue_combined_cones(w, scalar + 5, scalar + 4);
    cone_division(w->lambda->d_data, w->Ds->d_data, u2, d->l, d->nsoc, q, starts);
    nt_multiply(nt, ntidx, starts, u2, u1, d->l, m, d->nsoc, q);
    if (m) qoco_axpy(u1, rhs + n + p, rhs + n + p, -1.0, m);
#ifdef SPACEPDHCG_QOCO_QUEUED_CENTERING_METADATA
    // No production host code consumes sigma before the next synchronous solve
    // result or scope completion. Default-stream order protects the scratch
    // source against subsequent reuse. Audit downloads below also complete it.
    CUDA_CHECK(cudaMemcpyAsync(&w->sigma, scalar + 5, sizeof(double), cudaMemcpyDeviceToHost));
#else
    CUDA_CHECK(cudaMemcpy(&w->sigma, scalar + 5, sizeof(double), cudaMemcpyDeviceToHost));
#endif
    if (audit) {
        std::vector<double> got_ds(m), got_rhs(total);
        if (m) CUDA_CHECK(cudaMemcpy(got_ds.data(), w->Ds->d_data, m * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got_rhs.data(), rhs, total * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < m; ++i) compare(got_ds[i], expected_ds[i]);
        for (int i = 0; i < total; ++i) compare(got_rhs[i], expected_rhs[i]);
    }
    qoco_gpu_end_reduction_scope();
}
