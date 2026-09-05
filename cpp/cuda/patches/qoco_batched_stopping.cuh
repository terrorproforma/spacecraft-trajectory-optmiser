// SPDX-License-Identifier: Apache-2.0
// Included at the end of prepared CUDA algebra. Reuse the existing sparse
// operators and cuBLAS summation; only scalar result placement is changed.
namespace qoco_batched_stopping {
enum Slot { B, S, C, H, ATY, GTZ, PX, AX, GX, EQ, CONIC, DUAL,
            XPX, GAP, CTX, BTY, HTZ, Count, RAW_XPX = Count, XX, SZ };
inline void blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "QOCO batched stopping cuBLAS error %d\n", static_cast<int>(status));
        exit(1);
    }
}
inline void norm(const double* input, int count, double* output, double* partial) {
    if (!count) { CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(double))); return; }
    const int blocks = count <= 4096 ? 1 : std::min(256, (count - 1) / 256 + 1);
    using namespace qoco_device_scalar;
    reduce<Maximum, false><<<blocks, 256>>>(input, count, blocks == 1 ? output : partial);
    if (blocks > 1) reduce<Maximum, true><<<1, 256>>>(partial, blocks, output);
    CUDA_CHECK(cudaGetLastError());
}
template<bool Iteration>
__global__ void finish(const double* v, double kinv, double k, double reg, int m, double* out) {
    const double pres = qoco_max(v[EQ], v[CONIC]);
    double pres_rel = qoco_max(v[AX], v[B]);
    pres_rel = qoco_max(pres_rel, v[GX]);
    pres_rel = qoco_max(pres_rel, v[H]);
    pres_rel = qoco_max(pres_rel, v[S]);
    double dres_rel = qoco_max(v[PX], v[ATY]);
    dres_rel = qoco_max(dres_rel, v[GTZ]);
    dres_rel = qoco_max(dres_rel, v[C]);
    const double half = __dmul_rn(0.5, v[XPX]);
    const double pobj = fabs(__dadd_rn(half, v[CTX]));
    const double dobj = fabs(__dadd_rn(__dadd_rn(-half, -v[BTY]), -v[HTZ]));
    const double gap_rel = qoco_max(qoco_max(1.0, pobj), dobj);
    out[0] = pres; out[1] = v[DUAL]; out[2] = __dmul_rn(v[GAP], kinv);
    out[3] = pres_rel; out[4] = __dmul_rn(dres_rel, kinv); out[5] = gap_rel;
    if constexpr (Iteration) {
        const double corrected = __dadd_rn(v[RAW_XPX], -__dmul_rn(reg, v[XX]));
        const double obj = __dadd_rn(v[CTX], __dmul_rn(0.5, corrected));
        out[6] = fabs(k) > 1e-15 ? __ddiv_rn(obj, k) : QOCOFloat_MAX;
        out[7] = m ? __ddiv_rn(v[SZ], static_cast<double>(m)) : 0.0;
    }
}
} // namespace qoco_batched_stopping

// Returns [primal, dual, gap, primal scale, dual scale, gap scale]. Scratch and
// cuBLAS pointer mode belong to the enclosing scope, including unscoped callers.
template<bool Iteration>
static void qoco_gpu_metrics(QOCOSolver* solver, double* output) {
    using namespace qoco_batched_stopping;
    if (qoco_gpu_begin_reduction_scope() != 0) { fprintf(stderr, "QOCO stopping scope failed\n"); exit(1); }
    double* scratch{}; int temporary{};
    constexpr int slots = Iteration ? SZ + 1 : Count;
    constexpr int results = Iteration ? 8 : 6;
    CUDA_CHECK(qoco_gpu_acquire_scalar_workspace(slots + results + 256, &scratch, &temporary));
    double* result = scratch + slots;
    double* partial = result + results;
    auto* funcs = get_cuda_funcs();
    bool temporary_handle{};
    auto handle = qoco_acquire_reduction_handle(&temporary_handle);
    cublasPointerMode_t mode{};
    blas(funcs->cublasGetPointerMode(handle, &mode));
    blas(funcs->cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    const auto dot = [&](const double* a, const double* b, int n, Slot slot) {
        if (n) blas(funcs->cublasDdot(handle, n, a, 1, b, 1, scratch + slot));
        else CUDA_CHECK(cudaMemsetAsync(scratch + slot, 0, sizeof(double)));
    };
    auto* w = solver->work; auto* d = w->data; auto* scale = w->scaling;
    auto *xb = w->xbuff->d_data, *yb = w->ybuff->d_data, *u1 = w->ubuff1->d_data,
         *u2 = w->ubuff2->d_data, *u3 = w->ubuff3->d_data, *res = w->kktres->d_data;
    auto *di = scale->Dinvruiz->d_data, *ei = scale->Einvruiz->d_data,
         *fi = scale->Finvruiz->d_data, *f = scale->Fruiz->d_data;
    auto *x = w->x->d_data, *y = w->y->d_data, *z = w->z->d_data, *s = w->s->d_data;
    auto *c = d->c->d_data, *b = d->b->d_data, *h = d->h->d_data;
    const auto product_norm = [&](double* a, const double* b, double* out, int n, Slot slot) {
        if (n) ew_product(a, b, out, n);
        norm(out, n, scratch + slot, partial);
    };
    product_norm(ei, b, yb, d->p, B);
    product_norm(f, s, u1, d->m, S);
    product_norm(di, c, xb, d->n, C);
    product_norm(fi, h, u3, d->m, H);
    if (d->p) SpMtv(d->A, y, xb);
    product_norm(xb, di, xb, d->p ? d->n : 0, ATY);
    if (d->m) SpMtv(d->G, z, xb);
    product_norm(xb, di, xb, d->m ? d->n : 0, GTZ);
    USpMv(d->P, x, xb);
    if constexpr (Iteration) dot(xb, x, d->n, RAW_XPX);
    qoco_axpy(x, xb, xb, -solver->settings->kkt_static_reg_P, d->n);
    product_norm(xb, di, xb, d->n, PX);
    dot(x, xb, d->n, XPX);
    if (d->p) SpMv(d->A, x, yb);
    product_norm(yb, ei, yb, d->p, AX);
    if (d->m) SpMv(d->G, x, u1);
    product_norm(u1, fi, u1, d->m, GX);
    product_norm(res + d->n, ei, yb, d->p, EQ);
    product_norm(res + d->n + d->p, fi, u1, d->m, CONIC);
    ew_product(res, di, xb, d->n);
    scale_arrayf(xb, xb, scale->kinv, d->n);
    norm(xb, d->n, scratch + DUAL, partial);
    if (d->m) { ew_product(s, f, u1, d->m); ew_product(z, f, u2, d->m); }
    dot(u1, u2, d->m, GAP);
    dot(c, x, d->n, CTX); dot(b, y, d->p, BTY); dot(h, z, d->m, HTZ);
    if constexpr (Iteration) { dot(x, x, d->n, XX); dot(s, z, d->m, SZ); }
    finish<Iteration><<<1, 1>>>(scratch, scale->kinv, scale->k,
        solver->settings->kkt_static_reg_P, d->m, result);
    CUDA_CHECK(cudaGetLastError());
    blas(funcs->cublasSetPointerMode(handle, mode));
    CUDA_CHECK(cudaMemcpy(output, result, results * sizeof(double), cudaMemcpyDeviceToHost));
    qoco_gpu_end_reduction_scope();
}
extern "C" void qoco_gpu_stopping_metrics(QOCOSolver* solver, double* output) {
    qoco_gpu_metrics<false>(solver, output);
}
#ifdef SPACEPDHCG_QOCO_BATCHED_ITERATION
// Appends the objective and mu, sharing P*x and c'*x with the stopping metrics.
extern "C" void qoco_gpu_iteration_metrics(QOCOSolver* solver, double* output) {
    qoco_gpu_metrics<true>(solver, output);
}
#endif
