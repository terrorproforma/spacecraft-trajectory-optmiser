// SPDX-License-Identifier: Apache-2.0
// Preserve the reference Jordan products and subtraction order while combining
// product, identity shift, negation and correction in one thread per cone.
__global__ void qoco_combined_cones(double* u1, const double* u2, double* u3,
    const double* lambda, double* ds, int l, int nsoc, int m, const int* q,
    const int* starts, const double* sigma, const double* sz) {
    const long long total = static_cast<long long>(l) + nsoc;
    const double mu = m > 0 ? __ddiv_rn(*sz, static_cast<double>(m)) : 0.0;
    const double sm = __dmul_rn(*sigma, mu);
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        if (i < l) {
            u3[i] = __dsub_rn(__dmul_rn(u1[i], u2[i]), sm);
            u1[i] = __dmul_rn(lambda[i], lambda[i]);
            ds[i] = __dsub_rn(-u1[i], u3[i]);
        } else {
            const int cone = i - l, start = starts[cone], size = q[cone];
            soc_product(u1 + start, u2 + start, u3 + start, size);
            u3[start] = __dsub_rn(u3[start], sm);
            soc_product(lambda + start, lambda + start, u1 + start, size);
            for (int k = 0; k < size; ++k)
                ds[start + k] = __dsub_rn(-u1[start + k], u3[start + k]);
        }
    }
}
extern "C" void qoco_gpu_enqueue_combined_cones(QOCOWorkspace* w,
    const double* sigma, const double* sz) {
    auto* d = w->data;
    const long long total = static_cast<long long>(d->l) + d->nsoc;
    if (total) qoco_combined_cones<<<static_cast<int>(std::min(256LL, (total - 1) / 256 + 1)), 256>>>(
        get_data_vectorf(w->ubuff1), get_data_vectorf(w->ubuff2), get_data_vectorf(w->ubuff3),
        get_data_vectorf(w->lambda), get_data_vectorf(w->Ds), d->l, d->nsoc, d->m,
        get_data_vectori(d->q), get_data_vectori(w->soc_idx), sigma, sz);
    CUDA_CHECK(cudaGetLastError());
}
