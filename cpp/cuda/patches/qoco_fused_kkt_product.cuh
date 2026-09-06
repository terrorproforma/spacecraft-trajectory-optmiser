// SPDX-License-Identifier: Apache-2.0
// Fuse only the sparse portion of kkt_multiply. NT products and refinement
// decisions retain their existing implementation and stopping thresholds.
#include "qoco.h"

namespace qoco_fused_kkt {
struct Matrix {
    const QOCOCscMatrix* csc;
    const int *offsets, *entries, *columns;
};
static Matrix matrix(const QOCOMatrix* m) {
    if (!m) return {};
    return {m->d_csc, m->gather->offsets, m->gather->entries, m->gather->columns};
}
__device__ double row(Matrix m, const double* x, int r) {
    double sum=0.0;
    for (int k=m.offsets[r];k<m.offsets[r+1];++k) {
        const int e=m.entries[k];
        sum+=m.csc->x[e]*x[m.columns[e]];
    }
    return sum;
}
__device__ double transpose(Matrix m, const double* x, int c) {
    double sum=0.0;
    for (int e=m.csc->p[c];e<m.csc->p[c+1];++e)
        sum+=m.csc->x[e]*x[m.csc->i[e]];
    return sum;
}
__global__ void product(Matrix P, Matrix A, Matrix G, int n, int p, int m,
                        const double* x, double* y) {
    const int r=blockIdx.x*blockDim.x+threadIdx.x;
    if (r<n) {
        double sum=0.0;
        if (P.csc) {
            sum=row(P,x,r);
            // Match the original symmetric gather's mirrored off-diagonal loop.
            for (int e=P.csc->p[r];e<P.csc->p[r+1];++e)
                if (P.csc->i[e]!=r) sum+=P.csc->x[e]*x[P.csc->i[e]];
        }
        if (p) sum=__dadd_rn(sum,transpose(A,x+n,r));
        if (m) sum=__dadd_rn(sum,transpose(G,x+n+p,r));
        y[r]=sum;
    } else if (r<n+p) y[r]=row(A,x,r-n);
    else if (r<n+p+m) y[r]=row(G,x,r-n-p);
}
} // namespace qoco_fused_kkt

extern "C" void qoco_gpu_kkt_sparse(QOCOFloat* x,QOCOFloat* y,QOCOProblemData* d) {
    // Compatibility views may have pending device transposes. Materialize them
    // just as the separate USpMv/SpMv/SpMtv calls do before reading their maps.
    if (d->P) qoco_materialize_device_transpose(d->P);
    if (d->p) qoco_materialize_device_transpose(d->A);
    if (d->m) qoco_materialize_device_transpose(d->G);
    const int count=d->n+d->p+d->m;
    if (count) {
        qoco_fused_kkt::product<<<(count+255)/256,256>>>(
            qoco_fused_kkt::matrix(d->P),qoco_fused_kkt::matrix(d->A),
            qoco_fused_kkt::matrix(d->G),d->n,d->p,d->m,x,y);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(qoco_complete_vector_operation());
    }
}
