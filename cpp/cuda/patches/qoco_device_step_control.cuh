// SPDX-License-Identifier: Apache-2.0
// Included after batched stopping in the CUDA algebra. Default-stream ordering
// keeps cone results and cuBLAS scalar outputs on device until their consumers.
#include <cmath>
#include <cstdlib>
extern "C" void qoco_gpu_enqueue_linesearch(double*, double*, double, QOCOSolver*, double*, double*);
extern "C" void qoco_reference_compute_centering(QOCOSolver*);

namespace qoco_device_step_control {
using qoco_batched_stopping::blas;
inline bool compare_enabled() {
    const char* value = std::getenv("SPACEPDHCG_TEST_QOCO_DEVICE_STEPS_COMPARE");
    return value && value[0] == '1';
}
inline void compare(double got, double expected) {
    if ((std::isnan(got) && std::isnan(expected)) ||
        (std::isfinite(got) && std::isfinite(expected) &&
         std::abs(got - expected) <= 2e-12 * std::max(1.0, std::abs(expected)))) return;
    fprintf(stderr, "QOCO device step scalar mismatch %.17g %.17g\n", got, expected);
    exit(1);
}
inline double* workspace(QOCOSolver* solver) {
    const auto* d = solver->work->data;
    const size_t partials = std::max((d->l + 255LL) / 256, (d->nsoc + 255LL) / 256);
    double* storage{}; int temporary{};
    CUDA_CHECK(qoco_gpu_acquire_scalar_workspace(partials + 7, &storage, &temporary));
    return storage;
}
inline void begin() {
    if (qoco_gpu_begin_reduction_scope()) { fprintf(stderr, "QOCO step scope failed\n"); exit(1); }
}
__global__ void centering_vectors(const double* z, const double* s, const double* dz,
    const double* ds, int m, const double* steps, double* u1, double* u2) {
    const double a = qoco_min(steps[0], steps[1]);
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < m; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        u1[i] = a * dz[i] + z[i];
        u2[i] = a * ds[i] + s[i];
    }
}
__global__ void finish_sigma(const double* scalars, double* sigma) {
    const double rho = __ddiv_rn(scalars[3], scalars[4]);
    double clipped = qoco_min(1.0, rho);
    clipped = qoco_max(0.0, clipped);
    *sigma = __dmul_rn(__dmul_rn(clipped, clipped), clipped);
}
__global__ void update_iterates(double* x, double* y, double* s, double* z,
    const double* xyz, const double* ds, int n, int p, int m, const double* steps, double* alpha) {
    const double a = qoco_min(steps[0], steps[1]);
    const long long total = static_cast<long long>(n) + p + 2LL * m;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        if (i < n) x[i] = a * xyz[i] + x[i];
        else if (i < static_cast<long long>(n) + p) { const int j = i - n; y[j] = a * xyz[n + j] + y[j]; }
        else if (i < static_cast<long long>(n) + p + m) { const int j = i - n - p; s[j] = a * ds[j] + s[j]; }
        else { const int j = i - n - p - m; z[j] = a * xyz[n + p + j] + z[j]; }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) *alpha = a;
}
} // namespace qoco_device_step_control

extern "C" void qoco_gpu_compute_centering(QOCOSolver* solver) {
    using namespace qoco_device_step_control;
    begin();
    const bool audit = compare_enabled();
    double reference{};
    if (audit) { qoco_reference_compute_centering(solver); reference = solver->work->sigma; }
    auto* w = solver->work; const auto* d = w->data;
    double* scalar = workspace(solver);
    auto* dz = w->xyz->d_data + d->n + d->p;
    qoco_gpu_enqueue_linesearch(w->z->d_data, dz, 1.0, solver, scalar + 6, scalar);
    qoco_gpu_enqueue_linesearch(w->s->d_data, w->Ds->d_data, 1.0, solver, scalar + 6, scalar + 1);
    if (d->m) centering_vectors<<<std::min(256, (d->m - 1) / 256 + 1), 256>>>(
        w->z->d_data, w->s->d_data, dz, w->Ds->d_data, d->m, scalar, w->ubuff1->d_data, w->ubuff2->d_data);
    CUDA_CHECK(cudaGetLastError());
    auto* f = get_cuda_funcs(); bool temporary{};
    auto handle = qoco_acquire_reduction_handle(&temporary);
    cublasPointerMode_t mode{};
    blas(f->cublasGetPointerMode(handle, &mode));
    blas(f->cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    if (d->m) {
        blas(f->cublasDdot(handle, d->m, w->ubuff1->d_data, 1, w->ubuff2->d_data, 1, scalar + 3));
        blas(f->cublasDdot(handle, d->m, w->z->d_data, 1, w->s->d_data, 1, scalar + 4));
    } else CUDA_CHECK(cudaMemsetAsync(scalar + 3, 0, 2 * sizeof(double)));
    finish_sigma<<<1, 1>>>(scalar, scalar + 5);
    CUDA_CHECK(cudaGetLastError());
    blas(f->cublasSetPointerMode(handle, mode));
    CUDA_CHECK(cudaMemcpy(&w->sigma, scalar + 5, sizeof(double), cudaMemcpyDeviceToHost));
    if (audit) compare(w->sigma, reference);
    qoco_gpu_end_reduction_scope();
}

extern "C" void qoco_gpu_take_step(QOCOSolver* solver) {
    using namespace qoco_device_step_control;
    begin();
    auto* w = solver->work; const auto* d = w->data;
    auto* dz = w->xyz->d_data + d->n + d->p;
    const bool audit = compare_enabled();
    double reference{};
    if (audit) reference = qoco_min(linesearch(w->s->d_data, w->Ds->d_data, .99, solver),
        linesearch(w->z->d_data, dz, .99, solver));
    double* scalar = workspace(solver);
    qoco_gpu_enqueue_linesearch(w->s->d_data, w->Ds->d_data, .99, solver, scalar + 6, scalar);
    qoco_gpu_enqueue_linesearch(w->z->d_data, dz, .99, solver, scalar + 6, scalar + 1);
    const long long total = static_cast<long long>(d->n) + d->p + 2LL * d->m;
    update_iterates<<<static_cast<int>(std::min(256LL, (total - 1) / 256 + 1)), 256>>>(
        w->x->d_data, w->y->d_data, w->s->d_data, w->z->d_data, w->xyz->d_data, w->Ds->d_data,
        d->n, d->p, d->m, scalar, scalar + 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&w->a, scalar + 2, sizeof(double), cudaMemcpyDeviceToHost));
    if (audit) compare(w->a, reference);
    qoco_gpu_end_reduction_scope();
}
