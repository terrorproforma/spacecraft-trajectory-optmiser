// SPDX-License-Identifier: Apache-2.0
// Final recovery and unscaling execute after the IPM WHILE, before host reports.
namespace qoco_device_control {
__global__ void terminal_restore(State* state) {
    state->restored = 0;
    if (state->status == QOCO_NUMERICAL_ERROR || state->status == QOCO_MAX_ITER)
        restore_decision_impl(state, state->status);
}
__global__ void terminal_unscale(int n, int p, int m, double* x, double* y,
    double* s, double* z, const double* d, const double* e, const double* f,
    const double* finv, const QocoIpmParameters* parameters) {
    const long long total = n + static_cast<long long>(p) + 2LL * m;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += blockDim.x * gridDim.x) {
        if (i < n) x[i] = __dmul_rn(x[i], d[i]);
        else if (i < n + static_cast<long long>(p)) {
            const int j = i - n;
            y[j] = __dmul_rn(__dmul_rn(y[j], e[j]), parameters->kinv);
        } else if (i < n + static_cast<long long>(p) + m) {
            const int j = i - n - p;
            s[j] = __dmul_rn(s[j], finv[j]);
        } else {
            const int j = i - n - p - m;
            z[j] = __dmul_rn(__dmul_rn(z[j], f[j]), parameters->kinv);
        }
    }
}
}
extern "C" void qoco_gpu_ipm_terminal(QOCOSolver* solver, const QocoIpmParameters* parameters) {
    auto* w = solver->work;
    auto* d = w->data;
    auto* scaling = w->scaling;
    qoco_device_control::terminal_restore<<<1, 1>>>(
        static_cast<qoco_device_control::State*>(w->gpu_control));
    qoco_device_control::copy<true>(w);
    const long long count = d->n + static_cast<long long>(d->p) + 2LL * d->m;
    if (count) qoco_device_control::terminal_unscale<<<std::min(256LL, (count + 255) / 256), 256>>>(
        d->n, d->p, d->m, w->x->d_data, w->y->d_data, w->s->d_data, w->z->d_data,
        scaling->Druiz->d_data, scaling->Eruiz->d_data, scaling->Fruiz->d_data,
        scaling->Finvruiz->d_data, parameters);
    CUDA_CHECK(cudaGetLastError());
}
