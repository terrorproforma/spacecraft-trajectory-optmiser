// SPDX-License-Identifier: Apache-2.0
// Same residual gate and shift as bring2cone; the ordinary finite path is parallel.
struct QocoInitialConeState { double residual, shift; int special, active; };
static __global__ void qoco_initial_cone_shift(QocoInitialConeState* state,
    const double* u, int l, int nsoc, const int* sizes, const int* starts) {
    state->active = state->residual >= 0.0;
    if (!state->active) return;
    double a = qoco_max(0.0, state->residual);
    // Preserve the legacy LP NaN order and SOC comparisons on exceptional data.
    // This path remains device-only; finite production trajectories do not scan serially.
    if (state->special) {
        a = 0.0;
        for (int i = 0; i < l; ++i) a = qoco_max(a, -u[i]);
        a = qoco_max(a, 0.0);
        for (int i = 0; i < nsoc; ++i) {
            const double value = soc_residual(u + starts[i], sizes[i]);
            if (value > a) a = value;
        }
    }
    state->shift = 1.0 + a;
}
static __global__ void qoco_initial_cone_apply(double* u, int l, int nsoc,
    const int* starts, const QocoInitialConeState* state) {
    if (!state->active) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < l + nsoc; i += blockDim.x * gridDim.x) {
        const int index = i < l ? i : starts[i - l];
        u[index] += state->shift;
    }
}
void bring2cone(QOCOFloat* u, QOCOInt* starts, QOCOProblemData* data) {
    if (getenv("SPACEPDHCG_TEST_QOCO_HOST_INITIAL_CONE")) {
        qoco_reference_bring2cone(u, starts, data); return;
    }
    const int count = data->l + data->nsoc;
    if (!count) return;
    const int blocks = (count + 255) / 256;
    double* storage = nullptr;
    int temporary = 0;
    CUDA_CHECK(qoco_gpu_acquire_scalar_workspace(size_t(blocks) + 3, &storage, &temporary));
    auto* state = reinterpret_cast<QocoInitialConeState*>(storage);
    static_assert(sizeof(QocoInitialConeState) == 3 * sizeof(double));
    CUDA_CHECK(cudaMemsetAsync(state, 0, sizeof(*state)));
    cone_residual_stage1<<<blocks, 256, 256 * sizeof(double)>>>(u, data->l, data->nsoc,
        get_data_vectori(data->q), starts, storage + 3, &state->special);
    qoco_finish_cone_reduction<true><<<1, 256>>>(storage + 3, blocks, &state->residual, nullptr, 1.0, false);
    qoco_initial_cone_shift<<<1, 1>>>(state, u, data->l, data->nsoc, get_data_vectori(data->q), starts);
    qoco_initial_cone_apply<<<std::min(256, blocks), 256>>>(u, data->l, data->nsoc, starts, state);
    CUDA_CHECK(cudaGetLastError());
    if (temporary) CUDA_CHECK(cudaFree(storage));
}
