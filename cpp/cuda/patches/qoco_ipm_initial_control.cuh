// SPDX-License-Identifier: Apache-2.0
namespace qoco_device_control {
__global__ void initial_state(State* state, const QocoIpmParameters* p) {
    *state = {};
    state->alpha = 1.0;
    state->dynamic_reg = p->dynamic_reg;
    state->best_iter = -1;
    state->status = QOCO_UNSOLVED;
}
}
extern "C" void qoco_gpu_ipm_allocate_control(QOCOSolver* solver) {
    const char* disabled = getenv("SPACEPDHCG_TEST_QOCO_DEVICE_CONTROL_DISABLE");
    if (disabled && disabled[0] == '1') {
        fprintf(stderr, "GPU initialisation requires device solver control\n"); exit(1);
    }
    if (!solver->work->gpu_control)
        CUDA_CHECK(cudaMalloc(&solver->work->gpu_control, sizeof(qoco_device_control::State)));
}
extern "C" void qoco_gpu_ipm_initial_control(QOCOSolver* solver, const QocoIpmParameters* p) {
    qoco_device_control::initial_state<<<1, 1>>>(
        static_cast<qoco_device_control::State*>(solver->work->gpu_control), p);
    CUDA_CHECK(cudaGetLastError());
}
