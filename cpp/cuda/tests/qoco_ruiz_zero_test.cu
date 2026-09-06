#include "qoco.h"
#include "../algebra/cuda/cuda_types.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
void check(cudaError_t code) { require(code == cudaSuccess, cudaGetErrorString(code)); }
std::vector<double> read(QOCOVectorf* vector) {
    std::vector<double> values(vector->len);
    if (vector->len) check(cudaMemcpy(values.data(), vector->d_data,
        values.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (double value : values) require(std::isfinite(value), "finite scaled vector");
    return values;
}
void run(int passes, bool zero_objective) {
    // min 0.5*(x*x+y*y) - 0.5*x + 0.2*y, with an empty equality,
    // an inactive constant inequality and SOC (2,x,y). Optimum (0.5,-0.2).
    // Explicit zeros exercise 0 * infinity as well as repeated scale overflow.
    int pp[]{0, 1, 2}, pi[]{0, 1}, ap[]{0, 1, 2}, ai[]{0, 0};
    int gp[]{0, 3, 6}, gi[]{0, 1, 2, 0, 1, 3}, cones[]{3};
    double px[]{1, 1}, ax[]{0, 0}, gx[]{0, 0, -1, 0, 0, -1};
    double c[]{-0.5, 0.2}, b[]{0}, h[]{1, 2, 0, 0};
    if (zero_objective) { px[0] = px[1] = 0; c[0] = c[1] = 0; }
    QOCOCscMatrix P{2, 2, 2, pi, pp, px}, A{1, 2, 2, ai, ap, ax}, G{4, 2, 6, gi, gp, gx};
    QOCOSettings settings{}; set_default_settings(&settings);
    settings.ruiz_iters = 0; settings.verbose = 0;
    settings.abstol = settings.reltol = 1e-8;
    auto* solver = static_cast<QOCOSolver*>(std::calloc(1, sizeof(QOCOSolver)));
    require(qoco_setup(solver, 2, 4, 1, &P, c, &A, b, &G, h, 1, 1, cones, &settings) == 0,
            "unscaled initial setup");
    using Create = int (*)(QOCOSolver*, int, int, int, void**);
    using Update = int (*)(void*, const double*, cudaStream_t);
    using Destroy = void (*)(void*);
    auto create = reinterpret_cast<Create>(dlsym(RTLD_DEFAULT, "qoco_gpu_create_numeric_update"));
    auto update = reinterpret_cast<Update>(dlsym(RTLD_DEFAULT, "qoco_gpu_update_numeric"));
    auto destroy = reinterpret_cast<Destroy>(dlsym(RTLD_DEFAULT, "qoco_gpu_destroy_numeric_update"));
    require(create && update && destroy, "device update interface");
    void* context{}; require(create(solver, 2, 2, 6, &context) == 0, "create update context");
    settings.ruiz_iters = passes;
    require(qoco_update_settings(solver, &settings) == 0, "select GPU Ruiz passes");
    std::vector<double> packed;
    for (auto range : std::initializer_list<std::pair<double*, int>>{
            {px, 2}, {ax, 2}, {gx, 6}, {c, 2}, {b, 1}, {h, 4}})
        packed.insert(packed.end(), range.first, range.first + range.second);
    double* device{}; check(cudaMalloc(&device, packed.size() * sizeof(double)));
    cudaStream_t stream{}; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaMemcpyAsync(device, packed.data(), packed.size() * sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    for (int repeat = 0; repeat < 3; ++repeat) {
        require(update(context, device, stream) == 0, "finite GPU update with zero norms");
        auto* scaling = solver->work->scaling;
        auto e = read(scaling->Eruiz), f = read(scaling->Fruiz);
        require(e[0] == 1 && f[0] == 1, "identity scaling on constant constraints");
        require(f[1] == f[2] && f[2] == f[3], "SOC has one common positive scale");
        require(f[1] > 0 && std::isfinite(scaling->k) && scaling->k > 0, "positive finite scales");
        for (auto* vector : {scaling->Druiz, scaling->Dinvruiz, scaling->Einvruiz,
                            scaling->Finvruiz, solver->work->data->c,
                            solver->work->data->b, solver->work->data->h}) read(vector);
        qoco_solve(solver);
        require(solver->sol->status == QOCO_SOLVED, "scaled solve status");
        const auto* x = solver->sol->x;
        require(std::isfinite(x[0]) && std::isfinite(x[1]), "finite unscaled solution");
        if (!zero_objective)
            require(std::abs(x[0] - 0.5) < 1e-7 && std::abs(x[1] + 0.2) < 1e-7,
                    "independently known optimum retained");
        require(std::hypot(x[0], x[1]) <= 2 + 1e-8, "unscaled SOC feasibility");
    }
    destroy(context); check(cudaStreamDestroy(stream)); check(cudaFree(device)); qoco_cleanup(solver);
}
}
int main() {
    for (int passes : {0, 1, 4, 12, 100}) for (bool zero : {false, true}) run(passes, zero);
    std::puts("{\"case\":\"qoco_ruiz_zero\",\"cases\":10,\"updates\":30,\"status\":\"ok\"}");
    return 0;
}
