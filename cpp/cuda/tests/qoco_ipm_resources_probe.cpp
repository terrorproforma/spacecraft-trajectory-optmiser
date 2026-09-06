// Retained scratch/BLAS lifetimes without conditional-graph instrumentation.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "qoco.h"
extern "C" int qoco_gpu_begin_reduction_scope();
extern "C" void qoco_gpu_end_reduction_scope();
extern "C" cudaError_t qoco_gpu_acquire_scalar_workspace(size_t, double**, int*);
extern "C" void* qoco_gpu_ipm_resources_enter(void*);
extern "C" void qoco_gpu_ipm_resources_leave(void*);
extern "C" void qoco_gpu_ipm_resources_destroy(void*);
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
int main() {
    void* resources[2]{};
    double* retained[2]{};
    for (int cycle = 0; cycle < 8; ++cycle) {
        REQUIRE(qoco_gpu_begin_reduction_scope() == 0);
        double* original{};
        int temporary{};
        REQUIRE(qoco_gpu_acquire_scalar_workspace(64, &original, &temporary) == cudaSuccess);
        REQUIRE(!temporary);
        const double host[4]{1, 2, 3, double(cycle + 4)};
        REQUIRE(cudaMemcpy(original, host, sizeof(host), cudaMemcpyHostToDevice) == cudaSuccess);
        for (int index = 0; index < 2; ++index) {
            REQUIRE(qoco_gpu_begin_reduction_scope() == 0);
            resources[index] = qoco_gpu_ipm_resources_enter(resources[index]);
            double* active{};
            REQUIRE(qoco_gpu_acquire_scalar_workspace(512, &active, &temporary) == cudaSuccess);
            REQUIRE(!temporary && active != original);
            if (cycle) REQUIRE(active == retained[index]);
            else {
                retained[index] = active;
                const double values[4]{1, 2, 3, double(index + 4)};
                REQUIRE(cudaMemcpy(active, values, sizeof(values), cudaMemcpyHostToDevice) == cudaSuccess);
            }
            REQUIRE(qoco_dot(active, active, 4) == 14.0 + (index + 4) * (index + 4));
            qoco_gpu_ipm_resources_leave(resources[index]);
            qoco_gpu_end_reduction_scope();
            double* restored{};
            REQUIRE(qoco_gpu_acquire_scalar_workspace(64, &restored, &temporary) == cudaSuccess);
            REQUIRE(restored == original);
            REQUIRE(qoco_dot(restored, restored, 4) == 14.0 + (cycle + 4) * (cycle + 4));
        }
        REQUIRE(retained[0] != retained[1]);
        qoco_gpu_end_reduction_scope();
    }
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    for (auto* resource : resources) qoco_gpu_ipm_resources_destroy(resource);
    std::puts("PASS: two retained workspaces, eight scope lifetimes, nested scopes, BLAS and scalar restoration");
}
