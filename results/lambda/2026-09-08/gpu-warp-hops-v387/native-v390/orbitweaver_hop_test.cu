#include "spacepdhcg/cuda/orbitweaver_gpu_c_api.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Request = spacepdhcg_orbitweaver_hop_request;
using Result = spacepdhcg_orbitweaver_hop_result;
static_assert(sizeof(Request) == 160 && sizeof(Result) == 72);
static void require(bool ok) { if (!ok) std::abort(); }
static void check(cudaError_t s) { require(s == cudaSuccess); }
static void pass(spacepdhcg_cuda_status s) { require(s == SPACEPDHCG_CUDA_SUCCESS); }

int main() {
    constexpr size_t n = 131;
    cudaStream_t native{};
    check(cudaStreamCreateWithFlags(&native,cudaStreamNonBlocking));
    spacepdhcg_accelerator_stream stream{{SPACEPDHCG_DEVICE_CUDA,0},reinterpret_cast<uintptr_t>(native)};
    spacepdhcg_orbitweaver_lambert_config config{1,0,n,0,256};
    spacepdhcg_orbitweaver_lambert_workspace* workspace = nullptr;
    pass(spacepdhcg_orbitweaver_lambert_workspace_create(&config,stream,&workspace));
    std::vector<Request> requests(n);
    std::vector<Result> host(n),device(n);
    for (size_t i=0;i<n;++i) {
        auto& q=requests[i].lambert;
        q.departure_position[0]=7e6; q.arrival_position[1]=8e6;
        q.time_of_flight=3600+i*10; q.gravitational_parameter=3.986004418e14;
        q.time_tolerance=1e-8; q.maximum_iterations=256;
        requests[i].departure_allowance=requests[i].arrival_allowance=i%2 ? 1e6 : 0;
    }
    requests[0].lambert.time_of_flight=-1;
    Request* dq=nullptr; Result* dr=nullptr;
    check(cudaMalloc(&dq,n*sizeof(Request))); check(cudaMalloc(&dr,n*sizeof(Result)));
    check(cudaMemcpyAsync(dq,requests.data(),n*sizeof(Request),cudaMemcpyHostToDevice,native));
    check(cudaStreamSynchronize(native));
    check(cudaStreamBeginCapture(native,cudaStreamCaptureModeThreadLocal));
    pass(spacepdhcg_orbitweaver_hop_launch_device(dq,n,256,dr,n,stream));
    cudaGraph_t graph{}; cudaGraphExec_t exec{};
    check(cudaStreamEndCapture(native,&graph));
    check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for (int replay=0;replay<3;++replay) {
        requests.back().lambert.time_of_flight += 100;
        pass(spacepdhcg_orbitweaver_hop_screening_host(workspace,requests.data(),n,host.data(),n));
        check(cudaMemcpyAsync(dq,requests.data(),n*sizeof(Request),cudaMemcpyHostToDevice,native));
        check(cudaGraphLaunch(exec,native));
        check(cudaMemcpyAsync(device.data(),dr,n*sizeof(Result),cudaMemcpyDeviceToHost,native));
        check(cudaStreamSynchronize(native));
        require(!device[0].feasible && std::isinf(device[0].departure_delta_v));
        for (size_t i=1;i<n;++i) {
            require(host[i].feasible && device[i].feasible);
            require(host[i].long_way==device[i].long_way);
            require(std::abs(host[i].departure_delta_v-device[i].departure_delta_v)<1e-8);
            require(std::abs(host[i].arrival_delta_v-device[i].arrival_delta_v)<1e-8);
            for (int k=0;k<3;++k) {
                require(std::abs(host[i].departure_velocity[k]-device[i].departure_velocity[k])<1e-8);
                require(std::abs(host[i].arrival_velocity[k]-device[i].arrival_velocity[k])<1e-8);
            }
            if (i%2) require(device[i].long_way==0 && device[i].departure_delta_v==0 && device[i].arrival_delta_v==0);
        }
    }
    spacepdhcg_orbitweaver_batch_telemetry telemetry{}; telemetry.abi_version=1;
    pass(spacepdhcg_orbitweaver_lambert_workspace_telemetry(workspace,&telemetry));
    require(telemetry.requests_submitted==3*n && telemetry.results_emitted==3*n);
    require(telemetry.input_bytes==3*n*sizeof(Request) && telemetry.output_bytes==3*n*sizeof(Result));
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
    check(cudaFree(dq));check(cudaFree(dr));
    pass(spacepdhcg_orbitweaver_lambert_workspace_destroy(&workspace));
    check(cudaStreamDestroy(native));
    std::puts("combined hops: device graph, host parity, ties, invalid inputs and telemetry passed");
}
