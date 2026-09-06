// Compare queued GPU completion/consumer chains with synchronous analytic QPs.
#define main qoco_synchronous_probe_main
#include "qoco_ipm_cache_probe.cpp"
#undef main
#include <cuda_runtime.h>
#include <cstring>
#include <thread>
#include <atomic>
#include <chrono>
#include "qoco_gpu_replay.h"
#define CUDA(x) REQUIRE((x)==cudaSuccess)
struct Snapshot { QocoGpuCompletion result; double vectors[7]; };
struct Gate { std::atomic<bool> entered{false}, release{false}, returned{false}, timeout{false}; };
static void CUDART_CB hold_stream(void* pointer) {
    auto& gate=*static_cast<Gate*>(pointer);
    gate.entered=true;
    while (!gate.release.load()) std::this_thread::yield();
}
__global__ void consume(QocoGpuOutput output, Snapshot* saved) {
    saved->result=*output.completion;
    saved->vectors[0]=output.x[0]; saved->vectors[1]=output.x[1];
    saved->vectors[2]=output.y[0]; saved->vectors[3]=output.s[0];
    saved->vectors[4]=output.s[1]; saved->vectors[5]=output.z[0]; saved->vectors[6]=output.z[1];
}
int main() {
    REQUIRE(qoco_gpu_begin_reduction_scope()==0);
    setenv("QOCO_IPM_PROBE_RUIZ","1",1);
    unsetenv("QOCO_IPM_PROBE_WARM");
    cudaStream_t stream,other;
    CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));
    Snapshot* device; CUDA(cudaMalloc(&device,4*sizeof(Snapshot)));
    int chains=0;
    {
        Fixture a,b;
        QocoGpuOutput output{};
        REQUIRE(qoco_gpu_ipm_replay_device(a.solver,stream,&output)==2 && !output.completion);
        REQUIRE(qoco_gpu_ipm_replay_device(nullptr,stream,&output)==1 && !output.completion);
        for (int repeat=0;repeat<16;++repeat) for (auto* f : {&a,&b}) {
            REQUIRE(qoco_gpu_primal_start(f->solver,0)==0);
            f->run(f==&a ? 0:1,repeat);
            if (std::getenv("QOCO_REPLAY_WARM") && repeat!=14) {
                REQUIRE(qoco_gpu_primal_start(f->solver,2)==0);
                REQUIRE(qoco_gpu_primal_start(f->solver,1)==0);
                const double initial_reg=f->solver->settings->kkt_dynamic_reg;
                const int status=qoco_solve(f->solver);
                REQUIRE(status==QOCO_SOLVED || status==QOCO_SOLVED_INACCURATE);
                f->solver->settings->kkt_dynamic_reg=initial_reg;
            }
            const auto reference=*f->solver->sol;
            const double expected[]{reference.x[0],reference.x[1],reference.y[0],
                reference.s[0],reference.s[1],reference.z[0],reference.z[1]};
            const int step=f->solver->work->ir_iters;
            // A changed static value must reject without queuing stale operands.
            const double static_g=f->solver->settings->kkt_static_reg_G;
            f->solver->settings->kkt_static_reg_G*=2;
            REQUIRE(qoco_gpu_ipm_replay_device(f->solver,stream,&output)==2 && !output.completion);
            f->solver->settings->kkt_static_reg_G=static_g;
            for (int chain=0;chain<4;++chain) {
                REQUIRE(qoco_gpu_ipm_replay_device(f->solver,stream,&output)==0);
                REQUIRE(output.n==2 && output.p==1 && output.m==2);
                REQUIRE(f->solver->sol->status==QOCO_UNSOLVED);
                consume<<<1,1,0,stream>>>(output,device+chain);
                CUDA(cudaGetLastError());
                QocoGpuOutput rejected{};
                REQUIRE(qoco_gpu_ipm_replay_device(f->solver,other,&rejected)==3 && !rejected.completion);
            }
            REQUIRE(qoco_gpu_ipm_finish_device(f->solver)==0);
            Snapshot got[4]; CUDA(cudaMemcpy(got,device,sizeof(got),cudaMemcpyDeviceToHost));
            for (const auto& v : got) {
                REQUIRE(v.result.abi_version==1 && v.result.status==reference.status);
                REQUIRE(v.result.iterations==reference.iters && v.result.ir_iterations==reference.ir_iters);
                REQUIRE(v.result.step_ir_iterations==step);
                REQUIRE(v.result.objective==reference.obj && v.result.primal_residual==reference.pres);
                REQUIRE(v.result.dual_residual==reference.dres && v.result.gap==reference.gap);
                REQUIRE(std::memcmp(v.vectors,expected,sizeof(expected))==0);
                REQUIRE(std::memcmp(&v,&got[0],sizeof(v))==0);
                ++chains;
            }
            std::swap(stream,other);
        }
        // Replay owns its captured resources even after the caller scope ends.
        qoco_gpu_end_reduction_scope();
        Gate gate;
        CUDA(cudaLaunchHostFunc(stream,hold_stream,&gate));
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        while (!gate.entered && std::chrono::steady_clock::now()<deadline) std::this_thread::yield();
        if (!gate.entered) { gate.release=true; REQUIRE(false); }
        std::thread watchdog([&] {
            const auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(1);
            while (!gate.returned && std::chrono::steady_clock::now()<limit)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (!gate.returned) { gate.timeout=true; gate.release=true; }
        });
        const int queued=qoco_gpu_ipm_replay_device(a.solver,stream,&output);
        gate.returned=true; gate.release=true;
        watchdog.join();
        REQUIRE(queued==0 && !gate.timeout);
        consume<<<1,1,0,stream>>>(output,device);
        REQUIRE(qoco_gpu_ipm_finish_device(a.solver)==0);
        Snapshot final{}; CUDA(cudaMemcpy(&final,device,sizeof(final),cudaMemcpyDeviceToHost));
        REQUIRE((final.result.status==QOCO_SOLVED || final.result.status==QOCO_SOLVED_INACCURATE)
            && final.result.abi_version==1);
        // Resource ownership is bound to the original host thread.
        int thread_status=0;
        std::thread worker([&] {
            QocoGpuOutput rejected{};
            thread_status=qoco_gpu_ipm_replay_device(a.solver,stream,&rejected);
        });
        worker.join(); REQUIRE(thread_status==2);
    }
    CUDA(cudaFree(device)); CUDA(cudaStreamDestroy(stream)); CUDA(cudaStreamDestroy(other));
    qoco_gpu_end_reduction_scope();
    std::printf("PASS: %d queued replays and GPU consumers; exact synchronous parity; stream/static/thread guards; held-stream submission returned without waiting after caller scope destruction\n",chains);
}
