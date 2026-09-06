// Numerical scaling -> guarded IPM -> consumer, with no intervening host wait.
#define main qoco_synchronous_probe_main
#include "qoco_ipm_cache_probe.cpp"
#undef main
#include <cuda_runtime.h>
#include "qoco_gpu_replay.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <utility>
#define CUDA(x) REQUIRE((x)==cudaSuccess)
extern "C" int qoco_gpu_create_numeric_update(QOCOSolver*,int,int,int,void**);
extern "C" int qoco_gpu_update_numeric(void*,const double*,cudaStream_t);
extern "C" void qoco_gpu_destroy_numeric_update(void*);
struct Snapshot { QocoGpuCompletion result; double vectors[7],update[9]; };
__global__ void consume(QocoGpuOutput output,const double* update,Snapshot* saved) {
    saved->result=*output.completion;
    saved->vectors[0]=output.x[0]; saved->vectors[1]=output.x[1]; saved->vectors[2]=output.y[0];
    saved->vectors[3]=output.s[0]; saved->vectors[4]=output.s[1];
    saved->vectors[5]=output.z[0]; saved->vectors[6]=output.z[1];
    for (int i=0;i<9;++i) saved->update[i]=update[i];
}
struct Gate { std::atomic<bool> entered{false},release{false},returned{false},timeout{false}; };
static void CUDART_CB hold(void* pointer) {
    auto& gate=*static_cast<Gate*>(pointer); gate.entered=true;
    while (!gate.release) std::this_thread::yield();
}
void values(int repeat,double* v) {
    const double scale=repeat%3==0 ? .01 : repeat%3==1 ? 1 : 100;
    const double data[]{scale*(1+.01*repeat),scale*(2+.005*repeat),1,1,-1,-1,
        -scale*(1-.001*repeat),-scale,1.5+.002*repeat,0,0};
    std::memcpy(v,data,sizeof(data));
}
int main() {
    REQUIRE(qoco_gpu_begin_reduction_scope()==0);
    setenv("QOCO_IPM_PROBE_RUIZ","1",1);
    cudaStream_t stream,other;
    CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));
    double* packed; Snapshot* device;
    CUDA(cudaMalloc(&packed,44*sizeof(double))); CUDA(cudaMalloc(&device,4*sizeof(Snapshot)));
    int checked=0,qualified=0,rejected=0;
    {
        Fixture reference,queued;
        for (auto* f:{&reference,&queued}) {
            f->settings.abstol=f->settings.reltol=1e-10;
            f->settings.abstol_inacc=f->settings.reltol_inacc=1e-10;
            REQUIRE(qoco_update_settings(f->solver,&f->settings)==0);
        }
        void *reference_update{},*queued_update{};
        REQUIRE(qoco_gpu_create_numeric_update(reference.solver,2,2,2,&reference_update)==0);
        REQUIRE(qoco_gpu_create_numeric_update(queued.solver,2,2,2,&queued_update)==0);
        const double* result{};
        REQUIRE(qoco_gpu_update_numeric_device(queued_update,packed,stream,&result)==2 && !result);
        double initial[11]; values(0,initial);
        CUDA(cudaMemcpy(packed,initial,sizeof(initial),cudaMemcpyHostToDevice));
        for (auto entry:{std::pair{&reference,reference_update},std::pair{&queued,queued_update}}) {
            REQUIRE(qoco_gpu_update_numeric(entry.second,packed,stream)==0);
            REQUIRE(qoco_solve(entry.first->solver)==QOCO_SOLVED);
        }
        for (int group=0;group<16;++group) {
            double inputs[44]; Snapshot expected[4]{};
            for (int i=0;i<4;++i) {
                values(4*group+i,inputs+11*i);
                CUDA(cudaMemcpy(packed,inputs+11*i,11*sizeof(double),cudaMemcpyHostToDevice));
                REQUIRE(qoco_update_settings(reference.solver,&reference.settings)==0);
                REQUIRE(qoco_gpu_update_numeric(reference_update,packed,stream)==0);
                const int status=qoco_solve(reference.solver);
                std::fprintf(stderr,"reference group=%d item=%d status=%d k=%.17g iterations=%d objective=%.17g residual=%.17g/%.17g\n",group,i,status,reference.solver->work->scaling->k,reference.solver->sol->iters,reference.solver->sol->obj,reference.solver->sol->pres,reference.solver->sol->dres);
                REQUIRE(status==QOCO_SOLVED || status==QOCO_SOLVED_INACCURATE || status==QOCO_NUMERICAL_ERROR);
                const auto* s=reference.solver->sol;
                expected[i].result={1,s->status,s->iters,s->ir_iters,reference.solver->work->ir_iters,0,
                    s->pres,s->dres,s->gap,s->obj,reference.solver->settings->kkt_dynamic_reg};
                double vectors[]{s->x[0],s->x[1],s->y[0],s->s[0],s->s[1],s->z[0],s->z[1]};
                std::memcpy(expected[i].vectors,vectors,sizeof(vectors));
                const double* v=inputs+11*i;
                const double x1=(v[0]*v[8]+v[6]-v[7])/(v[0]+v[1]),x0=v[8]-x1;
                const double objective=.5*(v[0]*x0*x0+v[1]*x1*x1)+v[6]*x0+v[7]*x1;
                REQUIRE(std::abs(s->obj-objective)<1e-7);
                std::fprintf(stderr,"analytic errors x=%.17g/%.17g objective=%.17g\n",s->x[0]-x0,s->x[1]-x1,s->obj-objective);
                if (status==QOCO_SOLVED || status==QOCO_SOLVED_INACCURATE) {
                    REQUIRE(std::abs(s->x[0]-x0)<1e-7 && std::abs(s->x[1]-x1)<1e-7); ++qualified;
                } else ++rejected;
            }
            CUDA(cudaMemcpyAsync(packed,inputs,sizeof(inputs),cudaMemcpyHostToDevice,stream));
            REQUIRE(qoco_update_settings(queued.solver,&queued.settings)==0);
            for (int i=0;i<4;++i) {
                REQUIRE(qoco_gpu_update_numeric_device(queued_update,packed+11*i,stream,&result)==0);
                const double* rejected=reinterpret_cast<const double*>(1);
                REQUIRE(qoco_gpu_update_numeric_device(queued_update,packed,other,&rejected)==2 && !rejected);
                QocoGpuOutput output{};
                REQUIRE(qoco_gpu_ipm_replay_updated_device(queued.solver,stream,result,&output)==0);
                consume<<<1,1,0,stream>>>(output,result,device+i);
            }
            REQUIRE(qoco_gpu_ipm_finish_device(queued.solver)==0);
            REQUIRE(qoco_gpu_finish_numeric_update(queued_update,0)==0);
            Snapshot got[4]; CUDA(cudaMemcpy(got,device,sizeof(got),cudaMemcpyDeviceToHost));
            for (int i=0;i<4;++i) {
                std::fprintf(stderr,"compare group=%d item=%d status=%d/%d iterations=%d/%d objective=%.17g/%.17g k=%.17g kinv=%.17g x=%.17g/%.17g\n",group,i,got[i].result.status,expected[i].result.status,got[i].result.iterations,expected[i].result.iterations,got[i].result.objective,expected[i].result.objective,got[i].update[0],got[i].update[1],got[i].vectors[0],expected[i].vectors[0]);
                REQUIRE(got[i].result.status==expected[i].result.status && got[i].result.abi_version==1);
                REQUIRE(got[i].result.iterations==expected[i].result.iterations);
                REQUIRE(got[i].result.ir_iterations==expected[i].result.ir_iterations);
                REQUIRE(got[i].result.objective==expected[i].result.objective);
                REQUIRE(got[i].result.primal_residual==expected[i].result.primal_residual);
                REQUIRE(got[i].result.dual_residual==expected[i].result.dual_residual);
                REQUIRE(got[i].result.gap==expected[i].result.gap);
                REQUIRE(std::memcmp(got[i].vectors,expected[i].vectors,sizeof(got[i].vectors))==0);
                REQUIRE(got[i].update[8]==0);
                ++checked;
            }
            // A second finish may explicitly materialise previously deferred scales.
            REQUIRE(qoco_gpu_finish_numeric_update(queued_update,1)==0);
            REQUIRE(queued.solver->work->scaling->k==got[3].update[0]);
            REQUIRE(queued.solver->work->scaling->kinv==got[3].update[1]);
        }
        qoco_gpu_end_reduction_scope();
        Gate gate;
        CUDA(cudaLaunchHostFunc(stream,hold,&gate));
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        while (!gate.entered && std::chrono::steady_clock::now()<deadline) std::this_thread::yield();
        if (!gate.entered) { gate.release=true; REQUIRE(false); }
        std::thread watchdog([&] {
            const auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(1);
            while (!gate.returned && std::chrono::steady_clock::now()<limit) std::this_thread::yield();
            if (!gate.returned) { gate.timeout=true; gate.release=true; }
        });
        const int submitted=qoco_gpu_update_numeric_device(queued_update,packed,stream,&result);
        QocoGpuOutput output{};
        const int replayed=qoco_gpu_ipm_replay_updated_device(queued.solver,stream,result,&output);
        gate.returned=true; gate.release=true; watchdog.join();
        REQUIRE(submitted==0 && replayed==0 && !gate.timeout);
        consume<<<1,1,0,stream>>>(output,result,device);
        REQUIRE(qoco_gpu_ipm_finish_device(queued.solver)==0);
        REQUIRE(qoco_gpu_finish_numeric_update(queued_update,0)==0);
        Snapshot valid{}; CUDA(cudaMemcpy(&valid,device,sizeof(valid),cudaMemcpyDeviceToHost));
        REQUIRE(valid.result.status==QOCO_SOLVED);
        // Invalid numeric update must bypass the entire captured IPM, keeping its
        // prior vectors and publishing explicit failure, even with stale host k.
        double poisoned[11]; values(1,poisoned); poisoned[6]=NAN;
        CUDA(cudaMemcpyAsync(packed,poisoned,sizeof(poisoned),cudaMemcpyHostToDevice,stream));
        REQUIRE(qoco_gpu_update_numeric_device(queued_update,packed,stream,&result)==0);
        REQUIRE(qoco_gpu_ipm_replay_updated_device(queued.solver,stream,result,&output)==0);
        consume<<<1,1,0,stream>>>(output,result,device);
        REQUIRE(qoco_gpu_ipm_finish_device(queued.solver)==0);
        REQUIRE(qoco_gpu_finish_numeric_update(queued_update,1)==3);
        Snapshot invalid{}; CUDA(cudaMemcpy(&invalid,device,sizeof(invalid),cudaMemcpyDeviceToHost));
        REQUIRE(invalid.result.status==QOCO_NUMERICAL_ERROR && invalid.result.iterations==0);
        REQUIRE(invalid.update[8]!=0 && !std::isfinite(invalid.result.primal_residual));
        REQUIRE(std::memcmp(invalid.vectors,valid.vectors,sizeof(valid.vectors))==0);
        qoco_gpu_destroy_numeric_update(queued_update); qoco_gpu_destroy_numeric_update(reference_update);
    }
    CUDA(cudaFree(packed)); CUDA(cudaFree(device));
    CUDA(cudaStreamDestroy(stream)); CUDA(cudaStreamDestroy(other));
    qoco_gpu_end_reduction_scope();
    std::printf("PASS: %d queued update/replay/consumer chains with exact synchronous vectors; %d analytic-qualified and %d retained numerical failures; held-stream submission; device rejection skips IPM\n",checked,qualified,rejected);
}
