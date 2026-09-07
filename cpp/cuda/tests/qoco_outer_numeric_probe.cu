// Changing coefficients -> GPU scaling -> nested IPM -> GPU consumer, driven
// by one outer graph launch. Compare every problem with the synchronous path.
#define QOCO_OUTER_GRAPH_NO_MAIN
#include "qoco_outer_graph_probe.cu"
extern "C" int qoco_gpu_create_numeric_update(QOCOSolver*,int,int,int,void**);
extern "C" int qoco_gpu_update_numeric(void*,const double*,cudaStream_t);
extern "C" void qoco_gpu_destroy_numeric_update(void*);
__global__ void select_problem(const double* inputs,double* packed,const int* index) {
    for(int j=threadIdx.x;j<11;j+=blockDim.x) packed[j]=inputs[11*(*index)+j];
}
int main() {
    REQUIRE(qoco_gpu_begin_reduction_scope()==0);
    setenv("QOCO_IPM_PROBE_RUIZ","1",1);
    cudaStream_t stream;CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    double *packed,*inputs;Snapshot* snapshots;int* index;
    CUDA(cudaMalloc(&packed,11*sizeof(double)));CUDA(cudaMalloc(&inputs,88*sizeof(double)));
    CUDA(cudaMalloc(&snapshots,8*sizeof(Snapshot)));CUDA(cudaMalloc(&index,sizeof(int)));
    int checked=0;
    {
        Fixture reference,queued;void *reference_update{},*queued_update{};
        double values[88];
        for(int i=0;i<8;++i) {
            const double scale=i%3==0 ? .1 : i%3==1 ? 1 : 10;
            const double row[]{scale*(1+.02*i),scale*(2+.01*i),1,1,-1,-1,
                -scale*(1-.003*i),-scale,1.5+.01*i,0,0};
            std::memcpy(values+11*i,row,sizeof(row));
        }
        CUDA(cudaMemcpy(inputs,values,sizeof(values),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(packed,values,11*sizeof(double),cudaMemcpyHostToDevice));
        REQUIRE(qoco_gpu_create_numeric_update(reference.solver,2,2,2,&reference_update)==0);
        REQUIRE(qoco_gpu_create_numeric_update(queued.solver,2,2,2,&queued_update)==0);
        for(auto entry:{std::pair{&reference,reference_update},std::pair{&queued,queued_update}}) {
            entry.first->settings.abstol=entry.first->settings.reltol=1e-10;
            entry.first->settings.abstol_inacc=entry.first->settings.reltol_inacc=1e-10;
            REQUIRE(qoco_update_settings(entry.first->solver,&entry.first->settings)==0);
            REQUIRE(qoco_gpu_update_numeric(entry.second,packed,stream)==0);
            REQUIRE(qoco_solve(entry.first->solver)==QOCO_SOLVED);
        }
        const double* numeric{};
        REQUIRE(qoco_gpu_capture_numeric_update(queued_update,packed,stream,&numeric)==2 && !numeric);
        cudaGraph_t root;CUDA(cudaGraphCreate(&root,0));cudaGraphConditionalHandle loop;
        CUDA(cudaGraphConditionalHandleCreate(&loop,root,1,cudaGraphCondAssignDefault));
        CUDA(cudaStreamBeginCaptureToGraph(cudaStreamPerThread,root,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
        CUDA(cudaMemsetAsync(index,0,sizeof(int),cudaStreamPerThread));auto deps=end_capture(root);
        cudaGraphNodeParams params{};params.type=cudaGraphNodeTypeConditional;
        params.conditional.handle=loop;params.conditional.type=cudaGraphCondTypeWhile;params.conditional.size=1;
        cudaGraphNode_t node;CUDA(cudaGraphAddNode(&node,root,deps.data(),deps.size(),&params));
        auto body=params.conditional.phGraph_out[0];
        CUDA(cudaStreamBeginCaptureToGraph(cudaStreamPerThread,body,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
        select_problem<<<1,32>>>(inputs,packed,index);
        REQUIRE(qoco_gpu_capture_numeric_update(queued_update,packed,cudaStreamPerThread,&numeric)==0);
        deps=end_capture(body);
        QocoGpuOutput output{};cudaGraphNode_t completed{};
        REQUIRE(qoco_gpu_ipm_emit_graph(queued.solver,body,deps.data(),deps.size(),numeric,&output,&completed)==0);
        CUDA(cudaStreamBeginCaptureToGraph(cudaStreamPerThread,body,&completed,nullptr,1,cudaStreamCaptureModeThreadLocal));
        consume_outer<<<1,1>>>(output,snapshots,index,loop);end_capture(body);
        cudaGraphExec_t executable;CUDA(cudaGraphInstantiate(&executable,root,0));
        for(int launch=0;launch<4;++launch) {
            Snapshot expected[8]{};
            for(int i=0;i<8;++i) {
                CUDA(cudaMemcpy(packed,values+11*i,11*sizeof(double),cudaMemcpyHostToDevice));
                REQUIRE(qoco_update_settings(reference.solver,&reference.settings)==0);
                REQUIRE(qoco_gpu_update_numeric(reference_update,packed,stream)==0);
                REQUIRE(qoco_solve(reference.solver)==QOCO_SOLVED);
                const auto* s=reference.solver->sol;
                expected[i].result={1,s->status,s->iters,s->ir_iters,reference.solver->work->ir_iters,0,
                    s->pres,s->dres,s->gap,s->obj,reference.solver->settings->kkt_dynamic_reg};
                expected[i].x[0]=s->x[0];expected[i].x[1]=s->x[1];
            }
            CUDA(cudaGraphLaunch(executable,stream));CUDA(cudaStreamSynchronize(stream));
            Snapshot got[8];int count{};CUDA(cudaMemcpy(got,snapshots,sizeof(got),cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(&count,index,sizeof(count),cudaMemcpyDeviceToHost));REQUIRE(count==8);
            for(int i=0;i<8;++i) {
                const auto& r=got[i].result;const auto& ref=expected[i].result;const double* v=values+11*i;
                const double x1=(v[0]*v[8]+v[6]-v[7])/(v[0]+v[1]),x0=v[8]-x1;
                const double objective=.5*(v[0]*x0*x0+v[1]*x1*x1)+v[6]*x0+v[7]*x1;
                REQUIRE(r.status==QOCO_SOLVED && r.iterations>0);
                REQUIRE(r.primal_residual<=1e-8 && r.dual_residual<=1e-8);
                REQUIRE(std::abs(r.objective-objective)<1e-7 && std::abs(got[i].x[0]-x0)<1e-7 && std::abs(got[i].x[1]-x1)<1e-7);
                REQUIRE(r.iterations==ref.iterations && r.ir_iterations==ref.ir_iterations);
                REQUIRE(r.objective==ref.objective && r.primal_residual==ref.primal_residual && r.dual_residual==ref.dual_residual);
                REQUIRE(std::memcmp(got[i].x,expected[i].x,sizeof(got[i].x))==0);++checked;
            }
        }
        CUDA(cudaGraphExecDestroy(executable));CUDA(cudaGraphDestroy(root));
        qoco_gpu_destroy_numeric_update(queued_update);qoco_gpu_destroy_numeric_update(reference_update);
    }
    CUDA(cudaFree(packed));CUDA(cudaFree(inputs));CUDA(cudaFree(snapshots));CUDA(cudaFree(index));CUDA(cudaStreamDestroy(stream));
    qoco_gpu_end_reduction_scope();
    std::printf("PASS: %d changing QPs under GPU outer control; exact synchronous parity and independent analytic optima\n",checked);
}
