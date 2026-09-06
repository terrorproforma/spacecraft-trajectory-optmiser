#include "spacepdhcg/cuda/gtoc12_scvx_c_api.h"
#include "spacepdhcg/cuda/gtoc12_discretisation_c_api.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "../internal/gtoc12_seed.cuh"

namespace {
using Settings = spacepdhcg_gtoc12_scvx_settings;
using Record = spacepdhcg_gtoc12_scvx_record;
using Result = spacepdhcg_gtoc12_scvx_result;
struct Command { int done, substeps, refresh, error; };
struct State {
    Command command;
    Result result;
    double merit, trust_state, trust_control;
    int polishing, polish_left, copy_candidate;
};
struct Metrics { double fuel, penalty, virtual_sum, defect, virtual_inf, step, invalid; };

__global__ void reduce_metrics(int nodes, int variables, const double* candidate,
    const double* controls, const double* virtuals, const double* reference,
    const double* reference_controls, const double* propagated, const int* invalid,
    const double* fuel, double tolerance, Metrics* partial, const int* enabled=nullptr) {
    if (enabled && !*enabled) return;
    __shared__ double values[7][256];
    const int tid = threadIdx.x;
    double v[7] = {};
    const int count = max(variables, 7*nodes);
    for (int i = blockIdx.x*blockDim.x+tid; i < count; i += blockDim.x*gridDim.x) {
        if (i < nodes) v[0] += fuel[i]*controls[4*i+3];
        if (i < 7*(nodes-1)) {
            const double defect = fabs(candidate[7+i]-propagated[i]);
            v[1] += fmax(defect-tolerance, 0.0);
            v[3] = fmax(v[3], defect);
            if (!isfinite(defect)) v[6] = 1.0;
            if (virtuals) { v[2] += fabs(virtuals[i]); v[4] = fmax(v[4], fabs(virtuals[i])); }
        }
        if (reference && i < 7*nodes) v[5] = fmax(v[5], fabs(candidate[i]-reference[i]));
        if (reference_controls && i < 4*nodes)
            v[5] = fmax(v[5], fabs(controls[i]-reference_controls[i]));
        if ((i < variables && !isfinite(candidate[i])) ||
            (i < 4*nodes && !isfinite(controls[i]))) v[6] = 1.0;
    }
    for (int j = 0; j < 7; ++j) values[j][tid] = v[j];
    __syncthreads();
    for (int offset = 128; offset; offset /= 2) {
        if (tid < offset) for (int j = 0; j < 7; ++j)
            values[j][tid] = j < 3 ? values[j][tid]+values[j][tid+offset]
                : fmax(values[j][tid], values[j][tid+offset]);
        __syncthreads();
    }
    if (!tid) partial[blockIdx.x] = {values[0][0], values[1][0], values[2][0],
        values[3][0], values[4][0], values[5][0], fmax(values[6][0], double(*invalid))};
}

__global__ void finish_metrics(int count, const Metrics* partial, Metrics* result, const int* enabled=nullptr) {
    if (enabled && !*enabled) return;
    __shared__ double values[7][256];
    const int tid = threadIdx.x;
    Metrics m = tid < count ? partial[tid] : Metrics{};
    values[0][tid]=m.fuel; values[1][tid]=m.penalty; values[2][tid]=m.virtual_sum;
    values[3][tid]=m.defect; values[4][tid]=m.virtual_inf; values[5][tid]=m.step; values[6][tid]=m.invalid;
    __syncthreads();
    for (int offset=128; offset; offset/=2) {
        if (tid<offset) for (int j=0; j<7; ++j)
            values[j][tid]=j<3 ? values[j][tid]+values[j][tid+offset]
                : fmax(values[j][tid],values[j][tid+offset]);
        __syncthreads();
    }
    if (!tid) *result={values[0][0],values[1][0],values[2][0],values[3][0],
        values[4][0],values[5][0],values[6][0]};
}

__global__ void initialize(State* s, Settings p, spacepdhcg_gtoc12_conic_parameters* conic) {
    *s={};
    s->command.substeps=p.substeps;
    s->polish_left=p.polish_iterations;
    s->result.virtual_inf=INFINITY;
    s->trust_state=p.initial_trust_state; s->trust_control=p.initial_trust_control;
    *conic={p.initial_trust_state,p.initial_trust_control,p.virtual_weight,p.minimum_mass,
        p.radius_floor,p.vinf_max,p.smoothness_weight};
}

__global__ void set_reference(State* s, const Metrics* m, Settings p, bool only_refresh=false) {
    if (only_refresh && !s->command.refresh) return;
    s->merit=m->fuel+p.virtual_weight*m->penalty;
    s->result.max_defect=m->defect;
    s->command.refresh=0;
    if (m->invalid || !isfinite(s->merit)) { s->command.error=3; s->command.done=1; }
}

__device__ void shrink(State* s, Settings p) {
    s->trust_state=fmax(s->trust_state*p.shrink_factor,p.minimum_trust);
    s->trust_control=fmax(s->trust_control*p.shrink_factor,p.minimum_trust);
}

__global__ void decide(State* s, Settings p, const Metrics* m, const double* x,
    int nodes, int free_dep, int free_arr, int qualified, int qoco_status,
    Record* records, spacepdhcg_gtoc12_conic_parameters* conic,
    const spacepdhcg_gtoc12_qoco_report* device_report=nullptr) {
    if (device_report) {
        qualified=device_report->qualified;
        qoco_status=device_report->qoco_status;
    }
    const int iteration=++s->result.iterations;
    auto& record=records[iteration-1];
    record={}; record.iteration=iteration; record.qoco_status=qoco_status;
    s->copy_candidate=0;
    if (!qualified || m->invalid || !isfinite(m->fuel) || !isfinite(m->virtual_sum)) {
        // Match the reference's unclamped shrink on a rejected conic solve.
        record.conic_rejected=1;
        s->trust_state*=p.shrink_factor; s->trust_control*=p.shrink_factor;
        if (s->trust_state<p.minimum_trust) {
            s->result.status=2; s->result.diagnostic=1; s->command.done=1;
        }
    } else {
        const double merit=m->fuel+p.virtual_weight*m->penalty;
        const double predicted=s->merit-(m->fuel+p.virtual_weight*m->virtual_sum);
        const double actual=s->merit-merit;
        const double ratio=predicted>1e-15 ? actual/predicted : (actual>=0.0 ? 1.0 : -1.0);
        record.merit=merit; record.final_mass_fraction=x[7*nodes-1]; record.max_defect=m->defect;
        record.virtual_inf=m->virtual_inf; record.ratio=ratio; record.step=m->step;
        record.trust_state=s->trust_state; record.trust_control=s->trust_control;
        if (s->polishing) --s->polish_left;
        if (ratio<p.ratio_reject) {
            shrink(s,p);
            if (s->trust_state<=p.minimum_trust && s->trust_control<=p.minimum_trust) {
                s->result.status=2; s->result.diagnostic=2; s->command.done=1;
            }
        } else {
            record.accepted=1; s->copy_candidate=1; ++s->result.accepted_iterations;
            s->merit=merit; s->result.max_defect=m->defect; s->result.virtual_inf=m->virtual_inf;
            const int ivd=11*nodes+14*(nodes-1), iva=ivd+3*free_dep;
            for (int j=0;j<3;++j) {
                if (free_dep) s->result.departure_vinf[j]=x[ivd+j];
                if (free_arr) s->result.arrival_vinf[j]=x[iva+j];
            }
            if (ratio<p.ratio_shrink) shrink(s,p);
            else if (ratio>p.ratio_grow) {
                s->trust_state=fmin(s->trust_state*p.grow_factor,p.maximum_trust);
                s->trust_control=fmin(s->trust_control*p.grow_factor,p.maximum_trust);
            }
            const bool feasible=m->defect<=p.defect_tolerance && m->virtual_inf<=10.0*p.defect_tolerance;
            if (feasible && (m->step<=p.step_tolerance || predicted<=p.objective_tolerance)) {
                if (s->polishing || p.polish_iterations==0 || p.polish_substeps<=p.substeps) {
                    s->result.status=1; s->command.done=1;
                } else {
                    s->polishing=1; s->command.substeps=p.polish_substeps; s->command.refresh=1;
                    s->trust_state=fmax(s->trust_state,1e-3); s->trust_control=fmax(s->trust_control,1e-2);
                }
            } else if (iteration>=p.max_iterations && !s->polishing) s->command.done=1;
        }
    }
    conic->trust_state=s->trust_state; conic->trust_control=s->trust_control;
    if (s->polishing && s->polish_left<=0) s->command.done=1;
}

__global__ void accept_candidate(const State* s,int nodes,const double* x,double* states,double* controls) {
    if (!s->copy_candidate) return;
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<7*nodes;i+=blockDim.x*gridDim.x) {
        states[i]=x[i];
        if (i<4*nodes) controls[i]=x[7*nodes+i];
    }
}

__global__ void finalize(State* s, Settings p, int timeout) {
    auto& r=s->result;
    if (timeout) r.status=4;
    const bool feasible=r.max_defect<=10.0*p.defect_tolerance && r.virtual_inf<=100.0*p.defect_tolerance;
    if (r.status==0 && feasible) { r.status=1; r.diagnostic=s->polishing ? 3 : 4; }
    if (r.status==2 && feasible) { r.status=1; r.diagnostic=5; }
    if ((r.status==0 || r.status==2) && r.virtual_inf>1e-4) { r.status=3; r.diagnostic=6; }
}

template<class T> bool allocate(T** p,size_t n) { return cudaMalloc(p,n*sizeof(T))==cudaSuccess; }
struct Workspace {
    gtoc12_seed::Scratch seed;
    cudaStream_t stream{};
    spacepdhcg_gtoc12_qoco* qoco{};
    spacepdhcg_gtoc12_discretisation* dynamics{};
    double *states{},*controls{},*fuel{};
    Metrics *partial{},*metrics{};
    State* state{};
    Record* records{};
    spacepdhcg_gtoc12_conic_parameters* parameters{};
    ~Workspace() {
        if (stream) cudaStreamSynchronize(stream);
        spacepdhcg_gtoc12_qoco_destroy(qoco); spacepdhcg_gtoc12_discretisation_destroy(dynamics);
        cudaFree(states); cudaFree(controls); cudaFree(fuel); cudaFree(partial); cudaFree(metrics);
        cudaFree(state); cudaFree(records); cudaFree(parameters);
        if (stream) cudaStreamDestroy(stream);
    }
};
bool valid(const Settings& p) {
    return p.substeps>0 && p.polish_substeps>0 && p.max_iterations>0 && p.polish_iterations>=0
        && p.max_iterations<=INT_MAX-p.polish_iterations
        && std::isfinite(p.time_limit_s) && p.time_limit_s>=0.0
        && std::isfinite(p.virtual_weight) && p.virtual_weight>=0.0
        && std::isfinite(p.smoothness_weight) && p.smoothness_weight>=0.0
        && std::isfinite(p.initial_trust_state) && p.initial_trust_state>0.0
        && std::isfinite(p.initial_trust_control) && p.initial_trust_control>0.0
        && std::isfinite(p.minimum_trust) && p.minimum_trust>0.0
        && std::isfinite(p.maximum_trust) && p.maximum_trust>=p.minimum_trust
        && std::isfinite(p.ratio_reject) && std::isfinite(p.ratio_shrink) && std::isfinite(p.ratio_grow)
        && p.ratio_reject<=p.ratio_shrink && p.ratio_shrink<=p.ratio_grow
        && std::isfinite(p.shrink_factor) && p.shrink_factor>0.0 && p.shrink_factor<1.0
        && std::isfinite(p.grow_factor) && p.grow_factor>1.0
        && std::isfinite(p.defect_tolerance) && p.defect_tolerance>0.0
        && std::isfinite(p.step_tolerance) && p.step_tolerance>=0.0
        && std::isfinite(p.objective_tolerance) && p.objective_tolerance>=0.0
        && std::isfinite(p.conic_tolerance) && p.conic_tolerance>0.0
        && std::isfinite(p.minimum_mass) && p.minimum_mass>0.0
        && std::isfinite(p.radius_floor) && p.radius_floor>=0.0
        && std::isfinite(p.vinf_max) && p.vinf_max>=0.0;
}
}

extern "C" int spacepdhcg_gtoc12_scvx_solve_host(int intervals,int hold,int free_dep,int free_arr,
    double kappa,double mass_flow,const double* times,const double* boundary,const double* fuel,
    const double* seed_states,const double* seed_controls,int ruiz,const Settings* settings,
    double* states,double* controls,Record* records,spacepdhcg_gtoc12_qoco_report* reports,Result* result) {
    if (!settings || !valid(*settings) || (bool(seed_states)!=bool(seed_controls)) || !states || !controls
        || !records || !reports || !result) return 1;
    *result={};
    const auto started=std::chrono::steady_clock::now();
    const auto p=*settings;
    Workspace w;
    int code=spacepdhcg_gtoc12_qoco_create(intervals,hold,free_dep,free_arr,kappa,mass_flow,times,boundary,
        fuel,p.conic_tolerance,ruiz,&w.qoco);
    if (code) return code;
    code=spacepdhcg_gtoc12_discretisation_create(intervals,hold,kappa,mass_flow,times,&w.dynamics);
    if (code) return code;
    spacepdhcg_gtoc12_conic_dimensions dimensions{};
    if (spacepdhcg_gtoc12_qoco_get_dimensions(w.qoco,&dimensions)) return 2;
    const int nodes=intervals+1, budget=p.max_iterations+p.polish_iterations;
    const int blocks=std::min(256,(dimensions.variables+255)/256);
    if (cudaStreamCreateWithFlags(&w.stream,cudaStreamNonBlocking)!=cudaSuccess
        || !allocate(&w.states,7*nodes) || !allocate(&w.controls,4*nodes) || !allocate(&w.fuel,nodes)
        || !allocate(&w.partial,blocks) || !allocate(&w.metrics,1) || !allocate(&w.state,1)
        || !allocate(&w.records,budget) || !allocate(&w.parameters,1)) return 2;
    if (cudaMemcpyAsync(w.fuel,fuel,nodes*sizeof(double),cudaMemcpyHostToDevice,w.stream)!=cudaSuccess) return 2;
    if (seed_states) {
        if (cudaMemcpyAsync(w.states,seed_states,7*nodes*sizeof(double),cudaMemcpyHostToDevice,w.stream)!=cudaSuccess
            || cudaMemcpyAsync(w.controls,seed_controls,4*nodes*sizeof(double),cudaMemcpyHostToDevice,w.stream)!=cudaSuccess) return 2;
    } else {
        if (!w.seed.create(nodes,times,boundary,w.stream)) return 2;
        if ((code=w.seed.launch(nodes,free_dep,free_arr,p.vinf_max,w.states,w.controls,w.stream))) return code;
    }
    const double *a{},*b{},*c{},*propagated{}; const int* invalid{};
    if (spacepdhcg_gtoc12_discretisation_outputs(w.dynamics,&a,&b,&c,&propagated,&invalid)) return 2;
    const auto measure=[&](const double* x,const double* u,int substeps,bool candidate) {
        int status=spacepdhcg_gtoc12_discretisation_launch_device(w.dynamics,x,u,substeps,0,w.stream);
        if (status) return status;
        reduce_metrics<<<blocks,256,0,w.stream>>>(nodes,candidate ? dimensions.variables : 7*nodes,
            x,u,candidate ? x+11*nodes : nullptr,candidate ? w.states : nullptr,
            candidate ? w.controls : nullptr,propagated,invalid,w.fuel,p.conic_tolerance,w.partial);
        finish_metrics<<<1,256,0,w.stream>>>(blocks,w.partial,w.metrics);
        return cudaGetLastError()==cudaSuccess ? 0 : 2;
    };
    struct ConsumerContext {
        const decltype(measure)* measure;
        Workspace* workspace;
        Settings settings;
        int nodes,free_dep,free_arr,substeps;
    } consumer_context{&measure,&w,p,nodes,free_dep,free_arr,p.substeps};
    const auto consume=+[](void* opaque,const spacepdhcg_gtoc12_qoco_report* report,
        const double* x,void* stream) -> int {
        auto& c=*static_cast<ConsumerContext*>(opaque);
        auto& w=*c.workspace;
        if (stream!=w.stream) return 2;
        // Speculative measurement handles invalid numbers on device; acceptance
        // is always gated by the independent conic qualification in decide.
        const int status=(*c.measure)(x,x+7*c.nodes,c.substeps,true);
        if (status) return status;
        decide<<<1,1,0,w.stream>>>(w.state,c.settings,w.metrics,x,c.nodes,c.free_dep,c.free_arr,
            0,-1,w.records,w.parameters,report);
        accept_candidate<<<std::min(256,(7*c.nodes+255)/256),256,0,w.stream>>>(
            w.state,c.nodes,x,w.states,w.controls);
        return cudaGetLastError()==cudaSuccess ? 0 : 2;
    };
    const char* device_qualification=std::getenv("SPACEPDHCG_TEST_GTOC12_DEVICE_QUALIFICATION");
    const bool consume_on_device=device_qualification && device_qualification[0]=='1';
    const char* refresh_option=std::getenv("SPACEPDHCG_TEST_GTOC12_DEVICE_REFRESH");
    const bool refresh_on_device=refresh_option && refresh_option[0]=='1';
    const auto refresh_reference=[&]() {
        const int* enabled=&w.state->command.refresh;
        const int status=spacepdhcg_gtoc12_discretisation_launch_controlled_device(
            w.dynamics,w.states,w.controls,&w.state->command.substeps,enabled,0,w.stream);
        if (status) return status;
        reduce_metrics<<<blocks,256,0,w.stream>>>(nodes,7*nodes,w.states,w.controls,nullptr,
            nullptr,nullptr,propagated,invalid,w.fuel,p.conic_tolerance,w.partial,enabled);
        finish_metrics<<<1,256,0,w.stream>>>(blocks,w.partial,w.metrics,enabled);
        set_reference<<<1,1,0,w.stream>>>(w.state,w.metrics,p,true);
        return cudaGetLastError()==cudaSuccess ? 0 : 2;
    };
    initialize<<<1,1,0,w.stream>>>(w.state,p,w.parameters);
    if ((code=measure(w.states,w.controls,p.substeps,false))) return code;
    set_reference<<<1,1,0,w.stream>>>(w.state,w.metrics,p);
    Command command{};
    uint64_t control_bytes=0;
    const auto read_command=[&]() {
        control_bytes+=sizeof(Command);
        return cudaGetLastError()==cudaSuccess
            && cudaMemcpyAsync(&command,&w.state->command,sizeof(command),cudaMemcpyDeviceToHost,w.stream)==cudaSuccess
            && cudaStreamSynchronize(w.stream)==cudaSuccess;
    };
    if (!read_command()) return 2;
    if (command.error) return command.error;
    int attempts=0,timeout=0;
    for (;attempts<budget && !command.done;++attempts) {
        if (std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()>p.time_limit_s) {
            timeout=1; break;
        }
        int consumed=0;
        consumer_context.substeps=command.substeps;
        code=spacepdhcg_gtoc12_qoco_solve_device_with_consumer(w.qoco,w.states,w.controls,w.parameters,
            command.substeps,w.stream,&reports[attempts],consume_on_device ? consume : nullptr,
            &consumer_context,&consumed);
        if (code!=0 && code!=4) return code;
        if (!consumed) {
        const double* x{};
        if (spacepdhcg_gtoc12_qoco_primal(w.qoco,&x)) return 2;
        if (code==0 && (code=measure(x,x+7*nodes,command.substeps,true))) return code;
        decide<<<1,1,0,w.stream>>>(w.state,p,w.metrics,x,nodes,free_dep,free_arr,
            reports[attempts].qualified,reports[attempts].qoco_status,w.records,w.parameters);
        accept_candidate<<<std::min(256,(7*nodes+255)/256),256,0,w.stream>>>(w.state,nodes,x,w.states,w.controls);
        }
        if (consumed && std::getenv("SPACEPDHCG_TEST_GTOC12_DEVICE_QUALIFICATION_TRACE"))
            std::fprintf(stderr,"DEVICE_QUALIFICATION iteration=%d qualified=%d status=%d\n",
                attempts+1,reports[attempts].qualified,reports[attempts].qoco_status);
        if (refresh_on_device && (code=refresh_reference())) return code;
        if (!read_command()) return 2;
        if (!refresh_on_device && command.refresh) {
            if ((code=measure(w.states,w.controls,command.substeps,false))) return code;
            set_reference<<<1,1,0,w.stream>>>(w.state,w.metrics,p);
            if (!read_command()) return 2;
        }
        if (command.error) return command.error;
    }
    finalize<<<1,1,0,w.stream>>>(w.state,p,timeout);
    if (cudaGetLastError()!=cudaSuccess
        || cudaMemcpyAsync(states,w.states,7*nodes*sizeof(double),cudaMemcpyDeviceToHost,w.stream)!=cudaSuccess
        || cudaMemcpyAsync(controls,w.controls,4*nodes*sizeof(double),cudaMemcpyDeviceToHost,w.stream)!=cudaSuccess
        || cudaMemcpyAsync(records,w.records,attempts*sizeof(Record),cudaMemcpyDeviceToHost,w.stream)!=cudaSuccess
        || cudaMemcpyAsync(result,&w.state->result,sizeof(Result),cudaMemcpyDeviceToHost,w.stream)!=cudaSuccess
        || cudaStreamSynchronize(w.stream)!=cudaSuccess) return 2;
    result->trajectory_upload_bytes=seed_states ? 11*nodes*sizeof(double) : 0;
    result->trajectory_download_bytes=11*nodes*sizeof(double);
    result->control_download_bytes=control_bytes;
    return 0;
}

extern "C" int spacepdhcg_gtoc12_seed_evaluate_host(int nodes,const double* times,const double* boundary,
    int free_dep,int free_arr,double vinf,double* states,double* controls) {
    if (nodes<2 || nodes>(INT_MAX-1024)/512 || !times || !boundary || !states || !controls
        || (free_dep!=0 && free_dep!=1) || (free_arr!=0 && free_arr!=1) || !std::isfinite(vinf) || vinf<0) return 1;
    for (int i=0;i<nodes;++i) if (!std::isfinite(times[i]) || (i && !(times[i]>times[i-1]))) return 1;
    for (int i=0;i<12;++i) if (!std::isfinite(boundary[i])) return 1;
    Workspace w;
    if (cudaStreamCreateWithFlags(&w.stream,cudaStreamNonBlocking)!=cudaSuccess
        || !allocate(&w.states,7*nodes) || !allocate(&w.controls,4*nodes)
        || !w.seed.create(nodes,times,boundary,w.stream)) return 2;
    int code=w.seed.launch(nodes,free_dep,free_arr,vinf,w.states,w.controls,w.stream);
    if (code) return code;
    int invalid=0;
    if (cudaMemcpyAsync(states,w.states,7*nodes*sizeof(double),cudaMemcpyDeviceToHost,w.stream)!=cudaSuccess
        || cudaMemcpyAsync(controls,w.controls,4*nodes*sizeof(double),cudaMemcpyDeviceToHost,w.stream)!=cudaSuccess
        || cudaMemcpyAsync(&invalid,w.seed.invalid,sizeof(int),cudaMemcpyDeviceToHost,w.stream)!=cudaSuccess
        || cudaStreamSynchronize(w.stream)!=cudaSuccess) return 2;
    return invalid ? 3 : 0;
}
