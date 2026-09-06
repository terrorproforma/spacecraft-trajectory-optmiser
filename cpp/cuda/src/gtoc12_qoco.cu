#include "spacepdhcg/cuda/gtoc12_qoco_c_api.h"
#include "native_qoco_adapter.h"
#include "gtoc12_qoco_qualification.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>
#include <thread>
#include <vector>

namespace {
struct Map { int source; double scale; };
struct Entry { int row, column; Map map; };
struct Matrix {
    std::vector<int> offsets, indices;
    std::vector<Map> maps;
    int *device_offsets{}, *device_indices{};
    double* values{};
};
void compile(Matrix& matrix, std::vector<Entry>& entries, int columns) {
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.column < b.column || (a.column == b.column && a.row < b.row);
    });
    matrix.offsets.resize(columns+1);
    for (const auto& e : entries) {
        ++matrix.offsets[e.column+1]; matrix.indices.push_back(e.row); matrix.maps.push_back(e.map);
    }
    for (int i=0;i<columns;++i) matrix.offsets[i+1]+=matrix.offsets[i];
}
template<class T> bool allocate(T** pointer, size_t n) {
    return cudaMalloc(pointer, n*sizeof(T))==cudaSuccess;
}
template<class T> bool upload(T* destination, const T* source, size_t n, cudaStream_t stream) {
    return cudaMemcpyAsync(destination,source,n*sizeof(T),cudaMemcpyHostToDevice,stream)==cudaSuccess;
}
template<class T> spacepdhcg_accelerator_buffer_view view(T* data, size_t n, int device) {
    spacepdhcg_accelerator_buffer_view v{};
    v.data=const_cast<std::remove_const_t<T>*>(data); v.elements=n; v.element_stride=1;
    v.device={SPACEPDHCG_DEVICE_CUDA,device};
    v.scalar_type=std::is_same_v<std::remove_const_t<T>,int> ? SPACEPDHCG_SCALAR_INT32 : SPACEPDHCG_SCALAR_FLOAT64;
    v.access=SPACEPDHCG_ACCESS_READ_ONLY; return v;
}
__global__ void convert_values(int count, const Map* maps, const double* input, double* output) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x)
        output[i]=maps[i].scale*input[maps[i].source];
}
__global__ void bounds(int equalities,int scalar_rows,int variables,const double* b,
    double* lower,double* variable_lower,double* variable_upper) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<max(scalar_rows,variables);i+=blockDim.x*gridDim.x) {
        if (i<scalar_rows) lower[i]=i<equalities ? b[i] : -CUDART_INF;
        if (i<variables) { variable_lower[i]=-CUDART_INF; variable_upper[i]=CUDART_INF; }
    }
}
__global__ void objective_partial(int qnnz,int variables,int scalar_rows,int affine_rows,
    const int* qrows,const int* qcols,const double* qvalues,const double* objective,
    const double* scalar_rhs,const double* affine_rhs,const double* x,const double* dual,double* partial) {
    __shared__ double sums[3][256];
    const int tid=threadIdx.x;
    double quadratic=0.0,linear=0.0,rhs=0.0;
    const int count=max(max(qnnz,variables),scalar_rows+affine_rows);
    for (int i=blockIdx.x*blockDim.x+tid;i<count;i+=blockDim.x*gridDim.x) {
        if (i<qnnz) quadratic+=x[qrows[i]]*qvalues[i]*x[qcols[i]];
        if (i<variables) linear+=objective[i]*x[i];
        if (i<scalar_rows) rhs+=scalar_rhs[i]*dual[i];
        else if (i<scalar_rows+affine_rows) rhs+=affine_rhs[i-scalar_rows]*dual[i];
    }
    sums[0][tid]=quadratic; sums[1][tid]=linear; sums[2][tid]=rhs;
    __syncthreads();
    for (int offset=128;offset;offset/=2) {
        if (tid<offset) for (int j=0;j<3;++j) sums[j][tid]+=sums[j][tid+offset];
        __syncthreads();
    }
    if (tid==0) for (int j=0;j<3;++j) partial[3*blockIdx.x+j]=sums[j][0];
}
__global__ void objective_finish(int blocks,const double* partial,double* output) {
    __shared__ double sums[3][256];
    const int tid=threadIdx.x;
    for (int j=0;j<3;++j) sums[j][tid]=tid<blocks ? partial[3*tid+j] : 0.0;
    __syncthreads();
    for (int offset=128;offset;offset/=2) {
        if (tid<offset) for (int j=0;j<3;++j) sums[j][tid]+=sums[j][tid+offset];
        __syncthreads();
    }
    if (tid==0) {
        const double primal=0.5*sums[0][0]+sums[1][0],dual=-0.5*sums[0][0]-sums[2][0];
        output[0]=primal; output[1]=dual; output[2]=fabs(primal-dual);
        output[3]=output[2]/fmax(1.0,fmax(fabs(primal),fabs(dual)));
    }
}
int status_code(spacepdhcg_cuda_status s) {
    if (s==SPACEPDHCG_CUDA_SUCCESS) return 0;
    if (s==SPACEPDHCG_CUDA_UNSUPPORTED) return 5;
    if (s==SPACEPDHCG_CUDA_NUMERICAL_FAILURE) return 4;
    return 2;
}
}

struct spacepdhcg_gtoc12_qoco {
    spacepdhcg_gtoc12_conic* conic{};
    spacepdhcg_gtoc12_conic_dimensions dimensions{};
    spacepdhcg_gtoc12_conic_device_outputs output{};
    Matrix q,a,f;
    std::vector<spacepdhcg_cuda_cone_descriptor> cones;
    std::vector<Map> host_maps;
    Map* maps{};
    double *values{},*lower{},*variable_lower{},*variable_upper{},*primal{},*dual{};
    double *states{},*controls{};
    int* quadratic_columns{};
    double *objective_partial{},*objective_result{};
    spacepdhcg_gtoc12_qoco_report* device_report{};
    int objective_blocks{};
    spacepdhcg_gtoc12_conic_parameters* parameters{};
    spacepdhcg_cuda_scvx_problem problem{};
    spacepdhcg_native_qoco* solver{};
    spacepdhcg_native_qoco_report native_report{};
    int intervals{},device{},ruiz{},count{};
    int scalar_rows{},affine_rows{};
    double tolerance{};
    cudaStream_t stream{},pending_stream{};
    bool pending{};
    std::thread::id pending_owner{};
};

namespace {
struct Consumer {
    spacepdhcg_gtoc12_qoco* w;
    spacepdhcg_gtoc12_qoco_consumer function;
    void* context;
    int* consumed;
};
cudaError_t consume_audit(void* opaque, const QocoReplayStatus* status,
    const QocoAuditResult* audit, cudaStream_t stream) {
    auto& c=*static_cast<Consumer*>(opaque);
    auto* w=c.w;
    objective_partial<<<w->objective_blocks,256,0,stream>>>(w->q.indices.size(),w->dimensions.variables,
        w->scalar_rows,w->affine_rows,w->q.device_indices,w->quadratic_columns,w->q.values,
        w->output.q,static_cast<const double*>(w->problem.numeric.scalar_upper.data),
        static_cast<const double*>(w->problem.numeric.affine_offset.data),w->primal,w->dual,w->objective_partial);
    objective_finish<<<1,256,0,stream>>>(w->objective_blocks,w->objective_partial,w->objective_result);
    gtoc12_qoco::qualify<<<1,1,0,stream>>>(status,audit,w->objective_result,w->tolerance,w->device_report);
    auto error=cudaGetLastError();
    if (error!=cudaSuccess) return error;
    if (c.function(c.context,w->device_report,w->primal,stream)) return cudaErrorUnknown;
    *c.consumed=1;
    return cudaSuccess;
}
bool correct_device(const spacepdhcg_gtoc12_qoco* w) {
    int device=-1; return w && cudaGetDevice(&device)==cudaSuccess && device==w->device;
}
void report_result(spacepdhcg_gtoc12_qoco* w, spacepdhcg_gtoc12_qoco_report* out) {
    const auto& r=w->native_report;
    out->qoco_status=r.status_code; out->iterations=r.iterations; out->failure=r.failure;
    out->primal_residual=r.primal_residual; out->dual_residual=r.dual_residual;
    out->absolute_primal_residual=r.absolute_primal_residual; out->absolute_dual_residual=r.absolute_dual_residual;
    out->setup_seconds=r.setup_seconds; out->update_seconds=r.update_seconds;
    out->solve_seconds=r.solve_seconds; out->residual_seconds=r.residual_seconds;
    out->workspace_creations=r.workspace_creations; out->numeric_updates=r.numeric_updates;
    out->device_numeric_updates=r.device_numeric_updates; out->solves=r.solves;
    out->adapter_d2h_count=r.d2h_copy_count; out->adapter_d2h_bytes=r.d2h_bytes;
}
}

extern "C" void spacepdhcg_gtoc12_qoco_destroy(spacepdhcg_gtoc12_qoco* w) {
    if (!w) return;
    int previous=-1; cudaGetDevice(&previous); cudaSetDevice(w->device);
    if (w->stream) cudaStreamSynchronize(w->stream);
    spacepdhcg_native_qoco_destroy(w->solver);
    spacepdhcg_gtoc12_conic_destroy(w->conic);
    for (auto* m : {&w->q,&w->a,&w->f}) { cudaFree(m->device_offsets); cudaFree(m->device_indices); }
    cudaFree(w->maps); cudaFree(w->values); cudaFree(w->lower); cudaFree(w->variable_lower);
    cudaFree(w->variable_upper); cudaFree(w->primal); cudaFree(w->dual);
    cudaFree(w->states); cudaFree(w->controls); cudaFree(w->parameters);
    cudaFree(w->quadratic_columns); cudaFree(w->objective_partial); cudaFree(w->objective_result);
    cudaFree(w->device_report);
    if (w->stream) cudaStreamDestroy(w->stream);
    if (previous>=0 && previous!=w->device) cudaSetDevice(previous);
    delete w;
}

extern "C" int spacepdhcg_gtoc12_qoco_create(int intervals,int hold,int free_dep,int free_arr,
    double kappa,double mass_flow,const double* times,const double* boundary,const double* fuel,
    double tolerance,int ruiz,spacepdhcg_gtoc12_qoco** output) {
    if (!output) return 1;
    *output=nullptr;
    if (!std::isfinite(tolerance) || tolerance<=0.0 || ruiz<0 || ruiz>100) return 1;
    auto* w=new(std::nothrow) spacepdhcg_gtoc12_qoco;
    if (!w) return 2;
    if (cudaGetDevice(&w->device)!=cudaSuccess) { delete w; return 2; }
    const auto failed=[&](int code) { spacepdhcg_gtoc12_qoco_destroy(w); return code; };
    w->intervals=intervals; w->tolerance=tolerance; w->ruiz=ruiz;
    try {
        const int created=spacepdhcg_gtoc12_conic_create(intervals,hold,free_dep,free_arr,kappa,mass_flow,times,boundary,fuel,&w->conic);
        if (created) return failed(created);
        spacepdhcg_gtoc12_conic_get_dimensions(w->conic,&w->dimensions);
        spacepdhcg_gtoc12_conic_outputs(w->conic,&w->output);
        const auto& d=w->dimensions;
        const int source_scalar_rows=d.equalities+d.inequalities;
        const int scalar_rows=source_scalar_rows-(hold ? 0 : 2);
        const int affine_rows=d.rows-source_scalar_rows-(hold ? 0 : 4);
        w->scalar_rows=scalar_rows; w->affine_rows=affine_rows;
        const int terminal_gamma_rows=d.equalities+26*intervals;
        std::vector<int> ao(d.variables+1),ai(d.a_nonzeros),po(d.variables+1),pi(d.p_nonzeros);
        spacepdhcg_gtoc12_conic_copy_topology_host(w->conic,ao.data(),ai.data(),po.data(),pi.data());
        std::vector<Entry> qe,ae,fe;
        const int p_start=d.a_nonzeros+d.rows+d.variables;
        for (int col=0;col<d.variables;++col) {
            for (int i=po[col];i<po[col+1];++i) {
                qe.push_back({pi[i],col,{p_start+i,1.0}});
                if (pi[i]!=col) qe.push_back({col,pi[i],{p_start+i,1.0}});
            }
            for (int i=ao[col];i<ao[col+1];++i) {
                if (ai[i]<source_scalar_rows) {
                    // ZOH already fixes every final control component to zero.
                    // Its Gamma bounds and zero thrust cone are redundant and
                    // force the IPM onto a cone boundary at every feasible point.
                    // Omit exactly these rows; keep all reference-dependent
                    // control trust rows, so no admissible set is enlarged.
                    if (!hold && (ai[i]==terminal_gamma_rows || ai[i]==terminal_gamma_rows+1)) continue;
                    const int row=ai[i]-(!hold && ai[i]>terminal_gamma_rows+1 ? 2 : 0);
                    ae.push_back({row,col,{i,1.0}});
                }
                else {
                    const int original=ai[i]-source_scalar_rows;
                    if (!hold && original>=4*intervals && original<4*(intervals+1)) continue;
                    const int row=original-(!hold && original>=4*(intervals+1) ? 4 : 0);
                    // Canonical SOC uses [vector,t], Clarabel uses [t,vector].
                    const int mapped=(row/4)*4+(row%4+3)%4;
                    fe.push_back({mapped,col,{i,-1.0}});
                }
            }
        }
        compile(w->q,qe,d.variables); compile(w->a,ae,d.variables); compile(w->f,fe,d.variables);
        std::vector<int> qcolumns(w->q.indices.size());
        for (int col=0;col<d.variables;++col)
            for (int i=w->q.offsets[col];i<w->q.offsets[col+1];++i) qcolumns[i]=col;
        w->objective_blocks=std::min(256,(std::max({static_cast<int>(qe.size()),d.variables,d.rows})+255)/256);
        for (const auto* m : {&w->q,&w->a,&w->f})
            w->host_maps.insert(w->host_maps.end(),m->maps.begin(),m->maps.end());
        const int matrix_values=static_cast<int>(w->host_maps.size());
        for (int row=0;row<affine_rows;++row) {
            const int compact=(row/4)*4+(row%4+1)%4;
            const int original=compact+(!hold && compact>=4*intervals ? 4 : 0);
            w->host_maps.push_back({d.a_nonzeros+source_scalar_rows+original,1.0});
        }
        for (int row=0;row<scalar_rows;++row) {
            const int original=row+(!hold && row>=terminal_gamma_rows ? 2 : 0);
            w->host_maps.push_back({d.a_nonzeros+original,1.0});
        }
        w->count=static_cast<int>(w->host_maps.size());
        // The canonical descriptor stores total size minus two for SOC/RSOC.
        for (int i=0;i<d.soc_count-(hold ? 0 : 1);++i) w->cones.push_back({SPACEPDHCG_CUDA_CONE_SECOND_ORDER,4*i,2,0.0});
        if (cudaStreamCreateWithFlags(&w->stream,cudaStreamNonBlocking)!=cudaSuccess
            || !allocate(&w->maps,w->count) || !allocate(&w->values,w->count)
            || !allocate(&w->lower,scalar_rows) || !allocate(&w->variable_lower,d.variables)
            || !allocate(&w->variable_upper,d.variables) || !allocate(&w->primal,d.variables)
            || !allocate(&w->dual,d.rows) || !allocate(&w->states,(intervals+1)*7)
            || !allocate(&w->controls,(intervals+1)*4) || !allocate(&w->parameters,1)
            || !allocate(&w->quadratic_columns,qcolumns.size())
            || !allocate(&w->objective_partial,3*w->objective_blocks)
            || !allocate(&w->objective_result,4) || !allocate(&w->device_report,1)) return failed(2);
        int offset=0;
        for (auto* m : {&w->q,&w->a,&w->f}) {
            m->values=w->values+offset; offset+=static_cast<int>(m->maps.size());
            if (!allocate(&m->device_offsets,m->offsets.size()) || !allocate(&m->device_indices,m->indices.size())
                || !upload(m->device_offsets,m->offsets.data(),m->offsets.size(),w->stream)
                || !upload(m->device_indices,m->indices.data(),m->indices.size(),w->stream)) return failed(2);
        }
        if (!upload(w->maps,w->host_maps.data(),w->count,w->stream)
            || !upload(w->quadratic_columns,qcolumns.data(),qcolumns.size(),w->stream)
            || cudaStreamSynchronize(w->stream)!=cudaSuccess) return failed(2);
        auto& p=w->problem;
        p.abi_version=SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;
        p.dynamics.model=SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST;
        // Topology is owned for the lifetime of this solver, with no caller mutation.
        p.topology_fingerprint=static_cast<uint64_t>(intervals)*8+hold*4+free_dep*2+free_arr;
        p.canonical_structure={SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,p.topology_fingerprint,
            d.variables,scalar_rows,affine_rows,w->q.indices.size(),w->a.indices.size(),w->f.indices.size(),
            w->cones.data(),w->cones.size(),nullptr,0};
        p.canonical_topology={view(w->q.device_offsets,w->q.offsets.size(),w->device),view(w->q.device_indices,w->q.indices.size(),w->device),
            view(w->a.device_offsets,w->a.offsets.size(),w->device),view(w->a.device_indices,w->a.indices.size(),w->device),
            view(w->f.device_offsets,w->f.offsets.size(),w->device),view(w->f.device_indices,w->f.indices.size(),w->device)};
        p.numeric={view(w->q.values,w->q.indices.size(),w->device),view(w->a.values,w->a.indices.size(),w->device),
            view(w->f.values,w->f.indices.size(),w->device),view(w->output.q,d.variables,w->device),
            view(w->lower,scalar_rows,w->device),view(w->values+matrix_values+affine_rows,scalar_rows,w->device),
            view(w->values+matrix_values,affine_rows,w->device),view(w->variable_lower,d.variables,w->device),
            view(w->variable_upper,d.variables,w->device)};
        *output=w; return 0;
    } catch (...) { return failed(2); }
}

extern "C" int spacepdhcg_gtoc12_qoco_get_dimensions(spacepdhcg_gtoc12_qoco* w,
    spacepdhcg_gtoc12_conic_dimensions* d) {
    if (!w || !d) return 1;
    *d=w->dimensions; return 0;
}
extern "C" int spacepdhcg_gtoc12_qoco_primal(spacepdhcg_gtoc12_qoco* w,const double** output) {
    if (!w || !output) return 1;
    *output=w->primal; return 0;
}
static int solve_device_with_consumer_impl(spacepdhcg_gtoc12_qoco* w,
    const double* states,const double* controls,const spacepdhcg_gtoc12_conic_parameters* parameters,
    int substeps,void* stream_pointer,spacepdhcg_gtoc12_qoco_report* report,
    spacepdhcg_gtoc12_qoco_consumer consumer, void* context, int* consumed,const int* device_substeps,bool defer=false) {
    if (!consumed) return 1;
    *consumed=0;
    if (!report) return 1;
    *report={}; report->qoco_status=-1;
    report->primal_residual=report->dual_residual=std::numeric_limits<double>::infinity();
    report->absolute_primal_residual=report->absolute_dual_residual=std::numeric_limits<double>::infinity();
    report->primal_objective=report->dual_objective=std::numeric_limits<double>::quiet_NaN();
    report->absolute_gap=report->relative_gap=std::numeric_limits<double>::infinity();
    if (!correct_device(w) || !states || !controls || !parameters || (!device_substeps && substeps<1)) return 1;
    if (w->pending) return 1;
    if (defer && (!consumer || !device_substeps || !spacepdhcg_native_qoco_can_enqueue(w->solver))) return 5;
    report->requested_tolerance=w->tolerance;
    auto stream=static_cast<cudaStream_t>(stream_pointer);
    if (defer) {
        cudaStreamCaptureStatus capture{};
        if (cudaStreamIsCapturing(stream,&capture)!=cudaSuccess || capture!=cudaStreamCaptureStatusNone) return 1;
    }
    const int assembled=device_substeps
        ? spacepdhcg_gtoc12_conic_launch_controlled_device(w->conic,states,controls,parameters,device_substeps,nullptr,stream)
        : spacepdhcg_gtoc12_conic_launch_device(w->conic,states,controls,parameters,substeps,stream);
    if (assembled) { cudaStreamSynchronize(stream); return assembled; }
    const auto* flag=std::getenv("SPACEPDHCG_TEST_GTOC12_DEVICE_ASSEMBLY_VALIDATION");
    const bool guarded=w->solver && flag && flag[0]=='1';
    if (!guarded && !defer) {
        // First setup still consumes host matrices. Later guarded calls combine
        // this producer flag with canonical validation before numerical replay.
        int invalid=0;
        if (cudaMemcpyAsync(&invalid,w->output.invalid,sizeof(int),cudaMemcpyDeviceToHost,stream)!=cudaSuccess) {
            cudaStreamSynchronize(stream); return 2;
        }
        if (cudaStreamSynchronize(stream)!=cudaSuccess) return 2;
        if (invalid) return 3;
    }
    convert_values<<<std::min(1024,(w->count+255)/256),256,0,stream>>>(w->count,w->maps,w->output.a,w->values);
    const auto& d=w->dimensions;
    const int scalar_rows=w->scalar_rows;
    const auto* scalar_rhs=static_cast<const double*>(w->problem.numeric.scalar_upper.data);
    bounds<<<std::min(1024,(std::max(scalar_rows,d.variables)+255)/256),256,0,stream>>>(
        d.equalities,scalar_rows,d.variables,scalar_rhs,w->lower,w->variable_lower,w->variable_upper);
    if (cudaGetLastError()!=cudaSuccess) { cudaStreamSynchronize(stream); return 2; }
    if (!w->solver) {
        // Drive the internal IPM harder than the external audit. The audit
        // and nonlinear physics thresholds remain the caller's unchanged values.
        const double internal_tolerance=std::max(w->tolerance*0.01,std::numeric_limits<double>::min());
        const auto created=spacepdhcg_native_qoco_create_configured(&w->problem,stream,w->ruiz,internal_tolerance,true,&w->solver);
        if (created!=SPACEPDHCG_CUDA_SUCCESS) { cudaStreamSynchronize(stream); return status_code(created); }
    }
    Consumer callback{w,consumer,context,consumed};
    const auto solved=defer
        ? spacepdhcg_native_qoco_enqueue(w->solver,&w->problem,stream,w->primal,w->dual,
            consume_audit,&callback,w->output.invalid)
        : guarded
        ? spacepdhcg_native_qoco_update_solve_with_input_guard(w->solver,&w->problem,stream,
            w->primal,w->dual,&w->native_report,consumer ? consume_audit : nullptr,&callback,w->output.invalid)
        : consumer
        ? spacepdhcg_native_qoco_update_solve_with_consumer(w->solver,&w->problem,stream,
            w->primal,w->dual,&w->native_report,consume_audit,&callback)
        : spacepdhcg_native_qoco_update_solve(w->solver,&w->problem,stream,
            SPACEPDHCG_CUDA_WARM_START_NONE,w->primal,w->dual,&w->native_report);
    if (defer) {
        if (solved!=SPACEPDHCG_CUDA_SUCCESS) { cudaStreamSynchronize(stream); return status_code(solved); }
        w->pending=true; w->pending_stream=stream;
        w->pending_owner=std::this_thread::get_id();
        return 0;
    }
    report_result(w,report);
    if (guarded && std::getenv("SPACEPDHCG_TEST_GTOC12_DEVICE_ASSEMBLY_VALIDATION_TRACE"))
        std::fprintf(stderr,"GTOC12_ASSEMBLY_VALIDATION queued=%d invalid=%d\n",
            w->native_report.producer_validation_queued,w->native_report.producer_invalid);
    if (solved!=SPACEPDHCG_CUDA_SUCCESS) {
        // Failed solves have no fresh audit; never export the prior solve's residuals.
        report->primal_residual=report->dual_residual=std::numeric_limits<double>::infinity();
        report->absolute_primal_residual=report->absolute_dual_residual=std::numeric_limits<double>::infinity();
        cudaStreamSynchronize(stream);
        if (guarded && w->native_report.producer_invalid) {
            report->qoco_status=3; report->iterations=0;
            return 3;
        }
        return status_code(solved);
    }
    if (!*consumed) {
    objective_partial<<<w->objective_blocks,256,0,stream>>>(w->q.indices.size(),d.variables,
        scalar_rows,w->affine_rows,w->q.device_indices,w->quadratic_columns,w->q.values,
        w->output.q,scalar_rhs,static_cast<const double*>(w->problem.numeric.affine_offset.data),
        w->primal,w->dual,w->objective_partial);
    objective_finish<<<1,256,0,stream>>>(w->objective_blocks,w->objective_partial,w->objective_result);
    }
    double objective[4]{};
    if (*consumed && cudaMemcpyAsync(&report->qualified,&w->device_report->qualified,sizeof(int),
        cudaMemcpyDeviceToHost,stream)!=cudaSuccess) { cudaStreamSynchronize(stream); return 2; }
    if (cudaGetLastError()!=cudaSuccess
        || cudaMemcpyAsync(objective,w->objective_result,sizeof(objective),cudaMemcpyDeviceToHost,stream)!=cudaSuccess) {
        cudaStreamSynchronize(stream); return 2;
    }
    if (cudaStreamSynchronize(stream)!=cudaSuccess) return 2;
    report->primal_objective=objective[0]; report->dual_objective=objective[1];
    report->absolute_gap=objective[2]; report->relative_gap=objective[3];
    if (!*consumed) report->qualified=(report->qoco_status==1 || report->qoco_status==2)
        && std::isfinite(report->primal_residual) && std::isfinite(report->dual_residual)
        && report->primal_residual<=w->tolerance && report->dual_residual<=w->tolerance
        && std::isfinite(objective[0]) && std::isfinite(objective[1])
        && std::isfinite(objective[3]) && objective[3]<=w->tolerance;
    return report->qualified ? 0 : 4;
}
extern "C" int spacepdhcg_gtoc12_qoco_can_enqueue(spacepdhcg_gtoc12_qoco* w) {
    return correct_device(w) && !w->pending && spacepdhcg_native_qoco_can_enqueue(w->solver);
}
extern "C" int spacepdhcg_gtoc12_qoco_enqueue_controlled(spacepdhcg_gtoc12_qoco* w,
    const double* states,const double* controls,const spacepdhcg_gtoc12_conic_parameters* parameters,
    const int* substeps,void* stream,spacepdhcg_gtoc12_qoco_consumer consumer,void* context) {
    spacepdhcg_gtoc12_qoco_report unused{}; int consumed=0;
    return solve_device_with_consumer_impl(w,states,controls,parameters,0,stream,&unused,
        consumer,context,&consumed,substeps,true);
}
extern "C" int spacepdhcg_gtoc12_qoco_finish(spacepdhcg_gtoc12_qoco* w,
    void* stream_pointer,spacepdhcg_gtoc12_qoco_report* report) {
    if (!report) return 1;
    *report={}; report->qoco_status=-1;
    report->primal_residual=report->dual_residual=std::numeric_limits<double>::infinity();
    report->absolute_primal_residual=report->absolute_dual_residual=std::numeric_limits<double>::infinity();
    report->primal_objective=report->dual_objective=std::numeric_limits<double>::quiet_NaN();
    report->absolute_gap=report->relative_gap=std::numeric_limits<double>::infinity();
    auto stream=static_cast<cudaStream_t>(stream_pointer);
    if (!correct_device(w) || !w->pending || stream!=w->pending_stream
        || w->pending_owner!=std::this_thread::get_id()) return 1;
    report->requested_tolerance=w->tolerance;
    double objective[4]{}; int qualified=0;
    const auto copied=cudaMemcpyAsync(objective,w->objective_result,sizeof(objective),cudaMemcpyDeviceToHost,stream);
    const auto copied_gate=cudaMemcpyAsync(&qualified,&w->device_report->qualified,sizeof(int),cudaMemcpyDeviceToHost,stream);
    const auto solved=spacepdhcg_native_qoco_finish(w->solver,stream,&w->native_report);
    // Finish drains even on solver failure; a rejected ownership call must not
    // let these stack destinations expire while a transfer is outstanding.
    if (solved==SPACEPDHCG_CUDA_INVALID_STATE || copied!=cudaSuccess || copied_gate!=cudaSuccess) {
        cudaStreamSynchronize(stream);
        if (solved!=SPACEPDHCG_CUDA_INVALID_STATE) w->pending=false;
        return 2;
    }
    w->pending=false;
    report_result(w,report);
    if (solved!=SPACEPDHCG_CUDA_SUCCESS) {
        report->primal_residual=report->dual_residual=std::numeric_limits<double>::infinity();
        report->absolute_primal_residual=report->absolute_dual_residual=std::numeric_limits<double>::infinity();
        return w->native_report.producer_invalid ? 3 : status_code(solved);
    }
    report->qualified=qualified;
    report->primal_objective=objective[0]; report->dual_objective=objective[1];
    report->absolute_gap=objective[2]; report->relative_gap=objective[3];
    return report->qualified ? 0 : 4;
}
extern "C" int spacepdhcg_gtoc12_qoco_solve_device_with_consumer(spacepdhcg_gtoc12_qoco* w,
    const double* states,const double* controls,const spacepdhcg_gtoc12_conic_parameters* parameters,
    int substeps,void* stream,spacepdhcg_gtoc12_qoco_report* report,
    spacepdhcg_gtoc12_qoco_consumer consumer,void* context,int* consumed) {
    return solve_device_with_consumer_impl(w,states,controls,parameters,substeps,stream,report,consumer,context,consumed,nullptr);
}
extern "C" int spacepdhcg_gtoc12_qoco_solve_controlled_device_with_consumer(spacepdhcg_gtoc12_qoco* w,
    const double* states,const double* controls,const spacepdhcg_gtoc12_conic_parameters* parameters,
    const int* substeps,void* stream,spacepdhcg_gtoc12_qoco_report* report,
    spacepdhcg_gtoc12_qoco_consumer consumer,void* context,int* consumed) {
    return solve_device_with_consumer_impl(w,states,controls,parameters,0,stream,report,consumer,context,consumed,substeps);
}
extern "C" int spacepdhcg_gtoc12_qoco_solve_device(spacepdhcg_gtoc12_qoco* w,
    const double* states,const double* controls,const spacepdhcg_gtoc12_conic_parameters* parameters,
    int substeps,void* stream,spacepdhcg_gtoc12_qoco_report* report) {
    int consumed=0;
    return spacepdhcg_gtoc12_qoco_solve_device_with_consumer(w,states,controls,parameters,
        substeps,stream,report,nullptr,nullptr,&consumed);
}
extern "C" int spacepdhcg_gtoc12_qoco_solve_host(spacepdhcg_gtoc12_qoco* w,
    const double* states,const double* controls,const spacepdhcg_gtoc12_conic_parameters* parameters,
    int substeps,double* primal,spacepdhcg_gtoc12_qoco_report* report) {
    if (!report) return 1;
    *report={}; report->qoco_status=-1;
    report->primal_residual=report->dual_residual=std::numeric_limits<double>::infinity();
    report->absolute_primal_residual=report->absolute_dual_residual=std::numeric_limits<double>::infinity();
    report->primal_objective=report->dual_objective=std::numeric_limits<double>::quiet_NaN();
    report->absolute_gap=report->relative_gap=std::numeric_limits<double>::infinity();
    if (!correct_device(w) || w->pending || !states || !controls || !parameters || !primal || substeps<1) return 1;
    const auto failed=[&](int code) { cudaStreamSynchronize(w->stream); return code; };
    const size_t n=w->intervals+1;
    if (!upload(w->states,states,n*7,w->stream) || !upload(w->controls,controls,n*4,w->stream)
        || !upload(w->parameters,parameters,1,w->stream)) return failed(2);
    const int status=spacepdhcg_gtoc12_qoco_solve_device(w,w->states,w->controls,w->parameters,substeps,w->stream,report);
    if (status) return failed(status);
    if (cudaMemcpyAsync(primal,w->primal,w->dimensions.variables*sizeof(double),cudaMemcpyDeviceToHost,w->stream)!=cudaSuccess) return failed(2);
    return cudaStreamSynchronize(w->stream)==cudaSuccess ? 0 : 2;
}
