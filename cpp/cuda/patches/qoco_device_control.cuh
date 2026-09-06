// SPDX-License-Identifier: Apache-2.0
// Included after prepared metric/step/RHS implementations. Scalars and the best
// iterate decision belong to the solver, not the shared reduction scratch.
namespace qoco_device_control {
struct State {
    double alpha, dynamic_reg;
    int stop, status;
    double metrics[8];
    double best_pres, best_dres, best_gap, best_obj, best_metric;
    int best_iter, best_valid, save, restored;
};
__device__ double threshold(double a,double r,double scale) { return __dadd_rn(a,__dmul_rn(r,scale)); }
__global__ void initialize(State* s,double reg) {
    *s={}; s->alpha=1.0; s->dynamic_reg=reg; s->best_iter=-1; s->status=QOCO_UNSOLVED;
}
__global__ void decide(State* s,double absolute,double relative,double inaccurate_absolute,
    double inaccurate_relative,int iteration) {
    const double* v=s->metrics;
    const double pres=v[0],dres=v[1],gap=v[2];
    const double pt=threshold(inaccurate_absolute,inaccurate_relative,v[3]);
    const double dt=threshold(inaccurate_absolute,inaccurate_relative,v[4]);
    const double gt=threshold(inaccurate_absolute,inaccurate_relative,v[5]);
    const double metric=qoco_max(qoco_max(pres/pt,dres/dt),gap/gt);
    s->save=0; s->stop=0; s->restored=0;
    if (isfinite(metric) && (!s->best_valid || metric<s->best_metric)) {
        s->best_pres=pres; s->best_dres=dres; s->best_gap=gap; s->best_obj=v[6];
        s->best_metric=metric; s->best_iter=iteration; s->best_valid=1; s->save=1;
    }
    if (s->alpha<1e-8) {
        s->dynamic_reg*=10.0;
        if (s->dynamic_reg>1e-6) {
            s->status=pres<pt && dres<dt && gap<gt ? QOCO_SOLVED_INACCURATE : QOCO_NUMERICAL_ERROR;
            s->stop=1;
        }
        return;
    }
    if (pres<threshold(absolute,relative,v[3]) && dres<threshold(absolute,relative,v[4])
        && gap<threshold(absolute,relative,v[5])) {
        s->status=QOCO_SOLVED; s->stop=1;
    }
}
__global__ void restore_decision(State* s,int status) {
    s->status=status; s->restored=s->best_valid;
    if (!s->best_valid) return;
    s->metrics[0]=s->best_pres; s->metrics[1]=s->best_dres; s->metrics[2]=s->best_gap;
    s->metrics[6]=s->best_obj;
    if (s->best_metric<=1.0 && (status==QOCO_NUMERICAL_ERROR || status==QOCO_MAX_ITER))
        s->status=QOCO_SOLVED_INACCURATE;
}
template<bool Restore>
__global__ void best_vectors(const State* state,int n,int p,int m,
    double* x,double* y,double* s,double* z,double* bx,double* by,double* bs,double* bz) {
    if (!(Restore ? state->restored : state->save)) return;
    const long long total=n+static_cast<long long>(p)+2LL*m;
    for (long long i=blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=blockDim.x*gridDim.x) {
        double *live,*best; int j;
        if (i<n) {live=x;best=bx;j=i;}
        else if (i<n+static_cast<long long>(p)) {live=y;best=by;j=i-n;}
        else if (i<n+static_cast<long long>(p)+m) {live=s;best=bs;j=i-n-p;}
        else {live=z;best=bz;j=i-n-p-m;}
        if constexpr(Restore) live[j]=best[j]; else best[j]=live[j];
    }
}
#ifndef SPACEPDHCG_CONTROL_KERNEL_TEST
template<bool Restore> void copy(QOCOWorkspace* w) {
    auto* d=w->data; const long long count=d->n+static_cast<long long>(d->p)+2LL*d->m;
    if (!count) return;
    best_vectors<Restore><<<static_cast<int>(std::min(256LL,(count+255)/256)),256>>>(
        static_cast<State*>(w->gpu_control),d->n,d->p,d->m,w->x->d_data,w->y->d_data,w->s->d_data,w->z->d_data,
        w->best_x->d_data,w->best_y->d_data,w->best_s->d_data,w->best_z->d_data);
    CUDA_CHECK(cudaGetLastError());
}
State metadata(QOCOSolver* solver) {
    State h{}; auto* w=solver->work;
    // Transitional report packet. Its values are only copied into legacy host
    // metadata; all comparisons, thresholds and regularization arithmetic above
    // already ran on device. Native host dispatch still consumes stop/status/reg.
    CUDA_CHECK(cudaMemcpy(&h,w->gpu_control,sizeof(h),cudaMemcpyDeviceToHost));
    solver->sol->pres=h.metrics[0]; solver->sol->dres=h.metrics[1]; solver->sol->gap=h.metrics[2];
    solver->sol->obj=h.metrics[6]; w->mu=h.metrics[7]; w->a=h.alpha;
    solver->settings->kkt_dynamic_reg=h.dynamic_reg; solver->sol->status=h.status;
    w->best_pres=h.best_pres; w->best_dres=h.best_dres; w->best_gap=h.best_gap; w->best_obj=h.best_obj;
    w->best_metric=h.best_metric; w->best_iter=h.best_iter; w->best_valid=h.best_valid;
    return h;
}
int dispatch(QOCOSolver* solver) {
    struct Command { double dynamic_reg; int stop,status; };
    static_assert(offsetof(State,stop)==offsetof(State,dynamic_reg)+sizeof(double));
    static_assert(offsetof(State,status)==offsetof(State,stop)+sizeof(int));
    static_assert(sizeof(Command)==sizeof(double)+2*sizeof(int));
    auto* state=static_cast<State*>(solver->work->gpu_control);
    Command h{};
    CUDA_CHECK(cudaMemcpy(&h,&state->dynamic_reg,sizeof(h),cudaMemcpyDeviceToHost));
    solver->settings->kkt_dynamic_reg=h.dynamic_reg;
    solver->sol->status=h.status;
    // Production numerical consumers use device scalars. Full legacy telemetry
    // is materialized at completion, verbosity, or an explicit arithmetic audit.
    if (h.stop || solver->settings->verbose
        || getenv("SPACEPDHCG_TEST_QOCO_DEVICE_CONTROL_COMPARE")
        || getenv("SPACEPDHCG_TEST_QOCO_COMBINED_RHS_COMPARE")
        || getenv("SPACEPDHCG_TEST_QOCO_BATCHED_STOPPING_COMPARE")) metadata(solver);
    return h.stop;
}
#endif
}
#ifndef SPACEPDHCG_CONTROL_KERNEL_TEST
extern "C" void qoco_gpu_reset_control(QOCOSolver* solver) {
    const char* disable=getenv("SPACEPDHCG_TEST_QOCO_DEVICE_CONTROL_DISABLE");
    auto* w=solver->work;
    if (disable && disable[0]=='1') {
        if (w->gpu_control) { CUDA_CHECK(cudaFree(w->gpu_control)); w->gpu_control=nullptr; }
        return;
    }
    if (!w->gpu_control) CUDA_CHECK(cudaMalloc(&w->gpu_control,sizeof(qoco_device_control::State)));
    qoco_device_control::initialize<<<1,1>>>(static_cast<qoco_device_control::State*>(w->gpu_control),
        solver->settings->kkt_dynamic_reg);
    CUDA_CHECK(cudaGetLastError());
}
extern "C" void qoco_gpu_free_control(QOCOWorkspace* w) {
    CUDA_CHECK(cudaFree(w->gpu_control)); w->gpu_control=nullptr;
}
extern "C" void qoco_gpu_note_alpha(QOCOWorkspace* w,const double* alpha) {
    if (!w->gpu_control) return;
    auto* s=static_cast<qoco_device_control::State*>(w->gpu_control);
    if (alpha) CUDA_CHECK(cudaMemcpyAsync(&s->alpha,alpha,sizeof(double),cudaMemcpyDeviceToDevice));
    else CUDA_CHECK(cudaMemsetAsync(&s->alpha,0,sizeof(double)));
}
extern "C" void qoco_gpu_sync_control(QOCOSolver* solver) { qoco_device_control::metadata(solver); }
extern "C" unsigned char qoco_gpu_check_stopping(QOCOSolver* solver) {
    using namespace qoco_device_control;
    auto* state=static_cast<State*>(solver->work->gpu_control);
    qoco_gpu_metrics<true>(solver,state->metrics,true);
    auto* p=solver->settings;
    decide<<<1,1>>>(state,p->abstol,p->reltol,p->abstol_inacc,p->reltol_inacc,solver->sol->iters);
    copy<false>(solver->work);
    return static_cast<unsigned char>(dispatch(solver));
}
extern "C" unsigned char qoco_gpu_restore_best(QOCOSolver* solver) {
    using namespace qoco_device_control;
    restore_decision<<<1,1>>>(static_cast<State*>(solver->work->gpu_control),solver->sol->status);
    copy<true>(solver->work);
    return static_cast<unsigned char>(metadata(solver).restored);
}
#endif
