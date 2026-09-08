// Bounded tests of the opt-in algorithm, not a trajectory backend benchmark.
#include "cuda_test_support.hpp"
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>
namespace t=spacepdhcg::cuda::test;
namespace {
struct Fixture {
    t::ProblemStorage p{false,true};
    spacepdhcg_cuda_workspace* w{};
    int mode=1;
    ~Fixture(){if(w)spacepdhcg_cuda_workspace_destroy(&w);}
    void create(int equalities,int selected_mode) {
        mode=selected_mode;p.materialise();w=t::create_workspace(p);
        t::status_require(spacepdhcg_cuda_workspace_wait(w),"creation");
        t::status_require(spacepdhcg_cuda_workspace_set_execution_blocks(w,2),"blocks");
        const spacepdhcg_cuda_common_kkt_options common{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,1,equalities,0,1e-9,1e-8};
        t::status_require(spacepdhcg_cuda_workspace_set_common_kkt_options(w,&common),"common");
        auto h=options(mode);t::status_require(spacepdhcg_cuda_workspace_set_halpern_options(w,&h),"Halpern");
    }
    spacepdhcg_cuda_halpern_options options(int selected)const{return{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,selected,{0,0}};}
    void seed(const std::vector<double>& x,const std::vector<double>& y) {
        p.primal.upload(x,p.stream);p.dual.upload(y,p.stream);
        t::status_require(spacepdhcg_cuda_workspace_warm_start_async(w,SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL,&p.exchange.iterates,p.exchange.consumer_stream),"seed");
        t::status_require(spacepdhcg_cuda_workspace_wait(w),"seed wait");
    }
    spacepdhcg_cuda_halpern_diagnostics h() {
        spacepdhcg_cuda_halpern_diagnostics v{};t::status_require(spacepdhcg_cuda_workspace_halpern_diagnostics(w,&v),"Halpern report");return v;
    }
    spacepdhcg_cuda_common_kkt_diagnostics kkt() {
        spacepdhcg_cuda_common_kkt_diagnostics v{};t::status_require(spacepdhcg_cuda_workspace_common_kkt_diagnostics(w,&v),"KKT report");return v;
    }
    std::vector<double> copy(std::uintptr_t pointer,std::size_t count) {
        std::vector<double> v(count);t::cuda_require(cudaMemcpy(v.data(),reinterpret_cast<const void*>(pointer),count*sizeof(double),cudaMemcpyDeviceToHost),"internal vector");return v;
    }
    std::vector<double> scaling() {
        spacepdhcg_cuda_pointer_snapshot v{};t::status_require(spacepdhcg_cuda_workspace_pointer_snapshot(w,&v),"pointers");
        return copy(v.scaling,p.variables+p.scalar_rows+p.affine_rows);
    }
};
void scalar(Fixture& f,bool slow=false) {
    auto& p=f.p;p.variables=slow?2:1;p.scalar_rows=slow?2:1;p.affine_rows=0;
    p.h_q_offsets.assign(p.variables+1,0);p.h_f_offsets={};
    p.h_a_offsets=slow?std::vector<int>{0,2,4}:std::vector<int>{0,1};
    p.h_a_indices=slow?std::vector<int>{0,1,0,1}:std::vector<int>{0};
    p.h_a=slow?std::vector<double>{1,1,1,1+1e-6}:std::vector<double>{1};
    p.h_c=slow?std::vector<double>{0,0}:std::vector<double>{2};
    p.h_scalar_lower=p.h_scalar_upper=slow?std::vector<double>{1,2}:std::vector<double>{3};
    p.h_variable_lower.assign(p.variables,-INFINITY);p.h_variable_upper.assign(p.variables,INFINITY);
}
void mixed(Fixture& f) {
    auto& p=f.p;p.variables=2;p.scalar_rows=2;p.affine_rows=3;
    p.h_q_offsets={0,0,0};p.h_a_offsets={0,1,2};p.h_a_indices={1,0};p.h_a={1,1};
    p.h_f_offsets={0,1,2};p.h_f_indices={0,1};p.h_f={1,1};p.h_c={-2,-3};
    p.h_scalar_lower={0,-INFINITY};p.h_scalar_upper={0,1};p.h_affine_offset={0,0,1};
    p.h_variable_lower={-INFINITY,-INFINITY};p.h_variable_upper={INFINITY,INFINITY};
    p.affine_cones={{SPACEPDHCG_CUDA_CONE_SECOND_ORDER,0,1,0}};
}
void output(const char* name,Fixture& f,const spacepdhcg_cuda_diagnostics& d) {
    const auto h=f.h();const auto k=f.kkt();std::cout.precision(17);
    std::cout<<"HALPERN_TEST {\"case\":\""<<name<<"\",\"mode\":"<<f.mode<<",\"termination\":"<<d.termination<<",\"iterations\":"<<d.iterations
        <<",\"common_valid\":"<<k.valid<<",\"common_passes\":"<<k.passes<<",\"halpern_valid\":"<<h.valid<<",\"finite\":"<<h.finite
        <<",\"updates\":"<<h.updates<<",\"restarts\":"<<h.restarts<<",\"inner\":"<<h.inner_iterations<<",\"epoch_reference\":"<<h.epoch_reference_iteration
        <<",\"weight_updates\":"<<h.weight_updates<<",\"weight_fallbacks\":"<<h.weight_fallbacks;
    auto number=[](const char* key,double value){std::cout<<",\""<<key<<"\":";if(std::isfinite(value))std::cout<<value;else std::cout<<"null";};
    number("weight",h.primal_weight);number("fixed_point_error",h.fixed_point_error);number("gap",k.gap_relative);
    std::cout<<"}\n"<<std::flush;
}
void exact_and_transition(int mode) {
    Fixture f;mixed(f);f.create(1,mode);f.seed({1,0},{3,1,1,0,-1});
    const auto d=t::solve_and_wait(f.w,f.p,t::solve_options(1e-9,1));output("mixed_exact_zero_step",f,d);
    t::require(d.iterations==0 && d.termination==SPACEPDHCG_CUDA_TERMINATION_OPTIMAL && f.kkt().passes,"qualified seed advanced");
    t::require(f.p.primal.download(f.p.stream)==std::vector<double>({1,0}) && f.p.dual.download(f.p.stream)==std::vector<double>({3,1,1,0,-1}),"seed changed");
    t::require(spacepdhcg_cuda_workspace_set_execution_blocks(f.w,0)==SPACEPDHCG_CUDA_UNSUPPORTED,"enabled blocks0 accepted");
    const spacepdhcg_cuda_common_kkt_options off{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,0,1,0,1e-9,1e-8};
    t::require(spacepdhcg_cuda_workspace_set_common_kkt_options(f.w,&off)==SPACEPDHCG_CUDA_UNSUPPORTED,"common disabled under Halpern");
    t::require(spacepdhcg_cuda_workspace_checkpoint_async(f.w,{},f.p.exchange.consumer_stream)==SPACEPDHCG_CUDA_UNSUPPORTED
        && spacepdhcg_cuda_workspace_restore_async(f.w,f.p.fingerprint,{},f.p.exchange.consumer_stream)==SPACEPDHCG_CUDA_UNSUPPORTED,"checkpoint contract changed");
    auto invalid=f.options(3);t::require(spacepdhcg_cuda_workspace_set_halpern_options(f.w,&invalid)==SPACEPDHCG_CUDA_INVALID_ARGUMENT,"invalid mode");
}
void two_step_and_disable(int mode) {
    Fixture f;scalar(f);f.create(1,mode);
    const auto d=t::solve_and_wait(f.w,f.p,t::solve_options(1e-9,2));output("two_step_proximal_output",f,d);
    const auto scale=f.scaling();const double tau=f.h().eta*scale[0],sigma=f.h().eta*scale[1];
    const double x1=-tau*2,y1=sigma*(2*x1-3);
    const double x2=x1-tau*(2+y1),y2=y1+sigma*(2*x2-x1-3);
    const auto x=f.p.primal.download(f.p.stream),y=f.p.dual.download(f.p.stream);
    t::require(d.iterations==2 && std::abs(x[0]-x2)<1e-12 && std::abs(y[0]-y2)<1e-12,"returned anchored state or wrong primal-first map");
    t::require(std::abs(x[0]-(2*x2-x1)*2/3)>1e-3,"fixture fails to distinguish T from working");
    auto off=f.options(0);t::status_require(spacepdhcg_cuda_workspace_set_halpern_options(f.w,&off),"disable Halpern");
    t::require(!f.h().valid && f.h().mode==0,"disabled report stale");
    const auto normal=t::solve_and_wait(f.w,f.p,t::solve_options(1e-9,1));
    const double expected_y=y2+sigma*(x2-3),expected_x=x2-tau*(2+expected_y);
    t::require(normal.iterations==1 && std::abs(f.p.primal.download(f.p.stream)[0]-expected_x)<1e-12
        && std::abs(f.p.dual.download(f.p.stream)[0]-expected_y)<1e-12,"default transition used stale Halpern history");
    output("disabled_default_transition",f,normal);
    const auto on=f.options(mode);t::status_require(spacepdhcg_cuda_workspace_set_halpern_options(f.w,&on),"re-enable");
    t::require(!f.h().valid,"re-enable report stale");
}
void mixed_prox(int mode) {
    Fixture f;mixed(f);f.create(1,mode);
    const auto d=t::solve_and_wait(f.w,f.p,t::solve_options(1e-9,1));output("mixed_cold_proximal_step",f,d);
    const auto scale=f.scaling();const double eta=f.h().eta;
    const std::vector<double> expected_x{2*eta*scale[0],3*eta*scale[1]};
    const double scalar_y=eta*scale[2]*(2*expected_x[1]);
    const double scalar_z=std::max(0.0,eta*scale[3]*(2*expected_x[0]-1));
    // Native radius-last SOC: prox of the negative-cone indicator is obtained
    // via independent 3-vector Moreau projection in ordinary CPU doubles.
    const double sx=2*expected_x[0],sy=2*expected_x[1],radius=1;
    const double norm=std::hypot(sx,sy);
    double px=sx,py=sy,pr=radius;
    if(norm<=-radius)px=py=pr=0;
    else if(norm>radius){pr=(norm+radius)/2;px=sx*pr/norm;py=sy*pr/norm;}
    const std::vector<double> expected_y{scalar_y,scalar_z,eta*scale[4]*(sx-px),eta*scale[5]*(sy-py),eta*scale[6]*(radius-pr)};
    const auto x=f.p.primal.download(f.p.stream),y=f.p.dual.download(f.p.stream);
    t::require(d.iterations==1,"mixed cold skipped update");
    for(std::size_t j=0;j<x.size();++j)t::require(std::abs(x[j]-expected_x[j])<1e-12,"mixed primal prox mismatch");
    for(std::size_t i=0;i<y.size();++i)t::require(std::abs(y[i]-expected_y[i])<1e-12,"mixed scalar/SOC conjugate prox mismatch");
    t::require(y[1]>=0 && std::hypot(y[2],y[3])+y[4]<1e-12,"mixed returned dual outside its cones");
}
void restart_epoch(int mode) {
    Fixture f;scalar(f,true);f.create(2,mode);
    const auto d=t::solve_and_wait(f.w,f.p,t::solve_options(1e-9,201));output("first_restart_epoch",f,d);
    const auto h=f.h();t::require(d.iterations==201 && h.updates==201 && h.finite,"bounded restart fixture failed");
    if(mode==2)t::require(h.restarts==1 && h.last_restart_iteration==200 && h.epoch_reference_iteration==201
        && h.inner_iterations==1 && h.metric_evaluations==2,"restart used old metric or wrong counter");
    else t::require(h.restarts==0 && h.inner_iterations==201 && h.primal_weight==1 && h.metric_evaluations==1,"plain Halpern adapted");
}
void cancellation(int mode) {
    Fixture f;scalar(f);f.create(1,mode);f.seed({3},{-2});
    std::atomic<bool> release=false;
    t::cuda_require(cudaLaunchHostFunc(f.p.stream,[](void* v){while(!static_cast<std::atomic<bool>*>(v)->load(std::memory_order_acquire))std::this_thread::yield();},&release),"cancel fence");
    const auto options=t::solve_options(1e-9,1);
    const auto started=spacepdhcg_cuda_workspace_solve_async(f.w,&options,f.p.exchange.consumer_stream);
    const auto cancelled=spacepdhcg_cuda_workspace_cancel(f.w);release.store(true,std::memory_order_release);
    t::status_require(started,"launch");t::status_require(cancelled,"cancel");t::status_require(spacepdhcg_cuda_workspace_wait(f.w),"cancel wait");
    spacepdhcg_cuda_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_diagnostics(f.w,&d),"cancel report");output("cancel_before_initial",f,d);
    t::require(d.termination==SPACEPDHCG_CUDA_TERMINATION_CANCELLED && d.iterations==0 && !f.kkt().valid,"cancellation lost to qualification");
}
void nonfinite(int mode) {
    Fixture f;scalar(f);f.create(1,mode);f.seed({std::numeric_limits<double>::quiet_NaN()},{-2});
    const auto options=t::solve_options(1e-9,1);
    t::status_require(spacepdhcg_cuda_workspace_solve_async(f.w,&options,f.p.exchange.consumer_stream),"nonfinite launch");
    const auto waited=spacepdhcg_cuda_workspace_wait(f.w);
    t::require(waited==SPACEPDHCG_CUDA_SUCCESS || waited==SPACEPDHCG_CUDA_NUMERICAL_FAILURE,"nonfinite wait");
    spacepdhcg_cuda_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_diagnostics(f.w,&d),"nonfinite report");output("nonfinite_initial",f,d);
    t::require(d.termination==SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE && d.iterations==0 && !f.kkt().passes,"nonfinite certified");
}
void nonzero_q_rejected() {
    Fixture f;scalar(f);f.p.h_q_offsets={0,1};f.p.h_q_indices={0};f.p.h_q={std::numeric_limits<double>::denorm_min()};
    f.p.materialise();f.w=t::create_workspace(f.p);t::status_require(spacepdhcg_cuda_workspace_wait(f.w),"reject fixture create");
    const auto options=f.options(1);
    t::require(spacepdhcg_cuda_workspace_set_halpern_options(f.w,&options)==SPACEPDHCG_CUDA_UNSUPPORTED,"missing common policy accepted");
    t::status_require(spacepdhcg_cuda_workspace_set_execution_blocks(f.w,2),"reject fixture blocks");
    const spacepdhcg_cuda_common_kkt_options common{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,1,1,0,1e-9,1e-8};
    t::status_require(spacepdhcg_cuda_workspace_set_common_kkt_options(f.w,&common),"reject fixture common");
    t::require(spacepdhcg_cuda_workspace_set_halpern_options(f.w,&options)==SPACEPDHCG_CUDA_UNSUPPORTED,"tiny nonzero Q treated as zero");
    std::cout<<"HALPERN_VALIDATION {\"nonzero_q_rejected\":true}\n";
}
}
int main(int argc,char** argv) try {
    t::require(argc==2,"usage: persistent_halpern_test plain|adaptive");
    const std::string name=argv[1];t::require(name=="plain"||name=="adaptive","invalid test mode");const int mode=name=="plain"?1:2;
    exact_and_transition(mode);two_step_and_disable(mode);mixed_prox(mode);restart_epoch(mode);cancellation(mode);nonfinite(mode);nonzero_q_rejected();
    std::cout<<"HALPERN_SUMMARY {\"complete\":true,\"mode\":"<<mode<<",\"solve_calls\":7,\"maximum_requested_iterations\":208,\"expected_optimization_iterations\":205}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
