// Bounded diagnostic correctness/lifecycle tests. No trajectory benchmark.
#include "cuda_test_support.hpp"
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>
namespace t=spacepdhcg::cuda::test;
namespace {
struct Fixture {
    t::ProblemStorage p{false,true};spacepdhcg_cuda_workspace* w{};
    const spacepdhcg_cuda_l1_pair pair{1,0,1,2};
    ~Fixture(){if(w)spacepdhcg_cuda_workspace_destroy(&w);}
    void materialise(double cost=0,bool soc=true) {
        p.variables=3;p.scalar_rows=3;p.affine_rows=soc?3:0;
        // Include off-diagonal structural-zero Q entries touching eliminated t.
        p.h_q_offsets={0,2,4,4};p.h_q_indices={0,1,0,1};p.h_q={0,0,0,0};
        p.h_a_offsets={0,3,5,6};p.h_a_indices={0,1,2,1,2,0};p.h_a={1,1,-1,-1,-1,1};
        p.h_c={cost,4,1};p.h_scalar_lower={1,-INFINITY,-INFINITY};p.h_scalar_upper={1,0,0};
        if(soc){p.h_f_offsets={0,1,1,2};p.h_f_indices={0,1};p.h_f={1,1};p.h_affine_offset={0,0,1};p.affine_cones={{SPACEPDHCG_CUDA_CONE_SECOND_ORDER,0,1,0}};}
        p.h_variable_lower.assign(3,-INFINITY);p.h_variable_upper.assign(3,INFINITY);
    }
    void create(bool enable=true) {
        p.materialise();w=t::create_workspace(p);t::status_require(spacepdhcg_cuda_workspace_wait(w),"create");
        t::status_require(spacepdhcg_cuda_workspace_set_execution_blocks(w,2),"blocks");
        const spacepdhcg_cuda_common_kkt_options k{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,1,1,0,1e-9,1e-8};
        t::status_require(spacepdhcg_cuda_workspace_set_common_kkt_options(w,&k),"common");
        if(enable)on();
    }
    void on(){const spacepdhcg_cuda_l1_options o{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,1,1,0};t::status_require(spacepdhcg_cuda_workspace_set_l1_options(w,&o,&pair),"L1 enable");}
    void off(){const spacepdhcg_cuda_l1_options o{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,0,0,0};t::status_require(spacepdhcg_cuda_workspace_set_l1_options(w,&o,nullptr),"L1 disable");}
    void weight(double omega){const spacepdhcg_cuda_l1_weight_options o{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        omega==0.0?SPACEPDHCG_CUDA_L1_WEIGHT_CANCEL_GLOBAL:SPACEPDHCG_CUDA_L1_WEIGHT_FIXED,omega};
        t::status_require(spacepdhcg_cuda_workspace_set_l1_weight(w,&o),"L1 weight");}
    spacepdhcg_cuda_l1_weight_diagnostics weight_report(){spacepdhcg_cuda_l1_weight_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_l1_weight_diagnostics(w,&d),"weight report");return d;}
    void seed(const std::vector<double>& x,const std::vector<double>& y) {
        p.primal.upload(x,p.stream);p.dual.upload(y,p.stream);
        t::status_require(spacepdhcg_cuda_workspace_warm_start_async(w,SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL,&p.exchange.iterates,p.exchange.consumer_stream),"seed");
        t::status_require(spacepdhcg_cuda_workspace_wait(w),"seed wait");
    }
    spacepdhcg_cuda_l1_diagnostics l1(){spacepdhcg_cuda_l1_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_l1_diagnostics(w,&d),"L1 report");return d;}
    spacepdhcg_cuda_common_kkt_diagnostics kkt(){spacepdhcg_cuda_common_kkt_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_common_kkt_diagnostics(w,&d),"KKT report");return d;}
    std::vector<double> scales() {
        spacepdhcg_cuda_pointer_snapshot ptr{};t::status_require(spacepdhcg_cuda_workspace_pointer_snapshot(w,&ptr),"pointers");
        std::vector<double> s(p.variables+p.scalar_rows+p.affine_rows);
        t::cuda_require(cudaMemcpy(s.data(),reinterpret_cast<void*>(ptr.scaling),s.size()*sizeof(double),cudaMemcpyDeviceToHost),"scale read");return s;
    }
};
void output(const char* name,Fixture& f,const spacepdhcg_cuda_diagnostics& d) {
    const auto l=f.l1();const auto k=f.kkt();std::cout.precision(17);
    std::cout<<"L1_TEST {\"case\":\""<<name<<"\",\"termination\":"<<d.termination<<",\"iterations\":"<<d.iterations
        <<",\"common_valid\":"<<k.valid<<",\"common_passes\":"<<k.passes<<",\"l1_enabled\":"<<l.enabled<<",\"l1_valid\":"<<l.valid<<",\"finite\":"<<l.finite
        <<",\"updates\":"<<l.updates<<",\"completions\":"<<l.completions<<",\"active_variables\":"<<l.active_variables<<",\"active_rows\":"<<l.active_rows;
    auto number=[](const char* key,double v){std::cout<<",\""<<key<<"\":";if(std::isfinite(v))std::cout<<v;else std::cout<<"null";};
    number("eta",l.eta);number("B",l.bound_scale);number("O",l.objective_scale);number("threshold",l.minimum_threshold);number("gap",k.gap_relative);
    const auto weight=f.weight_report();number("omega",weight.omega);number("primal_base_step",weight.primal_base_step);number("dual_base_step",weight.dual_base_step);
    std::cout<<",\"weight_mode\":"<<weight.mode;
    std::cout<<"}\n"<<std::flush;
}
void close(const std::vector<double>& a,const std::vector<double>& b) {
    t::require(a.size()==b.size(),"oracle shape");for(std::size_t i=0;i<a.size();++i)t::require(std::isfinite(a[i])&&std::abs(a[i]-b[i])<2e-12,"CPU proximal oracle differs");
}
void prox(double cost,bool negative_zero,const char* name,double omega=1.0) {
    Fixture f;f.materialise(cost);f.create();if(omega!=1.0)f.weight(omega);
    std::vector<double> initial_x{negative_zero?-0.0:0.0,7,0},initial_y(6,0.0);f.seed(initial_x,initial_y);
    const auto d=t::solve_and_wait(f.w,f.p,t::solve_options(1e-9,2));output(name,f,d);
    const auto scale=f.scales();const auto l=f.l1();const double eta=l.eta;
    const bool cancel_global=omega==0.0;
    if(cancel_global)omega=l.objective_scale/l.bound_scale;
    const double primal_eta=eta/omega,dual_eta=eta*omega;
    const auto weight=f.weight_report();
    t::require(weight.valid&&weight.omega==omega&&weight.primal_base_step==primal_eta&&weight.dual_base_step==dual_eta,"weight metadata");
    t::require(weight.mode==(cancel_global?SPACEPDHCG_CUDA_L1_WEIGHT_CANCEL_GLOBAL:omega==1.0?SPACEPDHCG_CUDA_L1_WEIGHT_UNIT:SPACEPDHCG_CUDA_L1_WEIGHT_FIXED),"weight mode metadata");
    t::require(std::abs(l.minimum_threshold-4*primal_eta*scale[0])<1e-14,"weighted threshold");
    auto x=initial_x,bar=x,y=initial_y;
    for(int step=0;step<2;++step) {
        y[0]+=dual_eta*scale[3]*(bar[0]+bar[2]-1);y[1]=y[2]=0;
        const double sigma=dual_eta*scale[6];double sx=y[3]/sigma+bar[0],sy=y[4]/sigma+bar[2],sr=y[5]/sigma+1;
        const double norm=std::hypot(sx,sy);double px=sx,py=sy,pr=sr;
        if(norm<=-sr)px=py=pr=0;else if(norm>sr){pr=.5*(norm+sr);px=sx*pr/norm;py=sy*pr/norm;}
        y[3]+=sigma*(bar[0]-px);y[4]+=sigma*(bar[2]-py);y[5]+=sigma*(1-pr);
        const double gradient=cost+y[0]+y[3],old_v=x[0],old_w=x[2],tau=primal_eta*scale[0];
        const double arg=old_v-tau*gradient;
        x[0]=std::copysign(std::max(0.0,std::abs(arg)-4*tau),arg);
        x[2]=old_w-primal_eta*scale[2]*(1+y[0]+y[4]);x[1]=std::abs(x[0]);
        const double delta=x[0]>0?4:(x[0]<0?-4:std::clamp(-gradient,-4.0,4.0));
        y[1]=(4+delta)/2;y[2]=(4-delta)/2;bar={2*x[0]-old_v,x[1],2*x[2]-old_w};
    }
    t::require(d.iterations==2&&l.updates==2&&l.completions==2,"cold prox step count");
    close(f.p.primal.download(f.p.stream),x);close(f.p.dual.download(f.p.stream),y);
    t::require(scale[1]==1&&scale[4]==1&&scale[5]==1,"inactive scaling placeholder");
    t::require(l.active_variables==2&&l.active_rows==4&&l.retained_variables==3&&l.retained_rows==6,"logical versus retained dimensions");
    if(cost==0&&omega<=1.0) t::require(x[0]==0.0,"zero prox branch not covered");
    else if(omega==1.0)t::require(cost<0?x[0]>0:x[0]<0,"strict sign branch not covered");
    if(cost==-8&&omega==1.0) {
        // No reseed after mode-off: the next ordinary solve must rebuild
        // original scaling and use the exported point as its history.
        const auto exported_x=f.p.primal.download(f.p.stream),exported_y=f.p.dual.download(f.p.stream);
        const auto reduced=scale;f.off();t::require(!f.l1().valid,"disabled stale diagnostics");
        const auto normal=t::solve_and_wait(f.w,f.p,t::solve_options(1e-9,1));output("disabled_original_scaling_refresh",f,normal);
        t::require(normal.scaling_refreshed&&f.scales()!=reduced,"original scaling was not restored");
        Fixture baseline;baseline.materialise(cost);baseline.create(false);baseline.seed(exported_x,exported_y);
        const auto control=t::solve_and_wait(baseline.w,baseline.p,t::solve_options(1e-9,1));output("fresh_original_control",baseline,control);
        close(f.scales(),baseline.scales());close(f.p.primal.download(f.p.stream),baseline.p.primal.download(baseline.p.stream));
        close(f.p.dual.download(f.p.stream),baseline.p.dual.download(baseline.p.stream));
        f.on();t::require(!f.l1().valid,"re-enable stale diagnostics");
    }
    if(cost==-8&&omega==.25) {
        const auto exported_x=f.p.primal.download(f.p.stream),exported_y=f.p.dual.download(f.p.stream);
        f.weight(1.0);t::require(!f.l1().valid&&!f.weight_report().valid&&!f.kkt().valid,"weight change stale diagnostics");
        const auto switched=t::solve_and_wait(f.w,f.p,t::solve_options(1e-9,1));output("weight_switch_unit",f,switched);
        Fixture fresh;fresh.materialise(cost);fresh.create();fresh.seed(exported_x,exported_y);
        const auto control=t::solve_and_wait(fresh.w,fresh.p,t::solve_options(1e-9,1));output("fresh_unit_l1_control",fresh,control);
        t::require(switched.scaling_refreshed&&switched.iterations==1&&control.iterations==1,"weight refresh/control count");
        close(f.scales(),fresh.scales());close(f.scales(),scale);
        close(f.p.primal.download(f.p.stream),fresh.p.primal.download(fresh.p.stream));
        close(f.p.dual.download(f.p.stream),fresh.p.dual.download(fresh.p.stream));
        f.weight(.25);f.off();f.on();t::require(f.weight_report().omega==1.0&&!f.weight_report().valid,"fresh enable retained old weight");
    }
}
void qualified_seed_and_disable(double omega=1.0) {
    Fixture f;f.materialise(0,false);f.create();if(omega!=1.0)f.weight(omega);
    // min 4t+w, v+w=1. Optimum v=t=0,w=1; y=-1 and pair duals2.5/1.5.
    const std::vector<double> x{1e-20,1e-20,1},y{-1,2.5,1.5};f.seed(x,y);
    const auto d=t::solve_and_wait(f.w,f.p,t::solve_options(1e-9,1));
    output(omega==0?"cancel_global_weak_seed_zero_step":omega==1?"original_weak_seed_zero_step":"weighted_weak_seed_zero_step",f,d);
    t::require(d.iterations==0&&f.kkt().passes&&f.l1().completions==0,"qualified original seed transformed");
    t::require(f.p.primal.download(f.p.stream)==x&&f.p.dual.download(f.p.stream)==y,"supplied point modified");
    t::require(spacepdhcg_cuda_workspace_set_execution_blocks(f.w,0)==SPACEPDHCG_CUDA_UNSUPPORTED,"L1 blocks0");
    const spacepdhcg_cuda_halpern_options h{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,1,{0,0}};
    t::require(spacepdhcg_cuda_workspace_set_halpern_options(f.w,&h)==SPACEPDHCG_CUDA_UNSUPPORTED,"L1 combined with Halpern");
    const spacepdhcg_cuda_common_kkt_options off{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,0,1,0,1e-9,1e-8};
    t::require(spacepdhcg_cuda_workspace_set_common_kkt_options(f.w,&off)==SPACEPDHCG_CUDA_UNSUPPORTED,"common disabled under L1");
    t::require(spacepdhcg_cuda_workspace_checkpoint_async(f.w,{},f.p.exchange.consumer_stream)==SPACEPDHCG_CUDA_UNSUPPORTED
        &&spacepdhcg_cuda_workspace_restore_async(f.w,f.p.fingerprint,{},f.p.exchange.consumer_stream)==SPACEPDHCG_CUDA_UNSUPPORTED,"L1 checkpoint contract");
    f.off();t::require(!f.l1().valid,"disabled stale diagnostics");
    f.on();t::require(!f.l1().valid,"re-enable stale diagnostics");
}
void cancellation(double omega=1.0) {
    Fixture f;f.materialise(0,false);f.create();if(omega!=1.0)f.weight(omega);f.seed({0,0,1},{-1,2.5,1.5});
    std::atomic<bool> release=false;
    t::cuda_require(cudaLaunchHostFunc(f.p.stream,[](void* v){while(!static_cast<std::atomic<bool>*>(v)->load(std::memory_order_acquire))std::this_thread::yield();},&release),"cancel fence");
    const auto options=t::solve_options(1e-9,1);const auto start=spacepdhcg_cuda_workspace_solve_async(f.w,&options,f.p.exchange.consumer_stream);
    const auto cancel=spacepdhcg_cuda_workspace_cancel(f.w);release.store(true,std::memory_order_release);
    t::status_require(start,"start");t::status_require(cancel,"cancel");t::status_require(spacepdhcg_cuda_workspace_wait(f.w),"cancel wait");
    spacepdhcg_cuda_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_diagnostics(f.w,&d),"cancel report");output("cancel_before_initial",f,d);
    t::require(d.iterations==0&&d.termination==SPACEPDHCG_CUDA_TERMINATION_CANCELLED&&!f.kkt().valid,"cancel lost to certificate");
    if(omega==0.0)t::require(!f.weight_report().valid,"cancel-before-initialization advertised a valid weight");
}
void rejection() {
    for(int mutation=0;mutation<3;++mutation) {
        Fixture f;f.materialise(0,false);
        if(mutation==0){f.p.h_q_offsets={0,1,1,1};f.p.h_q_indices={0};f.p.h_q={std::numeric_limits<double>::denorm_min()};}
        if(mutation==1){f.p.h_a_offsets={0,3,6,7};f.p.h_a_indices={0,1,2,0,1,2,0};f.p.h_a={1,1,-1,std::numeric_limits<double>::denorm_min(),-1,-1,1};}
        f.create(false);const spacepdhcg_cuda_l1_options o{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,1,1,0};auto pair=f.pair;
        if(mutation==2)pair.absolute_variable=1;
        const auto status=spacepdhcg_cuda_workspace_set_l1_options(f.w,&o,&pair);
        t::require(status==(mutation==2?SPACEPDHCG_CUDA_INVALID_ARGUMENT:SPACEPDHCG_CUDA_UNSUPPORTED),"invalid L1 map accepted");
    }
    std::cout<<"L1_VALIDATION {\"nonzero_q\":true,\"tiny_epigraph_coupling\":true,\"overlapping_roles\":true}\n";
}
void nonfinite(double omega=1.0) {
    Fixture f;f.materialise(0,false);f.create();if(omega!=1.0)f.weight(omega);f.seed({std::numeric_limits<double>::quiet_NaN(),0,1},{-1,2.5,1.5});
    const auto options=t::solve_options(1e-9,1);
    t::status_require(spacepdhcg_cuda_workspace_solve_async(f.w,&options,f.p.exchange.consumer_stream),"nonfinite launch");
    const auto waited=spacepdhcg_cuda_workspace_wait(f.w);
    t::require(waited==SPACEPDHCG_CUDA_SUCCESS||waited==SPACEPDHCG_CUDA_NUMERICAL_FAILURE,"nonfinite wait");
    spacepdhcg_cuda_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_diagnostics(f.w,&d),"nonfinite report");output("nonfinite_initial",f,d);
    t::require(d.termination==SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE&&d.iterations==0&&!f.kkt().passes,"nonfinite certified");
}
void weight_rejection() {
    Fixture f;f.materialise(0,false);f.create();
    for(double value:{0.0,-1.0,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::denorm_min()}) {
        const spacepdhcg_cuda_l1_weight_options option{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,SPACEPDHCG_CUDA_L1_WEIGHT_FIXED,value};
        t::require(spacepdhcg_cuda_workspace_set_l1_weight(f.w,&option)==SPACEPDHCG_CUDA_INVALID_ARGUMENT,"invalid fixed weight accepted");
        t::require(f.weight_report().omega==1.0,"invalid weight mutated setting");
    }
    for(const auto option:{spacepdhcg_cuda_l1_weight_options{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,99,1},
                         spacepdhcg_cuda_l1_weight_options{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,SPACEPDHCG_CUDA_L1_WEIGHT_CANCEL_GLOBAL,1}})
        t::require(spacepdhcg_cuda_workspace_set_l1_weight(f.w,&option)==SPACEPDHCG_CUDA_INVALID_ARGUMENT,"invalid weight mode accepted");
    f.off();const spacepdhcg_cuda_l1_weight_options option{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,SPACEPDHCG_CUDA_L1_WEIGHT_FIXED,4};
    t::require(spacepdhcg_cuda_workspace_set_l1_weight(f.w,&option)==SPACEPDHCG_CUDA_UNSUPPORTED,"weight set while L1 disabled");
    std::cout<<"L1_WEIGHT_VALIDATION {\"malformed_weights\":true,\"malformed_modes\":true,\"disabled_mode\":true}\n";
}
void effective_step_failure(bool primal) {
    Fixture f;f.materialise(0,false);const double rhs=primal?1024.0:1.0;
    f.p.h_scalar_lower[0]=f.p.h_scalar_upper[0]=rhs;f.create();
    f.weight(primal?std::numeric_limits<double>::min():std::numeric_limits<double>::max());
    const std::vector<double> x{0,0,rhs},y{-1,2.5,1.5};f.seed(x,y);
    const auto options=t::solve_options(1e-9,1);
    t::status_require(spacepdhcg_cuda_workspace_solve_async(f.w,&options,f.p.exchange.consumer_stream),"effective step launch");
    const auto waited=spacepdhcg_cuda_workspace_wait(f.w);
    t::require(waited==SPACEPDHCG_CUDA_SUCCESS||waited==SPACEPDHCG_CUDA_NUMERICAL_FAILURE,"effective step wait");
    spacepdhcg_cuda_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_diagnostics(f.w,&d),"effective step report");
    output(primal?"weighted_primal_step_overflow":"weighted_dual_step_overflow",f,d);
    t::require(d.termination==SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE&&d.iterations==0&&!f.l1().finite,"invalid weighted step accepted");
    t::require(f.p.primal.download(f.p.stream)==x&&f.p.dual.download(f.p.stream)==y,"invalid weighted step mutated seed");
}
}
int main(int argc,char** argv) try {
    if(argc==2&&std::string(argv[1])=="--weight-tests") {
        prox(-8,false,"weighted_quarter_prox",.25);prox(8,false,"weighted_four_prox",4);
        prox(0,false,"weighted_positive_zero",.25);prox(0,true,"weighted_negative_zero",.25);
        prox(-8,false,"cancel_global_prox",0);
        qualified_seed_and_disable(4);qualified_seed_and_disable(0);cancellation(0);nonfinite(4);
        effective_step_failure(true);effective_step_failure(false);weight_rejection();
        std::cout<<"L1_WEIGHT_SUMMARY {\"complete\":true,\"solve_calls\":13,\"maximum_requested_iterations\":18,\"expected_optimization_iterations\":12}\n";return 0;
    }
    t::require(argc==1,"usage: persistent_l1_test [--weight-tests]");
    prox(-8,false,"positive_prox");prox(8,false,"negative_prox");prox(0,false,"positive_zero_prox");prox(0,true,"negative_zero_prox");
    qualified_seed_and_disable();cancellation();nonfinite();rejection();
    std::cout<<"L1_SUMMARY {\"complete\":true,\"solve_calls\":9,\"maximum_requested_iterations\":13,\"expected_optimization_iterations\":10}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
