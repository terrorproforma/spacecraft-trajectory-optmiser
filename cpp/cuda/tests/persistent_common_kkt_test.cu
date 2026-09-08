// Bounded diagnostic-policy tests. No numerical defaults are changed.
#include "cuda_test_support.hpp"
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <thread>

namespace t=spacepdhcg::cuda::test;
namespace {
struct Fixture {
    t::ProblemStorage p{false,true};
    spacepdhcg_cuda_workspace* w{};
    ~Fixture(){if(w)spacepdhcg_cuda_workspace_destroy(&w);}
    void create(int blocks) {
        p.materialise();w=t::create_workspace(p);
        t::status_require(spacepdhcg_cuda_workspace_wait(w),"create wait");
        t::status_require(spacepdhcg_cuda_workspace_set_execution_blocks(w,blocks),"execution blocks");
    }
    spacepdhcg_cuda_common_kkt_options policy(int equalities,bool enabled=true) {
        return {SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,int(enabled),equalities,0,1e-9,1e-8};
    }
    void enable(int equalities) {
        const auto option=policy(equalities);
        t::status_require(spacepdhcg_cuda_workspace_set_common_kkt_options(w,&option),"common policy");
    }
    void seed(const std::vector<double>& x,const std::vector<double>& dual) {
        p.primal.upload(x,p.stream);p.dual.upload(dual,p.stream);
        t::status_require(spacepdhcg_cuda_workspace_warm_start_async(w,SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL,&p.exchange.iterates,p.exchange.consumer_stream),"seed");
        t::status_require(spacepdhcg_cuda_workspace_wait(w),"seed wait");
    }
    spacepdhcg_cuda_common_kkt_diagnostics common() {
        spacepdhcg_cuda_common_kkt_diagnostics result{};
        t::status_require(spacepdhcg_cuda_workspace_common_kkt_diagnostics(w,&result),"common diagnostics");return result;
    }
};
void mixed(Fixture& f) {
    auto& p=f.p;p.variables=2;p.scalar_rows=2;p.affine_rows=3;
    p.h_q_offsets={0,2,4};p.h_q_indices={0,1,0,1};p.h_q={2,1,1,2};
    p.h_a_offsets={0,1,2};p.h_a_indices={1,0};p.h_a={1,1};
    p.h_f_offsets={0,1,2};p.h_f_indices={0,1};p.h_f={1,1};p.h_c={-4,-4};
    p.h_scalar_lower={0,-INFINITY};p.h_scalar_upper={0,1};p.h_affine_offset={0,0,1};
    p.h_variable_lower={-INFINITY,-INFINITY};p.h_variable_upper={INFINITY,INFINITY};
    p.affine_cones={{SPACEPDHCG_CUDA_CONE_SECOND_ORDER,0,1,0}};
}
void one_variable(Fixture& f,int scalar,int affine) {
    auto& p=f.p;p.variables=1;p.scalar_rows=scalar;p.affine_rows=affine;
    p.h_q_offsets={0,0};p.h_a_offsets={0,0};p.h_f_offsets=affine?std::vector<int>{0,0}:std::vector<int>{};
    p.h_c={0};p.h_scalar_lower.assign(scalar,-INFINITY);p.h_scalar_upper.assign(scalar,0);
    p.h_affine_offset.assign(affine,0);p.h_variable_lower={-INFINITY};p.h_variable_upper={INFINITY};
    if(affine)p.affine_cones={{SPACEPDHCG_CUDA_CONE_SECOND_ORDER,0,1,0}};
}
void output(const char* name,const spacepdhcg_cuda_diagnostics& d,const spacepdhcg_cuda_common_kkt_diagnostics& c) {
    std::cout.precision(17);
    std::cout<<"COMMON_KKT_TEST {\"case\":\""<<name<<"\",\"termination\":"<<d.termination<<",\"iterations\":"<<d.iterations
        <<",\"valid\":"<<c.valid<<",\"finite\":"<<c.finite<<",\"passes\":"<<c.passes<<",\"evaluations\":"<<c.evaluations
        <<",\"cycles\":"<<c.evaluation_clock_cycles;
    auto number=[](const char* key,double value){std::cout<<",\""<<key<<"\":";if(std::isfinite(value))std::cout<<value;else std::cout<<"null";};
    number("dual",c.dual_relative);number("gap",c.gap_relative);number("block",c.block_complementarity_relative);
    std::cout<<"}\n"<<std::flush;
}
void exact_and_lifecycle(int blocks) {
    Fixture f;mixed(f);f.create(blocks);f.enable(1);
    t::require(!f.common().valid,"fresh policy has stale certificate");
    f.seed({1,0},{3,1,1,0,-1});
    auto d=t::solve_and_wait(f.w,f.p,t::solve_options(1e-9,1));auto c=f.common();
    output("mixed_exact_zero_step",d,c);
    t::require(d.termination==SPACEPDHCG_CUDA_TERMINATION_OPTIMAL && d.iterations==0 && c.passes && c.evaluations==1,"exact mixed start was advanced or rejected");
    t::require(f.p.primal.download(f.p.stream)==std::vector<double>({1,0}) && f.p.dual.download(f.p.stream)==std::vector<double>({3,1,1,0,-1}),"zero-step check changed iterates");
    t::status_require(spacepdhcg_cuda_workspace_reset_async(f.w,SPACEPDHCG_CUDA_RESET_FULL,f.p.exchange.consumer_stream),"reset");
    t::status_require(spacepdhcg_cuda_workspace_wait(f.w),"reset wait");
    t::require(!f.common().valid,"reset retained a valid common certificate");
    f.seed({1,0},{3,1,1,0,-1});t::require(!f.common().valid,"reseed retained certificate");
    const auto blocked=spacepdhcg_cuda_workspace_update_async(f.w,f.p.fingerprint,&f.p.exchange.numeric,f.p.exchange.consumer_stream);
    t::require(blocked==SPACEPDHCG_CUDA_UNSUPPORTED,"enabled policy allowed unchecked numeric update");
    auto invalid=f.policy(1);invalid.relative_tolerance=1e-8;
    t::require(spacepdhcg_cuda_workspace_set_common_kkt_options(f.w,&invalid)==SPACEPDHCG_CUDA_INVALID_ARGUMENT,"policy silently changed gate");
    invalid=f.policy(1);invalid.reserved=1;
    t::require(spacepdhcg_cuda_workspace_set_common_kkt_options(f.w,&invalid)==SPACEPDHCG_CUDA_INVALID_ARGUMENT,"reserved ABI field accepted");
    const auto off=f.policy(1,false);t::status_require(spacepdhcg_cuda_workspace_set_common_kkt_options(f.w,&off),"disable");
    t::require(!f.common().valid && !f.common().enabled,"disabled policy exposed certificate");
    d=t::solve_and_wait(f.w,f.p,t::solve_options(1e-9,1));
    t::require(d.iterations==1,"legacy policy unexpectedly performed new zero-step check");output("mixed_legacy_unchanged",d,f.common());
    f.p.h_variable_lower[0]=0;f.p.upload_numeric();
    t::status_require(spacepdhcg_cuda_workspace_update_async(f.w,f.p.fingerprint,&f.p.exchange.numeric,f.p.exchange.consumer_stream),"disabled policy numeric update");
    t::status_require(spacepdhcg_cuda_workspace_wait(f.w),"numeric update wait");
    const auto on=f.policy(1);
    t::require(spacepdhcg_cuda_workspace_set_common_kkt_options(f.w,&on)==SPACEPDHCG_CUDA_UNSUPPORTED,"domain validation reused a stale bound classification");
    f.p.h_variable_lower[0]=-INFINITY;f.p.upload_numeric();
    t::status_require(spacepdhcg_cuda_workspace_update_async(f.w,f.p.fingerprint,&f.p.exchange.numeric,f.p.exchange.consumer_stream),"restore free domain");
    t::status_require(spacepdhcg_cuda_workspace_wait(f.w),"restore wait");f.enable(1);
    t::require(f.common().enabled && !f.common().valid,"reenabled policy exposed stale result");
}
void grouping(int blocks,bool combined) {
    const double m=std::ldexp(1.0,40);Fixture f;one_variable(f,combined?1:2,combined?3:0);
    if(combined) {
        f.p.h_a_offsets={0,1};f.p.h_a_indices={0};f.p.h_a={1};
        f.p.h_f_offsets={0,1};f.p.h_f_indices={2};f.p.h_f={1};f.p.h_c={1};
    } else {
        f.p.h_a_offsets={0,2};f.p.h_a_indices={0,1};f.p.h_a={1,-1};f.p.h_scalar_lower[0]=0;
    }
    f.create(blocks);f.enable(combined?0:1);f.seed({0},combined?std::vector<double>{m,0,0,-m}:std::vector<double>{m+1,m});
    const auto d=t::solve_and_wait(f.w,f.p,t::solve_options(1e-9,1));const auto c=f.common();
    output(combined?"combined_G_denominator":"separate_equality_denominator",d,c);
    if(combined)t::require(d.iterations==1 && c.evaluations==2,"split G normalization falsely certified initial point");
    else t::require(d.iterations==0 && c.passes && c.dual_relative<1e-9,"equality contribution was merged into G denominator");
}
void block_cancellation(int blocks) {
    Fixture f;one_variable(f,2,0);f.p.h_scalar_upper={-1e-8,1};f.create(blocks);f.enable(0);f.seed({0},{1e12,1e4});
    const auto d=t::solve_and_wait(f.w,f.p,t::solve_options(1e-9,1));const auto c=f.common();output("block_complementarity_cancellation",d,c);
    t::require(d.iterations==1 && !c.passes && c.block_complementarity_relative>1,"global cancellation hid nonzero block products");
}
void unsupported(int blocks) {
    for(int domain=0;domain<3;++domain) {
        Fixture f;mixed(f);
        if(domain==0)f.p.h_variable_lower[0]=0;
        if(domain==1){f.p.h_scalar_lower[1]=0;f.p.h_scalar_upper[1]=INFINITY;}
        if(domain==2)f.p.h_scalar_lower[1]=0;
        f.create(blocks);const auto policy=f.policy(1);
        t::require(spacepdhcg_cuda_workspace_set_common_kkt_options(f.w,&policy)==SPACEPDHCG_CUDA_UNSUPPORTED,"unsupported domain accepted");
        t::require(!f.common().enabled && !f.common().valid,"rejected policy changed active state");
    }
    std::cout<<"COMMON_KKT_VALIDATION {\"unsupported_domains_rejected\":3}\n"<<std::flush;
}
void cancellation(int blocks) {
    Fixture f;mixed(f);f.create(blocks);f.enable(1);f.seed({1,0},{3,1,1,0,-1});
    // The stream callback prevents any solve work until cancel has been recorded.
    // Cancellation before solve_async is intentionally not used: solve clears it.
    std::atomic<bool> release=false;
    t::cuda_require(cudaLaunchHostFunc(f.p.stream,[](void* value){auto& ready=*static_cast<std::atomic<bool>*>(value);while(!ready.load(std::memory_order_acquire))std::this_thread::yield();},&release),"queued cancellation fence");
    const auto options=t::solve_options(1e-9,1);
    const auto launched=spacepdhcg_cuda_workspace_solve_async(f.w,&options,f.p.exchange.consumer_stream);
    const auto cancelled=spacepdhcg_cuda_workspace_cancel(f.w);release.store(true,std::memory_order_release);
    t::status_require(launched,"cancel solve launch");t::status_require(cancelled,"cancel");
    t::status_require(spacepdhcg_cuda_workspace_wait(f.w),"cancel wait");
    spacepdhcg_cuda_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_diagnostics(f.w,&d),"cancel diagnostics");
    const auto c=f.common();output("cancellation_before_initial_certificate",d,c);
    t::require(d.termination==SPACEPDHCG_CUDA_TERMINATION_CANCELLED && d.iterations==0 && !c.valid,"initial certificate overrode cancellation");
}
void nonfinite_seed(int blocks) {
    Fixture f;mixed(f);f.create(blocks);f.enable(1);f.seed({std::numeric_limits<double>::quiet_NaN(),0},{3,1,1,0,-1});
    const auto options=t::solve_options(1e-9,1);
    t::status_require(spacepdhcg_cuda_workspace_solve_async(f.w,&options,f.p.exchange.consumer_stream),"nonfinite launch");
    const auto waited=spacepdhcg_cuda_workspace_wait(f.w);
    t::require(waited==SPACEPDHCG_CUDA_SUCCESS || waited==SPACEPDHCG_CUDA_NUMERICAL_FAILURE,"nonfinite wait failed unexpectedly");
    spacepdhcg_cuda_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_diagnostics(f.w,&d),"nonfinite diagnostics");
    const auto c=f.common();output("nonfinite_seed_never_certified",d,c);
    t::require(d.iterations==0 && d.termination==SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE && c.valid && !c.finite && !c.passes,"nonfinite reduction produced a certificate");
}
}
int main(int argc,char** argv) try {
    int blocks=0;if(argc==3 && std::string(argv[1])=="--execution-blocks")blocks=std::stoi(argv[2]);
    else t::require(argc==1,"usage: persistent_common_kkt_test [--execution-blocks 0|2]");
    t::require(blocks==0 || blocks==2,"test blocks must be 0 or 2");
    exact_and_lifecycle(blocks);grouping(blocks,false);grouping(blocks,true);block_cancellation(blocks);unsupported(blocks);cancellation(blocks);nonfinite_seed(blocks);
    std::cout<<"COMMON_KKT_SUMMARY {\"complete\":true,\"execution_blocks\":"<<blocks<<",\"solve_calls\":7}\n";return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
