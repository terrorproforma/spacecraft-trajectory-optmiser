// Bounded arithmetic/lifecycle test; seven solve APIs, no trajectory benchmark.
#include "cuda_test_support.hpp"
#include "persistent_mass_snapshot.hpp"
#include "persistent_mass_fixture.hpp"
#include "spacepdhcg/cuda/mass_causal_arithmetic.hpp"
#include <atomic>
#include <iostream>
#include <thread>
namespace t=spacepdhcg::cuda::test;
namespace s=spacepdhcg::snapshot;
namespace f=mass_fixture;
namespace {
const std::vector<int> active{4,5,6,7,8,9,10,11};
const std::vector<int> retained_rows{4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,27,28,29};
const std::vector<spacepdhcg_cuda_mass_node> nodes{{0,0,-1,-1},{1,1,4,7},{2,2,5,8},{3,3,6,9}};
s::Csc dense(const std::vector<std::vector<double>>& rows) {
    s::Csc out;out.rows=static_cast<int>(rows.size());out.columns=15;out.offsets.push_back(0);
    for(int j=0;j<15;++j) {
        for(int r=0;r<out.rows;++r)if(rows[r][j]!=0.0 || (j==0&&r==out.rows-1)) {
            out.indices.push_back(r);out.values.push_back(rows[r][j]);
        }
        out.offsets.push_back(static_cast<int>(out.indices.size()));
    }
    return out;
}
s::Canonical canonical(bool seed=false) {
    s::Canonical c;c.Q.rows=c.Q.columns=15;c.Q.offsets.assign(16,0);
    c.A=dense(f::A);c.F=dense(f::F);c.c=seed?f::seed_c:f::c;c.upper=f::scalar_upper;c.lower=c.upper;
    for(int i=6;i<27;++i)c.lower[i]=-INFINITY;c.offset=f::affine_offset;
    c.variable_lower.assign(15,-INFINITY);c.variable_upper.assign(15,INFINITY);
    c.cones={{SPACEPDHCG_CUDA_CONE_SECOND_ORDER,0,1,0}};return c;
}
s::Snapshot shape(){s::Snapshot q;q.n=15;q.p=6;q.m=24;q.nonnegative=21;return q;}
void close(const std::vector<double>& actual,const std::vector<double>& expected,double tolerance=3e-12) {
    t::require(actual.size()==expected.size(),"oracle vector shape");
    for(std::size_t i=0;i<actual.size();++i)t::require(std::isfinite(actual[i])&&std::abs(actual[i]-expected[i])<=tolerance,"independent mass oracle mismatch");
}
void same_bits(const std::vector<double>& actual,const std::vector<double>& expected) {
    t::require(actual.size()==expected.size()&&std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(double))==0,"original point bits changed");
}
void cpu_checks() {
    const auto q=shape();const auto c=canonical();const auto l=s::l1::detect(q,c);const auto map=s::mass::detect(q,c,l);
    t::require(l.pairs.size()==3&&map.size()==4,"CPU mass/L1 map shape");
    for(std::size_t i=0;i<map.size();++i)t::require(std::memcmp(&map[i],&nodes[i],sizeof(map[i]))==0,"CPU automatic causal order");
    for(int mutation=0;mutation<5;++mutation) {
        auto changed=c;auto changed_q=q;bool rejected=false;
        if(mutation==0)changed_q.shifted=true;
        if(mutation==1)changed.Q.values={std::numeric_limits<double>::denorm_min()};
        if(mutation==2)changed.c[0]=1;
        if(mutation==3){auto rows=f::A;rows[4][0]=std::numeric_limits<double>::denorm_min();changed.A=dense(rows);}
        if(mutation==4){auto rows=f::F;rows[0][0]=std::numeric_limits<double>::denorm_min();changed.F=dense(rows);}
        try{static_cast<void>(s::mass::detect(changed_q,changed,l));}catch(const std::exception&){rejected=true;}
        t::require(rejected,"CPU unsupported mass map accepted");
    }
    namespace mm=spacepdhcg::cuda::mass;
    t::require(mm::step_down(0)==mm::theta,"zero denominator policy");
    for(double v:{1.0,3.0,1000.0,1e100})t::require(mm::step_down(v)>0.0&&static_cast<long double>(mm::step_down(v))*v<=.95L,"CPU inward diagonal step");
    std::cout<<"MASS_CPU {\"passed\":true,\"nodes\":4,\"pairs\":3,\"GPU_calls\":0}\n";
}
struct Fixture {
    t::ProblemStorage p{false,true};spacepdhcg_cuda_workspace* w{};s::Canonical c;
    explicit Fixture(bool seed=false):c(canonical(seed)){}
    ~Fixture(){if(w)spacepdhcg_cuda_workspace_destroy(&w);}
    void create(bool mass=true) {
        p.variables=15;p.scalar_rows=27;p.affine_rows=3;
        p.h_q_offsets={0};p.h_q_indices.clear();p.h_q.clear();
        for(int j=0;j<15;++j) {
            if(j==0||j==12){p.h_q_indices.push_back(j==0?12:0);p.h_q.push_back(0.0);}
            p.h_q_offsets.push_back(static_cast<int>(p.h_q.size()));
        }
        p.h_a_offsets=c.A.offsets;p.h_a_indices=c.A.indices;p.h_a=c.A.values;
        p.h_f_offsets=c.F.offsets;p.h_f_indices=c.F.indices;p.h_f=c.F.values;
        p.h_c=c.c;p.h_scalar_lower=c.lower;p.h_scalar_upper=c.upper;p.h_affine_offset=c.offset;
        p.h_variable_lower=c.variable_lower;p.h_variable_upper=c.variable_upper;p.affine_cones=c.cones;
        p.materialise();w=t::create_workspace(p);t::status_require(spacepdhcg_cuda_workspace_wait(w),"create wait");
        t::status_require(spacepdhcg_cuda_workspace_set_execution_blocks(w,2),"blocks");
        const spacepdhcg_cuda_common_kkt_options k{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,1,6,0,1e-9,1e-8};
        t::status_require(spacepdhcg_cuda_workspace_set_common_kkt_options(w,&k),"common");
        const auto l=s::l1::detect(shape(),c);std::vector<spacepdhcg_cuda_l1_pair> pairs;
        for(const auto pair:l.pairs)pairs.push_back({pair.epigraph,pair.variable,pair.positive_row,pair.negative_row});
        const spacepdhcg_cuda_l1_options option{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,1,static_cast<int>(pairs.size()),0};
        t::status_require(spacepdhcg_cuda_workspace_set_l1_options(w,&option,pairs.data()),"L1");
        if(mass)on();
    }
    void on(){const spacepdhcg_cuda_mass_options o{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,1,4,0};t::status_require(spacepdhcg_cuda_workspace_set_mass_options(w,&o,nodes.data()),"mass enable");}
    void off(){const spacepdhcg_cuda_mass_options o{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,0,0,0};t::status_require(spacepdhcg_cuda_workspace_set_mass_options(w,&o,nullptr),"mass disable");}
    void seed(const std::vector<double>& x,const std::vector<double>& y) {
        p.primal.upload(x,p.stream);p.dual.upload(y,p.stream);
        t::status_require(spacepdhcg_cuda_workspace_warm_start_async(w,SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL,&p.exchange.iterates,p.exchange.consumer_stream),"seed");
        t::status_require(spacepdhcg_cuda_workspace_wait(w),"seed wait");
    }
    spacepdhcg_cuda_mass_diagnostics mass(){spacepdhcg_cuda_mass_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_mass_diagnostics(w,&d),"mass diagnostics");return d;}
    spacepdhcg_cuda_common_kkt_diagnostics common(){spacepdhcg_cuda_common_kkt_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_common_kkt_diagnostics(w,&d),"common diagnostics");return d;}
    std::vector<double> scales() {
        spacepdhcg_cuda_pointer_snapshot ptr{};t::status_require(spacepdhcg_cuda_workspace_pointer_snapshot(w,&ptr),"pointers");
        std::vector<double> result(45);t::cuda_require(cudaMemcpy(result.data(),reinterpret_cast<void*>(ptr.scaling),result.size()*sizeof(double),cudaMemcpyDeviceToHost),"actual direct steps");return result;
    }
};
void output(const char* name,Fixture& fixture,const spacepdhcg_cuda_diagnostics& d) {
    const auto m=fixture.mass();const auto k=fixture.common();std::cout.precision(17);
    std::cout<<"MASS_TEST {\"case\":\""<<name<<"\",\"termination\":"<<d.termination<<",\"iterations\":"<<d.iterations
        <<",\"common_valid\":"<<k.valid<<",\"common_passes\":"<<k.passes<<",\"mass_enabled\":"<<m.enabled<<",\"mass_valid\":"<<m.valid
        <<",\"finite\":"<<m.finite<<",\"updates\":"<<m.updates<<",\"completions\":"<<m.completions;
    auto number=[](const char* key,double value){std::cout<<",\""<<key<<"\":";if(std::isfinite(value))std::cout<<value;else std::cout<<"null";};
    number("norm_squared_upper",m.norm_squared_upper);number("gap",k.gap_relative);std::cout<<"}\n"<<std::flush;
}
void cold_and_transition() {
    Fixture a;a.create();a.seed(f::initial_x,f::initial_y);
    const auto d=t::solve_and_wait(a.w,a.p,t::solve_options(1e-9,3));output("three_interval_oracle",a,d);
    t::require(d.termination==SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT&&d.iterations==3&&a.mass().updates==3,"three-step budget");
    close(a.p.primal.download(a.p.stream),f::expected_x[2]);close(a.p.dual.download(a.p.stream),f::expected_y[2]);
    const auto scale=a.scales();long double alpha=0,beta=0;
    for(std::size_t i=0;i<active.size();++i) {const double step=scale[active[i]];t::require(std::abs(step-f::tau[i])<2e-15,"actual primal diagonal");beta=std::max(beta,static_cast<long double>(step)*f::column_sums[i]);}
    for(std::size_t i=0;i<retained_rows.size();++i) {const double step=scale[15+retained_rows[i]];t::require(std::abs(step-f::sigma[i])<2e-15,"actual dual diagonal");alpha=std::max(alpha,static_cast<long double>(step)*f::row_sums[i]);}
    t::require(alpha*beta<1.0L&&a.mass().norm_squared_upper<1.0,"actual step product bound");
    t::require(scale[42]==scale[43]&&scale[43]==scale[44],"SOC steps untied");
    t::require(a.mass().active_variables==8&&a.mass().active_rows==20&&a.mass().retained_variables==15,"logical versus retained layout");
    const auto x=a.p.primal.download(a.p.stream),y=a.p.dual.download(a.p.stream);a.off();t::require(!a.mass().valid,"disabled mass record stale");
    const auto switched=t::solve_and_wait(a.w,a.p,t::solve_options(1e-9,1));output("disabled_to_L1",a,switched);
    Fixture b;b.create(false);b.seed(x,y);const auto control=t::solve_and_wait(b.w,b.p,t::solve_options(1e-9,1));output("fresh_L1_control",b,control);
    t::require(switched.scaling_refreshed&&switched.iterations==1&&control.iterations==1,"mode-off scaling refresh");
    close(a.scales(),b.scales());close(a.p.primal.download(a.p.stream),b.p.primal.download(b.p.stream));close(a.p.dual.download(a.p.stream),b.p.dual.download(b.p.stream));
    a.on();t::require(!a.mass().valid&&!a.common().valid,"re-enable stale certificate");
}
void qualified_seed() {
    Fixture a(true);a.create();a.seed(f::seed_x,f::seed_y);const auto d=t::solve_and_wait(a.w,a.p,t::solve_options(1e-9,1));output("original_seed_zero_step",a,d);
    t::require(d.termination==SPACEPDHCG_CUDA_TERMINATION_OPTIMAL&&d.iterations==0&&a.common().passes&&a.mass().completions==0,"original seed transformed");
    same_bits(a.p.primal.download(a.p.stream),f::seed_x);same_bits(a.p.dual.download(a.p.stream),f::seed_y);
    t::require(spacepdhcg_cuda_workspace_set_execution_blocks(a.w,0)==SPACEPDHCG_CUDA_UNSUPPORTED,"mass single block accepted");
    const spacepdhcg_cuda_l1_options off{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,0,0,0};
    t::require(spacepdhcg_cuda_workspace_set_l1_options(a.w,&off,nullptr)==SPACEPDHCG_CUDA_UNSUPPORTED,"L1 disabled underneath mass");
    const spacepdhcg_cuda_l1_weight_options weight{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,SPACEPDHCG_CUDA_L1_WEIGHT_FIXED,1.0};
    t::require(spacepdhcg_cuda_workspace_set_l1_weight(a.w,&weight)==SPACEPDHCG_CUDA_UNSUPPORTED,"mass weight changed");
    t::require(spacepdhcg_cuda_workspace_checkpoint_async(a.w,{},a.p.exchange.consumer_stream)==SPACEPDHCG_CUDA_UNSUPPORTED,"mass checkpoint accepted");
    t::status_require(spacepdhcg_cuda_workspace_reset_async(a.w,SPACEPDHCG_CUDA_RESET_FULL,a.p.exchange.consumer_stream),"reset");
    t::status_require(spacepdhcg_cuda_workspace_wait(a.w),"reset wait");t::require(!a.mass().valid&&!a.common().valid,"reset stale diagnostics");
    a.seed(f::seed_x,f::seed_y);t::require(!a.mass().valid,"reseed stale diagnostics");
}
void cancellation() {
    Fixture a(true);a.create();a.seed(f::seed_x,f::seed_y);std::atomic<bool> release=false;
    t::cuda_require(cudaLaunchHostFunc(a.p.stream,[](void* p){while(!static_cast<std::atomic<bool>*>(p)->load(std::memory_order_acquire))std::this_thread::yield();},&release),"cancel fence");
    const auto options=t::solve_options(1e-9,1);const auto started=spacepdhcg_cuda_workspace_solve_async(a.w,&options,a.p.exchange.consumer_stream);
    const auto cancelled=spacepdhcg_cuda_workspace_cancel(a.w);release.store(true,std::memory_order_release);
    t::status_require(started,"start");t::status_require(cancelled,"cancel");t::status_require(spacepdhcg_cuda_workspace_wait(a.w),"cancel wait");
    spacepdhcg_cuda_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_diagnostics(a.w,&d),"cancel result");output("cancel_before_initial",a,d);
    t::require(d.termination==SPACEPDHCG_CUDA_TERMINATION_CANCELLED&&d.iterations==0&&!a.common().valid&&!a.mass().valid,"cancel lost to initial metric/certificate");
    same_bits(a.p.primal.download(a.p.stream),f::seed_x);same_bits(a.p.dual.download(a.p.stream),f::seed_y);
}
void failure(bool overflow) {
    Fixture a(true);
    if(overflow){auto rows=f::A;rows[1][4]=std::numeric_limits<double>::max();a.c.A=dense(rows);}
    a.create();auto x=f::seed_x;if(!overflow)x[4]=std::numeric_limits<double>::quiet_NaN();a.seed(x,f::seed_y);
    const auto options=t::solve_options(1e-9,1);t::status_require(spacepdhcg_cuda_workspace_solve_async(a.w,&options,a.p.exchange.consumer_stream),"failure launch");
    const auto waited=spacepdhcg_cuda_workspace_wait(a.w);t::require(waited==SPACEPDHCG_CUDA_SUCCESS||waited==SPACEPDHCG_CUDA_NUMERICAL_FAILURE,"failure status");
    spacepdhcg_cuda_diagnostics d{};t::status_require(spacepdhcg_cuda_workspace_diagnostics(a.w,&d),"failure report");output(overflow?"metric_overflow":"nonfinite_seed",a,d);
    t::require(d.termination==SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE&&d.iterations==0&&!a.mass().finite,"invalid metric/point accepted");
    same_bits(a.p.primal.download(a.p.stream),x);same_bits(a.p.dual.download(a.p.stream),f::seed_y);
}
void rejection() {
    for(int mutation=0;mutation<5;++mutation) {
        Fixture a;auto rows=f::A;
        if(mutation==0)a.c.c[0]=std::numeric_limits<double>::denorm_min();
        if(mutation==1){rows[4][0]=std::numeric_limits<double>::denorm_min();a.c.A=dense(rows);}
        if(mutation==2){auto af=f::F;af[0][0]=std::numeric_limits<double>::denorm_min();a.c.F=dense(af);}
        if(mutation==3){rows[1][4]=-0.125;a.c.A=dense(rows);}
        a.create(false);auto map=nodes;if(mutation==4)map[2].gamma_variable=map[1].gamma_variable;
        const spacepdhcg_cuda_mass_options option{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,1,4,0};
        const auto status=spacepdhcg_cuda_workspace_set_mass_options(a.w,&option,map.data());
        t::require(status==(mutation==4?SPACEPDHCG_CUDA_INVALID_ARGUMENT:SPACEPDHCG_CUDA_UNSUPPORTED),"unsupported causal map accepted");
    }
    std::cout<<"MASS_VALIDATION {\"mass_cost\":true,\"extra_equality\":true,\"extra_SOC\":true,\"negative_gamma\":true,\"overlap\":true}\n";
}
}
int main(int argc,char** argv) {
    cpu_checks();if(argc==2&&std::string(argv[1])=="--cpu-only")return 0;
    t::require(argc==1,"usage: persistent_mass_test [--cpu-only]");
    cold_and_transition();qualified_seed();cancellation();failure(false);failure(true);rejection();
    std::cout<<"MASS_TEST_SUMMARY {\"passed\":true,\"solve_API_calls\":7,\"iteration_caps\":9,\"actual_updates\":5}\n";
}
