// Diagnostic importer for the production persistent C ABI. No GTOC12 backend alias.
#include "persistent_snapshot.hpp"
#include "cuda_test_support.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#if defined(__linux__)
#include <dlfcn.h>
#endif

#ifndef SPACEPDHCG_SNAPSHOT_SOURCE_SHA256
#define SPACEPDHCG_SNAPSHOT_SOURCE_SHA256 "unrecorded"
#endif
#ifndef SPACEPDHCG_SOURCE_COMMIT
#define SPACEPDHCG_SOURCE_COMMIT "unrecorded"
#endif

namespace s=spacepdhcg::snapshot;
namespace test=spacepdhcg::cuda::test;
using Clock=std::chrono::steady_clock;
namespace {
double elapsed(Clock::time_point start) {return std::chrono::duration<double>(Clock::now()-start).count();}
void number(long double v) {if(std::isfinite(v))std::cout<<v;else std::cout<<"null";}
void quoted(const std::string& value) {
    std::cout<<'"';
    for(unsigned char c:value) {
        if(c=='"'||c=='\\')std::cout<<'\\'<<c;
        else if(c<32) {const char* digits="0123456789abcdef";std::cout<<"\\u00"<<digits[c>>4]<<digits[c&15];}
        else std::cout<<c;
    }
    std::cout<<'"';
}
void vector_json(const std::vector<double>& v) {
    std::cout<<'[';for(std::size_t i=0;i<v.size();++i) {if(i)std::cout<<',';number(v[i]);}std::cout<<']';
}
void metric(const char* name,long double v,bool first=false) {
    if(!first)std::cout<<',';
    quoted(name);std::cout<<':';number(v);
}
void audit_json(const s::Quality& q) {
    std::cout<<'{';metric("primal",q.primal,true);metric("dual",q.dual);metric("gap",q.gap);
    metric("objective",q.objective);metric("dual_objective",q.dual_objective);metric("signed_gap",q.signed_gap);
    metric("equality_absolute",q.equality_absolute);metric("conic_equation_absolute",q.conic_equation_absolute);
    metric("stationarity_absolute",q.stationarity_absolute);metric("primal_cone_violation",q.primal_cone_violation);
    metric("dual_cone_violation",q.dual_cone_violation);metric("complementarity",q.complementarity);
    metric("global_complementarity",q.global_complementarity);
    metric("block_complementarity_normalized",q.block_complementarity_normalized);
    metric("global_complementarity_normalized",q.global_complementarity_normalized);
    std::cout<<",\"finite\":"<<(q.finite?"true":"false")<<'}';
}
struct Arguments {
    std::string path,mode="cold";
    double tolerance{},deadline{},audit_tolerance=1e-9,cone_tolerance=1e-8;
    std::uint64_t iterations{};
    int repeats=1;
    bool validate_only=false,fold_singleton_bounds=false;
};
Arguments arguments(int argc,char** argv) {
    s::require(argc>=2,"usage: persistent_snapshot_replay SNAPSHOT --tolerance T --iterations N --deadline-seconds S [--repeats N] [--mode cold|reuse|full-retained] [--fold-singleton-bounds] [--audit-tolerance T] [--cone-tolerance T], or SNAPSHOT --validate-only");
    Arguments out;out.path=argv[1];
    auto number_arg=[](const std::string& text) {
        std::size_t consumed=0;const double v=std::stod(text,&consumed);
        s::require(consumed==text.size() && std::isfinite(v) && v>0,"invalid positive numeric option");return v;
    };
    auto integer_arg=[](const std::string& text,std::uint64_t maximum) {
        s::require(!text.empty() && text.find_first_not_of("0123456789")==std::string::npos,"invalid integer option");
        std::size_t consumed=0;const auto v=std::stoull(text,&consumed);
        s::require(consumed==text.size() && v>0 && v<=maximum,"integer option exceeds diagnostic limit");return v;
    };
    std::vector<std::string> seen;
    for(int i=2;i<argc;++i) {
        const std::string name(argv[i]);s::require(std::find(seen.begin(),seen.end(),name)==seen.end(),"duplicate option");seen.push_back(name);
        if(name=="--validate-only") {out.validate_only=true;continue;}
        if(name=="--fold-singleton-bounds") {out.fold_singleton_bounds=true;continue;}
        s::require(i+1<argc,"missing option value");const std::string value(argv[++i]);
        if(name=="--tolerance")out.tolerance=number_arg(value);
        else if(name=="--iterations")out.iterations=integer_arg(value,100'000'000);
        else if(name=="--deadline-seconds")out.deadline=number_arg(value);
        else if(name=="--audit-tolerance")out.audit_tolerance=number_arg(value);
        else if(name=="--cone-tolerance")out.cone_tolerance=number_arg(value);
        else if(name=="--repeats")out.repeats=static_cast<int>(integer_arg(value,1000));
        else if(name=="--mode") {s::require(value=="cold"||value=="reuse"||value=="full-retained","unsupported repeat mode");out.mode=value;}
        else throw std::runtime_error("unknown option: "+name);
    }
    s::require(out.validate_only || (out.tolerance>0 && out.iterations>0 && out.deadline>0),"solve requires explicit tolerance, iteration cap and deadline");
    s::require(out.deadline<=3600,"deadline exceeds one-hour diagnostic limit");return out;
}
void api(spacepdhcg_cuda_status status,const char* operation,spacepdhcg_cuda_workspace* workspace=nullptr) {
    if(status==SPACEPDHCG_CUDA_SUCCESS)return;
    char message[1024]{};
    if(workspace)static_cast<void>(spacepdhcg_cuda_workspace_last_error(workspace,message,sizeof(message)));
    throw std::runtime_error(std::string(operation)+": status "+std::to_string(status)+" "+message);
}
struct Owner {
    // Destroy workspace before its borrowed input/output storage and stream.
    test::ProblemStorage storage{false,true};
    spacepdhcg_cuda_workspace* workspace{};
    explicit Owner(const s::Snapshot& q,const s::Canonical& c) {
        auto& p=storage;p.variables=q.n;p.scalar_rows=c.A.rows;p.affine_rows=c.F.rows;
        p.fingerprint=std::stoull((c.fold_singleton_bounds?s::sha256(q.input_sha256+"fold-singleton-bounds"):q.input_sha256).substr(0,16),nullptr,16);
        p.h_q_offsets=c.Q.offsets;p.h_q_indices=c.Q.indices;p.h_q=c.Q.values;
        p.h_a_offsets=c.A.offsets;p.h_a_indices=c.A.indices;p.h_a=c.A.values;
        // The C ABI requires an empty affine-offset view when there are no SOC rows.
        p.h_f_offsets=c.F.rows>0?c.F.offsets:std::vector<int>{};p.h_f_indices=c.F.indices;p.h_f=c.F.values;
        p.h_c=c.c;p.h_scalar_lower=c.lower;p.h_scalar_upper=c.upper;p.h_affine_offset=c.offset;
        p.h_variable_lower=c.variable_lower;p.h_variable_upper=c.variable_upper;p.affine_cones=c.cones;
        p.materialise();const auto options=test::create_options();
        api(spacepdhcg_cuda_workspace_create(&p.structure,&p.exchange,&options,&workspace),"create");
    }
    ~Owner() {if(workspace)static_cast<void>(spacepdhcg_cuda_workspace_destroy(&workspace));}
};
const char* termination_name(spacepdhcg_cuda_termination termination) {
    switch(termination) {
        case SPACEPDHCG_CUDA_TERMINATION_UNSPECIFIED:return "unspecified";
        case SPACEPDHCG_CUDA_TERMINATION_OPTIMAL:return "optimal";
        case SPACEPDHCG_CUDA_TERMINATION_ITERATION_LIMIT:return "iteration_limit";
        case SPACEPDHCG_CUDA_TERMINATION_CANCELLED:return "cancelled";
        case SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE:return "numerical_failure";
    }
    return "unknown";
}
std::string library_path() {
#if defined(__linux__)
    Dl_info info{};
    s::require(dladdr(reinterpret_cast<void*>(&spacepdhcg_cuda_workspace_create),&info)!=0 && info.dli_fname,"cannot identify loaded persistent library");
    return info.dli_fname;
#else
    throw std::runtime_error("runtime library fingerprinting currently requires Linux");
#endif
}
} // namespace

int main(int argc,char** argv) try {
    const auto process_begin=Clock::now();
    const auto args=arguments(argc,argv);
    const auto prepare=Clock::now();
    const auto snapshot=s::read(s::file_bytes(args.path,128ULL*1024*1024));
    const auto canonical=s::canonical(snapshot,args.fold_singleton_bounds);
    const double conversion_seconds=elapsed(prepare);
    const auto library=args.validate_only?std::string{}:library_path();
    const auto library_sha=args.validate_only?std::string{}:s::sha256(s::file_bytes(library));
    // Everything above, including malformed-input rejection, is CPU-only.
    std::cout<<std::setprecision(21)<<std::boolalpha;
    std::cout<<"PERSISTENT_REPLAY_META {\"input_sha256\":";quoted(snapshot.input_sha256);
    std::cout<<",\"source_commit\":";quoted(SPACEPDHCG_SOURCE_COMMIT);
    std::cout<<",\"source_sha256\":";quoted(SPACEPDHCG_SNAPSHOT_SOURCE_SHA256);
    std::cout<<",\"coordinate_system\":\"original\",\"slack_source\":\"reconstructed_h_minus_Gx\",\"shifted\":"<<snapshot.shifted;
    std::cout<<",\"convexity_evidence\":";quoted(canonical.convexity_evidence);
    std::cout<<",\"variables\":"<<snapshot.n<<",\"equalities\":"<<snapshot.p<<",\"inequalities\":"<<snapshot.m;
    std::cout<<",\"soc_count\":"<<snapshot.soc.size()<<",\"symmetric_quadratic_entries\":"<<canonical.Q.values.size();
    std::cout<<",\"fold_singleton_bounds\":"<<args.fold_singleton_bounds<<",\"representation\":";
    quoted(args.fold_singleton_bounds?"exact_singletons_as_native_variable_bounds":"all_nonnegative_rows_as_scalar_duals");
    std::cout<<",\"singleton_rows\":"<<canonical.singleton_rows<<",\"folded_rows\":"<<canonical.folded_bounds.size();
    std::cout<<",\"retained_nonexact_singleton_ratios\":"<<canonical.retained_nonexact_ratios<<",\"retained_out_of_range_singleton_ratios\":"<<canonical.retained_out_of_range_ratios;
    std::cout<<",\"native_scalar_rows\":"<<canonical.A.rows<<",\"native_affine_rows\":"<<canonical.F.rows;
    std::cout<<",\"folded_dual_activity_rule\":\"exact_local_primal_equals_tight_native_bound\"";
    metric("conversion_seconds",conversion_seconds);metric("audit_tolerance",args.audit_tolerance);metric("cone_tolerance",args.cone_tolerance);
    std::cout<<",\"repeat_mode\":";quoted(args.mode);
    std::cout<<",\"repeat_seconds_scope\":\"workspace_setup_solve_download_audit_and_cold_or_failed_cleanup_excludes_shared_initialization_and_output\"";
    std::cout<<",\"peak_workspace_bytes_scope\":\"native_workspace_only_excludes_borrowed_input_output_storage\"";
    std::cout<<",\"scaling_policy\":\"refresh_if_needed\",\"matrix_change_threshold\":0.25,\"vector_change_threshold\":0.5,\"scaling_reuse_limit\":4,\"residual_check_frequency\":25";
    std::cout<<",\"complementarity_gate\":\"max_absolute_scalar_or_SOC_dot_over_max_1_abs_primal_objective_abs_dual_objective\"";
    std::cout<<",\"captured_qoco_settings_used_by_persistent\":false,\"validate_only\":"<<args.validate_only;
    if(args.validate_only) {std::cout<<"}\n";return 0;}
    std::cout<<",\"library_path\":";quoted(library);
    std::cout<<",\"library_sha256\":";quoted(library_sha);std::cout<<"}\n"<<std::flush;
    // The test storage helper binds device zero explicitly. Select it deliberately.
    test::cuda_require(cudaSetDevice(0),"select CUDA device zero");
    std::unique_ptr<Owner> owner;
    for(int repeat=0;repeat<args.repeats;++repeat) {
        const auto complete_begin=Clock::now();const auto setup_begin=Clock::now();
        const bool fresh=!owner || args.mode=="cold";
        if(fresh) {owner.reset();owner=std::make_unique<Owner>(snapshot,canonical);}
        auto& p=owner->storage;auto* w=owner->workspace;
        if(!fresh) {
            if(args.mode=="reuse")api(spacepdhcg_cuda_workspace_reset_async(w,SPACEPDHCG_CUDA_RESET_ITERATES,p.exchange.consumer_stream),"reset iterates",w);
            else api(spacepdhcg_cuda_workspace_warm_start_async(w,SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED,nullptr,p.exchange.consumer_stream),"full retained warm start",w);
            api(spacepdhcg_cuda_workspace_wait(w),"prepare repeat",w);
        }
        const double setup_seconds=elapsed(setup_begin);
        const auto options=test::solve_options(args.tolerance,args.iterations);
        const auto begin=Clock::now();
        api(spacepdhcg_cuda_workspace_solve_async(w,&options,p.exchange.consumer_stream),"solve launch",w);
        std::mutex mutex;std::condition_variable wake;bool finished=false;
        std::atomic<bool> deadline_requested=false;
        std::atomic<int> cancellation_status{SPACEPDHCG_CUDA_SUCCESS};
        std::thread watchdog([&] {
            std::unique_lock lock(mutex);
            const auto deadline=begin+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(args.deadline));
            if(!wake.wait_until(lock,deadline,[&]{return finished;})) {
                deadline_requested=true;cancellation_status=spacepdhcg_cuda_workspace_cancel(w);
            }
        });
        const auto status=spacepdhcg_cuda_workspace_wait(w);
        {std::lock_guard lock(mutex);finished=true;}wake.notify_one();watchdog.join();
        const double wall=elapsed(begin);
        // Numerical failures still have an auditable termination and iterate.
        s::require(status==SPACEPDHCG_CUDA_SUCCESS || status==SPACEPDHCG_CUDA_NUMERICAL_FAILURE,"persistent wait failed: "+std::to_string(status));
        spacepdhcg_cuda_diagnostics diagnostic{};
        api(spacepdhcg_cuda_workspace_diagnostics(w,&diagnostic),"diagnostics",w);
        const auto download=Clock::now();const auto primal=p.primal.download(p.stream),dual=p.dual.download(p.stream);
        const double download_seconds=elapsed(download);const auto audit_begin=Clock::now();
        const auto original=s::original_vectors(snapshot,canonical,primal,dual);
        const auto quality=s::audit(snapshot,original,args.audit_tolerance,args.cone_tolerance);
        const double audit_seconds=elapsed(audit_begin);
        const bool optimal=diagnostic.termination==SPACEPDHCG_CUDA_TERMINATION_OPTIMAL;
        const auto cleanup_begin=Clock::now();
        // Independently unqualified iterates never seed a retained solve.
        if(!optimal || !quality.qualified || args.mode=="cold")owner.reset();
        const double cleanup_seconds=elapsed(cleanup_begin),repeat_seconds=elapsed(complete_begin);
        std::cout<<"PERSISTENT_REPLAY {\"repeat\":"<<repeat<<",\"termination\":"<<diagnostic.termination<<",\"termination_name\":";quoted(termination_name(diagnostic.termination));
        std::cout<<",\"api_status\":"<<status<<",\"solver_optimal\":"<<optimal;
        std::cout<<",\"fresh_workspace\":"<<fresh<<",\"within_requested_wall_deadline\":"<<(wall<=args.deadline);
        std::cout<<",\"folded_dual_reconstruction_supported\":"<<original.folded_dual_reconstruction_supported<<",\"reconstructed_bound_duals\":"<<original.reconstructed_bound_duals;
        std::cout<<",\"off_contact_bound_normals\":"<<original.off_contact_bound_normals<<",\"off_contact_one_ulp_normals\":"<<original.off_contact_one_ulp_normals;
        metric("max_off_contact_bound_distance",original.max_off_contact_bound_distance);
        std::cout<<",\"kkt_qualified_original\":"<<quality.qualified<<",\"qualified_original\":"<<(optimal&&quality.qualified);
        std::cout<<",\"deadline_requested\":"<<deadline_requested.load()<<",\"cancellation_api_status\":"<<cancellation_status.load();
        std::cout<<",\"iteration_limit\":"<<args.iterations<<",\"iterations\":"<<diagnostic.iterations<<",\"recovery_iterations\":"<<diagnostic.recovery_iterations;
        metric("requested_tolerance",args.tolerance);metric("deadline_seconds",args.deadline);
        metric("setup_seconds",setup_seconds);metric("wall_seconds",wall);metric("scaling_seconds",diagnostic.scaling_seconds);
        metric("solve_seconds",diagnostic.solve_seconds);metric("recovery_seconds",diagnostic.recovery_seconds);
        metric("download_seconds",download_seconds);metric("audit_seconds",audit_seconds);metric("cleanup_seconds",cleanup_seconds);metric("repeat_seconds",repeat_seconds);
        metric("native_objective",diagnostic.objective);metric("native_natural_residual",diagnostic.natural_residual_inf);
        std::cout<<",\"peak_workspace_bytes\":"<<diagnostic.peak_active_bytes<<",\"audit\":";audit_json(quality);
        std::cout<<",\"x\":";vector_json(original.x);std::cout<<",\"x_solver\":";vector_json(primal);
        std::cout<<",\"y\":";vector_json(original.y);std::cout<<",\"z\":";vector_json(original.z);std::cout<<",\"s\":";vector_json(original.s);
        std::cout<<"}\n"<<std::flush;
    }
    owner.reset();
    std::cout<<"PERSISTENT_REPLAY_SUMMARY {\"repeats\":"<<args.repeats<<",\"process_seconds\":"<<elapsed(process_begin)
             <<",\"process_seconds_scope\":\"argument_parse_through_final_workspace_cleanup_includes_all_prior_output_excludes_this_summary\"}\n";
    return 0; // Completion includes explicit unqualified outcomes; consume the quality fields.
} catch(const std::exception& e) {std::cerr<<"persistent snapshot replay: "<<e.what()<<'\n';return 1;}
