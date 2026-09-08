// Diagnostic importer for the production persistent C ABI. No GTOC12 backend alias.
#include "persistent_snapshot.hpp"
#include "persistent_l1_snapshot.hpp"
#include "persistent_mass_snapshot.hpp"
#include "persistent_warm_snapshot.hpp"
#include "cuda_test_support.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
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
#ifndef SPACEPDHCG_SOURCE_TREE_SHA256
#define SPACEPDHCG_SOURCE_TREE_SHA256 ""
#endif
#ifndef SPACEPDHCG_SOURCE_BASE_COMMIT
#define SPACEPDHCG_SOURCE_BASE_COMMIT ""
#endif

namespace s=spacepdhcg::snapshot;
namespace test=spacepdhcg::cuda::test;
using Clock=std::chrono::steady_clock;
namespace {
double elapsed(Clock::time_point start) {return std::chrono::duration<double>(Clock::now()-start).count();}
void number(long double v) {if(std::isfinite(v))std::cout<<v;else std::cout<<"null";}
// Avoid the name quoted: ADL can select std::quoted(string&) and discard its manipulator.
void json_string(const std::string& value) {
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
    json_string(name);std::cout<<':';number(v);
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
    std::string path,mode="cold",initial_point,warm_start_source,halpern="off";
    double tolerance{},deadline{},audit_tolerance=1e-9,cone_tolerance=1e-8,l1_weight=1.0;
    std::uint64_t iterations{};
    int repeats=1,execution_blocks=-1,l1_weight_mode=SPACEPDHCG_CUDA_L1_WEIGHT_UNIT;
    bool validate_only=false,fold_singleton_bounds=false,common_kkt_stop=false,l1_prox=false,l1_weight_explicit=false,mass_eliminate=false;
};
Arguments arguments(int argc,char** argv) {
    s::require(argc>=2,"usage: persistent_snapshot_replay SNAPSHOT --tolerance T --iterations N --deadline-seconds S [--repeats N] [--mode cold|reuse|full-retained] [--fold-singleton-bounds] [--initial-point PATH] [--audit-tolerance T] [--cone-tolerance T], or SNAPSHOT --validate-only");
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
        if(name=="--common-kkt-stop") {out.common_kkt_stop=true;continue;}
        if(name=="--l1-prox") {out.l1_prox=true;continue;}
        if(name=="--mass-eliminate") {out.mass_eliminate=true;continue;}
        s::require(i+1<argc,"missing option value");const std::string value(argv[++i]);
        if(name=="--tolerance")out.tolerance=number_arg(value);
        else if(name=="--iterations")out.iterations=integer_arg(value,100'000'000);
        else if(name=="--deadline-seconds")out.deadline=number_arg(value);
        else if(name=="--audit-tolerance")out.audit_tolerance=number_arg(value);
        else if(name=="--cone-tolerance")out.cone_tolerance=number_arg(value);
        else if(name=="--repeats")out.repeats=static_cast<int>(integer_arg(value,1000));
        else if(name=="--execution-blocks")out.execution_blocks=value=="0"?0:static_cast<int>(integer_arg(value,1000000));
        else if(name=="--l1-weight") {
            out.l1_weight_explicit=true;
            if(value=="cancel-global") {out.l1_weight_mode=SPACEPDHCG_CUDA_L1_WEIGHT_CANCEL_GLOBAL;out.l1_weight=0.0;}
            else {out.l1_weight_mode=SPACEPDHCG_CUDA_L1_WEIGHT_FIXED;out.l1_weight=number_arg(value);}
        }
        else if(name=="--halpern") {s::require(value=="off"||value=="plain"||value=="adaptive","unsupported Halpern mode");out.halpern=value;}
        else if(name=="--mode") {s::require(value=="cold"||value=="reuse"||value=="full-retained","unsupported repeat mode");out.mode=value;}
        else if(name=="--initial-point") {s::require(!value.empty(),"empty initial-point path");out.initial_point=value;}
        else if(name=="--warm-start-source") {s::require(!value.empty(),"empty warm-start source path");out.warm_start_source=value;}
        else throw std::runtime_error("unknown option: "+name);
    }
    s::require(out.validate_only || (out.tolerance>0 && out.iterations>0 && out.deadline>0),"solve requires explicit tolerance, iteration cap and deadline");
    s::require(out.deadline<=3600,"deadline exceeds one-hour diagnostic limit");
    s::require(out.warm_start_source.empty() || (!out.initial_point.empty() && out.common_kkt_stop
        && !out.fold_singleton_bounds && out.execution_blocks>0 && out.mode=="cold" && out.repeats==1
        && out.halpern=="off" && !out.mass_eliminate && !out.l1_weight_explicit),
        "warm-source replay requires a predecessor point, common-KKT, positive blocks, one fresh workspace and no optional weighting, mass elimination or Halpern");
    return out;
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
struct SolveWait {
    spacepdhcg_cuda_status status{};
    double wall_seconds{};
    bool deadline_requested{};
    int cancellation_status{};
};
SolveWait bounded_solve(spacepdhcg_cuda_workspace* w,test::ProblemStorage& p,
                        const spacepdhcg_cuda_solve_options& options,double deadline_seconds) {
    const auto begin=Clock::now();
    api(spacepdhcg_cuda_workspace_solve_async(w,&options,p.exchange.consumer_stream),"solve launch",w);
    std::mutex mutex;std::condition_variable wake;bool finished=false;
    std::atomic<bool> deadline_requested=false;
    std::atomic<int> cancellation_status{SPACEPDHCG_CUDA_SUCCESS};
    std::thread watchdog([&] {
        std::unique_lock lock(mutex);
        const auto deadline=begin+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(deadline_seconds));
        if(!wake.wait_until(lock,deadline,[&]{return finished;})) {
            deadline_requested=true;cancellation_status=spacepdhcg_cuda_workspace_cancel(w);
        }
    });
    const auto status=spacepdhcg_cuda_workspace_wait(w);
    {std::lock_guard lock(mutex);finished=true;}wake.notify_one();watchdog.join();
    const double wall=elapsed(begin);
    s::require(status==SPACEPDHCG_CUDA_SUCCESS || status==SPACEPDHCG_CUDA_NUMERICAL_FAILURE,"persistent wait failed: "+std::to_string(status));
    return {status,wall,deadline_requested.load(),cancellation_status.load()};
}
std::vector<double> internal_copy(std::uintptr_t address,std::size_t count,cudaStream_t stream) {
    std::vector<double> values(count);
    if(count) {
        s::require(address!=0,"null internal iterate pointer");
        test::cuda_require(cudaMemcpyAsync(values.data(),reinterpret_cast<const void*>(address),count*sizeof(double),cudaMemcpyDeviceToHost,stream),"internal iterate readback");
        test::cuda_require(cudaStreamSynchronize(stream),"internal iterate readback wait");
    }
    return values;
}
} // namespace

int main(int argc,char** argv) try {
    const auto process_begin=Clock::now();
    const auto args=arguments(argc,argv);
    const auto prepare=Clock::now();
    const auto snapshot=s::read(s::file_bytes(args.path,128ULL*1024*1024));
    s::require(!args.common_kkt_stop || (!snapshot.shifted && !args.fold_singleton_bounds
        && args.audit_tolerance==1e-9 && args.cone_tolerance==1e-8),
        "common-KKT stopping requires an unshifted generic capture and fixed 1e-9 relative/1e-8 cone audit gates");
    const auto canonical=s::canonical(snapshot,args.fold_singleton_bounds);
    s::require(args.halpern=="off" || (args.common_kkt_stop && args.execution_blocks>0
        && std::all_of(canonical.Q.values.begin(),canonical.Q.values.end(),[](double v){return v==0.0;})),
        "Halpern requires common-KKT, explicit positive cooperative blocks and exactly zero Q");
    s::require(!args.l1_prox || (args.common_kkt_stop && args.execution_blocks>0 && args.halpern=="off"),
        "L1 prox requires common-KKT, positive explicit blocks and Halpern off");
    s::require(!args.l1_weight_explicit || (args.l1_prox && (args.l1_weight_mode==SPACEPDHCG_CUDA_L1_WEIGHT_CANCEL_GLOBAL
        ||(std::isfinite(1.0/args.l1_weight)&&1.0/args.l1_weight>0.0))),
        "L1 weight requires L1 prox and a representable positive reciprocal or cancel-global policy");
    std::optional<s::l1::Reduction> l1_map;
    if(args.l1_prox) {
        l1_map=s::l1::detect(snapshot,canonical);
        s::require(!l1_map->pairs.empty(),"L1 prox requested but no provable isolated epigraph pair exists");
    }
    s::require(!args.mass_eliminate || (args.l1_prox && args.l1_weight==1.0),
        "mass elimination requires exact L1 with unit weight");
    std::vector<spacepdhcg_cuda_mass_node> mass_nodes;
    if(args.mass_eliminate)mass_nodes=s::mass::detect(snapshot,canonical,*l1_map);
    const double conversion_seconds=elapsed(prepare);
    const auto initial_begin=Clock::now();std::optional<s::InitialPoint> initial;
    std::optional<s::WarmSnapshotPoint> warm;
    if(!args.warm_start_source.empty()) {
        const auto predecessor=s::read(s::file_bytes(args.warm_start_source,128ULL*1024*1024));
        warm=s::warm_snapshot_point(s::file_bytes(args.initial_point,128ULL*1024*1024),predecessor,snapshot,canonical);
        initial=warm->successor;
    } else if(!args.initial_point.empty())initial=s::initial_point(s::file_bytes(args.initial_point,128ULL*1024*1024),snapshot,canonical);
    const double initial_validation_seconds=initial?elapsed(initial_begin):0;
    const auto library=args.validate_only?std::string{}:library_path();
    const auto library_sha=args.validate_only?std::string{}:s::sha256(s::file_bytes(library));
    // Everything above, including malformed-input rejection, is CPU-only.
    std::cout<<std::setprecision(21)<<std::boolalpha;
    std::cout<<"PERSISTENT_REPLAY_META {\"input_sha256\":";json_string(snapshot.input_sha256);
    std::cout<<",\"source_commit\":";json_string(SPACEPDHCG_SOURCE_COMMIT);
    if(SPACEPDHCG_SOURCE_TREE_SHA256[0]!='\0') {
        std::cout<<",\"source_commit_scope\":\"uncommitted_frozen_source_tree\",\"source_dirty\":true";
        std::cout<<",\"source_tree_sha256\":";json_string(SPACEPDHCG_SOURCE_TREE_SHA256);
        std::cout<<",\"base_commit\":";json_string(SPACEPDHCG_SOURCE_BASE_COMMIT);
    }
    std::cout<<",\"source_sha256\":";json_string(SPACEPDHCG_SNAPSHOT_SOURCE_SHA256);
    std::cout<<",\"initial_point_role\":";json_string(warm?"qualified_predecessor_iterate":initial?"qualified_target_point":"none");
    std::cout<<",\"coordinate_system\":\"original\",\"slack_source\":\"reconstructed_h_minus_Gx\",\"shifted\":"<<snapshot.shifted;
    std::cout<<",\"native_objective_coordinates\":";json_string(snapshot.shifted?"translated":"original");metric("objective_offset",snapshot.offset);
    std::cout<<",\"stopping_policy\":";json_string(args.common_kkt_stop?"gpu_common_kkt_original_equations":"native_absolute_natural_residual");
    std::cout<<",\"common_kkt_initial_check\":"<<args.common_kkt_stop<<",\"common_kkt_recovery_disabled\":"<<args.common_kkt_stop;
    std::cout<<",\"requested_execution_blocks\":";if(args.execution_blocks<0)std::cout<<"null";else std::cout<<args.execution_blocks;
    std::cout<<",\"halpern_mode\":";json_string(args.halpern);
    std::cout<<",\"l1_prox\":"<<args.l1_prox;
    std::cout<<",\"mass_eliminate\":"<<args.mass_eliminate;
    std::cout<<",\"requested_l1_weight\":";if(args.l1_weight_mode==SPACEPDHCG_CUDA_L1_WEIGHT_FIXED)number(args.l1_weight);else std::cout<<"null";
    std::cout<<",\"l1_weight_policy\":";json_string(args.l1_weight_mode==SPACEPDHCG_CUDA_L1_WEIGHT_CANCEL_GLOBAL
        ?"cancel_global_normalization":args.l1_weight_explicit?"fixed_user_input":"unit_default");
    if(l1_map) {
        std::cout<<",\"l1_representation\":\"original_layout_with_masked_working_operator\",\"l1_pairs\":"<<l1_map->pairs.size();
        std::cout<<",\"l1_active_variables\":"<<snapshot.n-l1_map->pairs.size()-mass_nodes.size()
            <<",\"l1_active_rows\":"<<canonical.A.rows+canonical.F.rows-2*l1_map->pairs.size()-mass_nodes.size();
        std::cout<<",\"l1_scaling\":";json_string(args.mass_eliminate?"direct_original_coordinate_diagonal_abs_sums_no_B_O_D_R":
            "reduced_Ruiz_B_and_O_norm_including_smooth_c_over_D_and_lambda_over_target_D");
        std::cout<<",\"l1_weight\":";if(args.l1_weight_mode==SPACEPDHCG_CUDA_L1_WEIGHT_CANCEL_GLOBAL)std::cout<<"null";else number(args.l1_weight);
        std::cout<<",\"l1_weight_rule\":\"primal_eta_div_omega_dual_eta_times_omega_cancel_global_chooses_O_div_B_once_after_scaling\"";
        std::cout<<",\"l1_natural_residual_metric\":";json_string(args.mass_eliminate?
            "original_problem_diagnostic_with_direct_retained_steps_and_unit_eliminated_placeholders":"legacy_unweighted_original_problem_diagnostic");
        std::cout<<",\"l1_dual_completion\":\"strict_sign_for_nonzero_v_clipped_retained_gradient_only_at_exact_zero\"";
        std::cout<<",\"l1_initial_point\":\"original_x_t_y_z_checked_before_any_completion\",\"l1_map\":[";
        for(std::size_t i=0;i<l1_map->pairs.size();++i) {
            if(i)std::cout<<',';const auto p=l1_map->pairs[i];
            std::cout<<'['<<p.epigraph<<','<<p.variable<<','<<p.positive_row<<','<<p.negative_row<<',';number(p.lambda);std::cout<<']';
        }
        std::cout<<']';
    }
    if(args.mass_eliminate) {
        std::cout<<",\"mass_metric\":\"upward_absolute_sums_SOC_tied_downward_theta_division\",\"mass_theta\":0.95";
        std::cout<<",\"mass_metric_refresh\":\"every_solve\"";
        std::cout<<",\"mass_affine_convention\":\"linear_prefix_and_once_shifted_upper_bounds\",\"mass_map\":[";
        for(std::size_t i=0;i<mass_nodes.size();++i) {
            if(i)std::cout<<',';const auto node=mass_nodes[i];
            std::cout<<'['<<node.mass_variable<<','<<node.equality_row<<','<<node.gamma_variable<<','<<node.virtual_variable<<']';
        }
        std::cout<<']';
    }
    if(args.halpern!="off") {
        std::cout<<",\"halpern_output_point\":\"proximal_T_not_anchored_working_state\",\"halpern_restart_frequency\":200";
        std::cout<<",\"halpern_weight_guard_residuals\":\"common_normalized_primal_dual\",\"halpern_spectral_policy\":\"unchanged_20_power_heuristic_not_a_proved_upper_bound\"";
        std::cout<<",\"halpern_repeat_state\":\"fresh_anchor_and_unit_weight_each_solve_including_bootstrap\"";
    }
    std::cout<<",\"convexity_evidence\":";json_string(canonical.convexity_evidence);
    std::cout<<",\"variables\":"<<snapshot.n<<",\"equalities\":"<<snapshot.p<<",\"inequalities\":"<<snapshot.m;
    std::cout<<",\"soc_count\":"<<snapshot.soc.size()<<",\"symmetric_quadratic_entries\":"<<canonical.Q.values.size();
    std::cout<<",\"fold_singleton_bounds\":"<<args.fold_singleton_bounds<<",\"representation\":";
    json_string(args.fold_singleton_bounds?"exact_singletons_as_native_variable_bounds":"all_nonnegative_rows_as_scalar_duals");
    std::cout<<",\"singleton_rows\":"<<canonical.singleton_rows<<",\"folded_rows\":"<<canonical.folded_bounds.size();
    std::cout<<",\"retained_nonexact_singleton_ratios\":"<<canonical.retained_nonexact_ratios<<",\"retained_out_of_range_singleton_ratios\":"<<canonical.retained_out_of_range_ratios;
    std::cout<<",\"native_scalar_rows\":"<<canonical.A.rows<<",\"native_affine_rows\":"<<canonical.F.rows;
    std::cout<<",\"folded_dual_activity_rule\":\"exact_local_primal_equals_tight_native_bound\"";
    metric("conversion_seconds",conversion_seconds);metric("audit_tolerance",args.audit_tolerance);metric("cone_tolerance",args.cone_tolerance);
    std::cout<<",\"repeat_mode\":";json_string(args.mode);
    std::cout<<",\"initial_point_supplied\":"<<bool(initial);
    if(initial) {
        std::cout<<",\"initial_point_sha256\":";json_string(initial->file_sha256);
        std::cout<<",\"initial_point_coordinates\":";json_string(initial->coordinates);
        std::cout<<",\"initial_point_repeat_policy\":\"full_reset_and_seed_every_cold_or_reuse_repeat_and_only_fresh_full_retained_workspace\"";
        std::cout<<",\"initial_native_measurement\":\"one_iteration_bootstrap_if_needed_then_full_reset_seed_residual_only\"";
        metric("initial_point_audit_tolerance",s::initial_point_tolerance);metric("initial_point_cone_tolerance",s::initial_point_cone_tolerance);
    }
    metric("initial_point_validation_seconds",initial_validation_seconds);
    std::cout<<",\"repeat_seconds_scope\":\"workspace_setup_solve_download_audit_cleanup_includes_bootstrap_and_prestep_output_excludes_shared_initialization_and_final_record_output\"";
    std::cout<<",\"setup_seconds_scope\":\"workspace_creation_repeat_preparation_and_seed_measurement_including_bootstrap_and_prestep_output\"";
    std::cout<<",\"peak_workspace_bytes_scope\":\"native_workspace_only_excludes_borrowed_input_output_storage\"";
    std::cout<<",\"scaling_policy\":\"refresh_if_needed\",\"matrix_change_threshold\":0.25,\"vector_change_threshold\":0.5,\"scaling_reuse_limit\":4,\"residual_check_frequency\":25";
    std::cout<<",\"complementarity_gate\":\"max_absolute_scalar_or_SOC_dot_over_max_1_abs_primal_objective_abs_dual_objective\"";
    std::cout<<",\"captured_qoco_settings_used_by_persistent\":false,\"validate_only\":"<<args.validate_only;
    if(!args.validate_only) {
        std::cout<<",\"library_path\":";json_string(library);
        std::cout<<",\"library_sha256\":";json_string(library_sha);
    }
    std::cout<<"}\n";
    if(warm) {
        std::cout<<"PERSISTENT_REPLAY_WARM_SOURCE {\"predecessor_sha256\":";json_string(warm->predecessor_sha256);
        std::cout<<",\"predecessor_point_sha256\":";json_string(warm->predecessor.file_sha256);
        std::cout<<",\"successor_sha256\":";json_string(snapshot.input_sha256);
        std::cout<<",\"scope\":\"fresh_successor_workspace_with_predecessor_iterate\",\"retained_numeric_update_measured\":false";
        std::cout<<",\"same_topology\":true,\"same_variable_and_row_meanings_required_from_caller\":true,\"changed_coefficients\":true";
        std::cout<<",\"original_xyz_bits_preserved\":true,\"successor_slack_source\":\"reconstructed_h_minus_Gx\"";
        std::cout<<",\"predecessor_audit\":";audit_json(warm->predecessor.supplied_audit);
        std::cout<<",\"predecessor_qualified\":"<<warm->predecessor.roundtrip_audit.qualified;
        std::cout<<",\"successor_initial_audit\":";audit_json(initial->reconstructed_audit);
        std::cout<<",\"successor_initial_qualified\":"<<initial->reconstructed_audit.qualified<<"}\n";
    }
    if(initial) {
        std::cout<<"PERSISTENT_REPLAY_INITIAL_POINT {\"point_sha256\":";json_string(initial->file_sha256);
        std::cout<<",\"coordinate_system\":\"original\",\"phase\":\"cpu_initial_point_validation\",\"native_residual_measured_in_this_phase\":false,\"strict_reconstruction_is_import_gate\":false";
        std::cout<<",\"supplied_audit\":";audit_json(initial->supplied_audit);
        std::cout<<",\"mapped_reference_audit\":";audit_json(initial->roundtrip_audit);
        std::cout<<",\"supplied_qualified\":"<<initial->supplied_audit.qualified<<",\"mapped_reference_qualified\":"<<initial->roundtrip_audit.qualified;
        std::cout<<",\"strict_reconstructed_audit\":";audit_json(initial->reconstructed_audit);
        std::cout<<",\"strict_reconstructed_qualified\":"<<initial->reconstructed_audit.qualified;
        std::cout<<",\"reference_folded_multipliers_are_audit_only\":true,\"x_solver\":";vector_json(initial->primal);
        std::cout<<",\"dual_solver\":";vector_json(initial->dual);
        std::cout<<",\"x\":";vector_json(initial->reference.x);std::cout<<",\"y\":";vector_json(initial->reference.y);
        std::cout<<",\"z\":";vector_json(initial->reference.z);std::cout<<",\"s\":";vector_json(initial->reference.s);
        std::cout<<"}\n";
    }
    std::cout<<std::flush;if(args.validate_only)return 0;
    // The test storage helper binds device zero explicitly. Select it deliberately.
    test::cuda_require(cudaSetDevice(0),"select CUDA device zero");
    std::unique_ptr<Owner> owner;
    for(int repeat=0;repeat<args.repeats;++repeat) {
        const auto complete_begin=Clock::now();const auto setup_begin=Clock::now();
        const bool fresh=!owner || args.mode=="cold";
        if(fresh) {owner.reset();owner=std::make_unique<Owner>(snapshot,canonical);}
        auto& p=owner->storage;auto* w=owner->workspace;
        if(fresh && args.execution_blocks>=0) {
            api(spacepdhcg_cuda_workspace_wait(w),"complete creation before explicit execution strategy",w);
            api(spacepdhcg_cuda_workspace_set_execution_blocks(w,args.execution_blocks),"select explicit execution blocks",w);
        }
        if(fresh && args.common_kkt_stop) {
            api(spacepdhcg_cuda_workspace_wait(w),"complete creation before common policy setup",w);
            const spacepdhcg_cuda_common_kkt_options policy{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,1,snapshot.p,0,1e-9,1e-8};
            api(spacepdhcg_cuda_workspace_set_common_kkt_options(w,&policy),"configure GPU common-KKT stopping",w);
        }
        if(fresh && args.halpern!="off") {
            const spacepdhcg_cuda_halpern_options policy{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,args.halpern=="plain"?1:2,{0,0}};
            api(spacepdhcg_cuda_workspace_set_halpern_options(w,&policy),"configure experimental Halpern",w);
        }
        if(fresh && l1_map) {
            std::vector<spacepdhcg_cuda_l1_pair> pairs;
            for(const auto pair:l1_map->pairs)pairs.push_back({pair.epigraph,pair.variable,pair.positive_row,pair.negative_row});
            const spacepdhcg_cuda_l1_options policy{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,1,static_cast<int>(pairs.size()),0};
            api(spacepdhcg_cuda_workspace_set_l1_options(w,&policy,pairs.data()),"configure exact L1 GPU prox",w);
            if(args.l1_weight_explicit) {
                const spacepdhcg_cuda_l1_weight_options weight{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,args.l1_weight_mode,args.l1_weight};
                api(spacepdhcg_cuda_workspace_set_l1_weight(w,&weight),"configure fixed L1 reciprocal weight",w);
            }
        }
        if(fresh && args.mass_eliminate) {
            const spacepdhcg_cuda_mass_options policy{SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,1,static_cast<int>(mass_nodes.size()),0};
            api(spacepdhcg_cuda_workspace_set_mass_options(w,&policy,mass_nodes.data()),"configure exact causal mass elimination",w);
        }
        if(!fresh) {
            if(args.mode=="reuse")api(spacepdhcg_cuda_workspace_reset_async(w,SPACEPDHCG_CUDA_RESET_ITERATES,p.exchange.consumer_stream),"reset iterates",w);
            else api(spacepdhcg_cuda_workspace_warm_start_async(w,SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED,nullptr,p.exchange.consumer_stream),"full retained warm start",w);
            api(spacepdhcg_cuda_workspace_wait(w),"prepare repeat",w);
        }
        const bool initial_applied=initial && (fresh || args.mode=="reuse");
        double seed_seconds=0;
        if(initial_applied) {
            spacepdhcg_cuda_diagnostics epoch{};
            api(spacepdhcg_cuda_workspace_diagnostics(w,&epoch),"read solve epoch only",w);
            const bool bootstrap_needed=epoch.solve_epoch==0;
            SolveWait bootstrap_wait{};spacepdhcg_cuda_diagnostics bootstrap{};
            if(bootstrap_needed) {
                bootstrap_wait=bounded_solve(w,p,test::solve_options(args.tolerance,1),args.deadline);
                api(spacepdhcg_cuda_workspace_diagnostics(w,&bootstrap),"bootstrap diagnostics",w);
                std::cout<<"PERSISTENT_REPLAY_BOOTSTRAP {\"repeat\":"<<repeat<<",\"phase\":\"unseeded_report_epoch_bootstrap\",\"iteration_limit\":1,\"iterations\":"<<bootstrap.iterations;
                std::cout<<",\"recovery_iterations\":"<<bootstrap.recovery_iterations<<",\"termination\":"<<bootstrap.termination<<",\"api_status\":"<<bootstrap_wait.status;
                std::cout<<",\"deadline_requested\":"<<bootstrap_wait.deadline_requested<<",\"cancellation_api_status\":"<<bootstrap_wait.cancellation_status;
                metric("wall_seconds",bootstrap_wait.wall_seconds);metric("scaling_seconds",bootstrap.scaling_seconds);metric("solve_seconds",bootstrap.solve_seconds);
                std::cout<<"}\n"<<std::flush;
                s::require(bootstrap_wait.status==SPACEPDHCG_CUDA_SUCCESS && bootstrap.iterations<=1
                    && bootstrap.termination!=SPACEPDHCG_CUDA_TERMINATION_CANCELLED
                    && bootstrap.termination!=SPACEPDHCG_CUDA_TERMINATION_NUMERICAL_FAILURE,"bootstrap failed before seeded residual measurement");
            }
            const auto reset_begin=Clock::now();
            api(spacepdhcg_cuda_workspace_reset_async(w,SPACEPDHCG_CUDA_RESET_FULL,p.exchange.consumer_stream),"full reset after bootstrap",w);
            api(spacepdhcg_cuda_workspace_wait(w),"wait for full reset",w);
            const double reset_seconds=elapsed(reset_begin);const auto seed_begin=Clock::now();
            p.primal.upload(initial->primal,p.stream);p.dual.upload(initial->dual,p.stream);
            api(spacepdhcg_cuda_workspace_warm_start_async(w,SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL,&p.exchange.iterates,p.exchange.consumer_stream),"import validated primal-dual start",w);
            api(spacepdhcg_cuda_workspace_wait(w),"wait for imported start",w);
            seed_seconds=elapsed(seed_begin);
            spacepdhcg_cuda_pointer_snapshot pointers{};
            api(spacepdhcg_cuda_workspace_pointer_snapshot(w,&pointers),"internal iterate addresses",w);
            const auto verify_begin=Clock::now();
            const auto before_x=internal_copy(pointers.primal,initial->primal.size(),p.stream),before_dual=internal_copy(pointers.dual,initial->dual.size(),p.stream);
            s::require(s::same_fp64_bits(before_x,initial->primal) && s::same_fp64_bits(before_dual,initial->dual),"native warm start differs from mapped seed");
            double verify_seconds=elapsed(verify_begin);const auto residual_begin=Clock::now();
            api(spacepdhcg_cuda_workspace_residuals_async(w,p.exchange.consumer_stream),"seeded residual only",w);
            api(spacepdhcg_cuda_workspace_wait(w),"wait for seeded residual only",w);
            const double residual_seconds=elapsed(residual_begin);
            spacepdhcg_cuda_diagnostics measured{};
            api(spacepdhcg_cuda_workspace_diagnostics(w,&measured),"seeded residual diagnostics",w);
            s::require(measured.solve_epoch>0 && measured.termination==SPACEPDHCG_CUDA_TERMINATION_UNSPECIFIED
                && measured.state==SPACEPDHCG_CUDA_WARM_STARTED,"seeded residual inherited an accepted termination or lacked a report epoch");
            const auto recheck_begin=Clock::now();
            s::require(s::same_fp64_bits(before_x,internal_copy(pointers.primal,initial->primal.size(),p.stream))
                && s::same_fp64_bits(before_dual,internal_copy(pointers.dual,initial->dual.size(),p.stream)),"residual-only measurement changed internal seed");
            verify_seconds+=elapsed(recheck_begin);
            std::cout<<"PERSISTENT_REPLAY_PRESTEP {\"repeat\":"<<repeat<<",\"phase\":\"seeded_residual_only\",\"seeded_iterations\":0,\"native_pre_step_measured\":true";
            std::cout<<",\"termination\":"<<measured.termination<<",\"state\":"<<measured.state<<",\"solve_epoch\":"<<measured.solve_epoch;
            std::cout<<",\"warm_start_mode\":"<<measured.warm_start_mode<<",\"warm_start_accepted\":"<<bool(measured.warm_start_accepted);
            std::cout<<",\"reset_api_status\":0,\"seed_api_status\":0,\"residual_api_status\":0";
            std::cout<<",\"inherited_report_iterations\":"<<measured.iterations<<",\"inherited_report_counters_are_seed_work\":false,\"native_seed_verified_unchanged\":true";
            std::cout<<",\"bootstrap_performed\":"<<bootstrap_needed<<",\"bootstrap_iterations\":"<<bootstrap.iterations<<",\"bootstrap_termination\":"<<bootstrap.termination<<",\"bootstrap_api_status\":"<<bootstrap_wait.status;
            std::cout<<",\"bootstrap_deadline_requested\":"<<bootstrap_wait.deadline_requested;
            metric("bootstrap_wall_seconds",bootstrap_wait.wall_seconds);metric("bootstrap_scaling_seconds",bootstrap.scaling_seconds);metric("bootstrap_solve_seconds",bootstrap.solve_seconds);
            metric("reset_seconds",reset_seconds);metric("seed_seconds",seed_seconds);metric("residual_only_wall_seconds",residual_seconds);metric("internal_iterate_verification_seconds",verify_seconds);
            metric("native_objective",measured.objective);metric("native_natural_residual",measured.natural_residual_inf);
            metric("native_scalar_primal_violation",measured.scalar_primal_violation_inf);metric("native_box_violation",measured.box_violation_inf);
            metric("native_affine_cone_distance",measured.affine_cone_distance_inf);metric("native_stationarity",measured.stationarity_inf);metric("native_complementarity",measured.complementarity_inf);
            std::cout<<"}\n"<<std::flush;
        }
        spacepdhcg_cuda_diagnostics before{};
        api(spacepdhcg_cuda_workspace_diagnostics(w,&before),"pre-solve state only",w);
        if(initial_applied)s::require(before.state==SPACEPDHCG_CUDA_WARM_STARTED && before.warm_start_accepted
            && before.warm_start_mode==SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL,"C ABI did not accept explicit primal-dual seed");
        const double setup_seconds=elapsed(setup_begin);
        const auto options=test::solve_options(args.tolerance,args.iterations);
        const auto waited=bounded_solve(w,p,options,args.deadline);
        const auto status=waited.status;const double wall=waited.wall_seconds;
        spacepdhcg_cuda_diagnostics diagnostic{};
        api(spacepdhcg_cuda_workspace_diagnostics(w,&diagnostic),"diagnostics",w);
        spacepdhcg_cuda_common_kkt_diagnostics common{};
        if(args.common_kkt_stop)api(spacepdhcg_cuda_workspace_common_kkt_diagnostics(w,&common),"common-KKT diagnostics",w);
        spacepdhcg_cuda_halpern_diagnostics halpern{};
        spacepdhcg_cuda_l1_diagnostics l1{};
        if(args.l1_prox)api(spacepdhcg_cuda_workspace_l1_diagnostics(w,&l1),"L1 diagnostics",w);
        spacepdhcg_cuda_l1_weight_diagnostics l1_weight{};
        if(args.l1_prox)api(spacepdhcg_cuda_workspace_l1_weight_diagnostics(w,&l1_weight),"L1 weight diagnostics",w);
        spacepdhcg_cuda_mass_diagnostics mass{};
        if(args.mass_eliminate)api(spacepdhcg_cuda_workspace_mass_diagnostics(w,&mass),"mass diagnostics",w);
        if(args.halpern!="off")api(spacepdhcg_cuda_workspace_halpern_diagnostics(w,&halpern),"Halpern diagnostics",w);
        const auto download=Clock::now();const auto primal=p.primal.download(p.stream),dual=p.dual.download(p.stream);
        std::vector<double> mass_steps;
        if(args.mass_eliminate) {
            spacepdhcg_cuda_pointer_snapshot pointers{};
            api(spacepdhcg_cuda_workspace_pointer_snapshot(w,&pointers),"mass step pointer snapshot",w);
            mass_steps.resize(snapshot.n+canonical.A.rows+canonical.F.rows);
            test::cuda_require(cudaMemcpy(mass_steps.data(),reinterpret_cast<const void*>(pointers.scaling),
                mass_steps.size()*sizeof(double),cudaMemcpyDeviceToHost),"mass direct-step readback");
        }
        const double download_seconds=elapsed(download);const auto audit_begin=Clock::now();
        const auto original=s::original_vectors(snapshot,canonical,primal,dual);
        const auto quality=s::audit(snapshot,original,args.audit_tolerance,args.cone_tolerance);
        const double audit_seconds=elapsed(audit_begin);
        const bool optimal=diagnostic.termination==SPACEPDHCG_CUDA_TERMINATION_OPTIMAL;
        const auto cleanup_begin=Clock::now();
        // Independently unqualified iterates never seed a retained solve.
        if(!optimal || !quality.qualified || args.mode=="cold")owner.reset();
        const double cleanup_seconds=elapsed(cleanup_begin),repeat_seconds=elapsed(complete_begin);
        std::cout<<"PERSISTENT_REPLAY {\"repeat\":"<<repeat<<",\"termination\":"<<diagnostic.termination<<",\"termination_name\":";json_string(termination_name(diagnostic.termination));
        std::cout<<",\"api_status\":"<<status<<",\"solver_optimal\":"<<optimal;
        std::cout<<",\"execution_blocks\":";if(args.execution_blocks<0)std::cout<<"null";else std::cout<<args.execution_blocks;
        std::cout<<",\"fresh_workspace\":"<<fresh<<",\"within_requested_wall_deadline\":"<<(wall<=args.deadline);
        std::cout<<",\"initial_point_applied\":"<<initial_applied<<",\"state_before_solve\":"<<before.state;
        std::cout<<",\"solve_epoch_before\":"<<before.solve_epoch<<",\"warm_start_mode_before\":"<<before.warm_start_mode<<",\"warm_start_accepted_before\":"<<bool(before.warm_start_accepted);
        metric("initial_point_upload_and_warm_start_seconds",seed_seconds);
        std::cout<<",\"folded_dual_reconstruction_supported\":"<<original.folded_dual_reconstruction_supported<<",\"reconstructed_bound_duals\":"<<original.reconstructed_bound_duals;
        std::cout<<",\"off_contact_bound_normals\":"<<original.off_contact_bound_normals<<",\"off_contact_one_ulp_normals\":"<<original.off_contact_one_ulp_normals;
        metric("max_off_contact_bound_distance",original.max_off_contact_bound_distance);
        std::cout<<",\"kkt_qualified_original\":"<<quality.qualified<<",\"qualified_original\":"<<(optimal&&quality.qualified);
        std::cout<<",\"deadline_requested\":"<<waited.deadline_requested<<",\"cancellation_api_status\":"<<waited.cancellation_status;
        std::cout<<",\"iteration_limit\":"<<args.iterations<<",\"iterations\":"<<diagnostic.iterations<<",\"recovery_iterations\":"<<diagnostic.recovery_iterations;
        metric("requested_tolerance",args.tolerance);metric("deadline_seconds",args.deadline);
        metric("setup_seconds",setup_seconds);metric("wall_seconds",wall);metric("scaling_seconds",diagnostic.scaling_seconds);
        metric("solve_seconds",diagnostic.solve_seconds);metric("recovery_seconds",diagnostic.recovery_seconds);
        metric("download_seconds",download_seconds);metric("audit_seconds",audit_seconds);metric("cleanup_seconds",cleanup_seconds);metric("repeat_seconds",repeat_seconds);
        metric("native_objective",diagnostic.objective);metric("native_natural_residual",diagnostic.natural_residual_inf);
        if(args.l1_prox) {
            std::cout<<",\"l1\":{\"enabled\":"<<bool(l1.enabled)<<",\"valid\":"<<bool(l1.valid)<<",\"finite\":"<<bool(l1.finite);
            std::cout<<",\"pairs\":"<<l1.pairs<<",\"active_variables\":"<<l1.active_variables<<",\"active_rows\":"<<l1.active_rows;
            std::cout<<",\"retained_variables\":"<<l1.retained_variables<<",\"retained_rows\":"<<l1.retained_rows;
            std::cout<<",\"updates\":"<<l1.updates<<",\"completions\":"<<l1.completions;
            metric("eta",l1.eta);metric("bound_scale",l1.bound_scale);metric("objective_scale",l1.objective_scale);
            metric("omega",l1_weight.omega);metric("primal_base_step",l1_weight.primal_base_step);metric("dual_base_step",l1_weight.dual_base_step);
            std::cout<<",\"weight_valid\":"<<bool(l1_weight.valid)<<",\"weight_mode\":"<<l1_weight.mode;
            metric("minimum_threshold",l1.minimum_threshold);metric("maximum_threshold",l1.maximum_threshold);std::cout<<'}';
        }
        if(args.mass_eliminate) {
            std::cout<<",\"mass\":{\"enabled\":"<<bool(mass.enabled)<<",\"valid\":"<<bool(mass.valid)<<",\"finite\":"<<bool(mass.finite);
            std::cout<<",\"nodes\":"<<mass.nodes<<",\"active_variables\":"<<mass.active_variables<<",\"active_rows\":"<<mass.active_rows;
            std::cout<<",\"retained_variables\":"<<mass.retained_variables<<",\"retained_rows\":"<<mass.retained_rows;
            std::cout<<",\"updates\":"<<mass.updates<<",\"completions\":"<<mass.completions;
            metric("theta",mass.theta);metric("row_factor_upper",mass.row_factor_upper);metric("column_factor_upper",mass.column_factor_upper);
            metric("norm_squared_upper",mass.norm_squared_upper);metric("minimum_primal_step",mass.minimum_primal_step);metric("maximum_primal_step",mass.maximum_primal_step);
            metric("minimum_dual_step",mass.minimum_dual_step);metric("maximum_dual_step",mass.maximum_dual_step);
            metric("minimum_threshold",mass.minimum_threshold);metric("maximum_threshold",mass.maximum_threshold);
            std::cout<<",\"steps_original_layout\":";vector_json(mass_steps);
            std::cout<<",\"step_layout\":\"primal_then_scalar_then_affine; eliminated_slots_are_unit_dummies\"}";
        }
        if(args.halpern!="off") {
            std::cout<<",\"halpern\":{\"mode\":"<<halpern.mode<<",\"valid\":"<<bool(halpern.valid)<<",\"finite\":"<<bool(halpern.finite);
            std::cout<<",\"updates\":"<<halpern.updates<<",\"inner_iterations\":"<<halpern.inner_iterations<<",\"restarts\":"<<halpern.restarts;
            std::cout<<",\"weight_updates\":"<<halpern.weight_updates<<",\"weight_fallbacks\":"<<halpern.weight_fallbacks;
            std::cout<<",\"metric_evaluations\":"<<halpern.metric_evaluations<<",\"last_restart_iteration\":"<<halpern.last_restart_iteration;
            std::cout<<",\"epoch_reference_iteration\":"<<halpern.epoch_reference_iteration;
            metric("primal_weight",halpern.primal_weight);metric("minimum_primal_weight",halpern.minimum_primal_weight);metric("maximum_primal_weight",halpern.maximum_primal_weight);
            metric("fixed_point_error",halpern.fixed_point_error);metric("epoch_initial_error",halpern.epoch_initial_error);metric("eta",halpern.eta);std::cout<<'}';
        }
        if(args.common_kkt_stop) {
            std::cout<<",\"gpu_common_kkt\":{\"enabled\":"<<bool(common.enabled)<<",\"valid\":"<<bool(common.valid)
                <<",\"finite\":"<<bool(common.finite)<<",\"passes\":"<<bool(common.passes)
                <<",\"evaluations\":"<<common.evaluations<<",\"evaluated_iteration\":"<<common.evaluated_iteration
                <<",\"evaluation_clock_cycles\":"<<common.evaluation_clock_cycles;
            metric("primal",common.primal_relative);metric("dual",common.dual_relative);metric("gap",common.gap_relative);
            metric("block_complementarity_normalized",common.block_complementarity_relative);
            metric("primal_cone_violation",common.primal_cone_violation);metric("dual_cone_violation",common.dual_cone_violation);
            metric("primal_absolute",common.primal_absolute);metric("conic_equation_absolute",common.conic_equation_absolute);
            metric("stationarity_absolute",common.dual_absolute);metric("objective",common.objective);metric("dual_objective",common.dual_objective);
            std::cout<<'}';
        }
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
