// Identical-input diagnostic for the pinned upstream C API, outside the production backend.
#include "persistent_snapshot.hpp"
#include <pdhcg.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <optional>

#ifndef SPACEPDHCG_SOURCE_COMMIT
#define SPACEPDHCG_SOURCE_COMMIT "unrecorded"
#endif
#ifndef SPACEPDHCG_UPSTREAM_REPLAY_SOURCE_SHA256
#define SPACEPDHCG_UPSTREAM_REPLAY_SOURCE_SHA256 "unrecorded"
#endif

namespace s = spacepdhcg::snapshot;
using Clock = std::chrono::steady_clock;
namespace {
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
void number(long double value) {
    if (std::isfinite(value)) std::cout << value;
    else std::cout << "null";
}
void json_string(const std::string& value) {
    std::cout << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') std::cout << '\\' << c;
        else if (c < 32) {
            constexpr char digits[] = "0123456789abcdef";
            std::cout << "\\u00" << digits[c >> 4] << digits[c & 15];
        } else std::cout << c;
    }
    std::cout << '"';
}
void metric(const char* name, long double value, bool first = false) {
    if (!first) std::cout << ',';
    json_string(name); std::cout << ':'; number(value);
}
void vector_json(const std::vector<double>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        number(values[i]);
    }
    std::cout << ']';
}
void audit_json(const s::Quality& q) {
    std::cout << '{';
    metric("primal", q.primal, true); metric("dual", q.dual); metric("gap", q.gap);
    metric("objective", q.objective); metric("dual_objective", q.dual_objective);
    metric("signed_gap", q.signed_gap); metric("equality_absolute", q.equality_absolute);
    metric("conic_equation_absolute", q.conic_equation_absolute);
    metric("stationarity_absolute", q.stationarity_absolute);
    metric("primal_cone_violation", q.primal_cone_violation);
    metric("dual_cone_violation", q.dual_cone_violation);
    metric("complementarity", q.complementarity);
    metric("global_complementarity", q.global_complementarity);
    metric("block_complementarity_normalized", q.block_complementarity_normalized);
    metric("global_complementarity_normalized", q.global_complementarity_normalized);
    std::cout << ",\"finite\":" << q.finite << ",\"qualified\":" << q.qualified << '}';
}
struct Arguments {
    std::string path, initial_point;
    int iterations{};
    double deadline{};
    bool validate_only{}, omit_zero_quadratic{};
};
Arguments arguments(int argc, char** argv) {
    s::require(argc >= 2, "usage: upstream_snapshot_replay SNAPSHOT --iterations N --deadline-seconds S [--initial-point PATH], or SNAPSHOT --validate-only");
    Arguments out; out.path = argv[1];
    std::vector<std::string> seen;
    for (int i = 2; i < argc; ++i) {
        const std::string name(argv[i]);
        s::require(std::find(seen.begin(), seen.end(), name) == seen.end(), "duplicate option");
        seen.push_back(name);
        if (name == "--validate-only") { out.validate_only = true; continue; }
        if (name == "--omit-zero-quadratic") { out.omit_zero_quadratic = true; continue; }
        s::require(i + 1 < argc, "missing option value");
        const std::string value(argv[++i]);
        std::size_t used = 0;
        if (name == "--iterations") {
            s::require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos, "invalid iteration limit");
            const auto parsed = std::stoull(value, &used);
            s::require(used == value.size() && parsed > 0 && parsed <= 100000, "iteration limit must be 1..100000");
            out.iterations = static_cast<int>(parsed);
        } else if (name == "--deadline-seconds") {
            out.deadline = std::stod(value, &used);
            s::require(used == value.size() && std::isfinite(out.deadline) && out.deadline > 0 && out.deadline <= 300,
                       "deadline must be finite and in (0,300]");
        } else if (name == "--initial-point") {
            s::require(!value.empty(), "empty initial point path"); out.initial_point = value;
        } else throw std::runtime_error("unknown option: " + name);
    }
    s::require(out.validate_only || (out.iterations > 0 && out.deadline > 0), "solve requires explicit iteration and time limits");
    return out;
}
matrix_desc_t matrix(const s::Csc& value) {
    matrix_desc_t out{};
    out.m = value.rows; out.n = value.columns; out.fmt = matrix_csc;
    out.data.csc = {static_cast<int>(value.values.size()), value.offsets.data(), value.indices.data(), value.values.data()};
    return out;
}
const char* termination_name(termination_reason_t reason) {
    switch (reason) {
        case TERMINATION_REASON_UNSPECIFIED: return "unspecified";
        case TERMINATION_REASON_OPTIMAL: return "optimal";
        case TERMINATION_REASON_PRIMAL_INFEASIBLE: return "primal_infeasible";
        case TERMINATION_REASON_DUAL_INFEASIBLE: return "dual_infeasible";
        case TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED: return "infeasible_or_unbounded";
        case TERMINATION_REASON_TIME_LIMIT: return "time_limit";
        case TERMINATION_REASON_ITERATION_LIMIT: return "iteration_limit";
        case TERMINATION_REASON_USER_INTERRUPT: return "user_interrupt";
        case TERMINATION_REASON_FEAS_POLISH_SUCCESS: return "feasibility_polish_success";
    }
    return "unknown";
}
} // namespace

int main(int argc, char** argv) try {
    const auto process_begin = Clock::now();
    const auto args = arguments(argc, argv);
    const auto prepare = Clock::now();
    const auto snapshot = s::read(s::file_bytes(args.path, 128ULL * 1024 * 1024));
    const auto canonical = s::canonical(snapshot);
    const bool quadratic_exactly_zero = std::all_of(canonical.Q.values.begin(), canonical.Q.values.end(),
        [](double value) { return value == 0.0; });
    s::require(!args.omit_zero_quadratic || quadratic_exactly_zero,
        "--omit-zero-quadratic requires every original quadratic coefficient to be exactly zero");
    std::optional<s::InitialPoint> initial;
    if (!args.initial_point.empty()) initial = s::initial_point(s::file_bytes(args.initial_point, 128ULL * 1024 * 1024), snapshot, canonical);
    const double conversion_seconds = elapsed(prepare);
    auto Q = matrix(canonical.Q), A = matrix(canonical.A), F = matrix(canonical.F);
    std::vector<cone_spec_t> cones;
    for (const auto& cone : canonical.cones) {
        s::require(cone.kind == SPACEPDHCG_CUDA_CONE_SECOND_ORDER, "unsupported upstream cone conversion");
        cones.push_back({CONE_STANDARD_SOC, static_cast<int>(cone.start), static_cast<int>(cone.vector_dimension), 0, nullptr});
    }
    pdhg_parameters_t parameters{};
    set_default_parameters(&parameters);
    parameters.verbose = 0;
    parameters.presolve = false;
    parameters.termination_criteria.eps_optimal_relative = 1e-9;
    parameters.termination_criteria.eps_feasible_relative = 1e-9;
    parameters.termination_criteria.iteration_limit = args.iterations;
    parameters.termination_criteria.time_sec_limit = args.deadline;
    // Upstream defaults, including restart, scaling, inner solves and norm, remain intact.
    // Its native termination is recorded separately from the common original-coordinate audit.
    std::cout << std::setprecision(21) << std::boolalpha;
    std::cout << "UPSTREAM_REPLAY_META {\"input_sha256\":"; json_string(snapshot.input_sha256);
    std::cout << ",\"source_commit\":"; json_string(SPACEPDHCG_SOURCE_COMMIT);
    std::cout << ",\"source_sha256\":"; json_string(SPACEPDHCG_UPSTREAM_REPLAY_SOURCE_SHA256);
    std::cout << ",\"executable_sha256\":"; json_string(s::sha256(s::file_bytes("/proc/self/exe")));
    std::cout << ",\"upstream_commit\":\"167c8b72b4b96d2f94d405b8763e485514192b81\"";
    std::cout << ",\"upstream_patch\":\"0001-free-quadratic-state.patch\",\"backend\":\"pinned_upstream_C_API\"";
    std::cout << ",\"coordinate_system\":\"original\",\"slack_source\":\"reconstructed_h_minus_Gx\"";
    std::cout << ",\"dual_mapping\":\"negate_upstream_Pi_then_scalar_identity_SOC_permutation_and_negation\"";
    std::cout << ",\"shifted\":" << snapshot.shifted << ",\"validate_only\":" << args.validate_only;
    std::cout << ",\"fold_singleton_bounds\":false,\"initial_point_supplied\":" << bool(initial);
    std::cout << ",\"omit_zero_quadratic\":" << args.omit_zero_quadratic;
    std::cout << ",\"quadratic_exactly_zero\":" << quadratic_exactly_zero;
    std::cout << ",\"quadratic_structural_entries\":" << canonical.Q.values.size();
    std::cout << ",\"quadratic_descriptor\":";
    json_string(args.omit_zero_quadratic ? "nullptr_exact_linear_objective" : "original_full_symmetric_CSC");
    std::cout << ",\"variables\":" << snapshot.n << ",\"native_scalar_rows\":" << A.m << ",\"native_affine_rows\":" << F.m;
    metric("conversion_and_initial_validation_seconds", conversion_seconds);
    metric("iteration_limit", args.iterations); metric("time_limit_seconds", args.deadline);
    std::cout << ",\"presolve\":false,\"native_feasible_tolerance\":1e-9,\"native_optimal_tolerance\":1e-9,\"audit_tolerance\":1e-9,\"cone_tolerance\":1e-8";
    metric("native_optimality_norm", parameters.optimality_norm);
    metric("termination_evaluation_frequency", parameters.termination_evaluation_frequency);
    metric("inner_iteration_limit", parameters.inner_solver_parameters.iteration_limit);
    metric("inner_initial_tolerance", parameters.inner_solver_parameters.initial_tolerance);
    metric("inner_minimum_tolerance", parameters.inner_solver_parameters.min_tolerance);
    std::cout << ",\"time_scope\":\"one_shot_native_C_API_including_setup_and_host_transfers_not_pure_GPU_time\"";
    if (initial) {
        std::cout << ",\"initial_point_sha256\":"; json_string(initial->file_sha256);
        std::cout << ",\"initial_audit\":"; audit_json(initial->reconstructed_audit);
    }
    std::cout << "}\n" << std::flush;
    if (args.validate_only) return 0; // No CUDA call or QP allocation above this line.

    const auto setup = Clock::now();
    const double objective_constant = snapshot.offset;
    using Problem = std::unique_ptr<qp_problem_t, decltype(&qp_problem_free)>;
    Problem problem(create_qp_problem(canonical.c.data(), args.omit_zero_quadratic ? nullptr : &Q, nullptr, nullptr, &A,
        canonical.lower.data(), canonical.upper.data(), canonical.variable_lower.data(), canonical.variable_upper.data(),
        &objective_constant, 0, nullptr, F.m ? &F : nullptr, F.m ? canonical.offset.data() : nullptr,
        static_cast<int>(cones.size()), cones.empty() ? nullptr : cones.data()), &qp_problem_free);
    s::require(bool(problem), "upstream rejected canonical input");
    s::require(problem->num_variables == snapshot.n && problem->num_constraints == A.m + F.m, "upstream input dimension mismatch");
    if (initial) {
        auto pi = initial->dual;
        for (auto& value : pi) value = -value;
        set_start_values(problem.get(), initial->primal.data(), pi.data());
    }
    const double setup_seconds = elapsed(setup);
    const auto solve = Clock::now();
    using Result = std::unique_ptr<pdhcg_result_t, decltype(&pdhcg_result_free)>;
    Result result(solve_qp_problem(problem.get(), &parameters), &pdhcg_result_free);
    s::require(bool(result), "upstream returned no result");
    s::require(cudaDeviceSynchronize() == cudaSuccess, "upstream final CUDA synchronization failed");
    const double solve_seconds = elapsed(solve);
    s::require(result->num_variables == snapshot.n && result->num_constraints == A.m + F.m, "upstream result dimension mismatch");
    s::require(result->primal_solution && (A.m + F.m == 0 || result->dual_solution), "upstream returned missing vectors");
    const auto audit_begin = Clock::now();
    std::vector<double> x(result->primal_solution, result->primal_solution + snapshot.n), normal(static_cast<std::size_t>(A.m + F.m));
    for (std::size_t i = 0; i < normal.size(); ++i) normal[i] = -result->dual_solution[i];
    const auto original = s::original_vectors(snapshot, canonical, x, normal);
    const auto quality = s::audit(snapshot, original, 1e-9, 1e-8);
    const double audit_seconds = elapsed(audit_begin);
    const bool accepted = result->termination_reason == TERMINATION_REASON_OPTIMAL && quality.qualified;
    std::cout << "UPSTREAM_REPLAY_RESULT {\"input_sha256\":"; json_string(snapshot.input_sha256);
    std::cout << ",\"termination\":"; json_string(termination_name(result->termination_reason));
    std::cout << ",\"termination_code\":" << result->termination_reason;
    std::cout << ",\"qualified\":" << accepted << ",\"passes_common_kkt_gate\":" << quality.qualified;
    metric("iterations", result->total_count); metric("inner_iterations", result->total_inner_count);
    metric("setup_seconds", setup_seconds); metric("solve_wall_seconds", solve_seconds); metric("audit_seconds", audit_seconds);
    metric("native_cumulative_seconds", result->cumulative_time_sec); metric("native_rescaling_seconds", result->rescaling_time_sec);
    metric("native_primal", result->relative_primal_residual); metric("native_dual", result->relative_dual_residual);
    metric("native_gap", result->relative_objective_gap); metric("native_primal_objective", result->primal_objective_value);
    metric("native_dual_objective", result->dual_objective_value);
    std::cout << ",\"audit\":"; audit_json(quality);
    std::cout << ",\"x_solver\":"; vector_json(x);
    std::cout << ",\"normal_dual_solver\":"; vector_json(normal);
    std::cout << ",\"x\":"; vector_json(original.x); std::cout << ",\"y\":"; vector_json(original.y);
    std::cout << ",\"z\":"; vector_json(original.z); std::cout << ",\"s\":"; vector_json(original.s);
    std::cout << "}\n" << std::flush;
    result.reset(); problem.reset();
    std::cout << "UPSTREAM_REPLAY_DONE {"; metric("process_seconds_before_final_record", elapsed(process_begin), true);
    std::cout << ",\"qualified\":" << accepted << "}\n";
    return 0; // Completed diagnostics retain unqualified results; qualification is explicit in JSON.
} catch (const std::exception& error) {
    std::cerr << "UPSTREAM_REPLAY_ERROR " << error.what() << '\n';
    return 2;
}
