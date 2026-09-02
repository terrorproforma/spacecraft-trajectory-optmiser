/*
 * spacepdhcg_plan: user-facing native planner on the persistent device SCvx stack.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Usage:
 *   spacepdhcg_plan <problem.json> [--output <result.json>] [--quiet]
 *   spacepdhcg_plan --describe <problem.json>   (host-only parse + resolved defaults)
 *   spacepdhcg_plan --capabilities
 *
 * Exit codes (deterministic):
 *   0   plan produced and independently certified
 *   2   plan produced but not certified (not converged, gate failure, trust exhausted,
 *       iteration budget, time limit)
 *   3   inner solver failure (native QOCO unavailable/numerical, PDHCG failure)
 *   64  invalid usage or invalid problem document
 *   65  input/output error
 *   66  CUDA runtime unavailable or device error
 *   70  internal error
 *
 * No exception crosses the C ABI: every native API status is converted to a JSON
 * status block and the exit codes above.
 */

#include "spacepdhcg/cuda/device_scvx_c_api.h"
#include "spacepdhcg/cuda/device_scvx_driver_c_api.h"
#include "spacepdhcg/planner/describe.hpp"
#include "spacepdhcg/planner/families.hpp"
#include "spacepdhcg/planner/json.hpp"
#include "spacepdhcg/planner/problem.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef SPACEPDHCG_SOURCE_COMMIT
#define SPACEPDHCG_SOURCE_COMMIT "0000000000000000000000000000000000000000"
#endif

namespace planner = spacepdhcg::planner;
namespace json = spacepdhcg::planner::json;

namespace {

enum ExitCode : int {
    exit_certified = 0,
    exit_not_certified = 2,
    exit_solver_failure = 3,
    exit_invalid_problem = 64,
    exit_io_error = 65,
    exit_cuda_error = 66,
    exit_internal_error = 70,
};

class PlanError final : public std::runtime_error {
  public:
    PlanError(int code, const std::string& message) : std::runtime_error(message), code_(code) {}
    [[nodiscard]] int code() const noexcept { return code_; }

  private:
    int code_;
};

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string cuda_status_name(const spacepdhcg_cuda_status status) {
    switch (status) {
        case SPACEPDHCG_CUDA_SUCCESS:
            return "success";
        case SPACEPDHCG_CUDA_INVALID_ARGUMENT:
            return "invalid_argument";
        case SPACEPDHCG_CUDA_INVALID_STATE:
            return "invalid_state";
        case SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH:
            return "topology_mismatch";
        case SPACEPDHCG_CUDA_POINTER_CONTRACT:
            return "pointer_contract";
        case SPACEPDHCG_CUDA_BUSY:
            return "busy";
        case SPACEPDHCG_CUDA_RUNTIME_ERROR:
            return "runtime_error";
        case SPACEPDHCG_CUDA_NUMERICAL_FAILURE:
            return "numerical_failure";
        case SPACEPDHCG_CUDA_UNSUPPORTED:
            return "unsupported";
        case SPACEPDHCG_CUDA_OUT_OF_MEMORY:
            return "out_of_memory";
        case SPACEPDHCG_CUDA_INTERNAL_ERROR:
            return "internal_error";
    }
    return "unknown";
}

std::string scvx_status_name(const spacepdhcg_cuda_scvx_status status) {
    switch (status) {
        case SPACEPDHCG_CUDA_SCVX_CONVERGED:
            return "converged";
        case SPACEPDHCG_CUDA_SCVX_MAXIMUM_ITERATIONS:
            return "maximum_iterations";
        case SPACEPDHCG_CUDA_SCVX_TRUST_REGION_EXHAUSTED:
            return "trust_region_exhausted";
        case SPACEPDHCG_CUDA_SCVX_INNER_FAILURE:
            return "inner_failure";
        case SPACEPDHCG_CUDA_SCVX_CANCELLED:
            return "cancelled";
        case SPACEPDHCG_CUDA_SCVX_INVALID:
            return "invalid";
    }
    return "unknown";
}

std::string phase_name(const spacepdhcg_cuda_scvx_phase phase) {
    switch (phase) {
        case SPACEPDHCG_CUDA_SCVX_REPAIR:
            return "repair";
        case SPACEPDHCG_CUDA_SCVX_PROGRESS:
            return "progress";
        case SPACEPDHCG_CUDA_SCVX_REFINEMENT:
            return "refinement";
        case SPACEPDHCG_CUDA_SCVX_POLISH:
            return "polish";
    }
    return "unknown";
}

std::string trust_action_name(const spacepdhcg_cuda_scvx_trust_action action) {
    switch (action) {
        case SPACEPDHCG_CUDA_SCVX_TRUST_RETAIN:
            return "retain";
        case SPACEPDHCG_CUDA_SCVX_TRUST_SHRINK:
            return "shrink";
        case SPACEPDHCG_CUDA_SCVX_TRUST_EXPAND:
            return "expand";
    }
    return "unknown";
}

std::string warm_start_name(const spacepdhcg_cuda_warm_start_mode mode) {
    switch (mode) {
        case SPACEPDHCG_CUDA_WARM_START_NONE:
            return "none";
        case SPACEPDHCG_CUDA_WARM_START_PRIMAL:
            return "primal";
        case SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL:
            return "primal_dual";
        case SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED:
            return "full_retained";
    }
    return "unknown";
}

std::string qoco_failure_name(const spacepdhcg_cuda_qoco_failure failure) {
    switch (failure) {
        case SPACEPDHCG_CUDA_QOCO_FAILURE_NONE:
            return "none";
        case SPACEPDHCG_CUDA_QOCO_FAILURE_UNAVAILABLE:
            return "unavailable";
        case SPACEPDHCG_CUDA_QOCO_FAILURE_UNSUPPORTED:
            return "unsupported";
        case SPACEPDHCG_CUDA_QOCO_FAILURE_OUT_OF_MEMORY:
            return "out_of_memory";
        case SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL:
            return "numerical";
        case SPACEPDHCG_CUDA_QOCO_FAILURE_INFEASIBLE:
            return "infeasible";
        case SPACEPDHCG_CUDA_QOCO_FAILURE_MAX_ITERATIONS:
            return "max_iterations";
        case SPACEPDHCG_CUDA_QOCO_FAILURE_ABI:
            return "abi";
    }
    return "unknown";
}

std::string recovery_reason_name(const spacepdhcg_cuda_recovery_reason reason) {
    switch (reason) {
        case SPACEPDHCG_CUDA_RECOVERY_NOT_TRIGGERED:
            return "not_triggered";
        case SPACEPDHCG_CUDA_RECOVERY_TIGHT_ITERATION_LIMIT:
            return "tight_iteration_limit";
        case SPACEPDHCG_CUDA_RECOVERY_QUALIFIED:
            return "qualified";
        case SPACEPDHCG_CUDA_RECOVERY_UNSUPPORTED_CONE:
            return "unsupported_cone";
        case SPACEPDHCG_CUDA_RECOVERY_NONFINITE_INPUT:
            return "nonfinite_input";
        case SPACEPDHCG_CUDA_RECOVERY_ZERO_CURVATURE:
            return "zero_curvature";
        case SPACEPDHCG_CUDA_RECOVERY_INCONSISTENT_ACTIVE_SET:
            return "inconsistent_active_set";
        case SPACEPDHCG_CUDA_RECOVERY_DUAL_INFEASIBLE:
            return "dual_infeasible";
        case SPACEPDHCG_CUDA_RECOVERY_EXHAUSTED:
            return "exhausted";
        case SPACEPDHCG_CUDA_RECOVERY_CANCELLED:
            return "cancelled";
    }
    return "unknown";
}

void cuda_check(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw PlanError(
            exit_cuda_error,
            std::string("CUDA failure during ") + operation + ": " + cudaGetErrorString(status)
        );
    }
}

void api_check(const spacepdhcg_cuda_status status, const char* operation) {
    if (status != SPACEPDHCG_CUDA_SUCCESS) {
        throw PlanError(
            status == SPACEPDHCG_CUDA_OUT_OF_MEMORY || status == SPACEPDHCG_CUDA_RUNTIME_ERROR
                ? exit_cuda_error
                : exit_internal_error,
            std::string("native API failure during ") + operation + ": "
                + cuda_status_name(status)
        );
    }
}

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    /// Allocates and zero-fills on `stream`.  The fill is issued on the consumer
    /// stream (never the legacy default stream) so it is ordered before every
    /// later upload/kernel on that non-blocking stream.
    DeviceBuffer(std::size_t elements, cudaStream_t stream) : elements_(elements) {
        if (elements_ == 0U) {
            return;
        }
        cuda_check(cudaMalloc(&pointer_, elements_ * sizeof(T)), "cudaMalloc");
        cuda_check(
            cudaMemsetAsync(pointer_, 0, elements_ * sizeof(T), stream), "cudaMemsetAsync"
        );
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)),
          elements_(std::exchange(other.elements_, 0U)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            pointer_ = std::exchange(other.pointer_, nullptr);
            elements_ = std::exchange(other.elements_, 0U);
        }
        return *this;
    }
    ~DeviceBuffer() { release(); }

    void upload(const std::vector<T>& values, cudaStream_t stream) {
        if (values.size() != elements_) {
            throw PlanError(exit_internal_error, "device upload size mismatch");
        }
        if (elements_ == 0U) {
            return;
        }
        cuda_check(
            cudaMemcpyAsync(
                pointer_, values.data(), elements_ * sizeof(T), cudaMemcpyHostToDevice, stream
            ),
            "cudaMemcpyAsync host-to-device"
        );
    }

    [[nodiscard]] std::vector<T> download(cudaStream_t stream) const {
        std::vector<T> values(elements_);
        if (elements_ > 0U) {
            cuda_check(
                cudaMemcpyAsync(
                    values.data(), pointer_, elements_ * sizeof(T), cudaMemcpyDeviceToHost, stream
                ),
                "cudaMemcpyAsync device-to-host"
            );
            cuda_check(cudaStreamSynchronize(stream), "download synchronize");
        }
        return values;
    }

    [[nodiscard]] T* get() noexcept { return pointer_; }
    [[nodiscard]] const T* get() const noexcept { return pointer_; }
    [[nodiscard]] std::size_t size() const noexcept { return elements_; }

    [[nodiscard]] spacepdhcg_accelerator_buffer_view view(
        spacepdhcg_accelerator_scalar_type type,
        spacepdhcg_accelerator_access access,
        std::size_t elements = static_cast<std::size_t>(-1)
    ) const noexcept {
        const std::size_t count = elements == static_cast<std::size_t>(-1) ? elements_ : elements;
        return spacepdhcg_accelerator_buffer_view{
            count == 0U ? nullptr : const_cast<T*>(pointer_),
            spacepdhcg_accelerator_device{SPACEPDHCG_DEVICE_CUDA, 0},
            type,
            count,
            0U,
            1,
            access,
        };
    }

  private:
    T* pointer_{nullptr};
    std::size_t elements_{0U};

    void release() noexcept {
        if (pointer_ != nullptr) {
            static_cast<void>(cudaFree(pointer_));
            pointer_ = nullptr;
        }
    }
};

spacepdhcg_accelerator_buffer_view null_int_view() noexcept {
    return spacepdhcg_accelerator_buffer_view{
        nullptr,
        spacepdhcg_accelerator_device{SPACEPDHCG_DEVICE_CUDA, 0},
        SPACEPDHCG_SCALAR_INT32,
        0U,
        0U,
        1,
        SPACEPDHCG_ACCESS_READ_ONLY,
    };
}

spacepdhcg_cuda_cone_kind cone_kind(const spacepdhcg::ConeKind kind) {
    switch (kind) {
        case spacepdhcg::ConeKind::second_order:
            return SPACEPDHCG_CUDA_CONE_SECOND_ORDER;
        case spacepdhcg::ConeKind::rotated_second_order:
            return SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER;
        case spacepdhcg::ConeKind::exponential:
            return SPACEPDHCG_CUDA_CONE_EXPONENTIAL;
        case spacepdhcg::ConeKind::power:
            return SPACEPDHCG_CUDA_CONE_POWER;
        case spacepdhcg::ConeKind::positive_semidefinite:
            return SPACEPDHCG_CUDA_CONE_POSITIVE_SEMIDEFINITE;
    }
    return SPACEPDHCG_CUDA_CONE_SECOND_ORDER;
}

spacepdhcg_cuda_dynamics_model model_of(const planner::Family family) {
    switch (family) {
        case planner::Family::hcw:
            return SPACEPDHCG_CUDA_DYNAMICS_HCW;
        case planner::Family::powered_descent_3dof:
            return SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_3DOF;
        case planner::Family::powered_descent_6dof:
            return SPACEPDHCG_CUDA_DYNAMICS_POWERED_DESCENT_6DOF;
        case planner::Family::low_thrust:
            return SPACEPDHCG_CUDA_DYNAMICS_LOW_THRUST;
    }
    return SPACEPDHCG_CUDA_DYNAMICS_HCW;
}

spacepdhcg_cuda_warm_start_mode warm_start_of(const std::string& name) {
    if (name == "none") {
        return SPACEPDHCG_CUDA_WARM_START_NONE;
    }
    if (name == "primal") {
        return SPACEPDHCG_CUDA_WARM_START_PRIMAL;
    }
    if (name == "primal_dual") {
        return SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL;
    }
    return SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED;
}

/// Solver policy derived from the backend/preset; documents exactly how each
/// planner backend maps onto the validated device driver.
struct BackendPolicy {
    spacepdhcg_cuda_scvx_policy policy{SPACEPDHCG_CUDA_SCVX_ADAPTIVE};
    std::string policy_name{};
    std::string description{};
};

BackendPolicy backend_policy(const planner::SolverOptions& solver) {
    BackendPolicy result{};
    switch (solver.backend) {
        case planner::Backend::pure_qoco:
            result.policy = SPACEPDHCG_CUDA_SCVX_PURE_QOCO;
            result.policy_name = "pure_qoco";
            result.description =
                "native pure-QOCO GPU interior-point inner solves with production nonlinear "
                "handback (qualified P1-C/P1-D/P1-E); one persistent QOCO workspace";
            break;
        case planner::Backend::pdhcg:
            result.policy = SPACEPDHCG_CUDA_SCVX_ADAPTIVE;
            result.policy_name = "adaptive_pdhcg";
            result.description =
                "persistent PDHCG workspace with the frozen adaptive forcing rule "
                "(repair/progress/refinement/polish tolerances)";
            break;
        case planner::Backend::pdhcg_recovery:
            result.policy = SPACEPDHCG_CUDA_SCVX_FIXED_TIGHT;
            result.policy_name = "fixed_tight_pdhcg_recovery";
            result.description =
                "persistent PDHCG workspace with tight fixed inner tolerance and the "
                "device projected-KKT/CGLS recovery path enabled (iteration limit >= 350000)";
            break;
        case planner::Backend::cpu_reference:
            throw PlanError(
                exit_invalid_problem,
                "backend cpu_reference is served by the Python reference solver, not by "
                "spacepdhcg_plan; choose pure_qoco, pdhcg, or pdhcg_recovery"
            );
    }
    return result;
}

spacepdhcg_cuda_scvx_options make_outer_options(
    const planner::PlannerProblem& problem,
    const BackendPolicy& backend
) {
    const auto& solver = problem.solver;
    const auto& trust = solver.trust;
    const auto& forcing = solver.forcing;
    spacepdhcg_cuda_scvx_options options{};
    options.abi_version = SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;
    options.maximum_outer_iterations = solver.maximum_outer_iterations;
    options.minimum_outer_iterations = solver.minimum_outer_iterations;
    options.maximum_resolves_per_iteration = forcing.maximum_resolves;
    options.convergence_tolerance = solver.tolerance;
    options.step_tolerance = solver.step_tolerance;
    options.acceptance_threshold = trust.acceptance_threshold;
    options.restoration_reduction = trust.restoration_reduction;
    options.feasibility_penalty = solver.penalty.feasibility_penalty;
    options.virtual_penalty = solver.penalty.virtual_penalty;
    options.initial_trust_radius = trust.initial_radius;
    options.minimum_trust_radius = trust.minimum_radius;
    options.maximum_trust_radius = trust.maximum_radius;
    options.shrink_factor = trust.shrink_factor;
    options.expansion_factor = trust.expansion_factor;
    options.strong_agreement_threshold = trust.strong_agreement_threshold;
    options.near_boundary_fraction = trust.near_boundary_fraction;
    options.fixed_inner_tolerance = forcing.fixed_inner_tolerance;
    options.fixed_inner_iteration_limit = forcing.fixed_inner_iteration_limit;
    options.policy = backend.policy;
    options.warm_start_mode = warm_start_of(solver.warm_start_mode);
    options.adaptive_epsilon_max = forcing.epsilon_max;
    options.adaptive_epsilon_floor = forcing.epsilon_floor;
    options.adaptive_epsilon_0 = forcing.epsilon_0;
    options.adaptive_coefficient = forcing.coefficient;
    options.adaptive_alpha = forcing.alpha;
    options.adaptive_gamma = forcing.gamma;
    options.repair_tolerance_ceiling = forcing.repair_ceiling;
    options.progress_tolerance_ceiling = forcing.progress_ceiling;
    options.refinement_tolerance_ceiling = forcing.refinement_ceiling;
    options.polish_tolerance_ceiling = forcing.polish_ceiling;
    options.repair_iteration_limit = forcing.repair_iterations;
    options.progress_iteration_limit = forcing.progress_iterations;
    options.refinement_iteration_limit = forcing.refinement_iterations;
    options.polish_iteration_limit = forcing.polish_iterations;
    options.resolve_trigger_multiple = forcing.resolve_trigger_multiple;
    options.resolve_refinement_factor = forcing.resolve_refinement_factor;
    options.resolve_minimum_tolerance = forcing.resolve_minimum_tolerance;
    options.final_polish_tolerance = forcing.final_polish_tolerance;
    options.final_polish_iteration_limit = forcing.final_polish_iteration_limit;
    return options;
}

json::Value iteration_json(const spacepdhcg_cuda_scvx_iteration& record) {
    json::Value value = json::Value::object();
    value.set("outer_iteration", static_cast<double>(record.outer_iteration));
    value.set("phase", phase_name(record.phase));
    value.set("requested_tolerance", record.requested_tolerance);
    value.set("achieved_residual", record.achieved_residual);
    value.set("inner_iterations", static_cast<double>(record.inner_iterations));
    value.set("trust_radius_before", record.trust_radius_before);
    value.set("trust_radius_after", record.trust_radius_after);
    value.set("trust_action", trust_action_name(record.trust_action));
    value.set("predicted_reduction", record.predicted_reduction);
    value.set("actual_reduction", record.actual_reduction);
    value.set("reduction_ratio", record.reduction_ratio);
    value.set("step_fraction", record.step_fraction);
    value.set("objective", record.objective);
    value.set("virtual_control", record.virtual_control);
    value.set("dynamics_defect", record.dynamics_defect);
    value.set("path_violation", record.path_violation);
    value.set("terminal_residual", record.terminal_residual);
    value.set("accepted", record.accepted != 0);
    value.set("restoration_accepted", record.restoration_accepted != 0);
    value.set("re_solved", record.re_solved != 0);
    value.set("scaling_refreshed", record.scaling_refreshed != 0);
    value.set("recovery_used", record.recovery_used != 0);
    value.set("forcing_satisfied", record.forcing_satisfied != 0);
    value.set("final_polish_handoff", record.final_polish_handoff != 0);
    value.set("warm_start_mode", warm_start_name(record.warm_start_mode));
    value.set("native_primal_residual", record.native_primal_residual);
    value.set("native_dual_residual", record.native_dual_residual);
    value.set("complementarity_residual", record.complementarity_residual);
    value.set("scalar_primal_residual", record.scalar_primal_residual);
    value.set("box_primal_residual", record.box_primal_residual);
    value.set("cone_primal_residual", record.cone_primal_residual);
    value.set("stationarity_residual", record.stationarity_residual);
    value.set("natural_residual", record.natural_residual);
    value.set("current_merit", record.current_merit);
    value.set("candidate_merit", record.candidate_merit);
    value.set("candidate_model_merit", record.candidate_model_merit);
    value.set("current_dynamics_defect", record.current_dynamics_defect);
    value.set("current_path_violation", record.current_path_violation);
    value.set("current_terminal_residual", record.current_terminal_residual);
    value.set("maximum_stage_trust_distance", record.maximum_stage_trust_distance);
    value.set("terminal_trust_distance", record.terminal_trust_distance);
    value.set("matvecs", static_cast<double>(record.matvecs));
    value.set("cone_projections", static_cast<double>(record.cone_projections));
    value.set("recovery_reason", recovery_reason_name(record.recovery_reason));
    value.set("recovery_attempt_count", static_cast<double>(record.recovery_attempt_count));
    value.set("recovery_accepted_count", static_cast<double>(record.recovery_accepted_count));
    value.set("recovery_rejected_count", static_cast<double>(record.recovery_rejected_count));
    value.set("recovery_seconds", record.recovery_seconds);
    value.set("recovery_iterations", static_cast<double>(record.recovery_iterations));
    char fingerprint[32];
    std::snprintf(
        fingerprint,
        sizeof(fingerprint),
        "%016llx",
        static_cast<unsigned long long>(record.cqp_numeric_fingerprint)
    );
    value.set("cqp_numeric_fingerprint", std::string(fingerprint));
    return value;
}

json::Value rows_json(const std::vector<double>& flat, std::size_t width) {
    json::Value rows = json::Value::array();
    for (std::size_t start = 0U; start + width <= flat.size(); start += width) {
        json::Value row = json::Value::array();
        for (std::size_t index = 0U; index < width; ++index) {
            row.push_back(json::Value(flat[start + index]));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

json::Value times_json(std::size_t count, double step) {
    json::Value times = json::Value::array();
    for (std::size_t index = 0U; index < count; ++index) {
        times.push_back(json::Value(static_cast<double>(index) * step));
    }
    return times;
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw PlanError(exit_io_error, "cannot read problem file: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

void write_output(const std::string& path, const std::string& text) {
    if (path.empty()) {
        std::fwrite(text.data(), 1U, text.size(), stdout);
        std::fputc('\n', stdout);
        std::fflush(stdout);
        return;
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw PlanError(exit_io_error, "cannot write result file: " + path);
    }
    stream << text << '\n';
    if (!stream) {
        throw PlanError(exit_io_error, "failed while writing result file: " + path);
    }
}

json::Value device_json(const bool cuda_ready) {
    json::Value device = json::Value::object();
    device.set("cuda_available", cuda_ready);
    if (!cuda_ready) {
        return device;
    }
    int index = 0;
    if (cudaGetDevice(&index) != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        return device;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, index) == cudaSuccess) {
        device.set("index", static_cast<double>(index));
        device.set("name", std::string(properties.name));
        device.set(
            "compute_capability",
            std::to_string(properties.major) + "." + std::to_string(properties.minor)
        );
        device.set(
            "total_memory_bytes", static_cast<double>(properties.totalGlobalMem)
        );
    }
    int runtime = 0;
    if (cudaRuntimeGetVersion(&runtime) == cudaSuccess) {
        device.set("cuda_runtime_version", static_cast<double>(runtime));
    }
    int driver = 0;
    if (cudaDriverGetVersion(&driver) == cudaSuccess) {
        device.set("cuda_driver_version", static_cast<double>(driver));
    }
    return device;
}

struct PlanOutcome {
    json::Value result{};
    int exit_code{exit_internal_error};
};

/// Owns the consumer stream; declared before every device buffer so it outlives them.
class StreamGuard {
  public:
    StreamGuard() {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    }
    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
    ~StreamGuard() {
        if (stream != nullptr) {
            static_cast<void>(cudaStreamSynchronize(stream));
            static_cast<void>(cudaStreamDestroy(stream));
        }
    }

    cudaStream_t stream{nullptr};
};

/// Owns the persistent workspace and SCvx driver; declared after the device
/// buffers they reference so teardown happens while those buffers are alive.
class Workspace {
  public:
    Workspace() = default;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    ~Workspace() {
        if (driver != nullptr) {
            static_cast<void>(spacepdhcg_cuda_scvx_driver_destroy(&driver));
        }
        if (workspace != nullptr) {
            static_cast<void>(spacepdhcg_cuda_workspace_destroy(&workspace));
        }
    }

    spacepdhcg_cuda_workspace* workspace{nullptr};
    spacepdhcg_cuda_scvx_driver* driver{nullptr};
};

PlanOutcome run_plan(
    const planner::PlannerProblem& problem,
    const double cuda_startup_seconds,
    const bool quiet
) {
    const auto wall_started = Clock::now();
    const auto backend = backend_policy(problem.solver);

    // 1. Host transcription from user values -------------------------------
    const auto topology_started = Clock::now();
    const auto adapter = planner::make_adapter(problem);
    const auto& structure = adapter->structure();
    const auto& info = adapter->layout();
    const std::size_t intervals = info.intervals;
    const std::size_t nx = info.state_dimension;
    const std::size_t nu = info.control_dimension;
    const auto reference = adapter->initial_reference();
    const auto values = adapter->values(reference, problem.solver.trust.initial_radius);
    const auto initial_evaluation = adapter->evaluate(reference.states, reference.controls);
    const double topology_seconds = seconds_since(topology_started);
    if (!quiet) {
        std::fprintf(
            stderr,
            "[spacepdhcg_plan] %s: N=%zu variables=%zu scalar_rows=%zu affine_rows=%zu "
            "backend=%s\n",
            std::string(planner::family_name(problem.family)).c_str(),
            intervals,
            info.variables,
            info.scalar_rows,
            info.affine_rows,
            backend.policy_name.c_str()
        );
    }

    // 2. Device storage ----------------------------------------------------
    StreamGuard stream_guard;
    auto* stream = stream_guard.stream;
    const auto to_int32 = [](const std::vector<spacepdhcg::Index>& source) {
        return std::vector<int>(source.begin(), source.end());
    };
    const std::vector<int> h_q_offsets = to_int32(structure.quadratic.offsets);
    const std::vector<int> h_q_indices = to_int32(structure.quadratic.indices);
    const std::vector<int> h_a_offsets = to_int32(structure.scalar_constraint.offsets);
    const std::vector<int> h_a_indices = to_int32(structure.scalar_constraint.indices);
    const std::vector<int> h_f_offsets = structure.affine_cone.has_value()
        ? to_int32(structure.affine_cone->offsets)
        : std::vector<int>(info.variables + 1U, 0);
    const std::vector<int> h_f_indices = structure.affine_cone.has_value()
        ? to_int32(structure.affine_cone->indices)
        : std::vector<int>{};

    DeviceBuffer<int> q_offsets(h_q_offsets.size(), stream);
    DeviceBuffer<int> q_indices(h_q_indices.size(), stream);
    DeviceBuffer<int> a_offsets(h_a_offsets.size(), stream);
    DeviceBuffer<int> a_indices(h_a_indices.size(), stream);
    DeviceBuffer<int> f_offsets(h_f_offsets.size(), stream);
    DeviceBuffer<int> f_indices(h_f_indices.size(), stream);
    DeviceBuffer<double> q(values.quadratic.size(), stream);
    DeviceBuffer<double> a(values.scalar_constraint.size(), stream);
    DeviceBuffer<double> f(values.affine_cone.size(), stream);
    DeviceBuffer<double> c(values.linear_objective.size(), stream);
    DeviceBuffer<double> scalar_lower(values.scalar_lower.size(), stream);
    DeviceBuffer<double> scalar_upper(values.scalar_upper.size(), stream);
    DeviceBuffer<double> affine_offset(values.affine_offset.size(), stream);
    DeviceBuffer<double> variable_lower(values.variable_lower.size(), stream);
    DeviceBuffer<double> variable_upper(values.variable_upper.size(), stream);
    DeviceBuffer<double> primal(info.variables, stream);
    DeviceBuffer<double> dual(info.scalar_rows + info.affine_rows, stream);
    q_offsets.upload(h_q_offsets, stream);
    q_indices.upload(h_q_indices, stream);
    a_offsets.upload(h_a_offsets, stream);
    a_indices.upload(h_a_indices, stream);
    f_offsets.upload(h_f_offsets, stream);
    f_indices.upload(h_f_indices, stream);
    q.upload(values.quadratic, stream);
    a.upload(values.scalar_constraint, stream);
    f.upload(values.affine_cone, stream);
    c.upload(values.linear_objective, stream);
    scalar_lower.upload(values.scalar_lower, stream);
    scalar_upper.upload(values.scalar_upper, stream);
    affine_offset.upload(values.affine_offset, stream);
    variable_lower.upload(values.variable_lower, stream);
    variable_upper.upload(values.variable_upper, stream);

    std::vector<spacepdhcg_cuda_cone_descriptor> affine_cones;
    for (const auto& cone : structure.affine_cones) {
        affine_cones.push_back(
            {cone_kind(cone.kind), cone.start, cone.vector_dimension, cone.power_alpha}
        );
    }
    std::vector<spacepdhcg_cuda_cone_descriptor> variable_cones;
    for (const auto& cone : structure.variable_cones) {
        variable_cones.push_back(
            {cone_kind(cone.kind), cone.start, cone.vector_dimension, cone.power_alpha}
        );
    }
    const std::uint64_t fingerprint = structure.fingerprint();
    const spacepdhcg_cuda_structure device_structure{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        fingerprint,
        static_cast<int32_t>(info.variables),
        static_cast<int32_t>(info.scalar_rows),
        static_cast<int32_t>(info.affine_rows),
        values.quadratic.size(),
        values.scalar_constraint.size(),
        values.affine_cone.size(),
        affine_cones.empty() ? nullptr : affine_cones.data(),
        affine_cones.size(),
        variable_cones.empty() ? nullptr : variable_cones.data(),
        variable_cones.size(),
    };
    const auto ro32 = [](const DeviceBuffer<int>& buffer) {
        return buffer.view(SPACEPDHCG_SCALAR_INT32, SPACEPDHCG_ACCESS_READ_ONLY);
    };
    const auto rw64 = [](const DeviceBuffer<double>& buffer) {
        return buffer.view(SPACEPDHCG_SCALAR_FLOAT64, SPACEPDHCG_ACCESS_READ_WRITE);
    };
    const auto ro64 = [](const DeviceBuffer<double>& buffer) {
        return buffer.view(SPACEPDHCG_SCALAR_FLOAT64, SPACEPDHCG_ACCESS_READ_ONLY);
    };
    spacepdhcg_cqp_accelerator_exchange exchange{};
    exchange.abi_version = SPACEPDHCG_ACCELERATOR_EXCHANGE_ABI_VERSION;
    exchange.topology_fingerprint = fingerprint;
    exchange.consumer_stream = spacepdhcg_accelerator_stream{
        {SPACEPDHCG_DEVICE_CUDA, 0},
        reinterpret_cast<std::uintptr_t>(stream),
    };
    exchange.topology = {
        ro32(q_offsets), ro32(q_indices), ro32(a_offsets), ro32(a_indices),
        ro32(f_offsets), ro32(f_indices),
    };
    exchange.numeric = {
        rw64(q), rw64(a), rw64(f), rw64(c), rw64(scalar_lower), rw64(scalar_upper),
        rw64(affine_offset), rw64(variable_lower), rw64(variable_upper),
    };
    exchange.iterates = {rw64(primal), rw64(dual)};

    // Trajectory, linearisation, and index buffers.
    DeviceBuffer<double> states(reference.states.size(), stream);
    DeviceBuffer<double> controls(reference.controls.size(), stream);
    DeviceBuffer<double> propagated(intervals * nx, stream);
    DeviceBuffer<double> transition(intervals * nx * nx, stream);
    DeviceBuffer<double> sensitivity(intervals * nx * nu, stream);
    DeviceBuffer<double> offset(intervals * nx, stream);
    DeviceBuffer<int> state_positions(info.state_positions.size(), stream);
    DeviceBuffer<int> control_positions(info.control_positions.size(), stream);
    DeviceBuffer<int> next_positions(info.next_positions.size(), stream);
    DeviceBuffer<int> virtual_positions(info.virtual_positions.size(), stream);
    DeviceBuffer<int> state_variables(info.state_variables.size(), stream);
    DeviceBuffer<int> control_variables(info.control_variables.size(), stream);
    DeviceBuffer<int> virtual_variables(info.virtual_variables.size(), stream);
    DeviceBuffer<int> q_positions(info.quadratic_diagonal_positions.size(), stream);
    DeviceBuffer<int> radial_positions(info.radial_positions.size(), stream);
    DeviceBuffer<int> quaternion_positions(info.quaternion_positions.size(), stream);
    DeviceBuffer<double> initial(nx, stream);
    DeviceBuffer<double> target(nx, stream);
    states.upload(reference.states, stream);
    controls.upload(reference.controls, stream);
    state_positions.upload(info.state_positions, stream);
    control_positions.upload(info.control_positions, stream);
    next_positions.upload(info.next_positions, stream);
    virtual_positions.upload(info.virtual_positions, stream);
    state_variables.upload(info.state_variables, stream);
    control_variables.upload(info.control_variables, stream);
    virtual_variables.upload(info.virtual_variables, stream);
    q_positions.upload(info.quadratic_diagonal_positions, stream);
    radial_positions.upload(info.radial_positions, stream);
    quaternion_positions.upload(info.quaternion_positions, stream);
    initial.upload(problem.initial_state, stream);
    target.upload(problem.target_state, stream);
    cuda_check(cudaStreamSynchronize(stream), "device upload synchronize");

    const auto dynamics = adapter->dynamics();
    const spacepdhcg_cuda_dynamics_config dynamics_config{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        model_of(problem.family),
        dynamics.step_seconds,
        dynamics.mean_motion,
        {dynamics.gravity[0U], dynamics.gravity[1U], dynamics.gravity[2U]},
        dynamics.gravitational_parameter,
        dynamics.thrust_to_acceleration,
        dynamics.mass_flow_coefficient,
        {dynamics.principal_inertia[0U], dynamics.principal_inertia[1U],
         dynamics.principal_inertia[2U]},
    };
    const spacepdhcg_cuda_variational_request linearise{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        intervals,
        states.view(SPACEPDHCG_SCALAR_FLOAT64, SPACEPDHCG_ACCESS_READ_ONLY, intervals * nx),
        ro64(controls),
        rw64(propagated),
        rw64(transition),
        rw64(sensitivity),
        rw64(offset),
    };
    const spacepdhcg_cuda_csc_dynamics_fill fill{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        intervals,
        nx,
        nu,
        ro64(transition),
        ro64(sensitivity),
        ro64(offset),
        ro32(state_positions),
        ro32(control_positions),
        ro32(next_positions),
        info.virtual_positions.empty() ? null_int_view() : ro32(virtual_positions),
        exchange.numeric.scalar_constraint,
        exchange.numeric.scalar_lower,
        exchange.numeric.scalar_upper,
        info.dynamics_row_start,
    };

    // 3. Persistent workspace ---------------------------------------------
    Workspace guard;
    const auto create_started = Clock::now();
    const spacepdhcg_cuda_create_options create_options{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        SPACEPDHCG_CUDA_SCALING_REFRESH_IF_NEEDED,
        0.25,
        0.50,
        4U,
        1,
        nullptr,
        nullptr,
        nullptr,
    };
    api_check(
        spacepdhcg_cuda_workspace_create(
            &device_structure, &exchange, &create_options, &guard.workspace
        ),
        "workspace create"
    );
    const double workspace_create_seconds = seconds_since(create_started);

    // 4. SCvx problem + driver ---------------------------------------------
    const auto limits = adapter->limits();
    spacepdhcg_cuda_scvx_numeric_update numeric_update{};
    numeric_update.abi_version = SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;
    numeric_update.quadratic_diagonal_positions = ro32(q_positions);
    numeric_update.radial_positions =
        info.radial_positions.empty() ? null_int_view() : ro32(radial_positions);
    numeric_update.quaternion_positions =
        info.quaternion_positions.empty() ? null_int_view() : ro32(quaternion_positions);
    numeric_update.terminal_row_start = info.terminal_row_start;
    numeric_update.radial_row_start = info.radial_row_start;
    numeric_update.quaternion_row_start = info.quaternion_row_start;
    numeric_update.stage_trust_row_start = info.stage_trust_row_start;
    numeric_update.stage_trust_stride = info.stage_trust_stride;
    numeric_update.terminal_trust_row_start = info.terminal_trust_row_start;
    numeric_update.virtual_variable_offset = info.virtual_variable_offset;
    numeric_update.epigraph_variable_offset = info.epigraph_variable_offset;
    for (std::size_t index = 0U; index < info.state_trust_scales.size() && index < 14U; ++index) {
        numeric_update.state_trust_scales[index] = info.state_trust_scales[index];
    }
    for (std::size_t index = 0U; index < info.control_trust_scales.size() && index < 7U; ++index) {
        numeric_update.control_trust_scales[index] = info.control_trust_scales[index];
    }
    numeric_update.fuel_weight = info.fuel_weight;
    numeric_update.virtual_l1_weight = info.virtual_l1_weight;
    numeric_update.maximum_thrust = limits.maximum_thrust;
    numeric_update.maximum_torque = limits.maximum_torque;
    numeric_update.maximum_angular_rate = limits.maximum_angular_rate;
    numeric_update.tilt_cosine = limits.tilt_cosine;
    numeric_update.glide_slope_tangent = limits.glide_slope_tangent;
    numeric_update.minimum_radius = limits.minimum_radius;
    numeric_update.conditioning_log10_span = 0.0;

    const spacepdhcg_cuda_scvx_problem outer_problem{
        SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION,
        guard.workspace,
        fingerprint,
        intervals,
        nx,
        nu,
        dynamics_config,
        exchange.numeric,
        linearise,
        fill,
        ro32(state_variables),
        ro32(control_variables),
        info.virtual_variables.empty() ? null_int_view() : ro32(virtual_variables),
        rw64(states),
        rw64(controls),
        ro64(initial),
        ro64(target),
        numeric_update,
        device_structure,
        exchange.topology,
    };

    // Initial analytic coefficient pass (device) so the workspace sees the same
    // coefficients the host transcription produced; timed as "coefficients".
    const auto coefficient_started = Clock::now();
    api_check(
        spacepdhcg_cuda_variational_rk4_async(
            &dynamics_config, &linearise, exchange.consumer_stream
        ),
        "initial variational coefficients"
    );
    api_check(
        spacepdhcg_cuda_fill_dynamics_csc_async(&fill, exchange.consumer_stream),
        "initial dynamics fill"
    );
    api_check(
        spacepdhcg_cuda_scvx_update_numeric_async(
            &outer_problem,
            problem.solver.trust.initial_radius,
            info.virtual_l1_weight,
            exchange.consumer_stream
        ),
        "initial numeric update"
    );
    cuda_check(cudaStreamSynchronize(stream), "initial coefficient synchronize");
    const double coefficient_seconds = seconds_since(coefficient_started);

    // Coefficient parity between the host transcription and the device kernels
    // (relative, matching the G3 production audit at 5e-12).
    double coefficient_parity = 0.0;
    {
        const auto compare = [&coefficient_parity](
            const std::vector<double>& actual, const std::vector<double>& wanted
        ) {
            if (actual.size() != wanted.size()) {
                throw PlanError(exit_internal_error, "coefficient parity size mismatch");
            }
            for (std::size_t index = 0U; index < actual.size(); ++index) {
                if (std::isinf(actual[index]) || std::isinf(wanted[index])) {
                    if (actual[index] != wanted[index]) {
                        coefficient_parity = std::numeric_limits<double>::infinity();
                    }
                    continue;
                }
                coefficient_parity = std::max(
                    coefficient_parity,
                    std::abs(actual[index] - wanted[index])
                        / std::max(1.0, std::abs(wanted[index]))
                );
            }
        };
        compare(q.download(stream), values.quadratic);
        compare(a.download(stream), values.scalar_constraint);
        compare(f.download(stream), values.affine_cone);
        compare(c.download(stream), values.linear_objective);
        compare(scalar_lower.download(stream), values.scalar_lower);
        compare(scalar_upper.download(stream), values.scalar_upper);
        compare(affine_offset.download(stream), values.affine_offset);
        compare(variable_lower.download(stream), values.variable_lower);
        compare(variable_upper.download(stream), values.variable_upper);
    }

    const auto outer_options = make_outer_options(problem, backend);
    api_check(
        spacepdhcg_cuda_scvx_driver_create(&outer_problem, &outer_options, &guard.driver),
        "SCvx driver create"
    );

    // 5. Solve with optional wall-clock deadline ---------------------------
    std::vector<spacepdhcg_cuda_scvx_iteration> records(problem.solver.maximum_outer_iterations);
    spacepdhcg_cuda_scvx_result outer{};
    std::mutex deadline_mutex;
    std::condition_variable deadline_condition;
    bool solve_finished = false;
    bool deadline_triggered = false;
    std::thread deadline_thread;
    if (problem.solver.time_limit_seconds > 0.0) {
        const auto deadline = Clock::now()
            + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(problem.solver.time_limit_seconds)
            );
        deadline_thread = std::thread([&]() {
            std::unique_lock lock(deadline_mutex);
            if (!deadline_condition.wait_until(lock, deadline, [&]() { return solve_finished; })) {
                deadline_triggered = true;
                static_cast<void>(spacepdhcg_cuda_scvx_driver_cancel(guard.driver));
            }
        });
    }
    const auto solve_started = Clock::now();
    const auto solve_status = spacepdhcg_cuda_scvx_driver_solve(
        guard.driver, exchange.consumer_stream, records.data(), records.size(), &outer
    );
    const double solve_wall_seconds = seconds_since(solve_started);
    if (deadline_thread.joinable()) {
        {
            std::lock_guard lock(deadline_mutex);
            solve_finished = true;
        }
        deadline_condition.notify_one();
        deadline_thread.join();
    }
    cuda_check(cudaStreamSynchronize(stream), "post-solve synchronize");

    spacepdhcg_cuda_scvx_path_inventory path_inventory{};
    static_cast<void>(spacepdhcg_cuda_scvx_driver_path_inventory(guard.driver, &path_inventory));
    spacepdhcg_cuda_diagnostics diagnostics{};
    static_cast<void>(spacepdhcg_cuda_workspace_diagnostics(guard.workspace, &diagnostics));

    // 6. Gather the accepted reference (device) -----------------------------
    const auto final_states = states.download(stream);
    const auto final_controls = controls.download(stream);
    const std::size_t recorded =
        std::min<std::size_t>(outer.outer_iterations, records.size());
    records.resize(recorded);

    if (!quiet) {
        std::fprintf(
            stderr,
            "[spacepdhcg_plan] solve api=%s status=%s outer=%u accepted=%u rejected=%u "
            "objective=%.9g canonical=%.3e dynamics=%.3e path=%.3e terminal=%.3e\n",
            cuda_status_name(solve_status).c_str(),
            scvx_status_name(outer.status).c_str(),
            outer.outer_iterations,
            outer.accepted_steps,
            outer.rejected_steps,
            outer.objective,
            outer.canonical_residual,
            outer.dynamics_defect,
            outer.path_violation,
            outer.terminal_residual
        );
    }

    // 7. Independent host replay and evaluation ----------------------------
    // The replay is evaluated defensively: a diverged device trajectory (for
    // example a non-physical mass) is reported as a failed gate, never as an
    // invalid problem.
    const auto replay_started = Clock::now();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> node_replay;
    std::vector<double> dense_replay;
    std::vector<planner::PathComponent> dense_path;
    planner::Evaluation independent{};
    planner::Evaluation model_evaluation{};
    double replay_parity = nan;
    double independent_dynamics = nan;
    double continuous_time_violation = nan;
    std::string replay_error;
    const std::size_t substeps = problem.output.dense_replay_substeps;
    independent.path_violation = nan;
    independent.terminal_residual = nan;
    independent.objective = nan;
    model_evaluation.objective = nan;
    try {
        node_replay = adapter->rollout(problem.initial_state, final_controls, 1U);
        replay_parity = planner::FamilyAdapter::infinity_distance(final_states, node_replay);
        independent_dynamics = adapter->scaled_state_defect(final_states, node_replay);
        independent = adapter->evaluate(node_replay, final_controls);
        model_evaluation = adapter->evaluate(final_states, final_controls);
        dense_replay = adapter->rollout(problem.initial_state, final_controls, substeps);
        std::vector<double> dense_controls;
        dense_controls.reserve(final_controls.size() * substeps);
        for (std::size_t interval = 0U; interval < intervals; ++interval) {
            for (std::size_t repeat = 0U; repeat < substeps; ++repeat) {
                dense_controls.insert(
                    dense_controls.end(),
                    final_controls.begin() + static_cast<std::ptrdiff_t>(interval * nu),
                    final_controls.begin() + static_cast<std::ptrdiff_t>((interval + 1U) * nu)
                );
            }
        }
        dense_path = adapter->path_components(dense_replay, dense_controls);
        continuous_time_violation = 0.0;
        for (const auto& component : dense_path) {
            continuous_time_violation =
                std::max(continuous_time_violation, component.normalised);
        }
    } catch (const std::exception& error) {
        replay_error = error.what();
        if (!quiet) {
            std::fprintf(stderr, "[spacepdhcg_plan] independent replay failed: %s\n", error.what());
        }
    }
    const double independent_replay_seconds = seconds_since(replay_started);

    // 8. Certificate -------------------------------------------------------
    const double tolerance = problem.solver.certificate_tolerance;
    const bool api_success = solve_status == SPACEPDHCG_CUDA_SUCCESS;
    const bool converged = api_success && outer.status == SPACEPDHCG_CUDA_SCVX_CONVERGED;
    struct Gate {
        std::string name;
        bool passed;
        double value;
        double limit;
    };
    const std::vector<Gate> gates{
        {"solver_api_success", api_success, api_success ? 0.0 : 1.0, 0.0},
        {"converged", converged, converged ? 0.0 : 1.0, 0.0},
        {"canonical_residual", api_success && outer.canonical_residual <= tolerance,
         outer.canonical_residual, tolerance},
        {"device_dynamics_defect", api_success && outer.dynamics_defect <= tolerance,
         outer.dynamics_defect, tolerance},
        {"device_path_violation", api_success && outer.path_violation <= tolerance,
         outer.path_violation, tolerance},
        {"device_terminal_residual", api_success && outer.terminal_residual <= tolerance,
         outer.terminal_residual, tolerance},
        {"virtual_control", api_success && outer.virtual_control <= tolerance,
         outer.virtual_control, tolerance},
        {"independent_replay_parity",
         std::isfinite(replay_parity) && replay_parity <= problem.solver.replay_parity_tolerance,
         replay_parity, problem.solver.replay_parity_tolerance},
        {"independent_dynamics_defect",
         std::isfinite(independent_dynamics) && independent_dynamics <= tolerance,
         independent_dynamics, tolerance},
        {"independent_path_violation",
         std::isfinite(independent.path_violation) && independent.path_violation <= tolerance,
         independent.path_violation, tolerance},
        {"independent_terminal_residual",
         std::isfinite(independent.terminal_residual)
             && independent.terminal_residual <= tolerance,
         independent.terminal_residual, tolerance},
        {"no_hidden_cpu_fallback", outer.hidden_cpu_fallback == 0,
         static_cast<double>(outer.hidden_cpu_fallback), 0.0},
        {"steady_state_residency",
         outer.topology_allocation_count_after_create == 0U
             && outer.topology_index_copy_count_after_create == 0U,
         static_cast<double>(
             outer.topology_allocation_count_after_create
             + outer.topology_index_copy_count_after_create
         ),
         0.0},
        {"coefficient_parity", coefficient_parity <= 5.0e-12, coefficient_parity, 5.0e-12},
        {"independent_replay_evaluated", replay_error.empty(), replay_error.empty() ? 0.0 : 1.0, 0.0},
    };
    bool certified = true;
    json::Value gate_json = json::Value::object();
    json::Value failed = json::Value::array();
    for (const auto& gate : gates) {
        certified = certified && gate.passed;
        json::Value entry = json::Value::object();
        entry.set("passed", gate.passed);
        entry.set("value", gate.value);
        entry.set("limit", gate.limit);
        gate_json.set(gate.name, entry);
        if (!gate.passed) {
            failed.push_back(json::Value(gate.name));
        }
    }

    // 9. Result document ---------------------------------------------------
    std::string status_code;
    std::string status_message;
    int exit_code = exit_not_certified;
    if (!api_success) {
        if (outer.qoco_failure == SPACEPDHCG_CUDA_QOCO_FAILURE_UNAVAILABLE) {
            status_code = "qoco_unavailable";
            status_message =
                "the native QOCO GPU library was not available (set SPACEPDHCG_QOCO_LIBRARY to "
                "the pinned libqoco.so); no CPU fallback was used";
        } else if (solve_status == SPACEPDHCG_CUDA_UNSUPPORTED) {
            status_code = "backend_unsupported";
            status_message = "the requested backend is unsupported in this build: "
                + cuda_status_name(solve_status) + " (QOCO failure: "
                + qoco_failure_name(outer.qoco_failure) + ")";
        } else if (deadline_triggered || outer.status == SPACEPDHCG_CUDA_SCVX_CANCELLED) {
            status_code = "time_limit";
            status_message = "the solver was cancelled at the requested time limit";
        } else {
            status_code = "solver_failure";
            status_message = "inner solver failure: " + cuda_status_name(solve_status)
                + " (outer status " + scvx_status_name(outer.status) + ", QOCO failure "
                + qoco_failure_name(outer.qoco_failure) + ")";
        }
        exit_code = (status_code == "time_limit") ? exit_not_certified : exit_solver_failure;
    } else if (deadline_triggered && outer.status == SPACEPDHCG_CUDA_SCVX_CANCELLED) {
        status_code = "time_limit";
        status_message = "the solver was cancelled at the requested time limit";
    } else if (certified) {
        status_code = "certified";
        status_message = "converged and independently certified";
        exit_code = exit_certified;
    } else if (outer.status == SPACEPDHCG_CUDA_SCVX_TRUST_REGION_EXHAUSTED) {
        status_code = "trust_region_exhausted";
        status_message = "the trust region shrank to its minimum radius without an accepted "
            "improving step; the retained reference did not meet the certificate gates";
    } else if (outer.status == SPACEPDHCG_CUDA_SCVX_MAXIMUM_ITERATIONS) {
        status_code = "maximum_iterations";
        status_message = "the outer iteration budget was exhausted before convergence";
    } else if (outer.status == SPACEPDHCG_CUDA_SCVX_CONVERGED) {
        status_code = "converged_not_certified";
        status_message = "the solver reported convergence but independent replay failed the "
            "certificate gates";
    } else {
        status_code = "not_certified";
        status_message = "the plan was produced but is not certified (outer status "
            + scvx_status_name(outer.status) + ")";
    }

    json::Value result = json::Value::object();
    result.set("schema_version", "1.0.0");
    result.set("result_kind", "spacepdhcg_plan_result");
    result.set("source_commit", std::string(SPACEPDHCG_SOURCE_COMMIT));
    {
        json::Value status = json::Value::object();
        status.set("code", status_code);
        status.set("message", status_message);
        status.set("exit_code", static_cast<double>(exit_code));
        status.set("solver_status", scvx_status_name(outer.status));
        status.set("api_status", cuda_status_name(solve_status));
        status.set("time_limit_triggered", deadline_triggered);
        result.set("status", status);
    }
    result.set("problem", planner::describe_problem(problem));
    {
        json::Value summary = json::Value::object();
        summary.set("objective", outer.objective);
        summary.set("objective_definition", model_evaluation.objective_definition);
        summary.set("outer_iterations", static_cast<double>(outer.outer_iterations));
        summary.set("accepted_steps", static_cast<double>(outer.accepted_steps));
        summary.set("rejected_steps", static_cast<double>(outer.rejected_steps));
        summary.set("resolved_steps", static_cast<double>(outer.resolved_steps));
        summary.set("inner_iterations", static_cast<double>(outer.inner_iterations));
        summary.set("final_trust_radius", outer.final_trust_radius);
        summary.set("trajectory_step", outer.trajectory_step);
        summary.set("propellant_used", independent.propellant_used);
        summary.set("final_mass", independent.final_mass);
        summary.set("terminal_position_error", independent.terminal_position_error);
        summary.set("terminal_velocity_error", independent.terminal_velocity_error);
        result.set("summary", summary);
    }
    {
        json::Value residuals = json::Value::object();
        residuals.set("canonical_residual", outer.canonical_residual);
        residuals.set("dynamics_defect", outer.dynamics_defect);
        residuals.set("path_violation", outer.path_violation);
        residuals.set("terminal_residual", outer.terminal_residual);
        residuals.set("virtual_control", outer.virtual_control);
        json::Value inventory = json::Value::object();
        inventory.set("thrust", path_inventory.thrust_violation);
        inventory.set("mass", path_inventory.mass_violation);
        inventory.set("altitude", path_inventory.altitude_violation);
        residuals.set("device_path_inventory", inventory);
        residuals.set("coefficient_parity_relative", coefficient_parity);
        result.set("solver_residuals", residuals);
    }
    {
        json::Value independent_json = planner::describe_evaluation(independent);
        independent_json.set("replay_error", replay_error.empty() ? json::Value(nullptr) : json::Value(replay_error));
        independent_json.set("replay_parity", replay_parity);
        independent_json.set("dynamics_defect", independent_dynamics);
        independent_json.set("continuous_time_violation", continuous_time_violation);
        json::Value dense_components = json::Value::object();
        json::Value dense_physical = json::Value::object();
        for (const auto& component : dense_path) {
            dense_components.set(component.name, component.normalised);
            dense_physical.set(component.name, component.physical);
        }
        independent_json.set("continuous_time_components", dense_components);
        independent_json.set("continuous_time_components_physical", dense_physical);
        independent_json.set("dense_replay_substeps", static_cast<double>(substeps));
        result.set("independent_replay", independent_json);
        result.set("model_evaluation", planner::describe_evaluation(model_evaluation));
        result.set("initial_reference_evaluation", planner::describe_evaluation(initial_evaluation));
    }
    {
        json::Value trajectory = json::Value::object();
        trajectory.set("times", times_json(intervals + 1U, problem.step_seconds));
        trajectory.set("states", rows_json(final_states, nx));
        trajectory.set("controls", rows_json(final_controls, nu));
        result.set("trajectory", trajectory);
        json::Value replay = json::Value::object();
        replay.set("integrator", problem.family == planner::Family::hcw
            ? "exact zero-order-hold HCW transition per substep"
            : "classical RK4 per substep, piecewise-constant controls");
        replay.set("substeps", static_cast<double>(substeps));
        replay.set(
            "times", times_json(intervals * substeps + 1U, problem.step_seconds / static_cast<double>(substeps))
        );
        replay.set("states", rows_json(dense_replay, nx));
        result.set("dense_replay", replay);
    }
    if (problem.output.include_iterations) {
        json::Value iterations = json::Value::array();
        for (const auto& record : records) {
            iterations.push_back(iteration_json(record));
        }
        result.set("iterations", iterations);
    }
    {
        json::Value timings = json::Value::object();
        timings.set("cuda_startup_seconds", cuda_startup_seconds);
        timings.set("topology_seconds", topology_seconds);
        timings.set("coefficient_seconds", coefficient_seconds);
        timings.set("workspace_create_seconds", workspace_create_seconds);
        timings.set("update_seconds", outer.update_seconds);
        timings.set("scaling_seconds", outer.scaling_seconds);
        timings.set("h2d_seconds", outer.h2d_seconds);
        timings.set("solve_seconds", outer.solve_seconds);
        timings.set("recovery_seconds", outer.recovery_seconds);
        timings.set("residual_seconds", outer.residual_seconds);
        timings.set("replay_seconds", outer.replay_seconds);
        timings.set("acceptance_seconds", outer.acceptance_seconds);
        timings.set("d2h_seconds", outer.d2h_seconds);
        timings.set("cqp_total_seconds", outer.cqp_total_seconds);
        timings.set("scvx_total_seconds", outer.scvx_total_seconds);
        timings.set("solve_wall_seconds", solve_wall_seconds);
        timings.set("qoco_conversion_seconds", outer.qoco_conversion_seconds);
        timings.set("qoco_setup_seconds", outer.qoco_setup_seconds);
        timings.set("qoco_update_seconds", outer.qoco_update_seconds);
        timings.set("qoco_solve_seconds", outer.qoco_solve_seconds);
        timings.set("independent_replay_seconds", independent_replay_seconds);
        timings.set("plan_wall_seconds", seconds_since(wall_started));
        result.set("timings", timings);
    }
    {
        json::Value disposition = json::Value::object();
        disposition.set("execution", "native_cuda");
        disposition.set("requested_backend", std::string(planner::backend_name(problem.solver.backend)));
        disposition.set("preset", std::string(planner::preset_name(problem.solver.preset)));
        disposition.set("device_policy", backend.policy_name);
        disposition.set("device_policy_code", static_cast<double>(backend.policy));
        disposition.set("description", backend.description);
        disposition.set("warm_start_mode", warm_start_name(outer_options.warm_start_mode));
        disposition.set("hidden_cpu_fallback", outer.hidden_cpu_fallback != 0);
        disposition.set("used_declared_stream", outer.used_declared_stream != 0);
        disposition.set("qoco_failure", qoco_failure_name(outer.qoco_failure));
        disposition.set("qoco_workspace_creations", static_cast<double>(outer.qoco_workspace_creations));
        disposition.set("qoco_numeric_updates", static_cast<double>(outer.qoco_numeric_updates));
        disposition.set("qoco_dual_discarded", outer.qoco_dual_discarded != 0);
        disposition.set("hybrid_handoff_eligible", outer.hybrid_handoff_eligible != 0);
        disposition.set("recovery_iterations", static_cast<double>(outer.recovery_iterations));
        disposition.set("recovery_count", static_cast<double>(diagnostics.recovery_count));
        disposition.set("recovery_rejected_count", static_cast<double>(diagnostics.recovery_rejected_count));
        disposition.set("recovery_attempt_count", static_cast<double>(diagnostics.recovery_attempt_count));
        disposition.set("allocation_count", static_cast<double>(outer.allocation_count));
        disposition.set("allocation_bytes", static_cast<double>(outer.allocation_bytes));
        disposition.set("h2d_copy_count", static_cast<double>(outer.h2d_copy_count));
        disposition.set("h2d_bytes", static_cast<double>(outer.h2d_bytes));
        disposition.set("d2h_copy_count", static_cast<double>(outer.d2h_copy_count));
        disposition.set("d2h_bytes", static_cast<double>(outer.d2h_bytes));
        disposition.set("device_copy_count", static_cast<double>(outer.device_copy_count));
        disposition.set("device_copy_bytes", static_cast<double>(outer.device_copy_bytes));
        disposition.set(
            "topology_allocation_count_after_create",
            static_cast<double>(outer.topology_allocation_count_after_create)
        );
        disposition.set(
            "topology_index_copy_count_after_create",
            static_cast<double>(outer.topology_index_copy_count_after_create)
        );
        char fingerprint_text[32];
        std::snprintf(
            fingerprint_text, sizeof(fingerprint_text), "%016llx",
            static_cast<unsigned long long>(fingerprint)
        );
        disposition.set("topology_fingerprint", std::string(fingerprint_text));
        disposition.set("variables", static_cast<double>(info.variables));
        disposition.set("scalar_rows", static_cast<double>(info.scalar_rows));
        disposition.set("affine_rows", static_cast<double>(info.affine_rows));
        disposition.set("device", device_json(true));
        result.set("backend", disposition);
    }
    {
        json::Value certificate = json::Value::object();
        certificate.set("certified", certified);
        certificate.set("tolerance", tolerance);
        certificate.set("replay_parity_tolerance", problem.solver.replay_parity_tolerance);
        certificate.set("gates", gate_json);
        certificate.set("failed_gates", failed);
        certificate.set("continuous_time_violation", continuous_time_violation);
        certificate.set("continuous_time_within_tolerance", continuous_time_violation <= tolerance);
        certificate.set(
            "definition",
            "certified only when the device SCvx driver converged with canonical, dynamics, path, "
            "terminal, and virtual-control residuals within tolerance AND an independent host "
            "RK4/ZOH replay of the returned controls reproduces the device trajectory to the "
            "parity tolerance, satisfies the nonlinear dynamics/path/terminal gates at the same "
            "tolerance, used no hidden CPU fallback, kept steady-state residency, and matched "
            "host/device coefficients; continuous-time violation is reported but not gated"
        );
        result.set("certificate", certificate);
    }
    return PlanOutcome{std::move(result), exit_code};
}

json::Value error_document(const std::string& code, const std::string& message, int exit_code) {
    json::Value result = json::Value::object();
    result.set("schema_version", "1.0.0");
    result.set("result_kind", "spacepdhcg_plan_result");
    result.set("source_commit", std::string(SPACEPDHCG_SOURCE_COMMIT));
    json::Value status = json::Value::object();
    status.set("code", code);
    status.set("message", message);
    status.set("exit_code", static_cast<double>(exit_code));
    result.set("status", status);
    json::Value certificate = json::Value::object();
    certificate.set("certified", false);
    result.set("certificate", certificate);
    return result;
}

json::Value capabilities_document() {
    json::Value capabilities = json::Value::object();
    capabilities.set("schema_version", "1.0.0");
    capabilities.set("executable", "spacepdhcg_plan");
    capabilities.set("source_commit", std::string(SPACEPDHCG_SOURCE_COMMIT));
    capabilities.set("problem_schema_version", std::string(planner::schema_version));
    capabilities.set(
        "families",
        json::Value(json::Array{
            json::Value("hcw"), json::Value("powered_descent_3dof"),
            json::Value("powered_descent_6dof"), json::Value("low_thrust"),
        })
    );
    capabilities.set(
        "backends",
        json::Value(json::Array{
            json::Value("pure_qoco"), json::Value("pdhcg"), json::Value("pdhcg_recovery"),
        })
    );
    capabilities.set(
        "presets",
        json::Value(json::Array{
            json::Value("frozen_adaptive_pure_qoco"), json::Value("frozen_adaptive_pdhcg"),
            json::Value("fixed_tight_pdhcg"),
        })
    );
    json::Value exit_codes = json::Value::object();
    exit_codes.set("certified", 0.0);
    exit_codes.set("not_certified", 2.0);
    exit_codes.set("solver_failure", 3.0);
    exit_codes.set("invalid_problem", 64.0);
    exit_codes.set("io_error", 65.0);
    exit_codes.set("cuda_error", 66.0);
    exit_codes.set("internal_error", 70.0);
    capabilities.set("exit_codes", exit_codes);
    const char* qoco = std::getenv("SPACEPDHCG_QOCO_LIBRARY");
    capabilities.set("qoco_library_configured", qoco != nullptr && qoco[0] != '\0');
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        count = 0;
    }
    capabilities.set("cuda_device_count", static_cast<double>(count));
    capabilities.set("device", device_json(count > 0));
    return capabilities;
}

int usage() {
    std::fprintf(
        stderr,
        "usage: spacepdhcg_plan <problem.json> [--output <result.json>] [--quiet]\n"
        "       spacepdhcg_plan --describe <problem.json>\n"
        "       spacepdhcg_plan --capabilities\n"
    );
    return exit_invalid_problem;
}

}  // namespace

int main(const int argc, char** argv) {
    std::string problem_path;
    std::string output_path;
    bool quiet = false;
    bool describe_only = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--capabilities") {
            const auto text = json::dump(capabilities_document(), 2);
            std::printf("%s\n", text.c_str());
            return 0;
        }
        if (argument == "--describe") {
            describe_only = true;
        } else if (argument == "--quiet") {
            quiet = true;
        } else if (argument == "--output") {
            if (index + 1 >= argc) {
                return usage();
            }
            output_path = argv[++index];
        } else if (!argument.empty() && argument.front() == '-') {
            return usage();
        } else if (problem_path.empty()) {
            problem_path = std::string(argument);
        } else {
            return usage();
        }
    }
    if (problem_path.empty()) {
        return usage();
    }

    int exit_code = exit_internal_error;
    json::Value document;
    try {
        const auto text = read_file(problem_path);
        planner::PlannerProblem problem;
        try {
            problem = planner::parse_problem_text(text);
        } catch (const json::ParseError& error) {
            throw PlanError(exit_invalid_problem, std::string("invalid problem JSON: ") + error.what());
        } catch (const json::TypeError& error) {
            throw PlanError(exit_invalid_problem, std::string("invalid problem document: ") + error.what());
        } catch (const std::invalid_argument& error) {
            throw PlanError(exit_invalid_problem, std::string("invalid problem document: ") + error.what());
        }
        if (describe_only) {
            document = json::Value::object();
            document.set("schema_version", "1.0.0");
            document.set("result_kind", "spacepdhcg_plan_describe");
            document.set("problem", planner::describe_problem(problem));
            try {
                const auto adapter = planner::make_adapter(problem);
                const auto& info = adapter->layout();
                json::Value dimensions = json::Value::object();
                dimensions.set("variables", static_cast<double>(info.variables));
                dimensions.set("scalar_rows", static_cast<double>(info.scalar_rows));
                dimensions.set("affine_rows", static_cast<double>(info.affine_rows));
                document.set("dimensions", dimensions);
            } catch (const std::invalid_argument& error) {
                throw PlanError(exit_invalid_problem, std::string("invalid problem document: ") + error.what());
            }
            exit_code = 0;
        } else {
            if (problem.solver.backend == planner::Backend::cpu_reference) {
                throw PlanError(
                    exit_invalid_problem,
                    "backend cpu_reference is served by the Python reference solver, not by "
                    "spacepdhcg_plan"
                );
            }
            const auto startup = Clock::now();
            int device_count = 0;
            const auto count_status = cudaGetDeviceCount(&device_count);
            if (count_status != cudaSuccess || device_count == 0) {
                throw PlanError(
                    exit_cuda_error,
                    std::string("no CUDA device is available: ")
                        + (count_status == cudaSuccess ? "device count is zero"
                                                       : cudaGetErrorString(count_status))
                );
            }
            cuda_check(cudaFree(nullptr), "CUDA context startup");
            const double startup_seconds = seconds_since(startup);
            try {
                auto outcome = run_plan(problem, startup_seconds, quiet);
                document = std::move(outcome.result);
                exit_code = outcome.exit_code;
            } catch (const std::invalid_argument& error) {
                throw PlanError(exit_invalid_problem, std::string("invalid problem document: ") + error.what());
            }
        }
    } catch (const PlanError& error) {
        exit_code = error.code();
        const std::string code = exit_code == exit_invalid_problem ? "invalid_problem"
            : exit_code == exit_io_error                            ? "io_error"
            : exit_code == exit_cuda_error                          ? "cuda_error"
                                                                    : "internal_error";
        document = error_document(code, error.what(), exit_code);
        std::fprintf(stderr, "spacepdhcg_plan: %s\n", error.what());
    } catch (const std::exception& error) {
        exit_code = exit_internal_error;
        document = error_document("internal_error", error.what(), exit_code);
        std::fprintf(stderr, "spacepdhcg_plan: internal error: %s\n", error.what());
    } catch (...) {
        exit_code = exit_internal_error;
        document = error_document("internal_error", "unknown native failure", exit_code);
        std::fprintf(stderr, "spacepdhcg_plan: unknown native failure\n");
    }
    try {
        write_output(output_path, json::dump(document));
    } catch (const PlanError& error) {
        std::fprintf(stderr, "spacepdhcg_plan: %s\n", error.what());
        return exit_io_error;
    }
    return exit_code;
}
