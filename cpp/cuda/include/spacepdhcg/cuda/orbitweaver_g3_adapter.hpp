#pragma once

#include "spacepdhcg/cuda/device_scvx_driver_c_api.h"
#include "spacepdhcg/orbitweaver/g7_orchestration.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver::g7 {

/// Per-arc public-G3 binding. Implementations own fixed device buffers and update their
/// numerical contents without replacing sparse topology or workspace allocations.
class G3ArcBinding {
  public:
    virtual ~G3ArcBinding() = default;
    [[nodiscard]] virtual const spacepdhcg_cuda_scvx_problem& problem() const = 0;
    [[nodiscard]] virtual const spacepdhcg_cuda_scvx_options& options() const = 0;
    virtual void update_numeric_in_place(
        const ScheduledArc& arc,
        spacepdhcg_accelerator_stream stream
    ) = 0;
    [[nodiscard]] virtual bool import_warm_token(
        std::uint64_t token,
        const ScheduledArc& arc,
        spacepdhcg_accelerator_stream stream
    ) = 0;
    [[nodiscard]] virtual std::optional<std::uint64_t> export_warm_token(
        const ScheduledArc& arc,
        spacepdhcg_accelerator_stream stream
    ) = 0;
    [[nodiscard]] virtual ArcSolution decode(
        const ScheduledArc& arc,
        const spacepdhcg_cuda_scvx_result& result,
        const spacepdhcg_cuda_scvx_path_inventory& path
    ) = 0;
    [[nodiscard]] virtual double independent_replay_residual(
        const ScheduledArc& arc,
        spacepdhcg_accelerator_stream stream
    ) = 0;
};

using G3ArcBindingFactory = std::function<std::unique_ptr<G3ArcBinding>(
    const TopologyFidelityKey&,
    Ownership
)>;

/// Concrete bounded adapter around the public G3 persistent device-SCvx C API.
///
/// One driver and one fixed-buffer binding are retained per compatible topology and
/// rank/device owner.  Every subsequent arc calls the public numerical-update API before
/// solve; sparse indices and workspace ownership never cross this boundary.
class G3PersistentTrajectoryAdapter final : public ArcBatchBackend {
  public:
    explicit G3PersistentTrajectoryAdapter(
        G3ArcBindingFactory factory,
        std::size_t maximum_workspaces = 16U
    )
        : factory_(std::move(factory)),
          maximum_workspaces_(maximum_workspaces) {
        if (!factory_ || maximum_workspaces_ == 0U) {
            throw std::invalid_argument("G3 adapter requires a factory and workspace bound");
        }
    }

    ~G3PersistentTrajectoryAdapter() override { clear(); }

    G3PersistentTrajectoryAdapter(const G3PersistentTrajectoryAdapter&) = delete;
    G3PersistentTrajectoryAdapter& operator=(const G3PersistentTrajectoryAdapter&) = delete;

    [[nodiscard]] std::vector<ArcExecution> evaluate(
        const TopologyFidelityKey& group,
        const std::vector<ScheduledArc>& batch,
        const Ownership owner,
        const std::atomic<bool>& cancelled
    ) override {
        auto& workspace = acquire(group, owner);
        std::vector<ArcExecution> output{};
        output.reserve(batch.size());
        for (const auto& arc : batch) {
            if (cancelled.load()) {
                static_cast<void>(spacepdhcg_cuda_scvx_driver_cancel(workspace.driver));
                output.push_back(failure(arc, ArcExecutionStatus::cancelled, "G3 cancelled"));
                continue;
            }
            if (arc.request.warm_start_token.has_value()
                && !workspace.binding->import_warm_token(
                    *arc.request.warm_start_token,
                    arc,
                    stream_
                )) {
                output.push_back(failure(
                    arc,
                    ArcExecutionStatus::warm_start_incompatible,
                    "G3 warm token is incompatible with this arc/topology"
                ));
                continue;
            }
            try {
                workspace.binding->update_numeric_in_place(arc, stream_);
            } catch (const std::exception& error) {
                output.push_back(failure(
                    arc,
                    ArcExecutionStatus::invalid_input,
                    error.what()
                ));
                continue;
            }
            const auto& problem = workspace.binding->problem();
            if (problem.topology_fingerprint != group.topology_fingerprint
                || problem.intervals != group.intervals) {
                output.push_back(failure(
                    arc,
                    ArcExecutionStatus::topology_mismatch,
                    "G3 binding changed frozen topology"
                ));
                continue;
            }
            auto status = spacepdhcg_cuda_scvx_update_numeric_async(
                &problem,
                workspace.binding->options().initial_trust_radius,
                workspace.binding->options().virtual_penalty,
                stream_
            );
            if (status != SPACEPDHCG_CUDA_SUCCESS) {
                output.push_back(api_failure(arc, status, "G3 numerical update failed"));
                continue;
            }
            std::vector<spacepdhcg_cuda_scvx_iteration> iterations(
                workspace.binding->options().maximum_outer_iterations
            );
            spacepdhcg_cuda_scvx_result result{};
            result.abi_version = 1U;
            status = spacepdhcg_cuda_scvx_driver_solve(
                workspace.driver,
                stream_,
                iterations.data(),
                iterations.size(),
                &result
            );
            if (status != SPACEPDHCG_CUDA_SUCCESS) {
                output.push_back(api_failure(arc, status, "G3 SCvx solve failed"));
                continue;
            }
            spacepdhcg_cuda_scvx_path_inventory path{};
            path.abi_version = 1U;
            status = spacepdhcg_cuda_scvx_driver_path_inventory(workspace.driver, &path);
            if (status != SPACEPDHCG_CUDA_SUCCESS) {
                output.push_back(api_failure(arc, status, "G3 path inventory failed"));
                continue;
            }
            output.push_back(convert(arc, workspace, result, path));
        }
        return output;
    }

    void set_stream(spacepdhcg_accelerator_stream stream) noexcept { stream_ = stream; }

    void clear() noexcept {
        for (auto& [_, workspace] : workspaces_) {
            static_cast<void>(spacepdhcg_cuda_scvx_driver_destroy(&workspace.driver));
        }
        workspaces_.clear();
    }

    [[nodiscard]] std::size_t workspace_count() const noexcept {
        return workspaces_.size();
    }

  private:
    struct WorkspaceKey {
        TopologyFidelityKey group{};
        Ownership owner{};
        [[nodiscard]] bool operator<(const WorkspaceKey& other) const noexcept {
            if (group < other.group) {
                return true;
            }
            if (other.group < group) {
                return false;
            }
            return owner < other.owner;
        }
    };

    struct Workspace {
        std::unique_ptr<G3ArcBinding> binding{};
        spacepdhcg_cuda_scvx_driver* driver{nullptr};
        std::uint64_t use_sequence{0U};
    };

    G3ArcBindingFactory factory_{};
    std::size_t maximum_workspaces_{16U};
    std::uint64_t sequence_{0U};
    spacepdhcg_accelerator_stream stream_{};
    std::map<WorkspaceKey, Workspace> workspaces_{};

    [[nodiscard]] Workspace& acquire(
        const TopologyFidelityKey& group,
        const Ownership owner
    ) {
        const WorkspaceKey key{group, owner};
        const auto existing = workspaces_.find(key);
        if (existing != workspaces_.end()) {
            existing->second.use_sequence = ++sequence_;
            return existing->second;
        }
        if (workspaces_.size() == maximum_workspaces_) {
            const auto victim = std::min_element(
                workspaces_.begin(),
                workspaces_.end(),
                [](const auto& left, const auto& right) {
                    return left.second.use_sequence < right.second.use_sequence;
                }
            );
            static_cast<void>(
                spacepdhcg_cuda_scvx_driver_destroy(&victim->second.driver)
            );
            workspaces_.erase(victim);
        }
        auto binding = factory_(group, owner);
        if (!binding) {
            throw std::runtime_error("G3 binding factory returned null");
        }
        const auto& problem = binding->problem();
        if (problem.topology_fingerprint != group.topology_fingerprint
            || problem.intervals != group.intervals) {
            throw std::runtime_error("G3 binding topology does not match scheduled group");
        }
        spacepdhcg_cuda_scvx_driver* driver{nullptr};
        const auto status = spacepdhcg_cuda_scvx_driver_create(
            &problem,
            &binding->options(),
            &driver
        );
        if (status != SPACEPDHCG_CUDA_SUCCESS || driver == nullptr) {
            throw std::runtime_error("public G3 driver creation failed");
        }
        const auto [iterator, inserted] = workspaces_.emplace(
            key,
            Workspace{std::move(binding), driver, ++sequence_}
        );
        if (!inserted) {
            static_cast<void>(spacepdhcg_cuda_scvx_driver_destroy(&driver));
            throw std::logic_error("duplicate G3 workspace key");
        }
        return iterator->second;
    }

    [[nodiscard]] ArcExecution convert(
        const ScheduledArc& arc,
        Workspace& workspace,
        const spacepdhcg_cuda_scvx_result& result,
        const spacepdhcg_cuda_scvx_path_inventory& path
    ) {
        if (result.status == SPACEPDHCG_CUDA_SCVX_CANCELLED) {
            return failure(arc, ArcExecutionStatus::cancelled, "G3 SCvx cancelled");
        }
        if (result.status == SPACEPDHCG_CUDA_SCVX_INNER_FAILURE
            || result.status == SPACEPDHCG_CUDA_SCVX_INVALID) {
            return failure(
                arc,
                ArcExecutionStatus::numerical_failure,
                "G3 SCvx numerical/inner failure"
            );
        }
        auto solution = workspace.binding->decode(arc, result, path);
        solution.lower_bound = std::max(
            arc.inherited_lower_bound,
            std::min(solution.cost, solution.lower_bound)
        );
        solution.warm_start_token =
            workspace.binding->export_warm_token(arc, stream_);
        solution.validate(arc.request);
        const ArcQuality quality{
            result.canonical_residual,
            workspace.binding->independent_replay_residual(arc, stream_),
            std::max(
                {result.path_violation,
                 path.thrust_violation,
                 path.mass_violation,
                 path.altitude_violation}
            ),
            result.terminal_residual,
        };
        if (!std::isfinite(quality.replay_residual)) {
            return failure(
                arc,
                ArcExecutionStatus::numerical_failure,
                "G3 independent replay residual is non-finite"
            );
        }
        return {
            arc.deterministic_id,
            ArcExecutionStatus::feasible,
            std::move(solution),
            0U,
            0U,
            0U,
            result.status == SPACEPDHCG_CUDA_SCVX_CONVERGED
                ? "G3 SCvx converged"
                : "G3 SCvx returned a bounded non-converged candidate",
            quality,
        };
    }

    [[nodiscard]] static ArcExecution failure(
        const ScheduledArc& arc,
        const ArcExecutionStatus status,
        std::string diagnostic
    ) {
        return {
            arc.deterministic_id,
            status,
            {},
            0U,
            0U,
            0U,
            std::move(diagnostic),
            {},
        };
    }

    [[nodiscard]] static ArcExecution api_failure(
        const ScheduledArc& arc,
        const spacepdhcg_cuda_status status,
        const char* diagnostic
    ) {
        auto classification = ArcExecutionStatus::backend_failure;
        if (status == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH) {
            classification = ArcExecutionStatus::topology_mismatch;
        } else if (status == SPACEPDHCG_CUDA_NUMERICAL_FAILURE) {
            classification = ArcExecutionStatus::numerical_failure;
        } else if (status == SPACEPDHCG_CUDA_UNSUPPORTED) {
            classification = ArcExecutionStatus::unsupported;
        } else if (status == SPACEPDHCG_CUDA_OUT_OF_MEMORY) {
            classification = ArcExecutionStatus::out_of_memory;
        }
        return failure(arc, classification, diagnostic);
    }
};

}  // namespace spacepdhcg::orbitweaver::g7
