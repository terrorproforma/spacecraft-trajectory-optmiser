#pragma once

#include "spacepdhcg/core/fixed_cqp.hpp"
#include "spacepdhcg/core/scaling_reuse.hpp"
#include "spacepdhcg/persistent_cqp.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace spacepdhcg::core {

struct SessionSolveRecord {
    SolveReport report{};
    RescalePolicy rescale_policy{RescalePolicy::reuse};
    NumericChangeMetrics change_metrics{};
    std::uint64_t numerical_update{0U};
    bool warm_start_applied{false};
};

/// Backend-independent owner for one persistent native CQP workspace.
///
/// The session retains a host mirror solely for scaling-change decisions and checkpointing.
/// Numerical pointers passed to `submit` are device-resident in production. A host-only fake may
/// interpret them as ordinary pointers for lifecycle tests.
class PersistentSolveSession {
  public:
    PersistentSolveSession(
        std::unique_ptr<PersistentCQP> workspace,
        FixedStructure structure,
        NumericValues initial_host_values,
        ScalingThresholds scaling_thresholds = {}
    )
        : workspace_(std::move(workspace)),
          structure_(std::move(structure)),
          host_values_(std::move(initial_host_values)),
          scaling_(scaling_thresholds),
          topology_fingerprint_(structure_.fingerprint()) {
        if (!workspace_) {
            throw std::invalid_argument("persistent solve session requires a workspace");
        }
        structure_.validate();
        host_values_.validate(structure_);
        validate_workspace_shape();
    }

    PersistentSolveSession(const PersistentSolveSession&) = delete;
    PersistentSolveSession& operator=(const PersistentSolveSession&) = delete;
    PersistentSolveSession(PersistentSolveSession&&) noexcept = default;
    PersistentSolveSession& operator=(PersistentSolveSession&&) noexcept = default;

    [[nodiscard]] const FixedStructure& structure() const noexcept { return structure_; }
    [[nodiscard]] const NumericValues& host_values() const noexcept { return host_values_; }
    [[nodiscard]] std::uint64_t topology_fingerprint() const noexcept {
        return topology_fingerprint_;
    }
    [[nodiscard]] std::uint64_t update_count() const noexcept { return update_count_; }
    [[nodiscard]] PersistentCQP& workspace() noexcept { return *workspace_; }
    [[nodiscard]] const PersistentCQP& workspace() const noexcept { return *workspace_; }
    [[nodiscard]] const std::optional<SessionSolveRecord>& last_record() const noexcept {
        return last_record_;
    }

    [[nodiscard]] SessionSolveRecord submit(
        NumericValues next_host_values,
        const NumericValuesView& next_device_values,
        const SolveOptions& solve_options,
        StreamHandle stream,
        const WarmStartView* warm_start = nullptr,
        std::optional<RescalePolicy> override_rescale_policy = std::nullopt
    ) {
        next_host_values.validate(structure_);
        validate_device_values(next_device_values);
        if (workspace_->state() == WorkspaceState::solving) {
            throw std::logic_error("cannot submit a numerical update while a solve is in flight");
        }
        const auto automatic_policy = scaling_.observe(
            host_values_,
            next_host_values,
            structure_
        );
        const auto policy = override_rescale_policy.value_or(automatic_policy);
        workspace_->update_values(next_device_values, policy, stream);
        bool warm_start_applied{false};
        if (warm_start != nullptr) {
            validate_warm_start(*warm_start);
            workspace_->set_warm_start(*warm_start, stream);
            warm_start_applied = true;
        }
        workspace_->solve_async(solve_options, stream);
        auto report = workspace_->synchronize();
        host_values_ = std::move(next_host_values);
        ++update_count_;
        last_record_ = SessionSolveRecord{
            report,
            policy,
            scaling_.last_metrics(),
            update_count_,
            warm_start_applied,
        };
        return *last_record_;
    }

    void reset_iterates(StreamHandle stream) {
        workspace_->reset_iterates(stream);
        last_record_.reset();
    }

    void request_cancel() noexcept { workspace_->request_cancel(); }

  private:
    std::unique_ptr<PersistentCQP> workspace_{};
    FixedStructure structure_{};
    NumericValues host_values_{};
    ScalingReuseController scaling_{};
    std::uint64_t topology_fingerprint_{0U};
    std::uint64_t update_count_{0U};
    std::optional<SessionSolveRecord> last_record_{};

    void validate_workspace_shape() const {
        if (workspace_->variables() != structure_.variables()
            || workspace_->scalar_constraints() != structure_.scalar_rows()
            || workspace_->affine_cone_rows() != structure_.affine_rows()) {
            throw std::invalid_argument(
                "persistent workspace dimensions do not match the fixed CQP structure"
            );
        }
    }

    void validate_device_values(const NumericValuesView& values) const {
        require_span(values.quadratic_values, structure_.quadratic.nonzeros(), "quadratic");
        require_span(
            values.scalar_constraint_values,
            structure_.scalar_constraint.nonzeros(),
            "scalar constraint"
        );
        require_span(
            values.affine_cone_values,
            structure_.affine_cone.has_value() ? structure_.affine_cone->nonzeros() : 0U,
            "affine cone"
        );
        require_span(
            values.linear_objective,
            static_cast<std::size_t>(structure_.variables()),
            "linear objective"
        );
        require_span(
            values.scalar_lower,
            static_cast<std::size_t>(structure_.scalar_rows()),
            "scalar lower"
        );
        require_span(
            values.scalar_upper,
            static_cast<std::size_t>(structure_.scalar_rows()),
            "scalar upper"
        );
        require_span(
            values.affine_offset,
            static_cast<std::size_t>(structure_.affine_rows()),
            "affine offset"
        );
        require_span(
            values.variable_lower,
            static_cast<std::size_t>(structure_.variables()),
            "variable lower"
        );
        require_span(
            values.variable_upper,
            static_cast<std::size_t>(structure_.variables()),
            "variable upper"
        );
    }

    void validate_warm_start(const WarmStartView& warm_start) const {
        if (!warm_start.primal.empty()) {
            require_span(
                warm_start.primal,
                static_cast<std::size_t>(structure_.variables()),
                "primal warm start"
            );
        }
        if (!warm_start.dual.empty()) {
            require_span(
                warm_start.dual,
                static_cast<std::size_t>(structure_.duals()),
                "dual warm start"
            );
        }
        if (warm_start.primal.empty() && warm_start.dual.empty()) {
            throw std::invalid_argument("at least one warm-start vector is required");
        }
    }

    template <typename T>
    static void require_span(
        const DeviceConstSpan<T>& span,
        std::size_t expected,
        const char* name
    ) {
        if (span.size != expected || (expected > 0U && span.data == nullptr)) {
            throw std::invalid_argument(std::string(name) + " device span has the wrong size");
        }
    }
};

}  // namespace spacepdhcg::core
