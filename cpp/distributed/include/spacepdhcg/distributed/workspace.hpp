/*
 * Persistent rank-local scenario workspace composition for Gate G5.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "spacepdhcg/cuda/persistent_pdhcg_c_api.h"
#include "spacepdhcg/distributed/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace spacepdhcg::distributed::g5 {

struct LocalScenarioCreate {
    std::size_t scenario{0};
    const spacepdhcg_cuda_structure* structure{nullptr};
    const spacepdhcg_cqp_accelerator_exchange* exchange{nullptr};
    const spacepdhcg_cuda_create_options* options{nullptr};
};

struct LocalScenarioDiagnostics {
    std::size_t scenario{0};
    spacepdhcg_cuda_diagnostics diagnostics{};
};

class DistributedWorkspace {
  public:
    DistributedWorkspace(
        RuntimeOptions runtime_options,
        PartitionPlan partition,
        ArrowheadMetadata arrowhead,
        std::span<const LocalScenarioCreate> local_scenarios
    );
    ~DistributedWorkspace();

    DistributedWorkspace(const DistributedWorkspace&) = delete;
    DistributedWorkspace& operator=(const DistributedWorkspace&) = delete;

    [[nodiscard]] MpiNcclRuntime& runtime() noexcept { return runtime_; }
    [[nodiscard]] const MpiNcclRuntime& runtime() const noexcept { return runtime_; }
    [[nodiscard]] const PartitionPlan& partition() const noexcept { return partition_; }
    [[nodiscard]] const ArrowheadMetadata& arrowhead() const noexcept { return arrowhead_; }
    [[nodiscard]] std::size_t local_scenario_count() const noexcept {
        return local_.size();
    }

    void update_local_async(
        std::size_t scenario,
        const spacepdhcg_cqp_numeric_accelerator_views& values,
        std::uint64_t topology_fingerprint
    );
    void warm_start_local_async(
        std::size_t scenario,
        spacepdhcg_cuda_warm_start_mode mode,
        const spacepdhcg_cqp_iterate_accelerator_views* iterates
    );
    void refresh_scaling_all_async();
    void solve_all_async(const spacepdhcg_cuda_solve_options& options);
    void residuals_all_async();
    void wait_all();
    void reduce_shared_arrowhead(
        double* device_values,
        std::size_t count,
        std::uint64_t frequency = 1
    );
    void reduce_global_residual_sums(
        double* device_values,
        std::size_t count,
        std::uint64_t frequency
    );
    void reduce_global_residual_maxima(
        double* device_values,
        std::size_t count,
        std::uint64_t frequency
    );
    void reduce_expected_risk(
        double* device_values,
        std::size_t count,
        std::uint64_t frequency
    );
    void reduce_worst_risk(
        double* device_values,
        std::size_t count,
        std::uint64_t frequency
    );
    void reduce_cvar_epigraph(
        double* device_values,
        std::size_t count,
        std::uint64_t frequency
    );
    [[nodiscard]] std::vector<LocalScenarioDiagnostics> diagnostics() const;
    [[nodiscard]] std::vector<std::byte> checkpoint();
    void restore(std::span<const std::byte> checkpoint);
    void cancel() noexcept;

  private:
    struct OwnedScenario {
        std::size_t scenario{0};
        std::uint64_t topology_fingerprint{0};
        std::size_t primal_elements{0};
        std::size_t dual_elements{0};
        spacepdhcg_cuda_workspace* workspace{nullptr};
    };

    MpiNcclRuntime runtime_;
    PartitionPlan partition_;
    ArrowheadMetadata arrowhead_;
    std::vector<OwnedScenario> local_{};
    WarmOwnership warm_ownership_{WarmOwnership::none};

    [[nodiscard]] spacepdhcg_accelerator_stream stream() const noexcept;
    [[nodiscard]] OwnedScenario& owned(std::size_t scenario);
    [[nodiscard]] const OwnedScenario& owned(std::size_t scenario) const;
    static void require_success(spacepdhcg_cuda_status status, const char* operation);
};

}  // namespace spacepdhcg::distributed::g5
