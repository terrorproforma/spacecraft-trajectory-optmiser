#pragma once

#include "spacepdhcg/core/fixed_cqp.hpp"
#include "spacepdhcg/persistent_cqp.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace spacepdhcg::core {

struct HostWarmStart {
    std::vector<double> primal{};
    std::vector<double> dual{};
};

struct HostCqpSolution {
    SolveStatus status{SolveStatus::internal_error};
    std::vector<double> primal{};
    std::vector<double> dual{};
    double objective{0.0};
    double primal_residual{0.0};
    double dual_residual{0.0};
    std::size_t outer_iterations{0U};
    std::size_t inner_iterations{0U};
    double setup_seconds{0.0};
    double update_seconds{0.0};
    double solve_seconds{0.0};

    [[nodiscard]] bool solved() const noexcept { return status == SolveStatus::optimal; }
};

/// Host-facing lifecycle implemented by CPU references and the future native CUDA adapter.
///
/// `update` may change numerical values only. The backend must reject a different topology.
class HostPersistentBackend {
  public:
    HostPersistentBackend(const HostPersistentBackend&) = delete;
    HostPersistentBackend& operator=(const HostPersistentBackend&) = delete;
    HostPersistentBackend(HostPersistentBackend&&) = delete;
    HostPersistentBackend& operator=(HostPersistentBackend&&) = delete;
    virtual ~HostPersistentBackend() = default;

    [[nodiscard]] virtual const FixedStructure& structure() const noexcept = 0;
    [[nodiscard]] virtual std::size_t update_count() const noexcept = 0;
    virtual void update(NumericValues values) = 0;
    virtual void warm_start(const HostWarmStart& start) = 0;
    [[nodiscard]] virtual HostCqpSolution solve(
        double tolerance,
        std::size_t iteration_limit
    ) = 0;

  protected:
    HostPersistentBackend() = default;
};

using HostBackendPointer = std::unique_ptr<HostPersistentBackend>;

}  // namespace spacepdhcg::core
