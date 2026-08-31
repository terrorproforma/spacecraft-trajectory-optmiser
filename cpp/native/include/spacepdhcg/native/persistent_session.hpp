#pragma once

#include "spacepdhcg/native/backend.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace spacepdhcg::native {

struct SessionDiagnostics {
    double backend_objective{0.0};
    double independent_objective{0.0};
    double objective_disagreement{0.0};
    CqpDiagnostics primal{};
};

struct SessionSolveResult {
    CqpSolveResult backend{};
    SessionDiagnostics independent{};
    std::size_t solve_index{0};
    bool warm_started{false};

    [[nodiscard]] bool acceptable(double tolerance) const;
};

class PersistentCqpSession {
  public:
    PersistentCqpSession(
        const OwnedCqp& initial_problem,
        CqpWorkspaceFactory factory
    );

    [[nodiscard]] const std::string& backend_name() const noexcept { return backend_name_; }
    [[nodiscard]] bool backend_is_persistent() const noexcept { return backend_is_persistent_; }
    [[nodiscard]] std::size_t update_count() const noexcept { return update_count_; }
    [[nodiscard]] std::size_t solve_count() const noexcept { return solve_count_; }
    [[nodiscard]] bool has_warm_start() const noexcept { return previous_primal_.has_value(); }

    [[nodiscard]] SessionSolveResult solve(
        const OwnedCqp& problem,
        const CqpSolveOptions& options,
        bool reuse_previous_warm_start = true
    );

    void set_warm_start(
        std::vector<double> primal,
        std::vector<double> dual
    );
    void clear_warm_start();

  private:
    OwnedCqp topology_reference_{};
    std::unique_ptr<CqpWorkspace> workspace_{};
    std::string backend_name_{};
    bool backend_is_persistent_{false};
    std::size_t update_count_{0};
    std::size_t solve_count_{0};
    std::optional<std::vector<double>> previous_primal_{};
    std::optional<std::vector<double>> previous_dual_{};

    void assert_compatible(const OwnedCqp& problem) const;
};

}  // namespace spacepdhcg::native
