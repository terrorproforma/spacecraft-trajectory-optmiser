#pragma once

#include "spacepdhcg/native/cqp.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace spacepdhcg::native {

enum class CqpSolveStatus {
    solved,
    solved_inaccurate,
    primal_infeasible,
    dual_infeasible,
    iteration_limit,
    interrupted,
    numerical_failure,
    internal_error,
};

struct CqpSolveOptions {
    double optimality_tolerance{1.0e-4};
    double feasibility_tolerance{1.0e-4};
    std::size_t iteration_limit{1'000'000};

    void validate() const;
};

struct CqpSolveResult {
    CqpSolveStatus status{CqpSolveStatus::internal_error};
    std::vector<double> primal{};
    std::vector<double> dual{};
    double objective{0.0};
    double primal_residual{0.0};
    double dual_residual{0.0};
    std::size_t iterations{0};
    double update_seconds{0.0};
    double solve_seconds{0.0};
    std::string backend_message{};

    [[nodiscard]] bool solved() const noexcept {
        return status == CqpSolveStatus::solved ||
               status == CqpSolveStatus::solved_inaccurate;
    }
};

class CqpWorkspace {
  public:
    CqpWorkspace(const CqpWorkspace&) = delete;
    CqpWorkspace& operator=(const CqpWorkspace&) = delete;
    CqpWorkspace(CqpWorkspace&&) = delete;
    CqpWorkspace& operator=(CqpWorkspace&&) = delete;
    virtual ~CqpWorkspace() = default;

    [[nodiscard]] virtual std::string backend_name() const = 0;
    [[nodiscard]] virtual bool persistent() const noexcept = 0;
    virtual void update(const OwnedCqp& problem) = 0;
    virtual void warm_start(
        std::span<const double> primal,
        std::span<const double> dual
    ) = 0;
    virtual void clear_warm_start() = 0;
    [[nodiscard]] virtual CqpSolveResult solve(const CqpSolveOptions& options) = 0;

  protected:
    CqpWorkspace() = default;
};

using CqpWorkspaceFactory =
    std::function<std::unique_ptr<CqpWorkspace>(const OwnedCqp& initial_problem)>;

[[nodiscard]] bool residual_qualified(
    const CqpSolveResult& result,
    double tolerance
);

}  // namespace spacepdhcg::native
