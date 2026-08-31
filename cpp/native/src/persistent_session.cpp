#include "spacepdhcg/native/persistent_session.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace spacepdhcg::native {
namespace {

[[nodiscard]] bool same_pattern(const CscMatrix& left, const CscMatrix& right) {
    return left.rows == right.rows && left.columns == right.columns &&
           left.offsets == right.offsets && left.indices == right.indices;
}

[[nodiscard]] bool same_cones(
    const std::vector<ConeBlockDescriptor>& left,
    const std::vector<ConeBlockDescriptor>& right
) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index].kind != right[index].kind || left[index].start != right[index].start ||
            left[index].vector_dimension != right[index].vector_dimension ||
            left[index].power_alpha != right[index].power_alpha) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool SessionSolveResult::acceptable(double tolerance) const {
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument("session acceptance tolerance must be finite and positive");
    }
    return residual_qualified(backend, tolerance) &&
           independent.primal.maximum_violation() <= tolerance &&
           independent.objective_disagreement <= tolerance *
               std::max(1.0, std::abs(independent.independent_objective));
}

PersistentCqpSession::PersistentCqpSession(
    const OwnedCqp& initial_problem,
    CqpWorkspaceFactory factory
)
    : topology_reference_(initial_problem) {
    topology_reference_.validate();
    if (!factory) {
        throw std::invalid_argument("persistent session requires a workspace factory");
    }
    workspace_ = factory(initial_problem);
    if (!workspace_) {
        throw std::runtime_error("workspace factory returned a null backend");
    }
    backend_name_ = workspace_->backend_name();
    backend_is_persistent_ = workspace_->persistent();
}

void PersistentCqpSession::assert_compatible(const OwnedCqp& problem) const {
    if (!same_pattern(problem.quadratic, topology_reference_.quadratic) ||
        !same_pattern(problem.scalar_constraint, topology_reference_.scalar_constraint) ||
        !same_pattern(problem.affine_cone, topology_reference_.affine_cone) ||
        !same_cones(problem.affine_cones, topology_reference_.affine_cones) ||
        !same_cones(problem.variable_cones, topology_reference_.variable_cones)) {
        throw std::invalid_argument("persistent session received a different CQP topology");
    }
}

SessionSolveResult PersistentCqpSession::solve(
    const OwnedCqp& problem,
    const CqpSolveOptions& options,
    bool reuse_previous_warm_start
) {
    options.validate();
    problem.validate();
    assert_compatible(problem);

    workspace_->update(problem);
    ++update_count_;
    bool warm_started = false;
    if (reuse_previous_warm_start && previous_primal_.has_value() &&
        previous_dual_.has_value()) {
        workspace_->warm_start(*previous_primal_, *previous_dual_);
        warm_started = true;
    }

    auto backend = workspace_->solve(options);
    ++solve_count_;
    if (backend.primal.size() != static_cast<std::size_t>(problem.variables())) {
        throw std::runtime_error("backend returned a primal vector with the wrong dimension");
    }

    const double independent_objective = problem.objective(backend.primal);
    const auto independent_primal = problem.diagnostics(backend.primal);
    const double objective_disagreement =
        std::isfinite(backend.objective)
            ? std::abs(backend.objective - independent_objective)
            : std::numeric_limits<double>::infinity();

    if (backend.solved()) {
        previous_primal_ = backend.primal;
        previous_dual_ = backend.dual;
    }
    return SessionSolveResult{
        std::move(backend),
        SessionDiagnostics{
            backend.objective,
            independent_objective,
            objective_disagreement,
            independent_primal,
        },
        solve_count_ - 1U,
        warm_started,
    };
}

void PersistentCqpSession::set_warm_start(
    std::vector<double> primal,
    std::vector<double> dual
) {
    if (primal.size() != static_cast<std::size_t>(topology_reference_.variables())) {
        throw std::invalid_argument("session primal warm start has the wrong dimension");
    }
    if (dual.size() != static_cast<std::size_t>(
            topology_reference_.scalar_constraint.rows + topology_reference_.affine_cone.rows
        )) {
        throw std::invalid_argument("session dual warm start has the wrong dimension");
    }
    previous_primal_ = std::move(primal);
    previous_dual_ = std::move(dual);
}

void PersistentCqpSession::clear_warm_start() {
    previous_primal_.reset();
    previous_dual_.reset();
    workspace_->clear_warm_start();
}

}  // namespace spacepdhcg::native
