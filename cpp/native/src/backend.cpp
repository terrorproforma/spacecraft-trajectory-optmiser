#include "spacepdhcg/native/backend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spacepdhcg::native {

void CqpSolveOptions::validate() const {
    if (!std::isfinite(optimality_tolerance) || optimality_tolerance <= 0.0 ||
        !std::isfinite(feasibility_tolerance) || feasibility_tolerance <= 0.0) {
        throw std::invalid_argument("CQP tolerances must be finite and positive");
    }
    if (iteration_limit == 0) {
        throw std::invalid_argument("CQP iteration limit must be positive");
    }
}

bool residual_qualified(const CqpSolveResult& result, double tolerance) {
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument("qualification tolerance must be finite and positive");
    }
    if (!result.solved()) {
        return false;
    }
    if (!std::isfinite(result.primal_residual) || !std::isfinite(result.dual_residual)) {
        return false;
    }
    return std::max(std::abs(result.primal_residual), std::abs(result.dual_residual)) <=
           tolerance;
}

}  // namespace spacepdhcg::native
