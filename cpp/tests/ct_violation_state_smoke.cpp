#include "spacepdhcg/backends/dense_admm.hpp"
#include "spacepdhcg/transcription/ct_violation_state.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace {

spacepdhcg::core::FixedCQP make_base_problem() {
    spacepdhcg::core::FixedStructure structure{};
    structure.quadratic = spacepdhcg::core::CscPattern{1, 1, {0, 1}, {0}};
    structure.scalar_constraint = spacepdhcg::core::CscPattern{0, 1, {0, 0}, {}};

    spacepdhcg::core::NumericValues values{};
    values.quadratic = {1.0};
    values.scalar_constraint = {};
    values.affine_cone = {};
    values.linear_objective = {-2.0};
    values.scalar_lower = {};
    values.scalar_upper = {};
    values.affine_offset = {};
    values.variable_lower = {0.0};
    values.variable_upper = {3.0};
    return spacepdhcg::core::FixedCQP(std::move(structure), std::move(values));
}

}  // namespace

int main() {
    using spacepdhcg::backends::DenseAdmmBackend;
    using spacepdhcg::transcription::AffinePathSample;
    using spacepdhcg::transcription::CtQuadratureInterval;
    using spacepdhcg::transcription::CtViolationStateCqp;

    const auto base = make_base_problem();
    const CtViolationStateCqp augmentation(
        base.structure(),
        {{0U}, {0U}},
        {CtQuadratureInterval{{0U, 1U}, {0.5, 0.5}}}
    );
    if (augmentation.structure().variables() != 5
        || augmentation.structure().scalar_rows() != 5) {
        return 1;
    }

    const std::vector<AffinePathSample> samples{
        AffinePathSample{{0U}, {1.0}, -1.0},
        AffinePathSample{{0U}, {1.0}, -1.0},
    };
    DenseAdmmBackend solver(
        augmentation.problem(base.values(), samples, {0.0}, 1.0)
    );
    const auto solution = solver.solve(1.0e-7, 200'000U);
    if (!solution.solved() || std::abs(solution.primal[0U] - 1.0) > 3.0e-4) {
        return 2;
    }
    for (std::size_t sample = 0; sample < augmentation.sample_count(); ++sample) {
        if (solution.primal[augmentation.lambda_index(sample)] > 3.0e-5) {
            return 3;
        }
    }
    if (solution.primal[augmentation.state_index(1U)] > 3.0e-5) {
        return 4;
    }
    const auto diagnostics = augmentation.diagnostics(solution.primal, samples, {0.0});
    if (diagnostics.maximum_positive_sample > 3.0e-4
        || diagnostics.maximum_integral_budget_violation > 3.0e-4) {
        return 5;
    }

    const std::vector<AffinePathSample> tightened{
        AffinePathSample{{0U}, {1.0}, -0.5},
        AffinePathSample{{0U}, {1.0}, -0.5},
    };
    solver.update(augmentation.values(base.values(), tightened, {0.0}, 1.0));
    const auto tightened_solution = solver.solve(1.0e-7, 200'000U);
    if (!tightened_solution.solved()
        || std::abs(tightened_solution.primal[0U] - 0.5) > 5.0e-4) {
        return 6;
    }
    if (solver.update_count() != 1U) {
        return 7;
    }
    return 0;
}
