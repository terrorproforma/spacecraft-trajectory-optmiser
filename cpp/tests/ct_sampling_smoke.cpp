#include "spacepdhcg/transcription/ct_sampling.hpp"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace {

spacepdhcg::core::FixedCQP make_base_problem() {
    spacepdhcg::core::FixedStructure structure{};
    structure.quadratic = spacepdhcg::core::CscPattern{
        3,
        3,
        {0, 1, 2, 3},
        {0, 1, 2},
    };
    structure.scalar_constraint = spacepdhcg::core::CscPattern{
        0,
        3,
        {0, 0, 0, 0},
        {},
    };
    spacepdhcg::core::NumericValues values{};
    values.quadratic = {1.0, 1.0, 1.0};
    values.linear_objective = {0.0, 0.0, 0.0};
    values.variable_lower = {-10.0, -10.0, -10.0};
    values.variable_upper = {10.0, 10.0, 10.0};
    return spacepdhcg::core::FixedCQP{
        std::move(structure),
        std::move(values),
    };
}

}  // namespace

int main() {
    using spacepdhcg::transcription::CtQuadratureRule;
    using spacepdhcg::transcription::CtSamplingPlan;
    using spacepdhcg::transcription::CtTrajectoryLayout;
    using spacepdhcg::transcription::CtVariableRange;
    using spacepdhcg::transcription::NonlinearPathLinearisation;
    using spacepdhcg::transcription::NonlinearPathLineariser;

    const auto base = make_base_problem();
    const CtTrajectoryLayout layout{
        {CtVariableRange{0U, 1U}, CtVariableRange{1U, 1U}},
        {CtVariableRange{2U, 1U}},
    };
    const CtSamplingPlan plan{
        3U,
        layout,
        1U,
        6.0,
        CtQuadratureRule::simpson,
        10.0,
    };
    if (plan.sample_count() != 3U || plan.interval_count() != 1U
        || plan.slots()[1U].time != 13.0) {
        return 1;
    }

    const NonlinearPathLineariser constraint = [](
        const std::span<const double> state,
        const std::span<const double> control,
        const double time
    ) {
        static_cast<void>(time);
        return NonlinearPathLinearisation{
            state[0U] * state[0U] + control[0U] * control[0U] - 2.0,
            {2.0 * state[0U]},
            {2.0 * control[0U]},
        };
    };
    const std::vector<std::vector<double>> states{{0.0}, {2.0}};
    const std::vector<std::vector<double>> controls{{1.0}};
    const std::vector<NonlinearPathLineariser> constraints{constraint};
    const auto samples = plan.linearise(states, controls, constraints);
    const std::vector<double> base_primal{0.0, 2.0, 1.0};
    const std::vector<double> expected{-1.0, 0.0, 3.0};
    for (std::size_t sample = 0; sample < samples.size(); ++sample) {
        if (std::abs(samples[sample].evaluate(base_primal) - expected[sample])
            > 1.0e-12) {
            return 2;
        }
        if (samples[sample].indices != plan.sample_patterns()[sample]) {
            return 3;
        }
    }

    const auto nonlinear = plan.evaluate_nonlinear(states, controls, constraints);
    if (std::abs(nonlinear.maximum_positive_sample - 3.0) > 1.0e-12
        || std::abs(nonlinear.total_positive_integral - 3.0) > 1.0e-12) {
        return 4;
    }

    const auto augmentation = plan.augment(base.structure());
    const auto augmented_values = augmentation.values(
        base.values(),
        samples,
        {3.0},
        10.0
    );
    augmented_values.validate(augmentation.structure());
    std::vector<double> augmented_primal{
        0.0,
        2.0,
        1.0,
        0.0,
        0.0,
        3.0,
        0.0,
        3.0,
    };
    const auto diagnostics = augmentation.diagnostics(
        augmented_primal,
        samples,
        {3.0}
    );
    if (std::abs(diagnostics.actual_interval_integrals[0U] - 3.0) > 1.0e-12
        || std::abs(diagnostics.state_interval_increments[0U] - 3.0) > 1.0e-12
        || diagnostics.maximum_integral_budget_violation > 1.0e-12) {
        return 5;
    }

    const std::vector<std::vector<double>> changed_states{{0.0}, {1.0}};
    const std::vector<std::vector<double>> changed_controls{{0.5}};
    const auto changed = plan.linearise(changed_states, changed_controls, constraints);
    for (std::size_t sample = 0; sample < changed.size(); ++sample) {
        if (changed[sample].indices != samples[sample].indices) {
            return 6;
        }
    }
    const auto changed_values = augmentation.values(
        base.values(),
        changed,
        {3.0},
        10.0
    );
    if (changed_values.scalar_constraint == augmented_values.scalar_constraint) {
        return 7;
    }
    return 0;
}
