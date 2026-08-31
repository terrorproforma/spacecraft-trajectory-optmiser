#pragma once

#include "spacepdhcg/core/fixed_cqp.hpp"
#include "spacepdhcg/transcription/ct_violation_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spacepdhcg::transcription {

struct CtVariableRange {
    std::size_t start{0U};
    std::size_t size{0U};

    [[nodiscard]] std::size_t stop() const noexcept { return start + size; }

    void validate(const std::size_t variables, const char* name) const {
        if (size == 0U || start > variables || size > variables - start) {
            throw std::invalid_argument(name);
        }
    }
};

struct CtTrajectoryLayout {
    std::vector<CtVariableRange> states{};
    std::vector<CtVariableRange> controls{};

    void validate(const std::size_t variables) const {
        if (controls.empty() || states.size() != controls.size() + 1U) {
            throw std::invalid_argument(
                "CT trajectory layout requires N controls and N+1 states"
            );
        }
        const auto state_dimension = states.front().size;
        const auto control_dimension = controls.front().size;
        for (const auto& state : states) {
            state.validate(variables, "CT state range is outside the base primal");
            if (state.size != state_dimension) {
                throw std::invalid_argument("CT state ranges must share one dimension");
            }
        }
        for (const auto& control : controls) {
            control.validate(variables, "CT control range is outside the base primal");
            if (control.size != control_dimension) {
                throw std::invalid_argument("CT control ranges must share one dimension");
            }
        }
        std::vector<std::pair<std::size_t, std::size_t>> ranges{};
        ranges.reserve(states.size() + controls.size());
        for (const auto& state : states) {
            ranges.emplace_back(state.start, state.stop());
        }
        for (const auto& control : controls) {
            ranges.emplace_back(control.start, control.stop());
        }
        std::sort(ranges.begin(), ranges.end());
        for (std::size_t index = 1U; index < ranges.size(); ++index) {
            if (ranges[index].first < ranges[index - 1U].second) {
                throw std::invalid_argument("CT trajectory variable ranges may not overlap");
            }
        }
    }

    [[nodiscard]] std::size_t intervals() const noexcept { return controls.size(); }
    [[nodiscard]] std::size_t state_dimension() const noexcept {
        return states.empty() ? 0U : states.front().size;
    }
    [[nodiscard]] std::size_t control_dimension() const noexcept {
        return controls.empty() ? 0U : controls.front().size;
    }
};

enum class CtQuadratureRule {
    midpoint,
    trapezoidal,
    simpson,
    gauss_lobatto_four,
};

struct CtQuadratureNode {
    double fraction{0.0};
    double normalised_weight{0.0};
};

[[nodiscard]] inline std::vector<CtQuadratureNode> ct_quadrature_nodes(
    const CtQuadratureRule rule
) {
    switch (rule) {
        case CtQuadratureRule::midpoint:
            return {{0.5, 1.0}};
        case CtQuadratureRule::trapezoidal:
            return {{0.0, 0.5}, {1.0, 0.5}};
        case CtQuadratureRule::simpson:
            return {{0.0, 1.0 / 6.0}, {0.5, 4.0 / 6.0}, {1.0, 1.0 / 6.0}};
        case CtQuadratureRule::gauss_lobatto_four: {
            constexpr double root_five = 2.23606797749978969641;
            return {
                {0.0, 1.0 / 12.0},
                {(5.0 - root_five) / 10.0, 5.0 / 12.0},
                {(5.0 + root_five) / 10.0, 5.0 / 12.0},
                {1.0, 1.0 / 12.0},
            };
        }
    }
    throw std::logic_error("unknown CT quadrature rule");
}

struct NonlinearPathLinearisation {
    double value{0.0};
    std::vector<double> state_gradient{};
    std::vector<double> control_gradient{};

    void validate(
        const std::size_t state_dimension,
        const std::size_t control_dimension
    ) const {
        if (!std::isfinite(value) || state_gradient.size() != state_dimension
            || control_gradient.size() != control_dimension) {
            throw std::invalid_argument("nonlinear path linearisation has invalid dimensions");
        }
        if (!std::all_of(
                state_gradient.begin(),
                state_gradient.end(),
                [](const double coefficient) { return std::isfinite(coefficient); }
            )
            || !std::all_of(
                control_gradient.begin(),
                control_gradient.end(),
                [](const double coefficient) { return std::isfinite(coefficient); }
            )) {
            throw std::invalid_argument("nonlinear path gradients must be finite");
        }
    }
};

using NonlinearPathLineariser = std::function<NonlinearPathLinearisation(
    std::span<const double> state,
    std::span<const double> control,
    double time
)>;

struct CtSampleSlot {
    std::size_t interval{0U};
    std::size_t quadrature_node{0U};
    std::size_t constraint{0U};
    double fraction{0.0};
    double weight{0.0};
    double time{0.0};
};

struct CtNonlinearSamplingDiagnostics {
    std::vector<double> sample_values{};
    std::vector<double> interval_positive_integrals{};
    double maximum_positive_sample{0.0};
    double total_positive_integral{0.0};
};

/// Fixed-grid constructor for the violation-state CT-SCvx augmentation.
///
/// State is linearly interpolated between shooting nodes and control is held piecewise
/// constant. Every nonlinear path function returns value and gradients at the reference
/// sample. The resulting affine samples preserve one immutable sparse pattern over all SCvx
/// iterations. Quadrature weights already include physical interval duration.
class CtSamplingPlan {
  public:
    CtSamplingPlan(
        const std::size_t base_variables,
        CtTrajectoryLayout layout,
        const std::size_t constraint_count,
        const double step_seconds,
        const CtQuadratureRule rule = CtQuadratureRule::gauss_lobatto_four,
        const double initial_time = 0.0
    )
        : base_variables_(base_variables),
          layout_(std::move(layout)),
          constraint_count_(constraint_count),
          step_seconds_(step_seconds),
          initial_time_(initial_time),
          nodes_(ct_quadrature_nodes(rule)) {
        if (base_variables_ == 0U) {
            throw std::invalid_argument("CT sampling requires base variables");
        }
        layout_.validate(base_variables_);
        if (constraint_count_ == 0U || !std::isfinite(step_seconds_)
            || step_seconds_ <= 0.0 || !std::isfinite(initial_time_)) {
            throw std::invalid_argument("CT sampling dimensions and time grid are invalid");
        }
        build_slots_and_patterns();
    }

    [[nodiscard]] std::size_t base_variables() const noexcept {
        return base_variables_;
    }
    [[nodiscard]] std::size_t constraint_count() const noexcept {
        return constraint_count_;
    }
    [[nodiscard]] std::size_t sample_count() const noexcept { return slots_.size(); }
    [[nodiscard]] std::size_t interval_count() const noexcept {
        return layout_.intervals();
    }
    [[nodiscard]] double step_seconds() const noexcept { return step_seconds_; }
    [[nodiscard]] const CtTrajectoryLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] const std::vector<CtSampleSlot>& slots() const noexcept {
        return slots_;
    }
    [[nodiscard]] const std::vector<std::vector<std::size_t>>& sample_patterns() const noexcept {
        return sample_patterns_;
    }
    [[nodiscard]] const std::vector<CtQuadratureInterval>& quadrature_intervals() const noexcept {
        return quadrature_intervals_;
    }

    [[nodiscard]] CtViolationStateCqp augment(
        core::FixedStructure base_structure
    ) const {
        if (static_cast<std::size_t>(base_structure.variables()) != base_variables_) {
            throw std::invalid_argument(
                "CT sampling plan and base CQP variable counts differ"
            );
        }
        return CtViolationStateCqp{
            std::move(base_structure),
            sample_patterns_,
            quadrature_intervals_,
        };
    }

    [[nodiscard]] std::vector<AffinePathSample> linearise(
        const std::vector<std::vector<double>>& reference_states,
        const std::vector<std::vector<double>>& reference_controls,
        const std::vector<NonlinearPathLineariser>& constraints
    ) const {
        validate_reference(reference_states, reference_controls, constraints);
        std::vector<AffinePathSample> result{};
        result.reserve(sample_count());
        for (std::size_t sample = 0; sample < slots_.size(); ++sample) {
            const auto& slot = slots_[sample];
            const auto state = interpolate_state(
                reference_states[slot.interval],
                reference_states[slot.interval + 1U],
                slot.fraction
            );
            const auto& control = reference_controls[slot.interval];
            const auto linearisation = constraints[slot.constraint](
                state,
                control,
                slot.time
            );
            linearisation.validate(
                layout_.state_dimension(),
                layout_.control_dimension()
            );

            std::map<std::size_t, double> coefficients{};
            const auto& previous_state = layout_.states[slot.interval];
            const auto& next_state = layout_.states[slot.interval + 1U];
            const auto& control_range = layout_.controls[slot.interval];
            for (std::size_t component = 0;
                 component < layout_.state_dimension();
                 ++component) {
                coefficients[previous_state.start + component] +=
                    (1.0 - slot.fraction)
                    * linearisation.state_gradient[component];
                coefficients[next_state.start + component] +=
                    slot.fraction * linearisation.state_gradient[component];
            }
            for (std::size_t component = 0;
                 component < layout_.control_dimension();
                 ++component) {
                coefficients[control_range.start + component] +=
                    linearisation.control_gradient[component];
            }

            AffinePathSample affine{};
            affine.indices = sample_patterns_[sample];
            affine.coefficients.reserve(affine.indices.size());
            for (const auto index : affine.indices) {
                affine.coefficients.push_back(coefficients[index]);
            }
            affine.offset = linearisation.value;
            for (std::size_t component = 0; component < state.size(); ++component) {
                affine.offset -= linearisation.state_gradient[component]
                                 * state[component];
            }
            for (std::size_t component = 0; component < control.size(); ++component) {
                affine.offset -= linearisation.control_gradient[component]
                                 * control[component];
            }
            affine.validate(base_variables_);
            result.push_back(std::move(affine));
        }
        return result;
    }

    [[nodiscard]] CtNonlinearSamplingDiagnostics evaluate_nonlinear(
        const std::vector<std::vector<double>>& states,
        const std::vector<std::vector<double>>& controls,
        const std::vector<NonlinearPathLineariser>& constraints
    ) const {
        validate_reference(states, controls, constraints);
        CtNonlinearSamplingDiagnostics result{};
        result.sample_values.reserve(sample_count());
        result.interval_positive_integrals.assign(interval_count(), 0.0);
        for (const auto& slot : slots_) {
            const auto state = interpolate_state(
                states[slot.interval],
                states[slot.interval + 1U],
                slot.fraction
            );
            const auto evaluation = constraints[slot.constraint](
                state,
                controls[slot.interval],
                slot.time
            );
            evaluation.validate(layout_.state_dimension(), layout_.control_dimension());
            result.sample_values.push_back(evaluation.value);
            const auto positive = std::max(0.0, evaluation.value);
            result.maximum_positive_sample = std::max(
                result.maximum_positive_sample,
                positive
            );
            result.interval_positive_integrals[slot.interval] +=
                slot.weight * positive;
        }
        result.total_positive_integral = std::accumulate(
            result.interval_positive_integrals.begin(),
            result.interval_positive_integrals.end(),
            0.0
        );
        return result;
    }

  private:
    std::size_t base_variables_{0U};
    CtTrajectoryLayout layout_{};
    std::size_t constraint_count_{0U};
    double step_seconds_{0.0};
    double initial_time_{0.0};
    std::vector<CtQuadratureNode> nodes_{};
    std::vector<CtSampleSlot> slots_{};
    std::vector<std::vector<std::size_t>> sample_patterns_{};
    std::vector<CtQuadratureInterval> quadrature_intervals_{};

    void build_slots_and_patterns() {
        quadrature_intervals_.resize(interval_count());
        for (std::size_t interval = 0; interval < interval_count(); ++interval) {
            auto& quadrature = quadrature_intervals_[interval];
            for (std::size_t node = 0; node < nodes_.size(); ++node) {
                for (std::size_t constraint = 0;
                     constraint < constraint_count_;
                     ++constraint) {
                    const auto sample = slots_.size();
                    slots_.push_back(CtSampleSlot{
                        interval,
                        node,
                        constraint,
                        nodes_[node].fraction,
                        step_seconds_ * nodes_[node].normalised_weight,
                        initial_time_
                            + (static_cast<double>(interval) + nodes_[node].fraction)
                                  * step_seconds_,
                    });
                    quadrature.sample_indices.push_back(sample);
                    quadrature.weights.push_back(
                        step_seconds_ * nodes_[node].normalised_weight
                    );
                    std::vector<std::size_t> pattern{};
                    const auto append_range = [&pattern](const CtVariableRange range) {
                        for (std::size_t offset = 0; offset < range.size; ++offset) {
                            pattern.push_back(range.start + offset);
                        }
                    };
                    append_range(layout_.states[interval]);
                    append_range(layout_.states[interval + 1U]);
                    append_range(layout_.controls[interval]);
                    std::sort(pattern.begin(), pattern.end());
                    pattern.erase(std::unique(pattern.begin(), pattern.end()), pattern.end());
                    sample_patterns_.push_back(std::move(pattern));
                }
            }
            quadrature.validate(sample_count());
        }
    }

    void validate_reference(
        const std::vector<std::vector<double>>& states,
        const std::vector<std::vector<double>>& controls,
        const std::vector<NonlinearPathLineariser>& constraints
    ) const {
        if (states.size() != interval_count() + 1U
            || controls.size() != interval_count()
            || constraints.size() != constraint_count_) {
            throw std::invalid_argument("CT sampling reference dimensions are invalid");
        }
        for (const auto& state : states) {
            if (state.size() != layout_.state_dimension()
                || !std::all_of(state.begin(), state.end(), [](const double value) {
                       return std::isfinite(value);
                   })) {
                throw std::invalid_argument("CT reference state is invalid");
            }
        }
        for (const auto& control : controls) {
            if (control.size() != layout_.control_dimension()
                || !std::all_of(control.begin(), control.end(), [](const double value) {
                       return std::isfinite(value);
                   })) {
                throw std::invalid_argument("CT reference control is invalid");
            }
        }
        if (!std::all_of(constraints.begin(), constraints.end(), [](const auto& value) {
                return static_cast<bool>(value);
            })) {
            throw std::invalid_argument("CT path linearisers may not be empty");
        }
    }

    [[nodiscard]] static std::vector<double> interpolate_state(
        const std::vector<double>& previous,
        const std::vector<double>& next,
        const double fraction
    ) {
        if (previous.size() != next.size() || !std::isfinite(fraction)
            || fraction < 0.0 || fraction > 1.0) {
            throw std::invalid_argument("CT state interpolation inputs are invalid");
        }
        std::vector<double> result(previous.size(), 0.0);
        for (std::size_t component = 0; component < result.size(); ++component) {
            result[component] = (1.0 - fraction) * previous[component]
                                + fraction * next[component];
        }
        return result;
    }
};

}  // namespace spacepdhcg::transcription
