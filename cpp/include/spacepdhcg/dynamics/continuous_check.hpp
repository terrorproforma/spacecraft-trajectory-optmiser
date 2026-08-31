#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace spacepdhcg::dynamics {

struct ContinuousTrajectoryDiagnostics {
    double maximum_node_mismatch{0.0};
    double maximum_path_violation{0.0};
    std::size_t propagated_substeps{0U};
};

/// Independently propagate a piecewise-constant control trajectory between knot points.
///
/// The model must expose `rk4_step(state, control, dt)` and
/// `path_diagnostics(states, controls).maximum_violation()`. This checker deliberately does not
/// reuse transcription defects or solver residuals.
template <typename Model, typename State, typename Control>
ContinuousTrajectoryDiagnostics check_piecewise_constant_trajectory(
    const Model& model,
    const std::vector<State>& knot_states,
    const std::vector<Control>& controls,
    double interval_seconds,
    std::size_t subdivisions = 8U
) {
    if (controls.empty() || knot_states.size() != controls.size() + 1U) {
        throw std::invalid_argument(
            "continuous checking requires N controls and N+1 knot states"
        );
    }
    if (!std::isfinite(interval_seconds) || interval_seconds <= 0.0) {
        throw std::invalid_argument("continuous-check interval must be finite and positive");
    }
    if (subdivisions == 0U) {
        throw std::invalid_argument("continuous-check subdivisions must be positive");
    }

    const auto substep_seconds = interval_seconds / static_cast<double>(subdivisions);
    std::vector<State> sampled_states{};
    std::vector<Control> sampled_controls{};
    sampled_states.reserve(controls.size() * subdivisions + 1U);
    sampled_controls.reserve(controls.size() * subdivisions);
    sampled_states.push_back(knot_states.front());

    ContinuousTrajectoryDiagnostics result{};
    for (std::size_t interval = 0; interval < controls.size(); ++interval) {
        auto propagated = knot_states[interval];
        for (std::size_t substep = 0; substep < subdivisions; ++substep) {
            propagated = model.rk4_step(propagated, controls[interval], substep_seconds);
            sampled_controls.push_back(controls[interval]);
            sampled_states.push_back(propagated);
            ++result.propagated_substeps;
        }
        for (std::size_t component = 0; component < propagated.size(); ++component) {
            result.maximum_node_mismatch = std::max(
                result.maximum_node_mismatch,
                std::abs(propagated[component] - knot_states[interval + 1U][component])
            );
        }
    }
    result.maximum_path_violation =
        model.path_diagnostics(sampled_states, sampled_controls).maximum_violation();
    return result;
}

}  // namespace spacepdhcg::dynamics
