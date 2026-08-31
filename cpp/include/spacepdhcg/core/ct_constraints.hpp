#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace spacepdhcg::core {

enum class ViolationPenalty {
    absolute,
    squared,
};

struct ContinuousTimeCertificate {
    std::vector<double> interval_increments;
    double maximum_increment{0.0};
    double total_violation{0.0};

    [[nodiscard]] bool certified(const double interval_tolerance) const {
        if (!std::isfinite(interval_tolerance) || interval_tolerance < 0.0) {
            throw std::invalid_argument(
                "continuous-time interval tolerance must be finite and non-negative"
            );
        }
        return maximum_increment <= interval_tolerance;
    }
};

[[nodiscard]] inline double violation_density(
    const double constraint_value,
    const ViolationPenalty penalty = ViolationPenalty::squared
) {
    if (!std::isfinite(constraint_value)) {
        throw std::invalid_argument("path-constraint samples must be finite");
    }
    const double positive = std::max(0.0, constraint_value);
    return penalty == ViolationPenalty::squared ? positive * positive : positive;
}

/// Integrate non-negative violation states over each fixed shooting interval.
///
/// `constraint_samples[node][constraint] <= 0` denotes feasibility.  The
/// returned increments are trapezoidal approximations to the violation-state
/// dynamics `y_dot = Lambda(g) >= 0`.  The topology is fixed: changing the
/// number of samples or constraints requires a new optimisation episode.
[[nodiscard]] inline ContinuousTimeCertificate certify_continuous_time_constraints(
    const std::vector<std::vector<double>>& constraint_samples,
    const double step_seconds,
    const ViolationPenalty penalty = ViolationPenalty::squared
) {
    if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("continuous-time step must be finite and positive");
    }
    if (constraint_samples.size() < 2U) {
        throw std::invalid_argument("at least two path-constraint sample nodes are required");
    }
    const std::size_t constraint_count = constraint_samples.front().size();
    if (constraint_count == 0U) {
        throw std::invalid_argument("at least one path constraint is required");
    }
    for (const auto& node : constraint_samples) {
        if (node.size() != constraint_count) {
            throw std::invalid_argument("path-constraint sample dimensions changed inside episode");
        }
    }

    ContinuousTimeCertificate result;
    result.interval_increments.resize(constraint_samples.size() - 1U, 0.0);
    for (std::size_t interval = 0; interval + 1U < constraint_samples.size(); ++interval) {
        double increment = 0.0;
        for (std::size_t constraint = 0; constraint < constraint_count; ++constraint) {
            const double left = violation_density(
                constraint_samples[interval][constraint],
                penalty
            );
            const double right = violation_density(
                constraint_samples[interval + 1U][constraint],
                penalty
            );
            increment += 0.5 * step_seconds * (left + right);
        }
        result.interval_increments[interval] = increment;
        result.maximum_increment = std::max(result.maximum_increment, increment);
        result.total_violation += increment;
    }
    return result;
}

/// Conservative midpoint-enhanced certificate without changing the shooting grid.
[[nodiscard]] inline ContinuousTimeCertificate certify_with_midpoints(
    const std::vector<std::vector<double>>& node_samples,
    const std::vector<std::vector<double>>& midpoint_samples,
    const double step_seconds,
    const ViolationPenalty penalty = ViolationPenalty::squared
) {
    if (node_samples.size() < 2U || midpoint_samples.size() + 1U != node_samples.size()) {
        throw std::invalid_argument("one midpoint sample is required per shooting interval");
    }
    const std::size_t constraint_count = node_samples.front().size();
    if (constraint_count == 0U) {
        throw std::invalid_argument("at least one path constraint is required");
    }
    for (const auto& node : node_samples) {
        if (node.size() != constraint_count) {
            throw std::invalid_argument("node sample dimensions changed inside episode");
        }
    }
    for (const auto& midpoint : midpoint_samples) {
        if (midpoint.size() != constraint_count) {
            throw std::invalid_argument("midpoint sample dimensions changed inside episode");
        }
    }
    if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
        throw std::invalid_argument("continuous-time step must be finite and positive");
    }

    ContinuousTimeCertificate result;
    result.interval_increments.resize(midpoint_samples.size(), 0.0);
    for (std::size_t interval = 0; interval < midpoint_samples.size(); ++interval) {
        double increment = 0.0;
        for (std::size_t constraint = 0; constraint < constraint_count; ++constraint) {
            const double left = violation_density(node_samples[interval][constraint], penalty);
            const double middle = violation_density(midpoint_samples[interval][constraint], penalty);
            const double right = violation_density(node_samples[interval + 1U][constraint], penalty);
            increment += step_seconds * (left + 4.0 * middle + right) / 6.0;
        }
        result.interval_increments[interval] = increment;
        result.maximum_increment = std::max(result.maximum_increment, increment);
        result.total_violation += increment;
    }
    return result;
}

}  // namespace spacepdhcg::core
