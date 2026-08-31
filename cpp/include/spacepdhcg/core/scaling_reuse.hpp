#pragma once

#include "spacepdhcg/core/fixed_cqp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace spacepdhcg::core {

struct NumericChangeMetrics {
    double maximum_relative_matrix_change{0.0};
    double maximum_relative_vector_change{0.0};
};

class ScalingReuseController {
  public:
    explicit ScalingReuseController(ScalingThresholds thresholds = {})
        : thresholds_(thresholds) {
        validate_thresholds();
    }

    [[nodiscard]] const ScalingThresholds& thresholds() const noexcept { return thresholds_; }
    [[nodiscard]] Index reused_updates() const noexcept { return reused_updates_; }
    [[nodiscard]] const NumericChangeMetrics& last_metrics() const noexcept {
        return last_metrics_;
    }

    [[nodiscard]] RescalePolicy observe(
        const NumericValues& previous,
        const NumericValues& current,
        const FixedStructure& structure
    ) {
        previous.validate(structure);
        current.validate(structure);
        last_metrics_ = NumericChangeMetrics{
            maximum_relative_change(
                previous.quadratic,
                current.quadratic,
                previous.scalar_constraint,
                current.scalar_constraint,
                previous.affine_cone,
                current.affine_cone
            ),
            maximum_relative_change(
                previous.linear_objective,
                current.linear_objective,
                previous.scalar_lower,
                current.scalar_lower,
                previous.scalar_upper,
                current.scalar_upper,
                previous.affine_offset,
                current.affine_offset,
                previous.variable_lower,
                current.variable_lower,
                previous.variable_upper,
                current.variable_upper
            ),
        };
        const auto refresh =
            last_metrics_.maximum_relative_matrix_change
                > thresholds_.maximum_relative_matrix_change
            || last_metrics_.maximum_relative_vector_change
                   > thresholds_.maximum_relative_vector_change
            || reused_updates_ >= thresholds_.maximum_reuse_updates;
        if (refresh) {
            reused_updates_ = 0;
            return RescalePolicy::force_refresh;
        }
        ++reused_updates_;
        return RescalePolicy::reuse;
    }

    void reset() noexcept {
        reused_updates_ = 0;
        last_metrics_ = NumericChangeMetrics{};
    }

  private:
    ScalingThresholds thresholds_{};
    Index reused_updates_{0};
    NumericChangeMetrics last_metrics_{};

    void validate_thresholds() const {
        if (!std::isfinite(thresholds_.maximum_relative_matrix_change)
            || thresholds_.maximum_relative_matrix_change < 0.0) {
            throw std::invalid_argument(
                "maximum relative matrix change must be finite and non-negative"
            );
        }
        if (!std::isfinite(thresholds_.maximum_relative_vector_change)
            || thresholds_.maximum_relative_vector_change < 0.0) {
            throw std::invalid_argument(
                "maximum relative vector change must be finite and non-negative"
            );
        }
        if (thresholds_.maximum_reuse_updates < 0) {
            throw std::invalid_argument("maximum reuse updates must be non-negative");
        }
    }

    static double relative_component_change(double previous, double current) noexcept {
        if (std::isinf(previous) || std::isinf(current)) {
            return previous == current ? 0.0 : std::numeric_limits<double>::infinity();
        }
        const auto denominator = std::max({1.0, std::abs(previous), std::abs(current)});
        return std::abs(current - previous) / denominator;
    }

    static double maximum_pair_change(
        const std::vector<double>& previous,
        const std::vector<double>& current
    ) {
        if (previous.size() != current.size()) {
            throw std::invalid_argument("numeric-change arrays must have identical sizes");
        }
        double maximum{0.0};
        for (std::size_t index = 0; index < previous.size(); ++index) {
            maximum = std::max(
                maximum,
                relative_component_change(previous[index], current[index])
            );
        }
        return maximum;
    }

    template <typename... Arrays>
    static double maximum_relative_change(
        const std::vector<double>& previous,
        const std::vector<double>& current,
        const Arrays&... arrays
    ) {
        static_assert(sizeof...(arrays) % 2U == 0U);
        auto maximum = maximum_pair_change(previous, current);
        if constexpr (sizeof...(arrays) > 0U) {
            maximum = std::max(maximum, maximum_relative_change(arrays...));
        }
        return maximum;
    }
};

}  // namespace spacepdhcg::core
