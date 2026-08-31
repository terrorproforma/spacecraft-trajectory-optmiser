#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spacepdhcg::distributed {

struct RiskSummary {
    double expected{0.0};
    double worst{0.0};
    double value_at_risk{0.0};
    double conditional_value_at_risk{0.0};
};

inline void validate_probability_distribution(
    const std::vector<double>& probabilities,
    double tolerance = 1.0e-12
) {
    if (probabilities.empty()) {
        throw std::invalid_argument("risk aggregation requires at least one scenario");
    }
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::invalid_argument("probability tolerance must be finite and non-negative");
    }
    double total{0.0};
    for (const auto probability : probabilities) {
        if (!std::isfinite(probability) || probability <= 0.0) {
            throw std::invalid_argument("scenario probabilities must be finite and positive");
        }
        total += probability;
    }
    if (std::abs(total - 1.0) > tolerance) {
        throw std::invalid_argument("scenario probabilities must sum to one");
    }
}

/// Weighted upper-tail VaR/CVaR for a loss distribution.
inline RiskSummary aggregate_scenario_risk(
    const std::vector<double>& losses,
    const std::vector<double>& probabilities,
    double confidence
) {
    if (losses.size() != probabilities.size() || losses.empty()) {
        throw std::invalid_argument("risk losses and probabilities must have equal nonzero size");
    }
    validate_probability_distribution(probabilities);
    if (!std::isfinite(confidence) || confidence < 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("CVaR confidence must lie in [0, 1)");
    }
    std::vector<std::pair<double, double>> distribution{};
    distribution.reserve(losses.size());
    double expected{0.0};
    double worst{-std::numeric_limits<double>::infinity()};
    for (std::size_t scenario = 0; scenario < losses.size(); ++scenario) {
        if (!std::isfinite(losses[scenario])) {
            throw std::invalid_argument("scenario losses must be finite");
        }
        expected += probabilities[scenario] * losses[scenario];
        worst = std::max(worst, losses[scenario]);
        distribution.emplace_back(losses[scenario], probabilities[scenario]);
    }
    std::sort(
        distribution.begin(),
        distribution.end(),
        [](const auto& left, const auto& right) {
            return left.first < right.first;
        }
    );

    double cumulative{0.0};
    double value_at_risk = distribution.back().first;
    for (const auto& [loss, probability] : distribution) {
        cumulative += probability;
        if (cumulative + 1.0e-15 >= confidence) {
            value_at_risk = loss;
            break;
        }
    }

    const auto tail_mass = 1.0 - confidence;
    double remaining = tail_mass;
    double tail_sum{0.0};
    for (auto iterator = distribution.rbegin();
         iterator != distribution.rend() && remaining > 1.0e-15;
         ++iterator) {
        const auto included = std::min(iterator->second, remaining);
        tail_sum += included * iterator->first;
        remaining -= included;
    }
    if (remaining > 1.0e-12) {
        throw std::logic_error("CVaR tail integration did not cover the requested probability");
    }
    return RiskSummary{
        expected,
        worst,
        value_at_risk,
        tail_sum / tail_mass,
    };
}

}  // namespace spacepdhcg::distributed
