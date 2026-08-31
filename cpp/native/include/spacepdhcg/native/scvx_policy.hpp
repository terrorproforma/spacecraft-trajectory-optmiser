#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace spacepdhcg::native {

struct OuterResidual {
    double dynamics{0.0};
    double path{0.0};
    double terminal{0.0};
    double step{0.0};

    [[nodiscard]] double feasibility() const noexcept {
        return std::max({dynamics, path, terminal});
    }

    [[nodiscard]] double maximum() const noexcept {
        return std::max(feasibility(), step);
    }

    void validate() const {
        for (double value : {dynamics, path, terminal, step}) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument("outer residuals must be finite and non-negative");
            }
        }
    }
};

enum class SolvePhase {
    repair,
    progress,
    refinement,
    polish,
};

struct ForcingRuleConfig {
    double maximum_tolerance{1.0e-2};
    double minimum_tolerance{1.0e-8};
    double forcing_coefficient{0.2};
    double forcing_exponent{1.25};
    double residual_floor{1.0e-12};
    double repair_threshold{2.0e-2};
    double refinement_threshold{2.0e-3};
    double polish_threshold{2.0e-4};
    double rejected_resolve_factor{0.1};
    int repair_iteration_limit{2'000};
    int progress_iteration_limit{10'000};
    int refinement_iteration_limit{100'000};
    int polish_iteration_limit{1'000'000};
    int minimum_polish_streak{2};

    void validate() const {
        if (!(minimum_tolerance > 0.0 && minimum_tolerance <= maximum_tolerance)) {
            throw std::invalid_argument("forcing tolerances are inconsistent");
        }
        if (!std::isfinite(forcing_coefficient) || forcing_coefficient <= 0.0 ||
            !std::isfinite(forcing_exponent) || forcing_exponent <= 0.0 ||
            !std::isfinite(residual_floor) || residual_floor <= 0.0) {
            throw std::invalid_argument("forcing parameters must be finite and positive");
        }
        if (!(repair_threshold >= refinement_threshold &&
              refinement_threshold >= polish_threshold && polish_threshold > 0.0)) {
            throw std::invalid_argument("forcing phase thresholds must be descending");
        }
        if (!(rejected_resolve_factor > 0.0 && rejected_resolve_factor < 1.0)) {
            throw std::invalid_argument("resolve factor must lie strictly between zero and one");
        }
        if (repair_iteration_limit <= 0 || progress_iteration_limit <= 0 ||
            refinement_iteration_limit <= 0 || polish_iteration_limit <= 0 ||
            minimum_polish_streak < 0) {
            throw std::invalid_argument("forcing iteration budgets must be positive");
        }
    }
};

struct ForcingDecision {
    double tolerance{0.0};
    double raw_target{0.0};
    int iteration_limit{0};
    SolvePhase phase{SolvePhase::repair};
    std::string reason{};
};

class AdaptiveForcingRule {
  public:
    explicit AdaptiveForcingRule(ForcingRuleConfig config = {}) : config_(config) {
        config_.validate();
    }

    [[nodiscard]] const ForcingRuleConfig& config() const noexcept { return config_; }

    [[nodiscard]] ForcingDecision request(
        int iteration,
        const OuterResidual& residual,
        int accepted_streak,
        std::optional<double> agreement = std::nullopt
    ) const {
        if (iteration < 0 || accepted_streak < 0) {
            throw std::invalid_argument("SCvx iteration counters must be non-negative");
        }
        residual.validate();
        if (agreement.has_value() && !std::isfinite(*agreement)) {
            throw std::invalid_argument("finite agreement is required when supplied");
        }

        const double measure = std::max(residual.feasibility(), config_.residual_floor);
        const double raw =
            config_.forcing_coefficient * std::pow(measure, config_.forcing_exponent);
        double tolerance = std::clamp(
            raw,
            config_.minimum_tolerance,
            config_.maximum_tolerance
        );

        SolvePhase phase = SolvePhase::progress;
        int iteration_limit = config_.progress_iteration_limit;
        std::string reason = "productive outer iteration";
        if (measure > config_.repair_threshold) {
            phase = SolvePhase::repair;
            iteration_limit = config_.repair_iteration_limit;
            tolerance = std::max(tolerance, 0.1 * config_.maximum_tolerance);
            reason = "large nonlinear defect; prioritise inexpensive restoration";
        } else if (
            measure <= config_.polish_threshold &&
            accepted_streak >= config_.minimum_polish_streak &&
            (!agreement.has_value() || *agreement >= 0.8)
        ) {
            phase = SolvePhase::polish;
            iteration_limit = config_.polish_iteration_limit;
            tolerance = config_.minimum_tolerance;
            reason = "small defect and stable accepted streak; request terminal polish";
        } else if (measure <= config_.refinement_threshold) {
            phase = SolvePhase::refinement;
            iteration_limit = config_.refinement_iteration_limit;
            tolerance = std::min(tolerance, std::sqrt(config_.minimum_tolerance));
            reason = "outer iterate is near feasibility; tighten the convex solve";
        }

        return ForcingDecision{tolerance, raw, iteration_limit, phase, std::move(reason)};
    }

    [[nodiscard]] bool should_resolve(
        bool accepted,
        double primal_residual,
        double dual_residual,
        double requested_tolerance
    ) const {
        if (accepted) {
            return false;
        }
        if (!(requested_tolerance > 0.0) || !std::isfinite(requested_tolerance)) {
            throw std::invalid_argument("requested tolerance must be finite and positive");
        }
        const double achieved = std::max(std::abs(primal_residual), std::abs(dual_residual));
        return std::isfinite(achieved) && achieved > 0.25 * requested_tolerance;
    }

    [[nodiscard]] double refined_tolerance(double current) const {
        if (!(current > 0.0) || !std::isfinite(current)) {
            throw std::invalid_argument("current tolerance must be finite and positive");
        }
        return std::max(
            config_.minimum_tolerance,
            config_.rejected_resolve_factor * current
        );
    }

  private:
    ForcingRuleConfig config_{};
};

enum class TrustAction {
    keep,
    expand,
    contract,
};

struct TrustRegionConfig {
    double initial_radius{1.0};
    double minimum_radius{1.0e-4};
    double maximum_radius{16.0};
    double contraction_factor{0.5};
    double expansion_factor{2.0};
    double expansion_agreement{0.9};
    double expansion_boundary_fraction{0.8};

    void validate() const {
        if (!(minimum_radius > 0.0 && initial_radius >= minimum_radius &&
              maximum_radius >= initial_radius)) {
            throw std::invalid_argument("trust-region radii are inconsistent");
        }
        if (!(contraction_factor > 0.0 && contraction_factor < 1.0) ||
            !(expansion_factor > 1.0) || !std::isfinite(expansion_factor)) {
            throw std::invalid_argument("trust-region scaling factors are invalid");
        }
        if (!(expansion_boundary_fraction >= 0.0 && expansion_boundary_fraction <= 1.0)) {
            throw std::invalid_argument("boundary fraction must lie in [0, 1]");
        }
    }
};

struct TrustRegionUpdate {
    double radius_before{0.0};
    double radius_after{0.0};
    TrustAction action{TrustAction::keep};
};

class TrustRegionController {
  public:
    explicit TrustRegionController(TrustRegionConfig config = {})
        : config_(config), radius_(config.initial_radius) {
        config_.validate();
    }

    [[nodiscard]] double radius() const noexcept { return radius_; }
    [[nodiscard]] bool exhausted() const noexcept {
        return radius_ <= config_.minimum_radius * (1.0 + 1.0e-12);
    }

    [[nodiscard]] TrustRegionUpdate update(
        bool accepted,
        double agreement,
        double step_fraction
    ) {
        if (!std::isfinite(step_fraction) || step_fraction < 0.0) {
            throw std::invalid_argument("step fraction must be finite and non-negative");
        }
        const double before = radius_;
        TrustAction action = TrustAction::keep;
        if (!accepted) {
            radius_ = std::max(config_.minimum_radius, config_.contraction_factor * radius_);
            action = TrustAction::contract;
        } else if (
            std::isfinite(agreement) && agreement >= config_.expansion_agreement &&
            step_fraction >= config_.expansion_boundary_fraction
        ) {
            radius_ = std::min(config_.maximum_radius, config_.expansion_factor * radius_);
            action = radius_ > before ? TrustAction::expand : TrustAction::keep;
        }
        return TrustRegionUpdate{before, radius_, action};
    }

  private:
    TrustRegionConfig config_{};
    double radius_{1.0};
};

struct InexactSolveSample {
    double outer_residual{0.0};
    double requested_tolerance{0.0};
    double achieved_residual{0.0};
};

class InexactErrorLedger {
  public:
    void record(InexactSolveSample sample) {
        if (!std::isfinite(sample.outer_residual) || sample.outer_residual < 0.0 ||
            !std::isfinite(sample.requested_tolerance) || sample.requested_tolerance <= 0.0 ||
            !std::isfinite(sample.achieved_residual) || sample.achieved_residual < 0.0) {
            throw std::invalid_argument("inexact-solve sample is invalid");
        }
        samples_.push_back(sample);
    }

    [[nodiscard]] double accumulated_achieved_error() const noexcept {
        double total = 0.0;
        for (const auto& sample : samples_) {
            total += sample.achieved_residual;
        }
        return total;
    }

    [[nodiscard]] double maximum_relative_forcing() const noexcept {
        double maximum = 0.0;
        for (const auto& sample : samples_) {
            const double denominator = std::max(sample.outer_residual, 1.0e-16);
            maximum = std::max(maximum, sample.achieved_residual / denominator);
        }
        return maximum;
    }

    [[nodiscard]] bool respects_requested_tolerances(double factor = 1.0) const {
        if (!std::isfinite(factor) || factor <= 0.0) {
            throw std::invalid_argument("tolerance factor must be finite and positive");
        }
        return std::ranges::all_of(samples_, [factor](const InexactSolveSample& sample) {
            return sample.achieved_residual <= factor * sample.requested_tolerance;
        });
    }

    [[nodiscard]] std::size_t size() const noexcept { return samples_.size(); }

  private:
    std::vector<InexactSolveSample> samples_{};
};

}  // namespace spacepdhcg::native
