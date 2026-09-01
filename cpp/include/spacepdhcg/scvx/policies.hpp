#pragma once

#include "spacepdhcg/scvx/g4_policy.generated.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::scvx {

struct OuterResidual {
    double dynamics{0.0};
    double path{0.0};
    double terminal{0.0};
    double step{0.0};

    [[nodiscard]] double feasibility() const noexcept {
        return std::max({std::abs(dynamics), std::abs(path), std::abs(terminal)});
    }

    [[nodiscard]] double maximum() const noexcept {
        return std::max(feasibility(), std::abs(step));
    }

    [[nodiscard]] bool finite() const noexcept {
        return std::isfinite(dynamics) && std::isfinite(path) && std::isfinite(terminal)
               && std::isfinite(step);
    }
};

enum class SolvePhase {
    repair,
    progress,
    refinement,
    polish,
};

struct ForcingDecision {
    double tolerance{1.0e-3};
    double raw_target{1.0e-3};
    std::size_t iteration_limit{10'000U};
    SolvePhase phase{SolvePhase::repair};
    std::string reason{};
};

struct ForcingRuleConfig {
    double minimum_tolerance{g4_policy::epsilon_floor};
    double maximum_tolerance{g4_policy::epsilon_max};
    double forcing_coefficient{g4_policy::coefficient};
    double forcing_exponent{g4_policy::alpha};
    double forcing_contraction{g4_policy::gamma};
    double residual_floor{1.0e-12};
    double repair_threshold{2.5e-1};
    double refinement_threshold{2.0e-2};
    double polish_threshold{5.0e-4};
    double repair_tolerance{g4_policy::repair_ceiling};
    double progress_tolerance{g4_policy::progress_ceiling};
    double refinement_tolerance{g4_policy::refinement_ceiling};
    double polish_tolerance{g4_policy::polish_ceiling};
    double resolve_factor{g4_policy::resolve_refinement_factor};
    double residual_overshoot_factor{g4_policy::resolve_trigger_multiple};
    std::size_t repair_iteration_limit{g4_policy::repair_iterations};
    std::size_t progress_iteration_limit{g4_policy::progress_iterations};
    std::size_t refinement_iteration_limit{g4_policy::refinement_iterations};
    std::size_t polish_iteration_limit{g4_policy::polish_iterations};
    std::size_t polish_accepted_streak{3U};

    void validate() const {
        require_positive(minimum_tolerance, "minimum tolerance must be positive");
        require_positive(maximum_tolerance, "maximum tolerance must be positive");
        if (minimum_tolerance > maximum_tolerance) {
            throw std::invalid_argument("minimum tolerance may not exceed maximum tolerance");
        }
        require_positive(forcing_coefficient, "forcing coefficient must be positive");
        require_positive(forcing_exponent, "forcing exponent must be positive");
        require_positive(forcing_contraction, "forcing contraction must be positive");
        require_positive(residual_floor, "residual floor must be positive");
        require_positive(repair_threshold, "repair threshold must be positive");
        require_positive(refinement_threshold, "refinement threshold must be positive");
        require_positive(polish_threshold, "polish threshold must be positive");
        if (!(repair_threshold > refinement_threshold
              && refinement_threshold > polish_threshold)) {
            throw std::invalid_argument("forcing thresholds must be strictly descending");
        }
        for (const auto tolerance : {
                 repair_tolerance,
                 progress_tolerance,
                 refinement_tolerance,
                 polish_tolerance,
             }) {
            require_positive(tolerance, "phase tolerances must be positive");
        }
        if (!(resolve_factor > 0.0 && resolve_factor < 1.0)) {
            throw std::invalid_argument("resolve factor must lie strictly between zero and one");
        }
        if (!std::isfinite(residual_overshoot_factor) || residual_overshoot_factor < 1.0) {
            throw std::invalid_argument("residual overshoot factor must be at least one");
        }
        if (repair_iteration_limit == 0U || progress_iteration_limit == 0U
            || refinement_iteration_limit == 0U || polish_iteration_limit == 0U) {
            throw std::invalid_argument("forcing iteration limits must be positive");
        }
    }

  private:
    static void require_positive(double value, const char* message) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(message);
        }
    }
};

class AdaptiveForcingRule {
  public:
    explicit AdaptiveForcingRule(ForcingRuleConfig config = {}) : config_(config) {
        config_.validate();
    }

    [[nodiscard]] const ForcingRuleConfig& config() const noexcept { return config_; }

    [[nodiscard]] ForcingDecision request(
        std::size_t iteration,
        const OuterResidual& residual,
        std::size_t accepted_streak,
        double agreement = std::numeric_limits<double>::quiet_NaN()
    ) const {
        static_cast<void>(iteration);
        if (!residual.finite()) {
            return phase_decision(
                SolvePhase::repair,
                config_.repair_tolerance,
                config_.repair_iteration_limit,
                config_.maximum_tolerance,
                "non-finite outer residual requires trajectory repair"
            );
        }

        const auto feasibility = std::max(residual.feasibility(), config_.residual_floor);
        const auto raw = std::clamp(
            config_.forcing_coefficient * std::pow(feasibility, config_.forcing_exponent),
            config_.minimum_tolerance,
            config_.maximum_tolerance
        );
        if (feasibility > config_.repair_threshold) {
            return phase_decision(
                SolvePhase::repair,
                std::min(raw, config_.repair_tolerance),
                config_.repair_iteration_limit,
                raw,
                "outer feasibility is in the repair regime"
            );
        }
        if (feasibility > config_.refinement_threshold || accepted_streak == 0U) {
            auto tolerance = std::min(raw, config_.progress_tolerance);
            if (std::isfinite(agreement) && agreement < 0.25) {
                tolerance = std::min(tolerance, 0.5 * config_.progress_tolerance);
            }
            return phase_decision(
                SolvePhase::progress,
                tolerance,
                config_.progress_iteration_limit,
                raw,
                "productive SCvx iterations use moderate inner accuracy"
            );
        }
        if (feasibility > config_.polish_threshold
            || accepted_streak < config_.polish_accepted_streak) {
            return phase_decision(
                SolvePhase::refinement,
                std::min(raw, config_.refinement_tolerance),
                config_.refinement_iteration_limit,
                raw,
                "accepted trajectory is entering the local refinement regime"
            );
        }
        return phase_decision(
            SolvePhase::polish,
            std::min(raw, config_.polish_tolerance),
            config_.polish_iteration_limit,
            raw,
            "small outer residual and stable accepted steps justify final polishing"
        );
    }

    [[nodiscard]] bool should_resolve(
        bool accepted,
        double primal_residual,
        double dual_residual,
        double requested_tolerance
    ) const {
        if (accepted || !std::isfinite(requested_tolerance) || requested_tolerance <= 0.0) {
            return false;
        }
        if (!std::isfinite(primal_residual) || !std::isfinite(dual_residual)) {
            return false;
        }
        return std::max(std::abs(primal_residual), std::abs(dual_residual))
               > config_.residual_overshoot_factor * requested_tolerance;
    }

    [[nodiscard]] double refined_tolerance(double current) const {
        if (!std::isfinite(current) || current <= 0.0) {
            throw std::invalid_argument("current tolerance must be finite and positive");
        }
        return std::max(config_.minimum_tolerance, config_.resolve_factor * current);
    }

  private:
    ForcingRuleConfig config_{};

    static ForcingDecision phase_decision(
        SolvePhase phase,
        double tolerance,
        std::size_t iteration_limit,
        double raw_target,
        std::string reason
    ) {
        return ForcingDecision{
            tolerance,
            raw_target,
            iteration_limit,
            phase,
            std::move(reason),
        };
    }
};

class FixedForcingRule {
  public:
    FixedForcingRule(double tolerance, std::size_t iteration_limit)
        : tolerance_(tolerance), iteration_limit_(iteration_limit) {
        if (!std::isfinite(tolerance_) || tolerance_ <= 0.0) {
            throw std::invalid_argument("fixed tolerance must be finite and positive");
        }
        if (iteration_limit_ == 0U) {
            throw std::invalid_argument("fixed iteration limit must be positive");
        }
    }

    [[nodiscard]] ForcingDecision request() const {
        return ForcingDecision{
            tolerance_,
            tolerance_,
            iteration_limit_,
            SolvePhase::progress,
            "fixed-accuracy comparison policy",
        };
    }

  private:
    double tolerance_{1.0e-4};
    std::size_t iteration_limit_{100'000U};
};

enum class QualityTier : std::size_t {
    coarse = 0U,
    medium = 1U,
    tight = 2U,
    ipm = 3U,
};

[[nodiscard]] inline constexpr double quality_tolerance(QualityTier tier) {
    return g4_policy::quality_tolerances.at(static_cast<std::size_t>(tier));
}

enum class TrustAction {
    retain,
    shrink,
    expand,
};

struct TrustRegionConfig {
    double initial_radius{g4_policy::trust_initial};
    double minimum_radius{g4_policy::trust_minimum};
    double maximum_radius{g4_policy::trust_maximum};
    double shrink_factor{g4_policy::trust_shrink};
    double expansion_factor{g4_policy::trust_expand};
    double poor_agreement{g4_policy::trust_acceptance};
    double strong_agreement{g4_policy::trust_strong_agreement};
    double boundary_fraction{g4_policy::trust_boundary_fraction};

    void validate() const {
        require_positive(initial_radius, "initial trust radius must be positive");
        require_positive(minimum_radius, "minimum trust radius must be positive");
        require_positive(maximum_radius, "maximum trust radius must be positive");
        if (!(minimum_radius <= initial_radius && initial_radius <= maximum_radius)) {
            throw std::invalid_argument("initial trust radius must lie within configured bounds");
        }
        if (!(shrink_factor > 0.0 && shrink_factor < 1.0)) {
            throw std::invalid_argument("trust-region shrink factor must lie in (0, 1)");
        }
        if (!std::isfinite(expansion_factor) || expansion_factor <= 1.0) {
            throw std::invalid_argument("trust-region expansion factor must exceed one");
        }
        if (!(poor_agreement >= 0.0 && strong_agreement > poor_agreement)) {
            throw std::invalid_argument("trust-region agreement thresholds are invalid");
        }
        if (!(boundary_fraction > 0.0 && boundary_fraction <= 1.0)) {
            throw std::invalid_argument("trust-region boundary fraction must lie in (0, 1]");
        }
    }

  private:
    static void require_positive(double value, const char* message) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(message);
        }
    }
};

struct TrustRegionUpdate {
    double radius_before{1.0};
    double radius_after{1.0};
    TrustAction action{TrustAction::retain};
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
        const auto before = radius_;
        auto action = TrustAction::retain;
        if (!accepted || !std::isfinite(agreement) || agreement < config_.poor_agreement) {
            radius_ = std::max(config_.minimum_radius, config_.shrink_factor * radius_);
            action = TrustAction::shrink;
        } else if (agreement >= config_.strong_agreement
                   && step_fraction >= config_.boundary_fraction) {
            radius_ = std::min(config_.maximum_radius, config_.expansion_factor * radius_);
            action = TrustAction::expand;
        }
        return TrustRegionUpdate{before, radius_, action};
    }

    void reset() noexcept { radius_ = config_.initial_radius; }

  private:
    TrustRegionConfig config_{};
    double radius_{1.0};
};

struct InexactSolveSample {
    double requested_tolerance{0.0};
    double achieved_residual{0.0};
    double outer_residual{0.0};
    double step_norm{0.0};
};

class InexactErrorLedger {
  public:
    void append(InexactSolveSample sample) {
        for (const auto value : {
                 sample.requested_tolerance,
                 sample.achieved_residual,
                 sample.outer_residual,
                 sample.step_norm,
             }) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument("inexact-solve ledger values must be finite and non-negative");
            }
        }
        samples_.push_back(sample);
        accumulated_error_ += sample.achieved_residual;
    }

    [[nodiscard]] const std::vector<InexactSolveSample>& samples() const noexcept {
        return samples_;
    }
    [[nodiscard]] double accumulated_error() const noexcept { return accumulated_error_; }

    [[nodiscard]] double maximum_relative_forcing() const noexcept {
        double maximum{0.0};
        for (const auto& sample : samples_) {
            const auto denominator = std::max(
                {sample.outer_residual, sample.step_norm, 1.0e-16}
            );
            maximum = std::max(maximum, sample.achieved_residual / denominator);
        }
        return maximum;
    }

    [[nodiscard]] bool within_summable_budget(double budget) const {
        if (!std::isfinite(budget) || budget < 0.0) {
            throw std::invalid_argument("summable-error budget must be finite and non-negative");
        }
        return accumulated_error_ <= budget;
    }

  private:
    std::vector<InexactSolveSample> samples_{};
    double accumulated_error_{0.0};
};

enum class SolverStage {
    first_order,
    interior_point_polish,
};

struct HybridSolvePlan {
    SolverStage stage{SolverStage::first_order};
    double tolerance{1.0e-4};
    std::size_t iteration_limit{100'000U};
    bool transfer_primal{false};
    bool transfer_dual{false};
};

inline HybridSolvePlan hybrid_plan(
    const ForcingDecision& forcing,
    bool final_accepted_model,
    bool interior_point_available
) {
    if (final_accepted_model && interior_point_available && forcing.phase == SolvePhase::polish) {
        return HybridSolvePlan{
            SolverStage::interior_point_polish,
            forcing.tolerance,
            forcing.iteration_limit,
            true,
            true,
        };
    }
    return HybridSolvePlan{
        SolverStage::first_order,
        forcing.tolerance,
        forcing.iteration_limit,
        false,
        false,
    };
}

}  // namespace spacepdhcg::scvx
