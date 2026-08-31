#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace spacepdhcg::core {

enum class SolvePhase {
    exploration,
    convergence,
    polish,
};

enum class RadiusAction {
    shrink,
    keep,
    grow,
};

enum class SolverFamily {
    pdhcg,
    qoco_gpu,
    cuclarabel,
};

struct OuterResidual {
    double dynamics{0.0};
    double path{0.0};
    double terminal{0.0};
    double step{0.0};

    void validate() const {
        const double values[]{dynamics, path, terminal, step};
        for (const auto value : values) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(
                    "outer-residual components must be finite and non-negative"
                );
            }
        }
    }

    [[nodiscard]] double maximum() const {
        validate();
        return std::max({dynamics, path, terminal, step});
    }
};

struct ForcingRuleConfig {
    double epsilon_max{1.0e-3};
    double epsilon_floor{1.0e-8};
    double epsilon_0{1.0e-3};
    double coefficient{0.2};
    double alpha{0.5};
    double gamma{0.6};
    std::size_t exploration_iterations{2};
    double switch_residual{2.0e-3};
    double switch_step{5.0e-2};
    double good_agreement{0.75};
    double polish_tolerance{1.0e-9};
    std::size_t exploration_iteration_limit{250};
    std::size_t convergence_iteration_limit{750};
    std::size_t polish_iteration_limit{2'000};
    double resolve_factor{0.1};
    double resolve_kkt_multiple{5.0};
    bool theoretical{false};

    void validate() const {
        const double positive[]{
            epsilon_max,
            epsilon_0,
            coefficient,
            alpha,
            switch_residual,
            switch_step,
            polish_tolerance,
            resolve_kkt_multiple,
        };
        for (const auto value : positive) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument("forcing-rule positive parameters are invalid");
            }
        }
        if (!std::isfinite(epsilon_floor) || epsilon_floor < 0.0 ||
            epsilon_floor > epsilon_max) {
            throw std::invalid_argument("forcing-rule epsilon floor is invalid");
        }
        if (!std::isfinite(gamma) || gamma <= 0.0 || gamma >= 1.0) {
            throw std::invalid_argument("forcing-rule gamma must lie in (0,1)");
        }
        if (!std::isfinite(resolve_factor) || resolve_factor <= 0.0 || resolve_factor >= 1.0) {
            throw std::invalid_argument("forcing-rule resolve factor must lie in (0,1)");
        }
        if (!std::isfinite(good_agreement) || good_agreement < 0.0 ||
            good_agreement > 1.0) {
            throw std::invalid_argument("forcing-rule agreement threshold is invalid");
        }
        if (exploration_iteration_limit == 0U || convergence_iteration_limit == 0U ||
            polish_iteration_limit == 0U) {
            throw std::invalid_argument("forcing-rule iteration limits must be positive");
        }
    }
};

struct ForcingDecision {
    double tolerance{0.0};
    double raw_target{0.0};
    std::size_t iteration_limit{0};
    SolvePhase phase{SolvePhase::exploration};
    std::string reason;
};

class AdaptiveForcingRule {
  public:
    explicit AdaptiveForcingRule(ForcingRuleConfig config = {}) : config_(config) {
        config_.validate();
    }

    [[nodiscard]] const ForcingRuleConfig& config() const noexcept { return config_; }

    [[nodiscard]] ForcingDecision request(
        const std::size_t iteration,
        const OuterResidual& residual,
        const std::size_t accepted_streak = 0U,
        const double agreement = std::numeric_limits<double>::quiet_NaN()
    ) const {
        const double maximum = residual.maximum();
        const double residual_target =
            config_.coefficient * std::pow(maximum, 1.0 + config_.alpha);
        const double geometric_target =
            config_.epsilon_0 * std::pow(config_.gamma, static_cast<double>(iteration));
        double raw_target = std::min({config_.epsilon_max, residual_target, geometric_target});
        if (maximum == 0.0) {
            raw_target = std::min(raw_target, config_.polish_tolerance);
        }

        const bool polish_ready = accepted_streak >= 2U && maximum < config_.switch_residual &&
            residual.step < config_.switch_step && std::isfinite(agreement) &&
            agreement > config_.good_agreement;
        if (polish_ready) {
            return ForcingDecision{
                config_.polish_tolerance,
                raw_target,
                config_.polish_iteration_limit,
                SolvePhase::polish,
                "accepted streak, residual, step and agreement passed the polish gate",
            };
        }

        const double floor = config_.theoretical ? 0.0 : config_.epsilon_floor;
        double tolerance = std::max(floor, raw_target);
        if (iteration < config_.exploration_iterations) {
            const double exploration_target = std::min(config_.epsilon_max, geometric_target);
            tolerance = std::max(tolerance, exploration_target);
            return ForcingDecision{
                tolerance,
                raw_target,
                config_.exploration_iteration_limit,
                SolvePhase::exploration,
                "early outer iteration: retain a coarse first-order solve",
            };
        }
        return ForcingDecision{
            tolerance,
            raw_target,
            config_.convergence_iteration_limit,
            SolvePhase::convergence,
            "tighten according to residual and geometric forcing terms",
        };
    }

    [[nodiscard]] bool should_resolve(
        const bool accepted,
        const double primal_residual,
        const double dual_residual,
        const double requested_tolerance
    ) const noexcept {
        if (accepted || !std::isfinite(requested_tolerance) || requested_tolerance <= 0.0 ||
            std::isnan(primal_residual) || std::isnan(dual_residual)) {
            return false;
        }
        return std::max(std::abs(primal_residual), std::abs(dual_residual)) >
            config_.resolve_kkt_multiple * requested_tolerance;
    }

    [[nodiscard]] double refined_tolerance(const double requested_tolerance) const {
        if (!std::isfinite(requested_tolerance) || requested_tolerance <= 0.0) {
            throw std::invalid_argument("requested tolerance must be finite and positive");
        }
        return std::max(
            config_.polish_tolerance,
            config_.resolve_factor * requested_tolerance
        );
    }

  private:
    ForcingRuleConfig config_;
};

struct TrustRegionConfig {
    double initial_radius{1.0};
    double minimum_radius{1.0e-4};
    double maximum_radius{8.0};
    double shrink_factor{0.5};
    double growth_factor{1.6};
    double rejection_threshold{0.05};
    double strong_agreement{0.8};
    double boundary_fraction{0.8};

    void validate() const {
        if (!std::isfinite(initial_radius) || !std::isfinite(minimum_radius) ||
            !std::isfinite(maximum_radius) || minimum_radius <= 0.0 ||
            initial_radius < minimum_radius || initial_radius > maximum_radius) {
            throw std::invalid_argument("trust-region radius bounds are invalid");
        }
        if (!std::isfinite(shrink_factor) || shrink_factor <= 0.0 || shrink_factor >= 1.0) {
            throw std::invalid_argument("trust-region shrink factor must lie in (0,1)");
        }
        if (!std::isfinite(growth_factor) || growth_factor <= 1.0) {
            throw std::invalid_argument("trust-region growth factor must exceed one");
        }
        if (!std::isfinite(rejection_threshold) || !std::isfinite(strong_agreement) ||
            rejection_threshold < 0.0 || rejection_threshold >= strong_agreement) {
            throw std::invalid_argument("trust-region agreement thresholds are invalid");
        }
        if (!std::isfinite(boundary_fraction) || boundary_fraction <= 0.0 ||
            boundary_fraction > 1.0) {
            throw std::invalid_argument("trust-region boundary fraction must lie in (0,1]");
        }
    }
};

struct TrustRegionUpdate {
    double radius_before{0.0};
    double radius_after{0.0};
    RadiusAction action{RadiusAction::keep};
    std::string reason;
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
        const bool accepted,
        const double agreement,
        const double step_fraction
    ) {
        if (!std::isfinite(step_fraction) || step_fraction < 0.0) {
            throw std::invalid_argument("step fraction must be finite and non-negative");
        }
        const double before = radius_;
        RadiusAction action = RadiusAction::keep;
        std::string reason;
        if (!accepted || !std::isfinite(agreement)) {
            radius_ = std::max(config_.minimum_radius, before * config_.shrink_factor);
            action = RadiusAction::shrink;
            reason = "candidate rejected or agreement was not finite";
        } else if (agreement < config_.rejection_threshold) {
            radius_ = std::max(config_.minimum_radius, before * config_.shrink_factor);
            action = RadiusAction::shrink;
            reason = "accepted safeguard step had poor model agreement";
        } else if (agreement >= config_.strong_agreement &&
                   step_fraction >= config_.boundary_fraction) {
            radius_ = std::min(config_.maximum_radius, before * config_.growth_factor);
            action = RadiusAction::grow;
            reason = "strong agreement on a boundary-active step";
        } else {
            reason = "agreement and step usage support retaining the radius";
        }
        return TrustRegionUpdate{before, radius_, action, reason};
    }

  private:
    TrustRegionConfig config_;
    double radius_{1.0};
};

struct HybridSolvePlan {
    SolverFamily primary{SolverFamily::pdhcg};
    SolverFamily polish{SolverFamily::qoco_gpu};
    bool use_polish{false};
    double first_order_tolerance{1.0e-4};
    double polish_tolerance{1.0e-9};
};

[[nodiscard]] inline HybridSolvePlan make_hybrid_plan(
    const ForcingDecision& forcing,
    const bool broad_cones_required
) {
    return HybridSolvePlan{
        SolverFamily::pdhcg,
        broad_cones_required ? SolverFamily::cuclarabel : SolverFamily::qoco_gpu,
        forcing.phase == SolvePhase::polish,
        forcing.tolerance,
        forcing.phase == SolvePhase::polish ? forcing.tolerance : 1.0e-9,
    };
}

}  // namespace spacepdhcg::core
