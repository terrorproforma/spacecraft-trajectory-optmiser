#pragma once

#include "spacepdhcg/orbitweaver/beam_search.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

enum class ArcFidelity : std::uint8_t {
    analytical_screening = 0U,
    coarse_convex = 1U,
    refined_scvx = 2U,
    robust_scvx = 3U,
    certified = 4U,
};

struct ArcRequest {
    std::size_t from_target{0U};
    std::size_t to_target{0U};
    double departure_epoch{0.0};
    std::optional<double> arrival_epoch{};
    double initial_mass{0.0};
    std::size_t spacecraft{0U};
    std::size_t scenario_count{1U};
    ArcFidelity fidelity{ArcFidelity::analytical_screening};
    double requested_tolerance{1.0e-4};
    std::string model_identifier{"default"};
    std::optional<std::uint64_t> warm_start_token{};

    void validate() const {
        if (from_target == to_target) {
            throw std::invalid_argument("trajectory-oracle endpoints must differ");
        }
        if (!std::isfinite(departure_epoch) || !std::isfinite(initial_mass)
            || initial_mass <= 0.0) {
            throw std::invalid_argument("trajectory-oracle epoch and mass must be finite and valid");
        }
        if (arrival_epoch.has_value()
            && (!std::isfinite(*arrival_epoch) || *arrival_epoch <= departure_epoch)) {
            throw std::invalid_argument("arrival epoch must be finite and later than departure");
        }
        if (scenario_count == 0U) {
            throw std::invalid_argument("trajectory-oracle scenario count must be positive");
        }
        if (!std::isfinite(requested_tolerance) || requested_tolerance <= 0.0) {
            throw std::invalid_argument("trajectory-oracle tolerance must be finite and positive");
        }
        if (model_identifier.empty()) {
            throw std::invalid_argument("trajectory-oracle model identifier may not be empty");
        }
    }
};

struct ArcSolution {
    bool feasible{false};
    ArcFidelity achieved_fidelity{ArcFidelity::analytical_screening};
    double cost{std::numeric_limits<double>::infinity()};
    double lower_bound{0.0};
    double duration{0.0};
    double delta_v{0.0};
    double propellant{0.0};
    double final_mass{0.0};
    double terminal_error{std::numeric_limits<double>::infinity()};
    double maximum_constraint_violation{std::numeric_limits<double>::infinity()};
    double achieved_tolerance{std::numeric_limits<double>::infinity()};
    std::size_t outer_iterations{0U};
    std::size_t inner_iterations{0U};
    double setup_seconds{0.0};
    double solve_seconds{0.0};
    std::optional<std::uint64_t> warm_start_token{};
    std::string diagnostics{};

    void validate(const ArcRequest& request) const {
        request.validate();
        if (!feasible) {
            return;
        }
        if (static_cast<std::uint8_t>(achieved_fidelity)
            < static_cast<std::uint8_t>(request.fidelity)) {
            throw std::runtime_error("trajectory oracle returned lower fidelity than requested");
        }
        for (const auto value : {
                 cost,
                 lower_bound,
                 duration,
                 delta_v,
                 propellant,
                 final_mass,
                 terminal_error,
                 maximum_constraint_violation,
                 achieved_tolerance,
                 setup_seconds,
                 solve_seconds,
             }) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("feasible trajectory-oracle result must be finite");
            }
        }
        if (cost < 0.0 || lower_bound < 0.0 || lower_bound > cost || duration <= 0.0
            || delta_v < 0.0 || propellant < 0.0 || final_mass < 0.0
            || terminal_error < 0.0 || maximum_constraint_violation < 0.0
            || achieved_tolerance <= 0.0 || setup_seconds < 0.0 || solve_seconds < 0.0) {
            throw std::runtime_error("feasible trajectory-oracle result contains invalid values");
        }
        if (propellant > request.initial_mass || final_mass > request.initial_mass) {
            throw std::runtime_error("trajectory-oracle mass accounting is inconsistent");
        }
        const auto mass_error = std::abs(
            request.initial_mass - propellant - final_mass
        );
        if (mass_error > 1.0e-8 * std::max(1.0, request.initial_mass)) {
            throw std::runtime_error("trajectory-oracle propellant and final mass do not close");
        }
    }

    [[nodiscard]] ArcEstimate beam_estimate() const {
        if (!feasible) {
            return {};
        }
        return ArcEstimate{
            true,
            cost,
            duration,
            delta_v,
            final_mass,
            lower_bound,
        };
    }
};

class TrajectoryOracle {
  public:
    TrajectoryOracle(const TrajectoryOracle&) = delete;
    TrajectoryOracle& operator=(const TrajectoryOracle&) = delete;
    TrajectoryOracle(TrajectoryOracle&&) = delete;
    TrajectoryOracle& operator=(TrajectoryOracle&&) = delete;
    virtual ~TrajectoryOracle() = default;

    [[nodiscard]] virtual ArcSolution evaluate(const ArcRequest& request) = 0;

    [[nodiscard]] virtual std::vector<ArcSolution> evaluate_batch(
        const std::vector<ArcRequest>& requests
    ) {
        std::vector<ArcSolution> results{};
        results.reserve(requests.size());
        for (const auto& request : requests) {
            results.push_back(evaluate(request));
        }
        return results;
    }

  protected:
    TrajectoryOracle() = default;
};

namespace trajectory_oracle_detail {

struct CacheKey {
    std::size_t from_target{0U};
    std::size_t to_target{0U};
    double departure_epoch{0.0};
    bool has_arrival_epoch{false};
    double arrival_epoch{0.0};
    double initial_mass{0.0};
    std::size_t spacecraft{0U};
    std::size_t scenario_count{1U};
    ArcFidelity fidelity{ArcFidelity::analytical_screening};
    double requested_tolerance{0.0};
    std::string model_identifier{};

    [[nodiscard]] auto tie() const noexcept {
        return std::tie(
            from_target,
            to_target,
            departure_epoch,
            has_arrival_epoch,
            arrival_epoch,
            initial_mass,
            spacecraft,
            scenario_count,
            fidelity,
            requested_tolerance,
            model_identifier
        );
    }

    [[nodiscard]] bool operator<(const CacheKey& other) const noexcept {
        return tie() < other.tie();
    }
};

inline CacheKey cache_key(const ArcRequest& request) {
    request.validate();
    return CacheKey{
        request.from_target,
        request.to_target,
        request.departure_epoch,
        request.arrival_epoch.has_value(),
        request.arrival_epoch.value_or(0.0),
        request.initial_mass,
        request.spacecraft,
        request.scenario_count,
        request.fidelity,
        request.requested_tolerance,
        request.model_identifier,
    };
}

}  // namespace trajectory_oracle_detail

/// Exact semantic cache around any native oracle. Warm-start tokens are deliberately excluded
/// from the key because they affect runtime, not the mathematical arc request.
class CachedTrajectoryOracle final : public TrajectoryOracle {
  public:
    explicit CachedTrajectoryOracle(std::shared_ptr<TrajectoryOracle> delegate)
        : delegate_(std::move(delegate)) {
        if (delegate_ == nullptr) {
            throw std::invalid_argument("cached trajectory oracle requires a delegate");
        }
    }

    [[nodiscard]] ArcSolution evaluate(const ArcRequest& request) override {
        const auto key = trajectory_oracle_detail::cache_key(request);
        const auto iterator = cache_.find(key);
        if (iterator != cache_.end()) {
            ++hits_;
            return iterator->second;
        }
        ++misses_;
        auto solution = delegate_->evaluate(request);
        solution.validate(request);
        cache_.emplace(key, solution);
        return solution;
    }

    [[nodiscard]] std::size_t cache_size() const noexcept { return cache_.size(); }
    [[nodiscard]] std::size_t hits() const noexcept { return hits_; }
    [[nodiscard]] std::size_t misses() const noexcept { return misses_; }
    void clear() noexcept {
        cache_.clear();
        hits_ = 0U;
        misses_ = 0U;
    }

  private:
    std::shared_ptr<TrajectoryOracle> delegate_{};
    std::map<trajectory_oracle_detail::CacheKey, ArcSolution> cache_{};
    std::size_t hits_{0U};
    std::size_t misses_{0U};
};

/// Ordered multi-fidelity oracle. Each stage can consume the previous stage's solution as a
/// warm-start/lower-bound object. All registered stages up to the requested fidelity execute.
class FidelityPipelineOracle final : public TrajectoryOracle {
  public:
    using Stage = std::function<ArcSolution(
        const ArcRequest&,
        const std::optional<ArcSolution>&
    )>;

    void register_stage(ArcFidelity fidelity, Stage stage) {
        if (!stage) {
            throw std::invalid_argument("trajectory-oracle stage may not be empty");
        }
        const auto [_, inserted] = stages_.emplace(fidelity, std::move(stage));
        if (!inserted) {
            throw std::invalid_argument("trajectory-oracle fidelity stage is already registered");
        }
    }

    [[nodiscard]] ArcSolution evaluate(const ArcRequest& request) override {
        request.validate();
        if (stages_.find(request.fidelity) == stages_.end()) {
            throw std::runtime_error("requested trajectory-oracle fidelity is not registered");
        }
        std::optional<ArcSolution> previous{};
        for (const auto& [fidelity, stage] : stages_) {
            if (static_cast<std::uint8_t>(fidelity)
                > static_cast<std::uint8_t>(request.fidelity)) {
                break;
            }
            auto stage_request = request;
            stage_request.fidelity = fidelity;
            if (previous.has_value() && previous->warm_start_token.has_value()) {
                stage_request.warm_start_token = previous->warm_start_token;
            }
            auto result = stage(stage_request, previous);
            result.validate(stage_request);
            if (!result.feasible) {
                return result;
            }
            previous = std::move(result);
        }
        if (!previous.has_value()) {
            throw std::runtime_error("trajectory-oracle pipeline contains no executable stage");
        }
        previous->validate(request);
        return *previous;
    }

    [[nodiscard]] std::size_t stage_count() const noexcept { return stages_.size(); }

  private:
    std::map<ArcFidelity, Stage> stages_{};
};

}  // namespace spacepdhcg::orbitweaver
