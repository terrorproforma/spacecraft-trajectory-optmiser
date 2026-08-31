#include "spacepdhcg/orbitweaver/low_thrust_oracle.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct BackendCounters {
    std::size_t creations{0U};
    std::size_t updates{0U};
    std::size_t warm_starts{0U};
    std::size_t solves{0U};
};

class EchoBackend final : public spacepdhcg::core::HostPersistentBackend {
  public:
    EchoBackend(
        spacepdhcg::core::FixedCQP problem,
        std::vector<double> primal,
        std::shared_ptr<BackendCounters> counters
    )
        : structure_(problem.structure()),
          values_(problem.values()),
          primal_(std::move(primal)),
          dual_(static_cast<std::size_t>(structure_.duals()), 0.0),
          counters_(std::move(counters)) {
        if (primal_.size() != static_cast<std::size_t>(structure_.variables())) {
            throw std::invalid_argument("low-thrust oracle echo primal has the wrong size");
        }
        ++counters_->creations;
    }

    [[nodiscard]] const spacepdhcg::core::FixedStructure& structure() const noexcept override {
        return structure_;
    }

    [[nodiscard]] std::size_t update_count() const noexcept override { return update_count_; }

    void update(spacepdhcg::core::NumericValues values) override {
        values.validate(structure_);
        values_ = std::move(values);
        ++update_count_;
        ++counters_->updates;
    }

    void warm_start(const spacepdhcg::core::HostWarmStart& start) override {
        if (!start.primal.empty()
            && start.primal.size() != static_cast<std::size_t>(structure_.variables())) {
            throw std::invalid_argument("low-thrust oracle warm primal has the wrong size");
        }
        if (!start.dual.empty()
            && start.dual.size() != static_cast<std::size_t>(structure_.duals())) {
            throw std::invalid_argument("low-thrust oracle warm dual has the wrong size");
        }
        ++counters_->warm_starts;
    }

    [[nodiscard]] spacepdhcg::core::HostCqpSolution solve(
        const double tolerance,
        const std::size_t iteration_limit
    ) override {
        if (!std::isfinite(tolerance) || tolerance <= 0.0 || iteration_limit == 0U) {
            throw std::invalid_argument("low-thrust oracle solve request is invalid");
        }
        ++counters_->solves;
        spacepdhcg::core::HostCqpSolution solution{};
        solution.status = spacepdhcg::SolveStatus::optimal;
        solution.primal = primal_;
        solution.dual = dual_;
        solution.primal_residual = 1.0e-10;
        solution.dual_residual = 1.0e-10;
        solution.outer_iterations = 1U;
        solution.inner_iterations = 2U;
        solution.setup_seconds = 1.0e-6;
        solution.update_seconds = 2.0e-6;
        solution.solve_seconds = 3.0e-6;
        return solution;
    }

  private:
    spacepdhcg::core::FixedStructure structure_{};
    spacepdhcg::core::NumericValues values_{};
    std::vector<double> primal_{};
    std::vector<double> dual_{};
    std::shared_ptr<BackendCounters> counters_{};
    std::size_t update_count_{0U};
};

bool near(const double actual, const double expected, const double tolerance) {
    return std::isfinite(actual) && std::abs(actual - expected) <= tolerance;
}

}  // namespace

int main() {
    using spacepdhcg::dynamics::LowThrustState;
    using spacepdhcg::dynamics::LowThrustTwoBodyConfig;
    using spacepdhcg::dynamics::LowThrustTwoBodyModel;
    using spacepdhcg::orbitweaver::ArcFidelity;
    using spacepdhcg::orbitweaver::ArcRequest;
    using spacepdhcg::orbitweaver::ArcSolution;
    using spacepdhcg::orbitweaver::CartesianEphemerisState;
    using spacepdhcg::orbitweaver::FidelityPipelineOracle;
    using spacepdhcg::orbitweaver::LowThrustArcOracleConfig;
    using spacepdhcg::orbitweaver::LowThrustOrbitStages;
    using spacepdhcg::transcription::LowThrustScvxConfig;
    using spacepdhcg::transcription::LowThrustSubproblem;

    LowThrustTwoBodyConfig dynamics_config{};
    dynamics_config.gravitational_parameter = 1.0e-12;
    dynamics_config.thrust_to_acceleration = 1.0;
    dynamics_config.mass_flow_coefficient = 1.0e-6;
    dynamics_config.minimum_mass = 100.0;
    dynamics_config.maximum_thrust = 1.0;
    dynamics_config.minimum_radius = 1.0;
    const LowThrustTwoBodyModel model{dynamics_config};

    LowThrustScvxConfig transcription_config{};
    transcription_config.intervals = 4U;
    transcription_config.step_seconds = 10.0;
    transcription_config.trust_radius = 1.0;
    const LowThrustState initial{
        7'000.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        500.0,
    };
    const LowThrustState target = initial;
    const auto reference = spacepdhcg::scvx::make_native_low_thrust_reference(
        model,
        initial,
        target,
        transcription_config.intervals,
        transcription_config.step_seconds,
        false
    );
    const LowThrustSubproblem subproblem{model, transcription_config};
    const auto primal = subproblem.reference_decision(
        reference.first,
        reference.second
    );
    const auto counters = std::make_shared<BackendCounters>();
    const auto backend_factory = [primal, counters](spacepdhcg::core::FixedCQP problem) {
        return std::make_unique<EchoBackend>(
            std::move(problem),
            primal,
            counters
        );
    };
    const auto ephemeris = [](const std::size_t target_id, const double epoch) {
        static_cast<void>(epoch);
        if (target_id > 1U) {
            throw std::invalid_argument("unexpected target in low-thrust oracle smoke test");
        }
        return CartesianEphemerisState{
            {7'000.0, 0.0, 0.0},
            {0.0, 0.0, 0.0},
        };
    };

    LowThrustArcOracleConfig config{};
    config.dynamics = dynamics_config;
    config.transcription = transcription_config;
    config.outer.maximum_iterations = 2U;
    config.outer.minimum_iterations = 2U;
    config.coarse_feasibility_tolerance = 1.0e-6;
    config.refined_feasibility_tolerance = 1.0e-6;
    config.cost_per_delta_v = 2.0;
    config.cost_per_second = 0.0;

    LowThrustOrbitStages stages{ephemeris, backend_factory, config};
    FidelityPipelineOracle pipeline;
    pipeline.register_stage(
        ArcFidelity::analytical_screening,
        [](const ArcRequest& request, const std::optional<ArcSolution>&) {
            return ArcSolution{
                true,
                ArcFidelity::analytical_screening,
                1.0,
                0.0,
                *request.arrival_epoch - request.departure_epoch,
                1.0,
                0.0,
                request.initial_mass,
                0.0,
                0.0,
                request.requested_tolerance,
                0U,
                0U,
                0.0,
                0.0,
                std::nullopt,
                "analytical predecessor for low-thrust native stages",
            };
        }
    );
    stages.register_stages(pipeline);

    const ArcRequest request{
        0U,
        1U,
        0.0,
        40.0,
        500.0,
        0U,
        1U,
        ArcFidelity::refined_scvx,
        1.0e-6,
        "low-thrust-native-smoke",
        std::nullopt,
    };
    const auto solution = pipeline.evaluate(request);
    if (!solution.feasible || solution.achieved_fidelity != ArcFidelity::refined_scvx
        || !solution.warm_start_token.has_value()) {
        return 1;
    }
    if (solution.outer_iterations != 2U || solution.inner_iterations != 6U
        || solution.maximum_constraint_violation > 1.0e-9
        || solution.terminal_error > 1.0e-9) {
        return 2;
    }
    if (!near(solution.final_mass + solution.propellant, request.initial_mass, 1.0e-10)
        || solution.cost < 0.0 || solution.delta_v < 0.0
        || solution.lower_bound > solution.cost) {
        return 3;
    }
    if (counters->creations != 2U || counters->updates != 1U
        || counters->warm_starts != 2U || counters->solves != 3U) {
        return 4;
    }
    if (stages.store()->size() != 2U || pipeline.stage_count() != 3U) {
        return 5;
    }
    return 0;
}
