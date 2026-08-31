#include "spacepdhcg/backends/dense_admm.hpp"
#include "spacepdhcg/orbitweaver/host_cqp_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>

namespace {

spacepdhcg::core::FixedCQP make_problem() {
    spacepdhcg::core::FixedStructure structure{};
    structure.quadratic = spacepdhcg::core::CscPattern{1, 1, {0, 1}, {0}};
    structure.scalar_constraint = spacepdhcg::core::CscPattern{0, 1, {0, 0}, {}};

    spacepdhcg::core::NumericValues values{};
    values.quadratic = {1.0};
    values.scalar_constraint = {};
    values.affine_cone = {};
    values.linear_objective = {-2.0};
    values.scalar_lower = {};
    values.scalar_upper = {};
    values.affine_offset = {};
    values.variable_lower = {0.0};
    values.variable_upper = {3.0};
    return spacepdhcg::core::FixedCQP(std::move(structure), std::move(values));
}

}  // namespace

int main() {
    using spacepdhcg::backends::DenseAdmmBackend;
    using spacepdhcg::orbitweaver::ArcFidelity;
    using spacepdhcg::orbitweaver::ArcRequest;
    using spacepdhcg::orbitweaver::ArcSolution;
    using spacepdhcg::orbitweaver::FidelityPipelineOracle;
    using spacepdhcg::orbitweaver::HostCqpArcProblem;
    using spacepdhcg::orbitweaver::HostCqpFidelityStage;
    using spacepdhcg::orbitweaver::HostCqpOracleConfig;

    const auto builder = [](const ArcRequest& request, const std::optional<ArcSolution>& previous) {
        if (!previous.has_value() || previous->warm_start_token != 42U) {
            throw std::runtime_error("coarse CQP stage did not receive screening context");
        }
        const auto decoder = [](
                                 const ArcRequest& decoder_request,
                                 const spacepdhcg::core::HostCqpSolution& solution,
                                 const std::optional<ArcSolution>& predecessor
                             ) {
            if (!predecessor.has_value()) {
                throw std::runtime_error("coarse decoder lost its predecessor");
            }
            const auto decision = solution.primal.at(0U);
            const auto propellant = 1.0;
            return ArcSolution{
                true,
                ArcFidelity::coarse_convex,
                std::abs(decision),
                0.0,
                *decoder_request.arrival_epoch - decoder_request.departure_epoch,
                std::abs(decision),
                propellant,
                decoder_request.initial_mass - propellant,
                std::abs(decision - 2.0),
                std::max(solution.primal_residual, solution.dual_residual),
                decoder_request.requested_tolerance,
                0U,
                0U,
                0.0,
                0.0,
                77U,
                "dense host CQP coarse arc",
            };
        };
        spacepdhcg::core::HostWarmStart warm{};
        warm.primal = {1.0};
        return HostCqpArcProblem(make_problem(), decoder, warm);
    };
    const auto backend_factory = [](spacepdhcg::core::FixedCQP problem) {
        return std::make_unique<DenseAdmmBackend>(std::move(problem));
    };
    HostCqpOracleConfig config{};
    config.fidelity = ArcFidelity::coarse_convex;
    config.iteration_limit = 100'000U;
    const HostCqpFidelityStage coarse(builder, backend_factory, config);

    FidelityPipelineOracle pipeline;
    pipeline.register_stage(
        ArcFidelity::analytical_screening,
        [](const ArcRequest& request, const std::optional<ArcSolution>&) {
            return ArcSolution{
                true,
                ArcFidelity::analytical_screening,
                5.0,
                0.0,
                *request.arrival_epoch - request.departure_epoch,
                5.0,
                2.0,
                request.initial_mass - 2.0,
                0.0,
                0.0,
                request.requested_tolerance,
                0U,
                0U,
                0.0,
                0.0,
                42U,
                "screening stage",
            };
        }
    );
    pipeline.register_stage(ArcFidelity::coarse_convex, coarse.pipeline_stage());

    const ArcRequest request{
        0U,
        1U,
        0.0,
        10.0,
        100.0,
        0U,
        1U,
        ArcFidelity::coarse_convex,
        1.0e-7,
        "host-cqp-smoke",
        std::nullopt,
    };
    const auto result = pipeline.evaluate(request);
    if (!result.feasible || result.achieved_fidelity != ArcFidelity::coarse_convex) {
        return 1;
    }
    if (std::abs(result.cost - 2.0) > 5.0e-4 || result.terminal_error > 5.0e-4) {
        return 2;
    }
    if (result.inner_iterations == 0U || result.solve_seconds < 0.0) {
        return 3;
    }
    if (result.warm_start_token != 77U || result.final_mass != 99.0) {
        return 4;
    }
    return 0;
}
