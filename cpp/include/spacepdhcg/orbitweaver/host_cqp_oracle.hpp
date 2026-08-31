#pragma once

#include "spacepdhcg/core/host_backend.hpp"
#include "spacepdhcg/orbitweaver/trajectory_oracle.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace spacepdhcg::orbitweaver {

using HostArcDecoder = std::function<ArcSolution(
    const ArcRequest& request,
    const core::HostCqpSolution& solution,
    const std::optional<ArcSolution>& previous
)>;

class HostCqpArcProblem {
  public:
    HostCqpArcProblem(
        core::FixedCQP problem,
        HostArcDecoder decoder,
        std::optional<core::HostWarmStart> warm_start = std::nullopt
    )
        : problem_(std::move(problem)),
          decoder_(std::move(decoder)),
          warm_start_(std::move(warm_start)) {
        if (!decoder_) {
            throw std::invalid_argument("host CQP arc problem requires a decoder");
        }
    }

    [[nodiscard]] core::FixedCQP take_problem() { return std::move(problem_); }
    [[nodiscard]] const HostArcDecoder& decoder() const noexcept { return decoder_; }
    [[nodiscard]] const std::optional<core::HostWarmStart>& warm_start() const noexcept {
        return warm_start_;
    }

  private:
    core::FixedCQP problem_;
    HostArcDecoder decoder_{};
    std::optional<core::HostWarmStart> warm_start_{};
};

using HostCqpArcBuilder = std::function<HostCqpArcProblem(
    const ArcRequest& request,
    const std::optional<ArcSolution>& previous
)>;
using HostCqpBackendFactory = std::function<core::HostBackendPointer(core::FixedCQP)>;

struct HostCqpOracleConfig {
    ArcFidelity fidelity{ArcFidelity::coarse_convex};
    std::size_t iteration_limit{100'000U};

    void validate() const {
        if (fidelity == ArcFidelity::analytical_screening) {
            throw std::invalid_argument(
                "host CQP oracle must represent a convex or higher fidelity stage"
            );
        }
        if (iteration_limit == 0U) {
            throw std::invalid_argument("host CQP oracle iteration limit must be positive");
        }
    }
};

/// Backend-independent native arc stage for coarse convex and host reference solves.
///
/// The builder owns mission-specific transcription. The backend factory can create the dense
/// CPU debug solver today and the persistent PDHCG implementation later. The decoder converts
/// solver output into the stable OrbitWeaver arc contract and independently supplies physical
/// feasibility, mass closure and objective semantics.
class HostCqpFidelityStage {
  public:
    HostCqpFidelityStage(
        HostCqpArcBuilder builder,
        HostCqpBackendFactory backend_factory,
        HostCqpOracleConfig config = HostCqpOracleConfig{}
    )
        : builder_(std::move(builder)),
          backend_factory_(std::move(backend_factory)),
          config_(config) {
        if (!builder_ || !backend_factory_) {
            throw std::invalid_argument(
                "host CQP fidelity stage requires builder and backend factory"
            );
        }
        config_.validate();
    }

    [[nodiscard]] ArcSolution solve(
        const ArcRequest& request,
        const std::optional<ArcSolution>& previous = std::nullopt
    ) const {
        request.validate();
        if (request.fidelity != config_.fidelity) {
            throw std::invalid_argument("host CQP stage received the wrong fidelity request");
        }
        auto arc_problem = builder_(request, previous);
        const auto decoder = arc_problem.decoder();
        auto backend = backend_factory_(arc_problem.take_problem());
        if (backend == nullptr) {
            throw std::runtime_error("host CQP backend factory returned a null backend");
        }
        if (arc_problem.warm_start().has_value()) {
            backend->warm_start(*arc_problem.warm_start());
        }
        const auto solution = backend->solve(
            request.requested_tolerance,
            config_.iteration_limit
        );
        if (!solution.solved()) {
            ArcSolution failed{};
            failed.achieved_fidelity = config_.fidelity;
            failed.outer_iterations = solution.outer_iterations;
            failed.inner_iterations = solution.inner_iterations;
            failed.setup_seconds = solution.setup_seconds;
            failed.solve_seconds = solution.solve_seconds;
            failed.diagnostics = "host CQP backend did not return an optimal solution";
            return failed;
        }
        auto result = decoder(request, solution, previous);
        result.achieved_fidelity = config_.fidelity;
        result.outer_iterations = std::max(
            result.outer_iterations,
            solution.outer_iterations
        );
        result.inner_iterations = std::max(
            result.inner_iterations,
            solution.inner_iterations
        );
        result.setup_seconds += solution.setup_seconds;
        result.solve_seconds += solution.solve_seconds;
        result.validate(request);
        return result;
    }

    [[nodiscard]] FidelityPipelineOracle::Stage pipeline_stage() const {
        return [this](const ArcRequest& request, const std::optional<ArcSolution>& previous) {
            return solve(request, previous);
        };
    }

    [[nodiscard]] const HostCqpOracleConfig& config() const noexcept { return config_; }

  private:
    HostCqpArcBuilder builder_{};
    HostCqpBackendFactory backend_factory_{};
    HostCqpOracleConfig config_{};
};

}  // namespace spacepdhcg::orbitweaver
