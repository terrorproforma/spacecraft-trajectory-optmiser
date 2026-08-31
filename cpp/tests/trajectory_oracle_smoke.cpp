#include "spacepdhcg/orbitweaver/trajectory_oracle.hpp"

#include <cmath>
#include <memory>
#include <optional>
#include <vector>

namespace {

class CountingOracle final : public spacepdhcg::orbitweaver::TrajectoryOracle {
  public:
    [[nodiscard]] spacepdhcg::orbitweaver::ArcSolution evaluate(
        const spacepdhcg::orbitweaver::ArcRequest& request
    ) override {
        ++calls;
        request.validate();
        const auto propellant = 10.0;
        return spacepdhcg::orbitweaver::ArcSolution{
            true,
            request.fidelity,
            12.0,
            6.0,
            20.0,
            2.0,
            propellant,
            request.initial_mass - propellant,
            1.0e-6,
            2.0e-6,
            request.requested_tolerance,
            1U,
            25U,
            1.0e-4,
            2.0e-3,
            99U,
            "counting oracle",
        };
    }

    std::size_t calls{0U};
};

spacepdhcg::orbitweaver::ArcSolution stage_solution(
    const spacepdhcg::orbitweaver::ArcRequest& request,
    double cost,
    double lower_bound,
    double propellant,
    std::uint64_t token
) {
    return spacepdhcg::orbitweaver::ArcSolution{
        true,
        request.fidelity,
        cost,
        lower_bound,
        15.0,
        cost * 0.1,
        propellant,
        request.initial_mass - propellant,
        1.0e-7,
        2.0e-7,
        request.requested_tolerance,
        2U,
        50U,
        2.0e-4,
        4.0e-3,
        token,
        "pipeline stage",
    };
}

}  // namespace

int main() {
    using spacepdhcg::orbitweaver::ArcFidelity;
    using spacepdhcg::orbitweaver::ArcRequest;
    using spacepdhcg::orbitweaver::CachedTrajectoryOracle;
    using spacepdhcg::orbitweaver::FidelityPipelineOracle;

    auto delegate = std::make_shared<CountingOracle>();
    CachedTrajectoryOracle cached(delegate);
    ArcRequest request{
        0U,
        1U,
        100.0,
        120.0,
        100.0,
        0U,
        1U,
        ArcFidelity::analytical_screening,
        1.0e-4,
        "two-body",
        std::nullopt,
    };
    const auto first = cached.evaluate(request);
    request.warm_start_token = 123U;
    const auto second = cached.evaluate(request);
    if (delegate->calls != 1U || cached.hits() != 1U || cached.misses() != 1U) {
        return 1;
    }
    if (first.cost != second.cost || cached.cache_size() != 1U) {
        return 2;
    }
    request.requested_tolerance = 1.0e-5;
    const auto third = cached.evaluate(request);
    if (delegate->calls != 2U || third.achieved_tolerance != 1.0e-5) {
        return 3;
    }
    const auto estimate = third.beam_estimate();
    if (!estimate.feasible || estimate.final_mass != 90.0 || estimate.lower_bound != 6.0) {
        return 4;
    }

    FidelityPipelineOracle pipeline;
    std::size_t calls{0U};
    pipeline.register_stage(
        ArcFidelity::analytical_screening,
        [&calls](const ArcRequest& stage_request, const auto& previous) {
            ++calls;
            if (previous.has_value()) {
                throw std::runtime_error("screening stage unexpectedly received a predecessor");
            }
            return stage_solution(stage_request, 12.0, 5.0, 10.0, 10U);
        }
    );
    pipeline.register_stage(
        ArcFidelity::coarse_convex,
        [&calls](const ArcRequest& stage_request, const auto& previous) {
            ++calls;
            if (!previous.has_value() || stage_request.warm_start_token != 10U) {
                throw std::runtime_error("coarse stage did not receive the screening warm start");
            }
            return stage_solution(stage_request, 10.0, 7.0, 9.0, 20U);
        }
    );
    pipeline.register_stage(
        ArcFidelity::refined_scvx,
        [&calls](const ArcRequest& stage_request, const auto& previous) {
            ++calls;
            if (!previous.has_value() || stage_request.warm_start_token != 20U) {
                throw std::runtime_error("refined stage did not receive the coarse warm start");
            }
            return stage_solution(stage_request, 9.0, 8.0, 8.0, 30U);
        }
    );

    request.fidelity = ArcFidelity::refined_scvx;
    request.requested_tolerance = 1.0e-6;
    request.warm_start_token.reset();
    const auto refined = pipeline.evaluate(request);
    if (calls != 3U || refined.achieved_fidelity != ArcFidelity::refined_scvx) {
        return 5;
    }
    if (refined.cost != 9.0 || refined.lower_bound != 8.0
        || refined.warm_start_token != 30U) {
        return 6;
    }

    const auto batch = pipeline.evaluate_batch(std::vector<ArcRequest>{request, request});
    if (batch.size() != 2U || calls != 9U) {
        return 7;
    }
    return 0;
}
