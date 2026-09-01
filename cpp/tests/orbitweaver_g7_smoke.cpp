#include "spacepdhcg/orbitweaver/g7_orchestration.hpp"
#include "spacepdhcg/orbitweaver/lambert_family.hpp"

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <vector>

namespace ow = spacepdhcg::orbitweaver;
namespace g7 = spacepdhcg::orbitweaver::g7;

#ifdef NDEBUG
#  undef assert
#  define assert(condition) ((condition) ? static_cast<void>(0) : std::abort())
#endif

g7::ScheduledArc request(const std::uint64_t id, const std::uint64_t topology) {
    ow::ArcRequest arc{};
    arc.from_target = 0U;
    arc.to_target = 1U;
    arc.departure_epoch = 0.0;
    arc.arrival_epoch = 10.0;
    arc.initial_mass = 100.0;
    arc.fidelity = ow::ArcFidelity::coarse_convex;
    return {id, arc, {topology, arc.fidelity, 8U, 1U}, 1.0};
}

g7::ArcExecution feasible(const g7::ScheduledArc& arc, const double cost) {
    ow::ArcSolution solution{};
    solution.feasible = true;
    solution.achieved_fidelity = arc.request.fidelity;
    solution.cost = cost;
    solution.lower_bound = 1.0;
    solution.duration = 10.0;
    solution.delta_v = 2.0;
    solution.propellant = 5.0;
    solution.final_mass = 95.0;
    solution.terminal_error = 1.0e-6;
    solution.maximum_constraint_violation = 1.0e-6;
    solution.achieved_tolerance = 1.0e-6;
    return {arc.deterministic_id, g7::ArcExecutionStatus::feasible, solution};
}

int main() {
    const auto family = ow::enumerate_lambert_families(
        {7.0e6, 0.0, 0.0},
        {0.0, 8.0e6, 0.0},
        3'600.0,
        3.986004418e14,
        1U,
        true,
        false,
        1.0e-8,
        256U,
        512U
    );
    assert(!family.empty() && family.front().revolutions == 0U);

    auto backend = std::make_shared<g7::PersistentArcCallbackBackend>(
        [](const g7::TopologyFidelityKey&,
           const std::vector<g7::ScheduledArc>& batch,
           g7::Ownership,
           const std::atomic<bool>&) {
            std::vector<g7::ArcExecution> output{};
            for (const auto& arc : batch) {
                output.push_back(
                    arc.deterministic_id == 3U
                        ? g7::ArcExecution{
                              arc.deterministic_id,
                              g7::ArcExecutionStatus::infeasible}
                        : feasible(arc, arc.deterministic_id == 1U ? 4.0 : 3.0)
                );
            }
            return output;
        }
    );
    g7::BoundedArcScheduler scheduler{
        backend,
        std::make_shared<g7::LogicalRankOwnership>(
            std::vector<std::size_t>{0U, 1U}
        ),
        {2U, 8U, 64U, 128U},
    };
    const auto executions =
        scheduler.run({request(3U, 22U), request(2U, 11U), request(1U, 11U)});
    assert(executions.size() == 3U && executions[0].deterministic_id == 1U);
    assert(scheduler.telemetry().batches == 3U);
    const auto selected = g7::deterministic_top_k(executions, 3U);
    assert(selected[0].deterministic_id == 2U);
    assert(selected[2].status == g7::ArcExecutionStatus::infeasible);

    const auto risk = g7::aggregate_risk(
        {{1U, 0.75, 2.0, 1.0, {1.0}}, {0U, 0.25, 6.0, 2.0, {1.0}}},
        g7::RiskMeasure::expected
    );
    assert(risk.feasible && std::abs(risk.objective - 3.0) < 1.0e-12);
    assert(
        !g7::aggregate_risk(
             {{0U, 0.5, 2.0, 1.0, {1.0}}, {1U, 0.5, 3.0, 1.0, {1.1}}},
             g7::RiskMeasure::worst_case
         )
             .feasible
    );

    const g7::Checkpoint checkpoint{1U, 42U, 2U, 3.0, 1.0, {1U, 2U}, {101U}};
    assert(g7::Checkpoint::decode(checkpoint.encode()).seed == 42U);
    const g7::IndependentCertifier certifier{
        [](const g7::ArcExecution&) {
            return g7::CertificationChecks{1e-7, 2e-7, 3e-7, 4e-7, 5e-7};
        },
        "independent-rk4-test",
        1.0e-6,
    };
    assert(certifier.certify(executions[0]).accepted);
    assert(!certifier.certify(executions[2]).accepted);
    return 0;
}
