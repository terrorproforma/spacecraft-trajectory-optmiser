#include "spacepdhcg/distributed/risk.hpp"
#include "spacepdhcg/distributed/orbitweaver_g5_adapter.hpp"
#include "spacepdhcg/distributed/runtime.hpp"
#include "spacepdhcg/distributed/scenario_layout.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace g5 = spacepdhcg::distributed::g5;
namespace distributed = spacepdhcg::distributed;
namespace g7 = spacepdhcg::orbitweaver::g7;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

std::vector<g5::ScenarioWork> make_work(std::size_t count) {
    std::vector<g5::ScenarioWork> work(count);
    for (std::size_t scenario = 0; scenario < count; ++scenario) {
        work[scenario] = g5::ScenarioWork{
            20U + scenario,
            40U + 3U * scenario,
            12U + 2U * scenario,
            4U + scenario % 5U,
            scenario % 3U,
            scenario % 2U,
            (scenario + 1U) % 3U,
            scenario % 2U,
            8U + scenario,
            30U + 7U * scenario,
            2U + scenario % 4U,
            10U + 5U * scenario,
        };
    }
    return work;
}

double full_model_imbalance(
    const g5::PartitionPlan& plan,
    std::span<const g5::ScenarioWork> work
) {
    std::vector<double> loads(plan.rank_scenarios.size(), 0.0);
    for (std::size_t rank = 0; rank < plan.rank_scenarios.size(); ++rank) {
        for (const auto scenario : plan.rank_scenarios[rank]) {
            loads[rank] += g5::scenario_cost(
                work[scenario],
                g5::ScenarioCostModel{},
                g5::PartitionKind::scenario_aware
            );
        }
    }
    const auto mean =
        std::accumulate(loads.begin(), loads.end(), 0.0) / static_cast<double>(loads.size());
    return *std::max_element(loads.begin(), loads.end()) / mean;
}

void test_partition_contracts() {
    const auto work = make_work(23);
    for (const std::size_t ranks : {1U, 2U, 4U, 8U}) {
        const auto first = g5::partition_scenarios(
            work,
            ranks,
            g5::PartitionKind::scenario_aware
        );
        const auto second = g5::partition_scenarios(
            work,
            ranks,
            g5::PartitionKind::scenario_aware
        );
        require(first.fingerprint == second.fingerprint, "partition fingerprint is nondeterministic");
        require(first.rank_scenarios == second.rank_scenarios, "partition ownership is nondeterministic");
        first.validate(work.size(), ranks);

        auto measured = std::vector<g5::MeasuredRankLoad>{};
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            measured.push_back(g5::MeasuredRankLoad{
                rank,
                0.001 * static_cast<double>(rank + 1U),
                0.0001 * static_cast<double>(rank),
                0.0,
                static_cast<std::uint64_t>(first.rank_scenarios[rank].size()),
            });
        }
        auto with_measurements = first;
        g5::record_measured_loads(with_measurements, measured);
        require(
            with_measurements.measured_rank_load.size() == ranks,
            "measured load interface lost a rank"
        );
    }

    std::vector<g5::ScenarioWork> comparison(4);
    for (auto& scenario : comparison) {
        scenario.q_nonzeros = 100;
    }
    comparison[0].replay_work = 1000;
    comparison[2].replay_work = 1000;
    const auto aware = g5::partition_scenarios(
        comparison,
        2,
        g5::PartitionKind::scenario_aware
    );
    const auto generic = g5::partition_scenarios(
        comparison,
        2,
        g5::PartitionKind::nonzero_balanced
    );
    require(
        full_model_imbalance(aware, comparison) < full_model_imbalance(generic, comparison),
        "scenario-aware comparison did not improve the predicted full-work balance"
    );
}

void test_orbitweaver_rank_local_ownership() {
    const std::vector<std::size_t> devices{3U, 7U, 11U};
    std::vector<g7::G5ArcPartitionMetadata> items{};
    for (std::size_t identifier = 0U; identifier < 12U; ++identifier) {
        auto work = g5::ScenarioWork{};
        work.q_nonzeros = 10U + identifier;
        work.time_nodes = 4U + identifier % 3U;
        work.replay_work = 2U + identifier;
        items.push_back({
            100U + identifier,
            identifier / 6U,
            (identifier / 3U) % 2U,
            identifier % 3U,
            work,
        });
    }
    const auto first = g7::partition_g5_orbitweaver_arcs(items, devices);
    const auto second = g7::partition_g5_orbitweaver_arcs(items, devices);
    require(
        first.scenario_plan.fingerprint == second.scenario_plan.fingerprint
            && first.owners == second.owners,
        "OrbitWeaver G5 ownership is nondeterministic"
    );
    g7::G5RankLocalOwnershipAdapter ownership(first);
    for (const auto& item : items) {
        g7::ScheduledArc arc{};
        arc.deterministic_id = item.deterministic_id;
        arc.route_index = item.route_index;
        arc.trajectory_arc_index = item.trajectory_arc_index;
        arc.scenario_index = item.scenario_index;
        require(
            ownership.owner(arc, 999U) == first.owners.at(item.deterministic_id),
            "OrbitWeaver G5 owner changed with batch sequence"
        );
    }
}

void test_arrowhead_and_nonanticipativity() {
    const std::vector<std::vector<double>> local{{1.0, 2.0}, {3.0, 4.0}, {-1.0, 5.0}};
    const auto reduced = g5::reduce_shared_arrowhead(local);
    require(reduced == std::vector<double>({3.0, 11.0}), "arrowhead sum is incorrect");

    g5::ArrowheadMetadata metadata{
        20,
        {16, 17},
        12,
        18,
        {19},
        0x1234,
    };
    metadata.validate();
    metadata.risk_excess_indices = {18};
    require_throws([&metadata] { metadata.validate(); }, "duplicate arrowhead ownership was accepted");

    const auto tree = distributed::ScenarioTree::common_open_loop(4, 3, 2);
    const distributed::BlockArrowLayout layout(tree, 2, 1);
    std::vector<double> primal(layout.total_variables(), 0.0);
    for (const auto& block : layout.consensus_blocks()) {
        primal[block.offset] = 2.5;
        for (const auto scenario : block.node.scenario_indices) {
            const auto control = layout.control_range(scenario, block.node.stage).first;
            primal[control] = 2.5;
        }
    }
    for (const auto& entry : layout.nonanticipativity_triplets()) {
        static_cast<void>(entry);
    }
    std::vector<double> row_values(layout.nonanticipativity_rows(), 0.0);
    for (const auto& entry : layout.nonanticipativity_triplets()) {
        row_values[entry.row] += entry.value * primal[entry.column];
    }
    require(
        std::all_of(row_values.begin(), row_values.end(), [](double value) {
            return std::abs(value) < 1.0e-15;
        }),
        "non-anticipativity arrowhead algebra is inconsistent"
    );
}

void test_risk_and_residual_reductions() {
    const std::vector<g5::ExpectedRiskPartial> expected{
        {0.2, 0.25},
        {1.5, 0.50},
        {0.8, 0.25},
    };
    require(std::abs(g5::reduce_expected_risk(expected) - 2.5) < 1.0e-15, "expected risk mismatch");

    const std::vector<g5::WorstRiskPartial> worst{
        {7.0, 4},
        {9.0, 8},
        {9.0, 2},
    };
    const auto worst_result = g5::reduce_worst_risk(worst);
    require(
        worst_result.loss == 9.0 && worst_result.scenario == 2,
        "worst-case deterministic tie handling is incorrect"
    );

    const std::vector<g5::CvarEpigraphPartial> cvar{
        {0.5, 0.0, 0.4},
        {0.75, 0.2, 0.6},
    };
    const auto cvar_result = g5::reduce_cvar_epigraph(cvar);
    require(
        std::abs(cvar_result.weighted_excess - 1.25) < 1.0e-15
            && std::abs(cvar_result.maximum_epigraph_violation - 0.2) < 1.0e-15
            && std::abs(cvar_result.threshold_dual_sum - 1.0) < 1.0e-15,
        "CVaR epigraph or dual reduction is incorrect"
    );

    const auto summary = distributed::aggregate_scenario_risk(
        {1.0, 2.0, 9.0},
        {0.25, 0.50, 0.25},
        0.75
    );
    require(
        std::abs(summary.expected - 3.5) < 1.0e-15
            && summary.worst == 9.0 && summary.value_at_risk == 2.0
            && summary.conditional_value_at_risk == 9.0,
        "CPU risk truth semantics changed"
    );

    const std::vector<g5::ResidualPartial> residuals{
        {1.0, 4.0, 9.0, 0.1, 0.3, 0.2},
        {3.0, 5.0, 7.0, 0.4, 0.2, 0.5},
    };
    const auto residual = g5::reduce_residuals(residuals);
    require(
        residual.squared_primal == 4.0 && residual.squared_dual == 9.0
            && residual.squared_gap == 16.0 && residual.maximum_cone_distance == 0.4
            && residual.maximum_nonanticipativity == 0.3
            && residual.maximum_risk_epigraph == 0.5,
        "global residual reduction is incorrect"
    );
}

void test_ordering_failure_and_cancellation() {
    g5::CollectiveOrdering ordering;
    ordering.begin();
    ordering.collective_wait();
    ordering.enqueue();
    ordering.collective_complete();
    ordering.compute_wait();
    ordering.finish();
    require(ordering.epoch() == 1, "collective ordering epoch was not advanced");
    require_throws([&ordering] { ordering.enqueue(); }, "invalid collective transition was accepted");

    for (const std::size_t ranks : {1U, 2U, 4U, 8U}) {
        std::vector<g5::RankStatus> statuses(ranks, g5::RankStatus::healthy);
        require(
            g5::aggregate_rank_status(statuses) == g5::RankStatus::healthy,
            "healthy rank aggregation failed"
        );
        statuses.back() = g5::RankStatus::cancelled;
        require(
            g5::aggregate_rank_status(statuses) == g5::RankStatus::cancelled,
            "cancellation did not propagate"
        );
        statuses.front() = g5::RankStatus::failed;
        require(
            g5::aggregate_rank_status(statuses) == g5::RankStatus::failed,
            "rank failure did not propagate"
        );
        statuses[ranks / 2U] = g5::RankStatus::rank_lost;
        require(
            g5::aggregate_rank_status(statuses) == g5::RankStatus::rank_lost,
            "rank loss did not dominate the global status"
        );
    }
}

void test_checkpoint_and_topology_validation() {
    const std::array<std::byte, 5> payload{
        std::byte{1},
        std::byte{2},
        std::byte{3},
        std::byte{4},
        std::byte{5},
    };
    g5::RankCheckpointHeader header{};
    header.topology_fingerprint = 0x11223344;
    header.partition_fingerprint = 0x55667788;
    header.local_workspace_bytes = payload.size();
    header.local_scenario_count = 3;
    header.primal_elements = 20;
    header.dual_elements = 12;
    header.scaling_elements = 20;
    header.world_size = 4;
    header.rank = 2;
    header.device = 2;
    header.warm_ownership = g5::WarmOwnership::full_state;
    const auto bytes = g5::pack_rank_checkpoint(header, payload);
    const auto restored = g5::validate_rank_checkpoint(
        bytes,
        header.topology_fingerprint,
        header.partition_fingerprint,
        4,
        2,
        2
    );
    require(
        restored.warm_ownership == g5::WarmOwnership::full_state
            && restored.primal_elements == 20 && restored.dual_elements == 12,
        "checkpoint lost warm-state ownership"
    );
    require_throws(
        [&bytes, &header] {
            static_cast<void>(g5::validate_rank_checkpoint(
                bytes,
                header.topology_fingerprint + 1U,
                header.partition_fingerprint,
                4,
                2,
                2
            ));
        },
        "topology mutation was accepted"
    );
    require_throws(
        [&bytes, &header] {
            static_cast<void>(g5::validate_rank_checkpoint(
                bytes,
                header.topology_fingerprint,
                header.partition_fingerprint,
                8,
                2,
                2
            ));
        },
        "rank-loss/world-size checkpoint incompatibility was accepted"
    );
}

void test_telemetry_schema_contract() {
    g5::RuntimeTelemetry telemetry{};
    telemetry.rank = 0;
    telemetry.world_size = 1;
    telemetry.local_rank = 0;
    telemetry.device = 0;
    telemetry.collectives.push_back(g5::CollectiveTelemetry{
        g5::CollectiveKind::shared_arrowhead_sum,
        4,
        32,
        256,
        0,
        1,
        "non-anticipativity shared gradient",
        0.001,
        0.001,
        0.0,
    });
    const auto& collective = telemetry.collectives.front();
    require(
        collective.call_count > 0 && collective.element_count > 0
            && collective.payload_bytes == collective.element_count * sizeof(double)
            && collective.frequency > 0 && !collective.purpose.empty(),
        "collective schema omitted count/bytes/frequency/purpose telemetry"
    );
    require(
        telemetry.rank == 0 && telemetry.world_size == 1 && telemetry.device == 0
            && telemetry.deterministic,
        "rank or deterministic-mode telemetry is incomplete"
    );
}

}  // namespace

int main() {
    try {
        test_partition_contracts();
        test_orbitweaver_rank_local_ownership();
        test_arrowhead_and_nonanticipativity();
        test_risk_and_residual_reductions();
        test_ordering_failure_and_cancellation();
        test_checkpoint_and_topology_validation();
        test_telemetry_schema_contract();
        std::cout << "G5 logical-rank contracts passed for 1/2/4/8 ranks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
