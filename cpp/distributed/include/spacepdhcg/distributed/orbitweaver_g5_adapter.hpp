#pragma once

#include "spacepdhcg/distributed/runtime.hpp"
#include "spacepdhcg/orbitweaver/g7_orchestration.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver::g7 {

struct G5ArcPartitionMetadata {
    std::uint64_t deterministic_id{0U};
    std::size_t route_index{0U};
    std::size_t trajectory_arc_index{0U};
    std::size_t scenario_index{0U};
    distributed::g5::ScenarioWork work{};

    [[nodiscard]] auto tie() const noexcept {
        return std::tie(
            route_index,
            trajectory_arc_index,
            scenario_index,
            deterministic_id
        );
    }
};

struct G5ArcPartition {
    distributed::g5::PartitionPlan scenario_plan{};
    std::vector<G5ArcPartitionMetadata> items{};
    std::map<std::uint64_t, Ownership> owners{};

    void validate(std::span<const std::size_t> devices) const {
        scenario_plan.validate(items.size(), devices.size());
        if (devices.empty() || owners.size() != items.size()) {
            throw std::invalid_argument("G5 OrbitWeaver ownership is incomplete");
        }
        for (std::size_t item = 0U; item < items.size(); ++item) {
            const auto iterator = owners.find(items[item].deterministic_id);
            const auto rank = scenario_plan.scenario_owner[item];
            if (iterator == owners.end() || iterator->second.rank != rank
                || iterator->second.device != devices[rank]) {
                throw std::invalid_argument("G5 OrbitWeaver ownership disagrees with partition");
            }
        }
    }
};

[[nodiscard]] inline G5ArcPartition partition_g5_orbitweaver_arcs(
    std::vector<G5ArcPartitionMetadata> items,
    std::span<const std::size_t> devices,
    const distributed::g5::ScenarioCostModel& cost_model = {}
) {
    if (items.empty() || devices.empty()) {
        throw std::invalid_argument("G5 OrbitWeaver partition requires arcs and devices");
    }
    std::stable_sort(items.begin(), items.end(), [](const auto& left, const auto& right) {
        return left.tie() < right.tie();
    });
    std::set<std::uint64_t> identifiers{};
    for (const auto& item : items) {
        if (!identifiers.insert(item.deterministic_id).second) {
            throw std::invalid_argument("G5 OrbitWeaver deterministic IDs must be unique");
        }
    }
    std::vector<distributed::g5::ScenarioWork> work{};
    work.reserve(items.size());
    for (const auto& item : items) {
        work.push_back(item.work);
    }
    auto plan = distributed::g5::partition_scenarios(
        work,
        devices.size(),
        distributed::g5::PartitionKind::scenario_aware,
        cost_model
    );
    std::map<std::uint64_t, Ownership> owners{};
    for (std::size_t index = 0U; index < items.size(); ++index) {
        const auto rank = plan.scenario_owner[index];
        owners.emplace(items[index].deterministic_id, Ownership{rank, devices[rank]});
    }
    G5ArcPartition result{std::move(plan), std::move(items), std::move(owners)};
    result.validate(devices);
    return result;
}

class G5RankLocalOwnershipAdapter final : public OwnershipPolicy {
  public:
    explicit G5RankLocalOwnershipAdapter(G5ArcPartition partition)
        : partition_(std::move(partition)) {}

    [[nodiscard]] Ownership owner(const ScheduledArc& arc, std::size_t) const override {
        const auto iterator = partition_.owners.find(arc.deterministic_id);
        if (iterator == partition_.owners.end()) {
            throw std::invalid_argument("arc is absent from frozen G5 ownership");
        }
        const auto metadata = std::find_if(
            partition_.items.begin(),
            partition_.items.end(),
            [&arc](const auto& item) {
                return item.deterministic_id == arc.deterministic_id;
            }
        );
        if (metadata == partition_.items.end()
            || metadata->route_index != arc.route_index
            || metadata->trajectory_arc_index != arc.trajectory_arc_index
            || metadata->scenario_index != arc.scenario_index) {
            throw std::invalid_argument("arc metadata changed after G5 partitioning");
        }
        return iterator->second;
    }

    [[nodiscard]] const G5ArcPartition& partition() const noexcept {
        return partition_;
    }

  private:
    G5ArcPartition partition_{};
};

/// Rank-local G5 backend wrapping a persistent G3 arc backend.
///
/// The adapter verifies that every scheduled batch belongs to this MPI rank/device,
/// propagates cancellation/failure through G5 status synchronization, and exposes the
/// G5 runtime's collective telemetry without claiming physical scaling evidence.
class G5RankLocalArcBackend final : public ArcBatchBackend {
  public:
    G5RankLocalArcBackend(
        std::shared_ptr<ArcBatchBackend> g3,
        distributed::g5::MpiNcclRuntime& runtime
    )
        : g3_(std::move(g3)), runtime_(&runtime) {
        if (!g3_) {
            throw std::invalid_argument("G5 rank-local adapter requires a G3 backend");
        }
    }

    [[nodiscard]] std::vector<ArcExecution> evaluate(
        const TopologyFidelityKey& group,
        const std::vector<ScheduledArc>& batch,
        const Ownership owner,
        const std::atomic<bool>& cancelled
    ) override {
        if (owner.rank != static_cast<std::size_t>(runtime_->rank())
            || owner.device != static_cast<std::size_t>(runtime_->device())) {
            throw std::invalid_argument("G5 batch was submitted to the wrong rank/device");
        }
        if (cancelled.load()) {
            runtime_->cancel();
        }
        try {
            auto result = g3_->evaluate(group, batch, owner, cancelled);
            if (result.size() != batch.size()) {
                runtime_->fail("G3 rank-local result count mismatch");
                throw std::runtime_error("G3 rank-local result count mismatch");
            }
            const auto failed = std::any_of(result.begin(), result.end(), [](const auto& item) {
                return item.status != ArcExecutionStatus::feasible
                       && item.status != ArcExecutionStatus::infeasible
                       && item.status != ArcExecutionStatus::unsupported;
            });
            if (failed) {
                runtime_->fail("rank-local trajectory execution failed");
            }
            static_cast<void>(runtime_->synchronize_status());
            return result;
        } catch (const std::exception& error) {
            runtime_->fail(error.what());
            static_cast<void>(runtime_->synchronize_status());
            throw;
        }
    }

    [[nodiscard]] const distributed::g5::RuntimeTelemetry& telemetry() const noexcept {
        return runtime_->telemetry();
    }

    void cancel() noexcept { runtime_->cancel(); }

  private:
    std::shared_ptr<ArcBatchBackend> g3_{};
    distributed::g5::MpiNcclRuntime* runtime_{nullptr};
};

}  // namespace spacepdhcg::orbitweaver::g7
