#pragma once

#include "spacepdhcg/persistent_cqp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace spacepdhcg::core {

struct Scenario {
    std::string name;
    double probability{0.0};
    std::vector<std::string> information_history;
};

struct InformationNode {
    std::size_t stage{0};
    std::vector<std::string> history;
    std::vector<std::size_t> scenario_indices;

    [[nodiscard]] bool shared() const noexcept { return scenario_indices.size() > 1U; }
};

class ScenarioTree {
  public:
    explicit ScenarioTree(
        std::vector<Scenario> scenarios,
        const double probability_tolerance = 1.0e-12
    )
        : scenarios_(std::move(scenarios)), probability_tolerance_(probability_tolerance) {
        validate();
        build_nodes();
    }

    [[nodiscard]] std::size_t scenario_count() const noexcept { return scenarios_.size(); }
    [[nodiscard]] std::size_t horizon() const noexcept {
        return scenarios_.front().information_history.size();
    }
    [[nodiscard]] const std::vector<Scenario>& scenarios() const noexcept { return scenarios_; }
    [[nodiscard]] const std::vector<InformationNode>& nodes() const noexcept { return nodes_; }

    [[nodiscard]] std::vector<const InformationNode*> shared_nodes() const {
        std::vector<const InformationNode*> result;
        for (const auto& node : nodes_) {
            if (node.shared()) {
                result.push_back(&node);
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<const InformationNode*> nodes_at_stage(
        const std::size_t stage
    ) const {
        if (stage >= horizon()) {
            throw std::out_of_range("scenario-tree stage is outside the horizon");
        }
        std::vector<const InformationNode*> result;
        for (const auto& node : nodes_) {
            if (node.stage == stage) {
                result.push_back(&node);
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<double> probabilities() const {
        std::vector<double> result;
        result.reserve(scenarios_.size());
        for (const auto& scenario : scenarios_) {
            result.push_back(scenario.probability);
        }
        return result;
    }

    [[nodiscard]] static ScenarioTree common_open_loop(
        const std::size_t scenario_count,
        const std::size_t horizon,
        const std::size_t common_prefix,
        std::vector<double> probabilities = {}
    ) {
        if (scenario_count == 0U || horizon == 0U) {
            throw std::invalid_argument("scenario count and horizon must be positive");
        }
        if (common_prefix > horizon) {
            throw std::invalid_argument("common prefix exceeds the scenario horizon");
        }
        if (probabilities.empty()) {
            probabilities.assign(scenario_count, 1.0 / static_cast<double>(scenario_count));
        }
        if (probabilities.size() != scenario_count) {
            throw std::invalid_argument("one probability is required per scenario");
        }
        std::vector<Scenario> scenarios;
        scenarios.reserve(scenario_count);
        for (std::size_t scenario_index = 0; scenario_index < scenario_count; ++scenario_index) {
            std::vector<std::string> history;
            history.reserve(horizon);
            for (std::size_t stage = 0; stage < horizon; ++stage) {
                if (stage < common_prefix) {
                    history.emplace_back("open-loop");
                } else {
                    history.push_back(
                        "scenario-" + std::to_string(scenario_index) + "/recourse-" +
                        std::to_string(stage)
                    );
                }
            }
            scenarios.push_back(Scenario{
                "scenario-" + std::to_string(scenario_index),
                probabilities[scenario_index],
                std::move(history),
            });
        }
        return ScenarioTree{std::move(scenarios)};
    }

    [[nodiscard]] static ScenarioTree common_open_loop(
        const std::size_t scenario_count,
        const std::size_t horizon
    ) {
        return common_open_loop(scenario_count, horizon, horizon);
    }

  private:
    void validate() const {
        if (scenarios_.empty()) {
            throw std::invalid_argument("scenario tree requires at least one scenario");
        }
        if (!std::isfinite(probability_tolerance_) || probability_tolerance_ < 0.0) {
            throw std::invalid_argument("probability tolerance must be finite and non-negative");
        }
        const auto expected_horizon = scenarios_.front().information_history.size();
        if (expected_horizon == 0U) {
            throw std::invalid_argument("scenario information history may not be empty");
        }
        std::vector<std::string> names;
        names.reserve(scenarios_.size());
        double probability_sum = 0.0;
        for (const auto& scenario : scenarios_) {
            if (scenario.name.empty()) {
                throw std::invalid_argument("scenario name may not be empty");
            }
            if (!std::isfinite(scenario.probability) || scenario.probability <= 0.0) {
                throw std::invalid_argument("scenario probability must be finite and positive");
            }
            if (scenario.information_history.size() != expected_horizon) {
                throw std::invalid_argument("all scenarios must share one information horizon");
            }
            if (std::any_of(
                    scenario.information_history.begin(),
                    scenario.information_history.end(),
                    [](const std::string& value) { return value.empty(); }
                )) {
                throw std::invalid_argument("information-history labels may not be empty");
            }
            names.push_back(scenario.name);
            probability_sum += scenario.probability;
        }
        std::sort(names.begin(), names.end());
        if (std::adjacent_find(names.begin(), names.end()) != names.end()) {
            throw std::invalid_argument("scenario names must be unique");
        }
        if (std::abs(probability_sum - 1.0) > probability_tolerance_) {
            throw std::invalid_argument("scenario probabilities must sum to one");
        }
    }

    void build_nodes() {
        for (std::size_t stage = 0; stage < horizon(); ++stage) {
            std::map<std::vector<std::string>, std::vector<std::size_t>> groups;
            for (std::size_t scenario_index = 0; scenario_index < scenarios_.size();
                 ++scenario_index) {
                const auto& full_history = scenarios_[scenario_index].information_history;
                std::vector<std::string> prefix(
                    full_history.begin(),
                    full_history.begin() + static_cast<std::ptrdiff_t>(stage + 1U)
                );
                groups[std::move(prefix)].push_back(scenario_index);
            }
            std::vector<InformationNode> stage_nodes;
            stage_nodes.reserve(groups.size());
            for (auto& [history, indices] : groups) {
                stage_nodes.push_back(InformationNode{stage, history, indices});
            }
            std::sort(
                stage_nodes.begin(),
                stage_nodes.end(),
                [](const InformationNode& left, const InformationNode& right) {
                    return std::tie(left.scenario_indices.front(), left.history) <
                        std::tie(right.scenario_indices.front(), right.history);
                }
            );
            nodes_.insert(nodes_.end(), stage_nodes.begin(), stage_nodes.end());
        }
    }

    std::vector<Scenario> scenarios_;
    double probability_tolerance_{1.0e-12};
    std::vector<InformationNode> nodes_;
};

struct ConsensusBlock {
    const InformationNode* node{nullptr};
    std::size_t offset{0};
    std::size_t dimension{0};
};

struct SparseTriplet {
    Index row{0};
    Index column{0};
    double value{0.0};
};

struct CommunicationProfile {
    std::size_t device_count{0};
    std::size_t shared_dimension{0};
    std::size_t payload_bytes{0};
    double bytes_per_device{0.0};
    double aggregate_bytes{0.0};
};

class BlockArrowLayout {
  public:
    BlockArrowLayout(
        const ScenarioTree& tree,
        const std::size_t state_dimension,
        const std::size_t control_dimension,
        const std::size_t local_auxiliary_dimension = 0U
    )
        : tree_(&tree),
          state_dimension_(state_dimension),
          control_dimension_(control_dimension),
          local_auxiliary_dimension_(local_auxiliary_dimension) {
        if (state_dimension == 0U || control_dimension == 0U) {
            throw std::invalid_argument("state and control dimensions must be positive");
        }
        state_variables_per_scenario_ = (tree.horizon() + 1U) * state_dimension_;
        control_variables_per_scenario_ = tree.horizon() * control_dimension_;
        local_variables_per_scenario_ = state_variables_per_scenario_ +
            control_variables_per_scenario_ + local_auxiliary_dimension_;
        local_dimension_ = tree.scenario_count() * local_variables_per_scenario_;
        std::size_t offset = local_dimension_;
        for (const auto* node : tree.shared_nodes()) {
            consensus_blocks_.push_back(ConsensusBlock{node, offset, control_dimension_});
            offset += control_dimension_;
        }
        total_variables_ = offset;
    }

    [[nodiscard]] std::size_t local_variables_per_scenario() const noexcept {
        return local_variables_per_scenario_;
    }
    [[nodiscard]] std::size_t local_dimension() const noexcept { return local_dimension_; }
    [[nodiscard]] std::size_t consensus_dimension() const noexcept {
        return total_variables_ - local_dimension_;
    }
    [[nodiscard]] std::size_t total_variables() const noexcept { return total_variables_; }
    [[nodiscard]] const std::vector<ConsensusBlock>& consensus_blocks() const noexcept {
        return consensus_blocks_;
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> scenario_range(
        const std::size_t scenario
    ) const {
        validate_scenario(scenario);
        const auto start = scenario * local_variables_per_scenario_;
        return {start, start + local_variables_per_scenario_};
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> control_range(
        const std::size_t scenario,
        const std::size_t stage
    ) const {
        validate_scenario(scenario);
        if (stage >= tree_->horizon()) {
            throw std::out_of_range("control stage is outside the scenario horizon");
        }
        const auto scenario_start = scenario * local_variables_per_scenario_;
        const auto start = scenario_start + state_variables_per_scenario_ +
            stage * control_dimension_;
        return {start, start + control_dimension_};
    }

    [[nodiscard]] std::size_t nonanticipativity_rows() const noexcept {
        std::size_t rows = 0U;
        for (const auto& block : consensus_blocks_) {
            rows += block.node->scenario_indices.size() * control_dimension_;
        }
        return rows;
    }

    [[nodiscard]] std::vector<SparseTriplet> nonanticipativity_triplets() const {
        std::vector<SparseTriplet> triplets;
        triplets.reserve(2U * nonanticipativity_rows());
        Index row = 0;
        for (const auto& block : consensus_blocks_) {
            for (const auto scenario : block.node->scenario_indices) {
                const auto local = control_range(scenario, block.node->stage);
                for (std::size_t component = 0; component < control_dimension_; ++component) {
                    triplets.push_back(SparseTriplet{
                        row,
                        static_cast<Index>(local.first + component),
                        1.0,
                    });
                    triplets.push_back(SparseTriplet{
                        row,
                        static_cast<Index>(block.offset + component),
                        -1.0,
                    });
                    ++row;
                }
            }
        }
        return triplets;
    }

    [[nodiscard]] CommunicationProfile communication_profile(
        const std::size_t device_count,
        const std::size_t scalar_bytes = sizeof(double)
    ) const {
        if (device_count == 0U || scalar_bytes == 0U) {
            throw std::invalid_argument("device count and scalar size must be positive");
        }
        const auto payload = consensus_dimension() * scalar_bytes;
        if (device_count == 1U || payload == 0U) {
            return CommunicationProfile{device_count, consensus_dimension(), payload, 0.0, 0.0};
        }
        const double per_device = 2.0 * static_cast<double>(device_count - 1U) /
            static_cast<double>(device_count) * static_cast<double>(payload);
        return CommunicationProfile{
            device_count,
            consensus_dimension(),
            payload,
            per_device,
            static_cast<double>(device_count) * per_device,
        };
    }

  private:
    void validate_scenario(const std::size_t scenario) const {
        if (scenario >= tree_->scenario_count()) {
            throw std::out_of_range("scenario index is outside the scenario tree");
        }
    }

    const ScenarioTree* tree_{nullptr};
    std::size_t state_dimension_{0};
    std::size_t control_dimension_{0};
    std::size_t local_auxiliary_dimension_{0};
    std::size_t state_variables_per_scenario_{0};
    std::size_t control_variables_per_scenario_{0};
    std::size_t local_variables_per_scenario_{0};
    std::size_t local_dimension_{0};
    std::size_t total_variables_{0};
    std::vector<ConsensusBlock> consensus_blocks_;
};

struct ScenarioPartition {
    std::vector<std::vector<std::size_t>> assignments;
    std::vector<double> loads;

    [[nodiscard]] double maximum_load() const noexcept {
        return loads.empty() ? 0.0 : *std::max_element(loads.begin(), loads.end());
    }

    [[nodiscard]] double mean_load() const noexcept {
        return loads.empty()
            ? 0.0
            : std::accumulate(loads.begin(), loads.end(), 0.0) /
                static_cast<double>(loads.size());
    }

    [[nodiscard]] double imbalance() const noexcept {
        const double mean = mean_load();
        return mean > 0.0 ? maximum_load() / mean : 1.0;
    }
};

[[nodiscard]] inline ScenarioPartition partition_scenarios(
    const std::vector<double>& weights,
    const std::size_t device_count
) {
    if (weights.empty() || device_count == 0U) {
        throw std::invalid_argument("scenario weights and device count must be non-empty");
    }
    for (const auto weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::invalid_argument("scenario weights must be finite and non-negative");
        }
    }
    std::vector<std::size_t> order(weights.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&weights](const auto left, const auto right) {
        if (weights[left] != weights[right]) {
            return weights[left] > weights[right];
        }
        return left < right;
    });

    ScenarioPartition result{
        std::vector<std::vector<std::size_t>>(device_count),
        std::vector<double>(device_count, 0.0),
    };
    for (const auto scenario : order) {
        const auto device = static_cast<std::size_t>(
            std::distance(result.loads.begin(), std::min_element(result.loads.begin(), result.loads.end()))
        );
        result.assignments[device].push_back(scenario);
        result.loads[device] += weights[scenario];
    }
    for (auto& assignment : result.assignments) {
        std::sort(assignment.begin(), assignment.end());
    }
    return result;
}

}  // namespace spacepdhcg::core
