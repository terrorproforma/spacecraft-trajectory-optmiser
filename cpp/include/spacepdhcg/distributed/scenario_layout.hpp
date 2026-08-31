#pragma once

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

namespace spacepdhcg::distributed {

struct Scenario {
    std::string name{};
    double probability{0.0};
    std::vector<std::string> information_history{};

    void validate() const {
        if (name.empty()) {
            throw std::invalid_argument("scenario name may not be empty");
        }
        if (!std::isfinite(probability) || probability <= 0.0) {
            throw std::invalid_argument("scenario probability must be finite and positive");
        }
        if (information_history.empty()) {
            throw std::invalid_argument("scenario information history may not be empty");
        }
        for (const auto& label : information_history) {
            if (label.empty()) {
                throw std::invalid_argument("information-history labels may not be empty");
            }
        }
    }
};

struct InformationNode {
    std::size_t stage{0U};
    std::vector<std::string> history{};
    std::vector<std::size_t> scenario_indices{};

    [[nodiscard]] bool shared() const noexcept { return scenario_indices.size() > 1U; }
};

class ScenarioTree {
  public:
    explicit ScenarioTree(
        std::vector<Scenario> scenarios,
        double probability_tolerance = 1.0e-12
    )
        : scenarios_(std::move(scenarios)) {
        if (scenarios_.empty()) {
            throw std::invalid_argument("a scenario tree requires at least one scenario");
        }
        if (!std::isfinite(probability_tolerance) || probability_tolerance < 0.0) {
            throw std::invalid_argument("probability tolerance must be finite and non-negative");
        }
        std::vector<std::string> names{};
        names.reserve(scenarios_.size());
        const auto expected_horizon = scenarios_.front().information_history.size();
        double probability_sum{0.0};
        for (const auto& scenario : scenarios_) {
            scenario.validate();
            if (scenario.information_history.size() != expected_horizon) {
                throw std::invalid_argument("all scenarios must have the same information horizon");
            }
            names.push_back(scenario.name);
            probability_sum += scenario.probability;
        }
        std::sort(names.begin(), names.end());
        if (std::adjacent_find(names.begin(), names.end()) != names.end()) {
            throw std::invalid_argument("scenario names must be unique");
        }
        if (std::abs(probability_sum - 1.0) > probability_tolerance) {
            throw std::invalid_argument("scenario probabilities must sum to one");
        }
        build_nodes();
    }

    [[nodiscard]] std::size_t scenario_count() const noexcept { return scenarios_.size(); }
    [[nodiscard]] std::size_t horizon() const noexcept {
        return scenarios_.front().information_history.size();
    }
    [[nodiscard]] const std::vector<Scenario>& scenarios() const noexcept { return scenarios_; }
    [[nodiscard]] const std::vector<InformationNode>& nodes() const noexcept { return nodes_; }

    [[nodiscard]] std::vector<const InformationNode*> nodes_at_stage(std::size_t stage) const {
        validate_stage(stage);
        std::vector<const InformationNode*> result{};
        for (const auto& node : nodes_) {
            if (node.stage == stage) {
                result.push_back(&node);
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<const InformationNode*> shared_nodes() const {
        std::vector<const InformationNode*> result{};
        for (const auto& node : nodes_) {
            if (node.shared()) {
                result.push_back(&node);
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<double> probabilities() const {
        std::vector<double> result{};
        result.reserve(scenarios_.size());
        for (const auto& scenario : scenarios_) {
            result.push_back(scenario.probability);
        }
        return result;
    }

    static ScenarioTree common_open_loop(
        std::size_t scenario_count,
        std::size_t horizon,
        std::size_t common_prefix,
        std::vector<double> probabilities = {}
    ) {
        if (scenario_count == 0U || horizon == 0U) {
            throw std::invalid_argument("scenario count and horizon must be positive");
        }
        if (common_prefix > horizon) {
            throw std::invalid_argument("common prefix may not exceed the horizon");
        }
        if (probabilities.empty()) {
            probabilities.assign(scenario_count, 1.0 / static_cast<double>(scenario_count));
        }
        if (probabilities.size() != scenario_count) {
            throw std::invalid_argument("one probability is required per scenario");
        }
        std::vector<Scenario> scenarios{};
        scenarios.reserve(scenario_count);
        for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
            std::vector<std::string> history{};
            history.reserve(horizon);
            for (std::size_t stage = 0; stage < horizon; ++stage) {
                if (stage < common_prefix) {
                    history.emplace_back("open-loop");
                } else {
                    history.push_back(
                        "scenario-" + std::to_string(scenario) + "/recourse-"
                        + std::to_string(stage)
                    );
                }
            }
            scenarios.push_back(
                Scenario{
                    "scenario-" + std::to_string(scenario),
                    probabilities[scenario],
                    std::move(history),
                }
            );
        }
        return ScenarioTree(std::move(scenarios));
    }

    static ScenarioTree common_open_loop(
        std::size_t scenario_count,
        std::size_t horizon
    ) {
        return common_open_loop(scenario_count, horizon, horizon);
    }

  private:
    std::vector<Scenario> scenarios_{};
    std::vector<InformationNode> nodes_{};

    void validate_stage(std::size_t stage) const {
        if (stage >= horizon()) {
            throw std::out_of_range("scenario-tree stage is outside the horizon");
        }
    }

    void build_nodes() {
        for (std::size_t stage = 0; stage < horizon(); ++stage) {
            std::map<std::vector<std::string>, std::vector<std::size_t>> groups{};
            for (std::size_t scenario = 0; scenario < scenarios_.size(); ++scenario) {
                const auto& full = scenarios_[scenario].information_history;
                std::vector<std::string> prefix(full.begin(), full.begin() + stage + 1U);
                groups[std::move(prefix)].push_back(scenario);
            }
            std::vector<InformationNode> stage_nodes{};
            stage_nodes.reserve(groups.size());
            for (auto& [history, indices] : groups) {
                stage_nodes.push_back(
                    InformationNode{stage, std::move(history), std::move(indices)}
                );
            }
            std::sort(
                stage_nodes.begin(),
                stage_nodes.end(),
                [](const InformationNode& left, const InformationNode& right) {
                    return std::tie(left.scenario_indices.front(), left.history)
                           < std::tie(right.scenario_indices.front(), right.history);
                }
            );
            nodes_.insert(
                nodes_.end(),
                std::make_move_iterator(stage_nodes.begin()),
                std::make_move_iterator(stage_nodes.end())
            );
        }
    }
};

struct ConsensusBlock {
    InformationNode node{};
    std::size_t offset{0U};
    std::size_t dimension{0U};

    [[nodiscard]] std::pair<std::size_t, std::size_t> range() const noexcept {
        return {offset, offset + dimension};
    }
};

struct SparseTriplet {
    std::size_t row{0U};
    std::size_t column{0U};
    double value{0.0};
};

struct CommunicationProfile {
    std::size_t device_count{0U};
    std::size_t shared_dimension{0U};
    std::size_t payload_bytes{0U};
    double bytes_per_device{0.0};
    double aggregate_bytes{0.0};
    std::size_t collective_count{0U};
};

class BlockArrowLayout {
  public:
    BlockArrowLayout(
        ScenarioTree tree,
        std::size_t state_dimension,
        std::size_t control_dimension,
        std::size_t local_auxiliary_dimension = 0U
    )
        : tree_(std::move(tree)),
          state_dimension_(state_dimension),
          control_dimension_(control_dimension),
          local_auxiliary_dimension_(local_auxiliary_dimension) {
        if (state_dimension_ == 0U || control_dimension_ == 0U) {
            throw std::invalid_argument("state and control dimensions must be positive");
        }
        state_variables_per_scenario_ = (tree_.horizon() + 1U) * state_dimension_;
        control_variables_per_scenario_ = tree_.horizon() * control_dimension_;
        local_variables_per_scenario_ = state_variables_per_scenario_
                                        + control_variables_per_scenario_
                                        + local_auxiliary_dimension_;
        local_dimension_ = tree_.scenario_count() * local_variables_per_scenario_;
        auto offset = local_dimension_;
        for (const auto* node : tree_.shared_nodes()) {
            consensus_blocks_.push_back(
                ConsensusBlock{*node, offset, control_dimension_}
            );
            offset += control_dimension_;
        }
        total_variables_ = offset;
    }

    [[nodiscard]] const ScenarioTree& tree() const noexcept { return tree_; }
    [[nodiscard]] std::size_t state_variables_per_scenario() const noexcept {
        return state_variables_per_scenario_;
    }
    [[nodiscard]] std::size_t control_variables_per_scenario() const noexcept {
        return control_variables_per_scenario_;
    }
    [[nodiscard]] std::size_t local_variables_per_scenario() const noexcept {
        return local_variables_per_scenario_;
    }
    [[nodiscard]] std::size_t local_dimension() const noexcept { return local_dimension_; }
    [[nodiscard]] std::size_t total_variables() const noexcept { return total_variables_; }
    [[nodiscard]] std::size_t consensus_dimension() const noexcept {
        return total_variables_ - local_dimension_;
    }
    [[nodiscard]] const std::vector<ConsensusBlock>& consensus_blocks() const noexcept {
        return consensus_blocks_;
    }

    [[nodiscard]] std::size_t nonanticipativity_rows() const noexcept {
        std::size_t rows{0U};
        for (const auto& block : consensus_blocks_) {
            rows += block.node.scenario_indices.size() * control_dimension_;
        }
        return rows;
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> scenario_range(
        std::size_t scenario
    ) const {
        validate_scenario(scenario);
        const auto begin = scenario * local_variables_per_scenario_;
        return {begin, begin + local_variables_per_scenario_};
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> state_range(
        std::size_t scenario,
        std::size_t node
    ) const {
        validate_scenario(scenario);
        if (node > tree_.horizon()) {
            throw std::out_of_range("state node is outside the trajectory horizon");
        }
        const auto begin = scenario * local_variables_per_scenario_ + node * state_dimension_;
        return {begin, begin + state_dimension_};
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> control_range(
        std::size_t scenario,
        std::size_t stage
    ) const {
        validate_scenario(scenario);
        if (stage >= tree_.horizon()) {
            throw std::out_of_range("control stage is outside the trajectory horizon");
        }
        const auto begin = scenario * local_variables_per_scenario_
                           + state_variables_per_scenario_ + stage * control_dimension_;
        return {begin, begin + control_dimension_};
    }

    [[nodiscard]] std::vector<SparseTriplet> nonanticipativity_triplets() const {
        std::vector<SparseTriplet> triplets{};
        triplets.reserve(2U * nonanticipativity_rows());
        std::size_t row{0U};
        for (const auto& block : consensus_blocks_) {
            for (const auto scenario : block.node.scenario_indices) {
                const auto [local_begin, local_end] = control_range(scenario, block.node.stage);
                static_cast<void>(local_end);
                for (std::size_t component = 0; component < control_dimension_; ++component) {
                    triplets.push_back(SparseTriplet{row, local_begin + component, 1.0});
                    triplets.push_back(SparseTriplet{row, block.offset + component, -1.0});
                    ++row;
                }
            }
        }
        return triplets;
    }

    [[nodiscard]] CommunicationProfile communication_profile(
        std::size_t device_count,
        std::size_t scalar_bytes = 8U,
        std::size_t collective_count = 1U
    ) const {
        if (device_count == 0U || scalar_bytes == 0U) {
            throw std::invalid_argument("device count and scalar byte width must be positive");
        }
        const auto payload = consensus_dimension() * scalar_bytes;
        double per_device{0.0};
        double aggregate{0.0};
        if (device_count > 1U && payload > 0U && collective_count > 0U) {
            const auto per_collective = 2.0
                                        * static_cast<double>(device_count - 1U)
                                        / static_cast<double>(device_count)
                                        * static_cast<double>(payload);
            per_device = static_cast<double>(collective_count) * per_collective;
            aggregate = static_cast<double>(device_count) * per_device;
        }
        return CommunicationProfile{
            device_count,
            consensus_dimension(),
            payload,
            per_device,
            aggregate,
            collective_count,
        };
    }

  private:
    ScenarioTree tree_;
    std::size_t state_dimension_{0U};
    std::size_t control_dimension_{0U};
    std::size_t local_auxiliary_dimension_{0U};
    std::size_t state_variables_per_scenario_{0U};
    std::size_t control_variables_per_scenario_{0U};
    std::size_t local_variables_per_scenario_{0U};
    std::size_t local_dimension_{0U};
    std::size_t total_variables_{0U};
    std::vector<ConsensusBlock> consensus_blocks_{};

    void validate_scenario(std::size_t scenario) const {
        if (scenario >= tree_.scenario_count()) {
            throw std::out_of_range("scenario index is outside the scenario tree");
        }
    }
};

struct ScenarioPartition {
    std::vector<std::vector<std::size_t>> assignments{};
    std::vector<double> loads{};

    [[nodiscard]] double maximum_load() const noexcept {
        return loads.empty() ? 0.0 : *std::max_element(loads.begin(), loads.end());
    }

    [[nodiscard]] double mean_load() const noexcept {
        if (loads.empty()) {
            return 0.0;
        }
        return std::accumulate(loads.begin(), loads.end(), 0.0)
               / static_cast<double>(loads.size());
    }

    [[nodiscard]] double imbalance() const noexcept {
        const auto mean = mean_load();
        return mean > 0.0 ? maximum_load() / mean : 1.0;
    }
};

inline ScenarioPartition partition_scenarios(
    const std::vector<double>& weights,
    std::size_t device_count
) {
    if (weights.empty()) {
        throw std::invalid_argument("scenario weights may not be empty");
    }
    if (device_count == 0U) {
        throw std::invalid_argument("device count must be positive");
    }
    for (const auto weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::invalid_argument("scenario weights must be finite and non-negative");
        }
    }
    std::vector<std::size_t> order(weights.size());
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(
        order.begin(),
        order.end(),
        [&weights](std::size_t left, std::size_t right) {
            if (weights[left] != weights[right]) {
                return weights[left] > weights[right];
            }
            return left < right;
        }
    );

    ScenarioPartition result{
        std::vector<std::vector<std::size_t>>(device_count),
        std::vector<double>(device_count, 0.0),
    };
    for (const auto scenario : order) {
        const auto owner = static_cast<std::size_t>(
            std::distance(
                result.loads.begin(),
                std::min_element(result.loads.begin(), result.loads.end())
            )
        );
        result.assignments[owner].push_back(scenario);
        result.loads[owner] += weights[scenario];
    }
    for (auto& assignment : result.assignments) {
        std::sort(assignment.begin(), assignment.end());
    }
    return result;
}

struct LogicalGpuGrid {
    std::size_t scenario_partitions{1U};
    std::size_t time_partitions{1U};

    void validate() const {
        if (scenario_partitions == 0U || time_partitions == 0U) {
            throw std::invalid_argument("logical GPU-grid dimensions must be positive");
        }
    }

    [[nodiscard]] std::size_t device_count() const {
        validate();
        return scenario_partitions * time_partitions;
    }

    [[nodiscard]] std::size_t rank(
        std::size_t scenario_partition,
        std::size_t time_partition
    ) const {
        validate();
        if (scenario_partition >= scenario_partitions || time_partition >= time_partitions) {
            throw std::out_of_range("logical GPU-grid coordinate is outside the grid");
        }
        return scenario_partition * time_partitions + time_partition;
    }
};

}  // namespace spacepdhcg::distributed
