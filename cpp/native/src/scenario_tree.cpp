#include "spacepdhcg/native/scenario_tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace spacepdhcg::native {
namespace {

[[nodiscard]] Index checked_index(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        throw std::overflow_error(std::string(name) + " exceeds native Index capacity");
    }
    return static_cast<Index>(value);
}

}  // namespace

std::string InformationNode::key() const {
    std::ostringstream stream;
    stream << "stage=" << stage << "|history=";
    for (std::size_t index = 0; index < history.size(); ++index) {
        if (index > 0) {
            stream << '/';
        }
        stream << history[index];
    }
    return stream.str();
}

ScenarioTree::ScenarioTree(
    std::vector<Scenario> scenarios,
    double probability_tolerance
)
    : scenarios_(std::move(scenarios)) {
    validate(probability_tolerance);
    nodes_ = build_nodes();
}

void ScenarioTree::validate(double probability_tolerance) const {
    if (scenarios_.empty()) {
        throw std::invalid_argument("a scenario tree requires at least one scenario");
    }
    if (!std::isfinite(probability_tolerance) || probability_tolerance < 0.0) {
        throw std::invalid_argument("probability tolerance must be finite and non-negative");
    }

    std::set<std::string> names;
    const std::size_t expected_horizon = scenarios_.front().information_history.size();
    if (expected_horizon == 0) {
        throw std::invalid_argument("scenario information histories may not be empty");
    }
    double probability_sum = 0.0;
    for (const auto& scenario : scenarios_) {
        if (scenario.name.empty() || !names.insert(scenario.name).second) {
            throw std::invalid_argument("scenario names must be non-empty and unique");
        }
        if (!std::isfinite(scenario.probability) || scenario.probability <= 0.0) {
            throw std::invalid_argument("scenario probabilities must be finite and positive");
        }
        if (scenario.information_history.size() != expected_horizon) {
            throw std::invalid_argument("all scenarios must have the same information horizon");
        }
        if (std::ranges::any_of(
                scenario.information_history,
                [](const std::string& label) { return label.empty(); }
            )) {
            throw std::invalid_argument("information-history labels may not be empty");
        }
        probability_sum += scenario.probability;
    }
    if (std::abs(probability_sum - 1.0) > probability_tolerance) {
        throw std::invalid_argument("scenario probabilities must sum to one");
    }
}

std::vector<InformationNode> ScenarioTree::build_nodes() const {
    std::vector<InformationNode> result;
    for (std::size_t stage = 0; stage < horizon(); ++stage) {
        std::map<std::vector<std::string>, std::vector<std::size_t>> groups;
        for (std::size_t scenario = 0; scenario < scenarios_.size(); ++scenario) {
            const auto& full_history = scenarios_[scenario].information_history;
            std::vector<std::string> prefix(
                full_history.begin(),
                full_history.begin() + static_cast<std::ptrdiff_t>(stage + 1U)
            );
            groups[std::move(prefix)].push_back(scenario);
        }

        std::vector<InformationNode> stage_nodes;
        stage_nodes.reserve(groups.size());
        for (auto& [history, scenario_indices] : groups) {
            stage_nodes.push_back(InformationNode{
                stage,
                std::move(history),
                std::move(scenario_indices),
            });
        }
        std::sort(
            stage_nodes.begin(),
            stage_nodes.end(),
            [](const InformationNode& left, const InformationNode& right) {
                const auto left_first = left.scenarios.front();
                const auto right_first = right.scenarios.front();
                if (left_first != right_first) {
                    return left_first < right_first;
                }
                return left.history < right.history;
            }
        );
        result.insert(
            result.end(),
            std::make_move_iterator(stage_nodes.begin()),
            std::make_move_iterator(stage_nodes.end())
        );
    }
    return result;
}

ScenarioTree ScenarioTree::common_open_loop(
    std::size_t scenario_count,
    std::size_t horizon,
    std::size_t common_prefix,
    std::span<const double> probabilities
) {
    if (scenario_count == 0 || horizon == 0) {
        throw std::invalid_argument("scenario count and horizon must be positive");
    }
    if (common_prefix > horizon) {
        throw std::invalid_argument("common prefix may not exceed the scenario horizon");
    }
    if (!probabilities.empty() && probabilities.size() != scenario_count) {
        throw std::invalid_argument("one probability is required per scenario");
    }

    std::vector<Scenario> scenarios;
    scenarios.reserve(scenario_count);
    for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
        std::vector<std::string> history;
        history.reserve(horizon);
        for (std::size_t stage = 0; stage < horizon; ++stage) {
            if (stage < common_prefix) {
                history.push_back("open-loop-stage-" + std::to_string(stage));
            } else {
                history.push_back(
                    "scenario-" + std::to_string(scenario) + "-stage-" +
                    std::to_string(stage)
                );
            }
        }
        scenarios.push_back(Scenario{
            "scenario-" + std::to_string(scenario),
            probabilities.empty() ? 1.0 / static_cast<double>(scenario_count)
                                  : probabilities[scenario],
            std::move(history),
        });
    }
    return ScenarioTree(std::move(scenarios));
}

std::size_t ScenarioTree::horizon() const noexcept {
    return scenarios_.front().information_history.size();
}

std::vector<double> ScenarioTree::probabilities() const {
    std::vector<double> result;
    result.reserve(scenarios_.size());
    for (const auto& scenario : scenarios_) {
        result.push_back(scenario.probability);
    }
    return result;
}

std::vector<InformationNode> ScenarioTree::nodes_at_stage(std::size_t stage) const {
    if (stage >= horizon()) {
        throw std::out_of_range("scenario-tree stage lies outside the horizon");
    }
    std::vector<InformationNode> result;
    for (const auto& node : nodes_) {
        if (node.stage == stage) {
            result.push_back(node);
        }
    }
    return result;
}

std::vector<InformationNode> ScenarioTree::shared_nodes() const {
    std::vector<InformationNode> result;
    for (const auto& node : nodes_) {
        if (node.shared()) {
            result.push_back(node);
        }
    }
    return result;
}

BlockArrowLayout::BlockArrowLayout(
    ScenarioTree tree,
    std::size_t state_dimension,
    std::size_t control_dimension,
    std::size_t local_auxiliary_dimension
)
    : tree_(std::move(tree)),
      state_dimension_(state_dimension),
      control_dimension_(control_dimension),
      local_auxiliary_dimension_(local_auxiliary_dimension) {
    if (state_dimension_ == 0 || control_dimension_ == 0) {
        throw std::invalid_argument("block-arrow state and control dimensions must be positive");
    }
    std::size_t offset = local_dimension();
    for (const auto& node : tree_.shared_nodes()) {
        consensus_blocks_.push_back(ConsensusBlock{
            node,
            VariableSlice{offset, control_dimension_},
        });
        offset += control_dimension_;
    }
    checked_index(offset, "block-arrow variable count");
    checked_index(nonanticipativity_rows(), "non-anticipativity row count");
}

std::size_t BlockArrowLayout::state_variables_per_scenario() const noexcept {
    return (tree_.horizon() + 1U) * state_dimension_;
}

std::size_t BlockArrowLayout::control_variables_per_scenario() const noexcept {
    return tree_.horizon() * control_dimension_;
}

std::size_t BlockArrowLayout::local_variables_per_scenario() const noexcept {
    return state_variables_per_scenario() + control_variables_per_scenario() +
           local_auxiliary_dimension_;
}

std::size_t BlockArrowLayout::local_dimension() const noexcept {
    return tree_.scenario_count() * local_variables_per_scenario();
}

std::size_t BlockArrowLayout::consensus_dimension() const noexcept {
    return consensus_blocks_.size() * control_dimension_;
}

std::size_t BlockArrowLayout::total_variables() const noexcept {
    return local_dimension() + consensus_dimension();
}

std::size_t BlockArrowLayout::nonanticipativity_rows() const noexcept {
    std::size_t rows = 0;
    for (const auto& block : consensus_blocks_) {
        rows += block.node.scenarios.size() * control_dimension_;
    }
    return rows;
}

void BlockArrowLayout::validate_scenario(std::size_t scenario) const {
    if (scenario >= tree_.scenario_count()) {
        throw std::out_of_range("scenario index lies outside the block-arrow layout");
    }
}

VariableSlice BlockArrowLayout::scenario_slice(std::size_t scenario) const {
    validate_scenario(scenario);
    return VariableSlice{
        scenario * local_variables_per_scenario(),
        local_variables_per_scenario(),
    };
}

VariableSlice BlockArrowLayout::state_slice(std::size_t scenario, std::size_t node) const {
    validate_scenario(scenario);
    if (node > tree_.horizon()) {
        throw std::out_of_range("state node lies outside the scenario horizon");
    }
    return VariableSlice{
        scenario_slice(scenario).offset + node * state_dimension_,
        state_dimension_,
    };
}

VariableSlice BlockArrowLayout::control_slice(
    std::size_t scenario,
    std::size_t stage
) const {
    validate_scenario(scenario);
    if (stage >= tree_.horizon()) {
        throw std::out_of_range("control stage lies outside the scenario horizon");
    }
    return VariableSlice{
        scenario_slice(scenario).offset + state_variables_per_scenario() +
            stage * control_dimension_,
        control_dimension_,
    };
}

VariableSlice BlockArrowLayout::auxiliary_slice(std::size_t scenario) const {
    validate_scenario(scenario);
    return VariableSlice{
        scenario_slice(scenario).offset + state_variables_per_scenario() +
            control_variables_per_scenario(),
        local_auxiliary_dimension_,
    };
}

CscMatrix BlockArrowLayout::nonanticipativity_operator() const {
    CscBuilder builder(
        checked_index(nonanticipativity_rows(), "non-anticipativity row count"),
        checked_index(total_variables(), "block-arrow variable count")
    );
    std::size_t row = 0;
    for (const auto& block : consensus_blocks_) {
        for (std::size_t scenario : block.node.scenarios) {
            const auto local = control_slice(scenario, block.node.stage);
            for (std::size_t component = 0; component < control_dimension_; ++component) {
                builder.add(
                    checked_index(row, "non-anticipativity row"),
                    checked_index(local.offset + component, "local control column"),
                    1.0
                );
                builder.add(
                    checked_index(row, "non-anticipativity row"),
                    checked_index(block.variables.offset + component, "consensus column"),
                    -1.0
                );
                ++row;
            }
        }
    }
    return builder.build();
}

double BlockArrowLayout::nonanticipativity_violation(
    std::span<const double> decision
) const {
    if (decision.size() != total_variables()) {
        throw std::invalid_argument("global scenario decision has the wrong dimension");
    }
    for (double value : decision) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("global scenario decision must be finite");
        }
    }
    const auto residual = nonanticipativity_operator().multiply(decision);
    double maximum = 0.0;
    for (double value : residual) {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

}  // namespace spacepdhcg::native
