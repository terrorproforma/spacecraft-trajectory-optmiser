#pragma once

#include "spacepdhcg/native/cqp.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace spacepdhcg::native {

struct Scenario {
    std::string name{};
    double probability{0.0};
    std::vector<std::string> information_history{};
};

struct InformationNode {
    std::size_t stage{0};
    std::vector<std::string> history{};
    std::vector<std::size_t> scenarios{};

    [[nodiscard]] bool shared() const noexcept { return scenarios.size() > 1U; }
    [[nodiscard]] std::string key() const;
};

class ScenarioTree {
  public:
    explicit ScenarioTree(
        std::vector<Scenario> scenarios,
        double probability_tolerance = 1.0e-12
    );

    [[nodiscard]] static ScenarioTree common_open_loop(
        std::size_t scenario_count,
        std::size_t horizon,
        std::size_t common_prefix,
        std::span<const double> probabilities = {}
    );

    [[nodiscard]] const std::vector<Scenario>& scenarios() const noexcept {
        return scenarios_;
    }
    [[nodiscard]] const std::vector<InformationNode>& nodes() const noexcept { return nodes_; }
    [[nodiscard]] std::size_t scenario_count() const noexcept { return scenarios_.size(); }
    [[nodiscard]] std::size_t horizon() const noexcept;
    [[nodiscard]] std::vector<double> probabilities() const;
    [[nodiscard]] std::vector<InformationNode> nodes_at_stage(std::size_t stage) const;
    [[nodiscard]] std::vector<InformationNode> shared_nodes() const;

  private:
    std::vector<Scenario> scenarios_{};
    std::vector<InformationNode> nodes_{};

    void validate(double probability_tolerance) const;
    [[nodiscard]] std::vector<InformationNode> build_nodes() const;
};

struct VariableSlice {
    std::size_t offset{0};
    std::size_t size{0};

    [[nodiscard]] std::size_t stop() const noexcept { return offset + size; }
};

struct ConsensusBlock {
    InformationNode node{};
    VariableSlice variables{};
};

class BlockArrowLayout {
  public:
    BlockArrowLayout(
        ScenarioTree tree,
        std::size_t state_dimension,
        std::size_t control_dimension,
        std::size_t local_auxiliary_dimension = 0
    );

    [[nodiscard]] const ScenarioTree& tree() const noexcept { return tree_; }
    [[nodiscard]] std::size_t state_dimension() const noexcept { return state_dimension_; }
    [[nodiscard]] std::size_t control_dimension() const noexcept { return control_dimension_; }
    [[nodiscard]] std::size_t state_variables_per_scenario() const noexcept;
    [[nodiscard]] std::size_t control_variables_per_scenario() const noexcept;
    [[nodiscard]] std::size_t local_variables_per_scenario() const noexcept;
    [[nodiscard]] std::size_t local_dimension() const noexcept;
    [[nodiscard]] std::size_t consensus_dimension() const noexcept;
    [[nodiscard]] std::size_t total_variables() const noexcept;
    [[nodiscard]] std::size_t nonanticipativity_rows() const noexcept;
    [[nodiscard]] const std::vector<ConsensusBlock>& consensus_blocks() const noexcept {
        return consensus_blocks_;
    }

    [[nodiscard]] VariableSlice scenario_slice(std::size_t scenario) const;
    [[nodiscard]] VariableSlice state_slice(std::size_t scenario, std::size_t node) const;
    [[nodiscard]] VariableSlice control_slice(std::size_t scenario, std::size_t stage) const;
    [[nodiscard]] VariableSlice auxiliary_slice(std::size_t scenario) const;
    [[nodiscard]] CscMatrix nonanticipativity_operator() const;
    [[nodiscard]] double nonanticipativity_violation(std::span<const double> decision) const;

  private:
    ScenarioTree tree_;
    std::size_t state_dimension_{0};
    std::size_t control_dimension_{0};
    std::size_t local_auxiliary_dimension_{0};
    std::vector<ConsensusBlock> consensus_blocks_{};

    void validate_scenario(std::size_t scenario) const;
};

}  // namespace spacepdhcg::native
