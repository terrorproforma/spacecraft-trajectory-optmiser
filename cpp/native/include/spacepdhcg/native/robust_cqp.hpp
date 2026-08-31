#pragma once

#include "spacepdhcg/native/cqp.hpp"
#include "spacepdhcg/native/scenario_tree.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace spacepdhcg::native {

enum class ScenarioRiskMode {
    expected_value,
    worst_case,
    conditional_value_at_risk,
};

struct RobustCqpLayout {
    std::size_t base_variables{0};
    std::size_t risk_offset{0};
    std::size_t risk_variables{0};
    std::size_t scenario_count{0};
    ScenarioRiskMode mode{ScenarioRiskMode::expected_value};

    [[nodiscard]] std::size_t variables() const noexcept {
        return base_variables + risk_variables;
    }

    [[nodiscard]] std::size_t threshold_index() const;
    [[nodiscard]] std::size_t excess_index(std::size_t scenario) const;
};

struct RobustPrimal {
    std::vector<std::vector<double>> local{};
    std::vector<std::vector<double>> consensus{};
    double threshold{0.0};
    std::vector<double> excess{};
};

class ScenarioCqpAssembler {
  public:
    ScenarioCqpAssembler(
        ScenarioTree tree,
        OwnedCqp local_prototype,
        std::size_t state_dimension,
        std::size_t control_dimension,
        std::size_t local_auxiliary_dimension = 0
    );

    [[nodiscard]] const ScenarioTree& tree() const noexcept { return tree_; }
    [[nodiscard]] const BlockArrowLayout& block_arrow() const noexcept { return block_arrow_; }
    [[nodiscard]] const OwnedCqp& local_prototype() const noexcept { return local_prototype_; }

    [[nodiscard]] OwnedCqp assemble(
        std::span<const OwnedCqp> local_problems,
        ScenarioRiskMode mode = ScenarioRiskMode::expected_value,
        double cvar_alpha = 0.95
    ) const;

    [[nodiscard]] RobustCqpLayout layout(ScenarioRiskMode mode) const;

    [[nodiscard]] RobustPrimal decode(
        std::span<const double> primal,
        ScenarioRiskMode mode
    ) const;

    [[nodiscard]] std::vector<double> local_objectives(
        const RobustPrimal& primal,
        std::span<const OwnedCqp> local_problems
    ) const;

    [[nodiscard]] double aggregate_risk(
        std::span<const double> scenario_losses,
        ScenarioRiskMode mode,
        double cvar_alpha = 0.95
    ) const;

  private:
    ScenarioTree tree_;
    OwnedCqp local_prototype_{};
    BlockArrowLayout block_arrow_;

    void validate_local_problems(std::span<const OwnedCqp> local_problems) const;
    [[nodiscard]] OwnedCqp assemble_expected(
        std::span<const OwnedCqp> local_problems
    ) const;
    [[nodiscard]] OwnedCqp assemble_epigraph_risk(
        std::span<const OwnedCqp> local_problems,
        ScenarioRiskMode mode,
        double cvar_alpha
    ) const;
};

}  // namespace spacepdhcg::native
