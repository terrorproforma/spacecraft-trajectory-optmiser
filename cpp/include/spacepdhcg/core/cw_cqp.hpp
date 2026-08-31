#pragma once

#include "spacepdhcg/core/cqp.hpp"
#include "spacepdhcg/core/hcw.hpp"
#include "spacepdhcg/core/sparse_builder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spacepdhcg::core {

enum class CWThrustConstraint {
    box,
    second_order_cone,
};

struct CWRendezvousCQPConfig {
    std::size_t intervals{40};
    double step_seconds{20.0};
    double mean_motion{1.13e-3};
    double maximum_acceleration{5.0e-2};
    CWThrustConstraint thrust_constraint{CWThrustConstraint::box};
    std::array<double, hcw_state_dimension> state_weights{
        1.0e-4,
        1.0e-4,
        1.0e-4,
        1.0e-2,
        1.0e-2,
        1.0e-2,
    };
    std::array<double, hcw_control_dimension> control_weights{1.0, 1.0, 1.0};

    void validate() const {
        if (intervals < 2U || intervals > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
            throw std::invalid_argument("CW interval count is outside supported range");
        }
        if (!std::isfinite(step_seconds) || step_seconds <= 0.0 ||
            !std::isfinite(mean_motion) || mean_motion <= 0.0 ||
            !std::isfinite(maximum_acceleration) || maximum_acceleration <= 0.0) {
            throw std::invalid_argument("CW physical parameters must be finite and positive");
        }
        for (const auto value : state_weights) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument("CW state weights must be finite and positive");
            }
        }
        for (const auto value : control_weights) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument("CW control weights must be finite and positive");
            }
        }
    }
};

class CWRendezvousCQPLayout {
  public:
    explicit CWRendezvousCQPLayout(const CWRendezvousCQPConfig& config)
        : intervals_(config.intervals), thrust_constraint_(config.thrust_constraint) {}

    [[nodiscard]] std::size_t intervals() const noexcept { return intervals_; }
    [[nodiscard]] std::size_t state_variable_count() const noexcept {
        return (intervals_ + 1U) * hcw_state_dimension;
    }
    [[nodiscard]] std::size_t control_variable_count() const noexcept {
        return intervals_ * hcw_control_dimension;
    }
    [[nodiscard]] std::size_t variables() const noexcept {
        return state_variable_count() + control_variable_count();
    }
    [[nodiscard]] std::size_t scalar_rows() const noexcept {
        const std::size_t core = hcw_state_dimension * (intervals_ + 2U);
        return thrust_constraint_ == CWThrustConstraint::box
            ? core + control_variable_count()
            : core;
    }
    [[nodiscard]] std::size_t affine_rows() const noexcept {
        return thrust_constraint_ == CWThrustConstraint::second_order_cone
            ? intervals_ * (hcw_control_dimension + 1U)
            : 0U;
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> state_range(
        const std::size_t node
    ) const {
        if (node > intervals_) {
            throw std::out_of_range("CW state node is outside trajectory");
        }
        const auto begin = node * hcw_state_dimension;
        return {begin, begin + hcw_state_dimension};
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> control_range(
        const std::size_t interval
    ) const {
        if (interval >= intervals_) {
            throw std::out_of_range("CW control interval is outside trajectory");
        }
        const auto begin = state_variable_count() + interval * hcw_control_dimension;
        return {begin, begin + hcw_control_dimension};
    }

    [[nodiscard]] std::size_t initial_row_start() const noexcept { return 0U; }
    [[nodiscard]] std::size_t dynamics_row_start() const noexcept {
        return hcw_state_dimension;
    }
    [[nodiscard]] std::size_t terminal_row_start() const noexcept {
        return dynamics_row_start() + intervals_ * hcw_state_dimension;
    }
    [[nodiscard]] std::size_t control_row_start() const noexcept {
        return terminal_row_start() + hcw_state_dimension;
    }

  private:
    std::size_t intervals_{0};
    CWThrustConstraint thrust_constraint_{CWThrustConstraint::box};
};

struct CWRendezvousCQPDiagnostics {
    double initial_error{0.0};
    double terminal_error{0.0};
    double dynamics_defect{0.0};
    double control_violation{0.0};

    [[nodiscard]] double maximum_violation() const noexcept {
        return std::max({initial_error, terminal_error, dynamics_defect, control_violation});
    }
};

class CWRendezvousCQP {
  public:
    explicit CWRendezvousCQP(CWRendezvousCQPConfig config = {})
        : CWRendezvousCQP(std::move(config), build(config)) {}

    [[nodiscard]] const CWRendezvousCQPConfig& config() const noexcept { return config_; }
    [[nodiscard]] const CWRendezvousCQPLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] const CQPStructure& structure() const noexcept { return structure_; }
    [[nodiscard]] const HCWDiscretisation& dynamics() const noexcept { return dynamics_; }

    [[nodiscard]] CQPValues values(
        const HCWState& initial_state,
        const HCWState& target_state
    ) const {
        validate_state(initial_state);
        validate_state(target_state);
        CQPValues result;
        result.quadratic = quadratic_values_;
        result.scalar_constraint = scalar_values_;
        result.affine_cone = affine_values_;
        result.linear_objective.assign(layout_.variables(), 0.0);
        result.scalar_lower.assign(layout_.scalar_rows(), 0.0);
        result.scalar_upper.assign(layout_.scalar_rows(), 0.0);
        result.affine_offset.assign(layout_.affine_rows(), 0.0);
        result.variable_lower.assign(
            layout_.variables(),
            -std::numeric_limits<double>::infinity()
        );
        result.variable_upper.assign(
            layout_.variables(),
            std::numeric_limits<double>::infinity()
        );

        for (std::size_t node = 0; node <= layout_.intervals(); ++node) {
            const double fraction = static_cast<double>(node) /
                static_cast<double>(layout_.intervals());
            const auto range = layout_.state_range(node);
            for (std::size_t component = 0; component < hcw_state_dimension; ++component) {
                const double reference =
                    (1.0 - fraction) * initial_state[component] + fraction * target_state[component];
                result.linear_objective[range.first + component] =
                    -2.0 * config_.state_weights[component] * reference;
            }
        }

        for (std::size_t component = 0; component < hcw_state_dimension; ++component) {
            result.scalar_lower[layout_.initial_row_start() + component] = initial_state[component];
            result.scalar_upper[layout_.initial_row_start() + component] = initial_state[component];
            result.scalar_lower[layout_.terminal_row_start() + component] = target_state[component];
            result.scalar_upper[layout_.terminal_row_start() + component] = target_state[component];
        }

        if (config_.thrust_constraint == CWThrustConstraint::box) {
            const auto start = layout_.control_row_start();
            for (std::size_t index = 0; index < layout_.control_variable_count(); ++index) {
                result.scalar_lower[start + index] = -config_.maximum_acceleration;
                result.scalar_upper[start + index] = config_.maximum_acceleration;
            }
        } else {
            constexpr std::size_t slots = hcw_control_dimension + 1U;
            for (std::size_t interval = 0; interval < layout_.intervals(); ++interval) {
                result.affine_offset[interval * slots + hcw_control_dimension] =
                    config_.maximum_acceleration;
            }
        }
        validate_values(structure_, result);
        return result;
    }

    [[nodiscard]] std::vector<HCWState> rollout(
        const HCWState& initial_state,
        const std::vector<HCWControl>& controls
    ) const {
        validate_state(initial_state);
        if (controls.size() != layout_.intervals()) {
            throw std::invalid_argument("CW rollout requires one control per interval");
        }
        std::vector<HCWState> states(layout_.intervals() + 1U);
        states.front() = initial_state;
        for (std::size_t interval = 0; interval < layout_.intervals(); ++interval) {
            validate_control(controls[interval]);
            states[interval + 1U] = propagate_hcw(
                dynamics_,
                states[interval],
                controls[interval]
            );
        }
        return states;
    }

    [[nodiscard]] std::vector<double> encode(
        const std::vector<HCWState>& states,
        const std::vector<HCWControl>& controls
    ) const {
        if (states.size() != layout_.intervals() + 1U ||
            controls.size() != layout_.intervals()) {
            throw std::invalid_argument("CW decision dimensions do not match the fixed horizon");
        }
        std::vector<double> decision(layout_.variables(), 0.0);
        for (std::size_t node = 0; node < states.size(); ++node) {
            validate_state(states[node]);
            const auto range = layout_.state_range(node);
            std::copy(states[node].begin(), states[node].end(), decision.begin() +
                static_cast<std::ptrdiff_t>(range.first));
        }
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            validate_control(controls[interval]);
            const auto range = layout_.control_range(interval);
            std::copy(controls[interval].begin(), controls[interval].end(), decision.begin() +
                static_cast<std::ptrdiff_t>(range.first));
        }
        return decision;
    }

    [[nodiscard]] CWRendezvousCQPDiagnostics diagnostics(
        const std::vector<double>& decision,
        const HCWState& initial_state,
        const HCWState& target_state
    ) const {
        if (decision.size() != layout_.variables()) {
            throw std::invalid_argument("CW decision vector has an incompatible size");
        }
        CWRendezvousCQPDiagnostics result{};
        std::vector<HCWState> states(layout_.intervals() + 1U);
        std::vector<HCWControl> controls(layout_.intervals());
        for (std::size_t node = 0; node < states.size(); ++node) {
            const auto range = layout_.state_range(node);
            std::copy(
                decision.begin() + static_cast<std::ptrdiff_t>(range.first),
                decision.begin() + static_cast<std::ptrdiff_t>(range.second),
                states[node].begin()
            );
        }
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            const auto range = layout_.control_range(interval);
            std::copy(
                decision.begin() + static_cast<std::ptrdiff_t>(range.first),
                decision.begin() + static_cast<std::ptrdiff_t>(range.second),
                controls[interval].begin()
            );
        }
        for (std::size_t component = 0; component < hcw_state_dimension; ++component) {
            result.initial_error = std::max(
                result.initial_error,
                std::abs(states.front()[component] - initial_state[component])
            );
            result.terminal_error = std::max(
                result.terminal_error,
                std::abs(states.back()[component] - target_state[component])
            );
        }
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            const auto predicted = propagate_hcw(dynamics_, states[interval], controls[interval]);
            for (std::size_t component = 0; component < hcw_state_dimension; ++component) {
                result.dynamics_defect = std::max(
                    result.dynamics_defect,
                    std::abs(states[interval + 1U][component] - predicted[component])
                );
            }
            if (config_.thrust_constraint == CWThrustConstraint::box) {
                for (const auto value : controls[interval]) {
                    result.control_violation = std::max(
                        result.control_violation,
                        std::abs(value) - config_.maximum_acceleration
                    );
                }
            } else {
                const double norm = std::sqrt(
                    controls[interval][0] * controls[interval][0] +
                    controls[interval][1] * controls[interval][1] +
                    controls[interval][2] * controls[interval][2]
                );
                result.control_violation = std::max(
                    result.control_violation,
                    norm - config_.maximum_acceleration
                );
            }
        }
        result.control_violation = std::max(0.0, result.control_violation);
        return result;
    }

  private:
    struct BuildData {
        CQPStructure structure;
        std::vector<double> quadratic_values;
        std::vector<double> scalar_values;
        std::vector<double> affine_values;
        HCWDiscretisation dynamics;
    };

    CWRendezvousCQP(CWRendezvousCQPConfig config, BuildData build_data)
        : config_(std::move(config)),
          layout_(config_),
          structure_(std::move(build_data.structure)),
          quadratic_values_(std::move(build_data.quadratic_values)),
          scalar_values_(std::move(build_data.scalar_values)),
          affine_values_(std::move(build_data.affine_values)),
          dynamics_(build_data.dynamics) {
        config_.validate();
    }

    [[nodiscard]] static BuildData build(const CWRendezvousCQPConfig& config) {
        config.validate();
        const CWRendezvousCQPLayout layout{config};
        const auto dynamics = discretise_hcw(config.mean_motion, config.step_seconds);
        CscBuilder quadratic_builder{
            static_cast<Index>(layout.variables()),
            static_cast<Index>(layout.variables()),
        };
        for (std::size_t node = 0; node <= layout.intervals(); ++node) {
            const auto range = layout.state_range(node);
            for (std::size_t component = 0; component < hcw_state_dimension; ++component) {
                const auto index = static_cast<Index>(range.first + component);
                quadratic_builder.add(index, index, 2.0 * config.state_weights[component]);
            }
        }
        for (std::size_t interval = 0; interval < layout.intervals(); ++interval) {
            const auto range = layout.control_range(interval);
            for (std::size_t component = 0; component < hcw_control_dimension; ++component) {
                const auto index = static_cast<Index>(range.first + component);
                quadratic_builder.add(index, index, 2.0 * config.control_weights[component]);
            }
        }

        CscBuilder scalar_builder{
            static_cast<Index>(layout.scalar_rows()),
            static_cast<Index>(layout.variables()),
        };
        const auto initial_range = layout.state_range(0U);
        for (std::size_t component = 0; component < hcw_state_dimension; ++component) {
            scalar_builder.add(
                static_cast<Index>(layout.initial_row_start() + component),
                static_cast<Index>(initial_range.first + component),
                1.0
            );
        }
        for (std::size_t interval = 0; interval < layout.intervals(); ++interval) {
            const auto current = layout.state_range(interval);
            const auto next = layout.state_range(interval + 1U);
            const auto control = layout.control_range(interval);
            const auto row_start = layout.dynamics_row_start() + interval * hcw_state_dimension;
            for (std::size_t row = 0; row < hcw_state_dimension; ++row) {
                for (std::size_t column = 0; column < hcw_state_dimension; ++column) {
                    scalar_builder.add(
                        static_cast<Index>(row_start + row),
                        static_cast<Index>(current.first + column),
                        -dynamics.state[hcw_state_index(row, column)]
                    );
                }
                scalar_builder.add(
                    static_cast<Index>(row_start + row),
                    static_cast<Index>(next.first + row),
                    1.0
                );
                for (std::size_t column = 0; column < hcw_control_dimension; ++column) {
                    scalar_builder.add(
                        static_cast<Index>(row_start + row),
                        static_cast<Index>(control.first + column),
                        -dynamics.control[hcw_control_index(row, column)]
                    );
                }
            }
        }
        const auto terminal_range = layout.state_range(layout.intervals());
        for (std::size_t component = 0; component < hcw_state_dimension; ++component) {
            scalar_builder.add(
                static_cast<Index>(layout.terminal_row_start() + component),
                static_cast<Index>(terminal_range.first + component),
                1.0
            );
        }
        if (config.thrust_constraint == CWThrustConstraint::box) {
            for (std::size_t interval = 0; interval < layout.intervals(); ++interval) {
                const auto control = layout.control_range(interval);
                for (std::size_t component = 0; component < hcw_control_dimension; ++component) {
                    scalar_builder.add(
                        static_cast<Index>(
                            layout.control_row_start() + interval * hcw_control_dimension + component
                        ),
                        static_cast<Index>(control.first + component),
                        1.0
                    );
                }
            }
        }

        auto quadratic = quadratic_builder.build();
        auto scalar = scalar_builder.build();
        std::optional<CscStructure> affine_structure;
        std::vector<double> affine_values;
        std::vector<ConeBlock> cones;
        if (config.thrust_constraint == CWThrustConstraint::second_order_cone) {
            CscBuilder affine_builder{
                static_cast<Index>(layout.affine_rows()),
                static_cast<Index>(layout.variables()),
            };
            constexpr std::size_t slots = hcw_control_dimension + 1U;
            for (std::size_t interval = 0; interval < layout.intervals(); ++interval) {
                const auto control = layout.control_range(interval);
                for (std::size_t component = 0; component < hcw_control_dimension; ++component) {
                    affine_builder.add(
                        static_cast<Index>(interval * slots + component),
                        static_cast<Index>(control.first + component),
                        1.0
                    );
                }
                cones.emplace_back(
                    ConeKind::second_order,
                    static_cast<Index>(interval * slots),
                    static_cast<Index>(hcw_control_dimension - 1U)
                );
            }
            auto affine = affine_builder.build();
            affine_structure = std::move(affine.structure);
            affine_values = std::move(affine.values);
        }
        return BuildData{
            CQPStructure{
                std::move(quadratic.structure),
                std::move(scalar.structure),
                std::move(affine_structure),
                std::move(cones),
            },
            std::move(quadratic.values),
            std::move(scalar.values),
            std::move(affine_values),
            dynamics,
        };
    }

    static void validate_state(const HCWState& state) {
        for (const auto value : state) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("CW state must be finite");
            }
        }
    }

    static void validate_control(const HCWControl& control) {
        for (const auto value : control) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("CW control must be finite");
            }
        }
    }

    CWRendezvousCQPConfig config_;
    CWRendezvousCQPLayout layout_;
    CQPStructure structure_;
    std::vector<double> quadratic_values_;
    std::vector<double> scalar_values_;
    std::vector<double> affine_values_;
    HCWDiscretisation dynamics_;
};

}  // namespace spacepdhcg::core
