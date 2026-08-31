#pragma once

#include "spacepdhcg/core/cqp.hpp"
#include "spacepdhcg/core/csc_operator.hpp"
#include "spacepdhcg/core/powered_descent_6dof.hpp"
#include "spacepdhcg/core/sparse_builder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace spacepdhcg::core {

struct PoweredDescent6DOFCQPConfig {
    std::size_t intervals{6};
    double step_seconds{0.5};
    double trust_radius{1.0};
    double finite_difference_epsilon{1.0e-6};
    double virtual_l1_weight{1.0e5};
    double virtual_quadratic_weight{1.0e-8};
    double virtual_epigraph_regularisation{1.0e-10};
    double fuel_weight{1.0e-3};
    std::array<double, powered_descent_6dof_state_dimension> state_tracking_weights{
        1.0e-4,
        1.0e-4,
        1.0e-4,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-8,
    };
    std::array<double, powered_descent_6dof_control_dimension> control_tracking_weights{
        1.0e-8,
        1.0e-8,
        1.0e-8,
        1.0e-6,
        1.0e-6,
        1.0e-6,
        1.0e-8,
    };
    std::array<double, powered_descent_6dof_state_dimension> state_trust_scales{
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0e-3,
    };
    std::array<double, powered_descent_6dof_control_dimension> control_trust_scales{
        1.0 / 15'000.0,
        1.0 / 15'000.0,
        1.0 / 15'000.0,
        1.0 / 2'000.0,
        1.0 / 2'000.0,
        1.0 / 2'000.0,
        1.0 / 15'000.0,
    };

    void validate() const {
        if (intervals < 2U ||
            intervals > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
            throw std::invalid_argument("6-DoF CQP interval count is outside supported range");
        }
        const double positive[]{step_seconds, trust_radius, finite_difference_epsilon};
        for (const auto value : positive) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument("6-DoF CQP step, radius, and epsilon must be positive");
            }
        }
        const double nonnegative[]{
            virtual_l1_weight,
            virtual_quadratic_weight,
            virtual_epigraph_regularisation,
            fuel_weight,
        };
        for (const auto value : nonnegative) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument("6-DoF CQP penalty weights must be non-negative");
            }
        }
        validate_positive(state_tracking_weights, "6-DoF state tracking weights");
        validate_positive(control_tracking_weights, "6-DoF control tracking weights");
        validate_positive(state_trust_scales, "6-DoF state trust scales");
        validate_positive(control_trust_scales, "6-DoF control trust scales");
    }

  private:
    template <std::size_t Size>
    static void validate_positive(
        const std::array<double, Size>& values,
        const char* name
    ) {
        for (const auto value : values) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument(std::string{name} + " must be finite and positive");
            }
        }
    }
};

class PoweredDescent6DOFCQPLayout {
  public:
    explicit PoweredDescent6DOFCQPLayout(const std::size_t intervals)
        : intervals_(intervals) {
        if (intervals_ < 2U) {
            throw std::invalid_argument("6-DoF layout requires at least two intervals");
        }
    }

    [[nodiscard]] std::size_t intervals() const noexcept { return intervals_; }
    [[nodiscard]] std::size_t state_count() const noexcept {
        return (intervals_ + 1U) * powered_descent_6dof_state_dimension;
    }
    [[nodiscard]] std::size_t control_count() const noexcept {
        return intervals_ * powered_descent_6dof_control_dimension;
    }
    [[nodiscard]] std::size_t virtual_count() const noexcept {
        return intervals_ * powered_descent_6dof_state_dimension;
    }
    [[nodiscard]] std::size_t control_offset() const noexcept { return state_count(); }
    [[nodiscard]] std::size_t virtual_offset() const noexcept {
        return control_offset() + control_count();
    }
    [[nodiscard]] std::size_t epigraph_offset() const noexcept {
        return virtual_offset() + virtual_count();
    }
    [[nodiscard]] std::size_t variables() const noexcept {
        return epigraph_offset() + virtual_count();
    }

    [[nodiscard]] std::size_t initial_row_start() const noexcept { return 0U; }
    [[nodiscard]] std::size_t dynamics_row_start() const noexcept {
        return powered_descent_6dof_state_dimension;
    }
    [[nodiscard]] std::size_t terminal_row_start() const noexcept {
        return dynamics_row_start() + intervals_ * powered_descent_6dof_state_dimension;
    }
    [[nodiscard]] std::size_t quaternion_row_start() const noexcept {
        return terminal_row_start() + 13U;
    }
    [[nodiscard]] std::size_t virtual_epigraph_row_start() const noexcept {
        return quaternion_row_start() + intervals_ + 1U;
    }
    [[nodiscard]] std::size_t scalar_rows() const noexcept {
        return virtual_epigraph_row_start() + 2U * virtual_count();
    }

    [[nodiscard]] std::size_t thrust_cone_row_start() const noexcept { return 0U; }
    [[nodiscard]] std::size_t angular_rate_cone_row_start() const noexcept {
        return 4U * intervals_;
    }
    [[nodiscard]] std::size_t stage_trust_cone_row_start() const noexcept {
        return angular_rate_cone_row_start() + 4U * (intervals_ + 1U);
    }
    [[nodiscard]] std::size_t terminal_trust_cone_row_start() const noexcept {
        return stage_trust_cone_row_start() + 22U * intervals_;
    }
    [[nodiscard]] std::size_t affine_rows() const noexcept {
        return terminal_trust_cone_row_start() + 15U;
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> state_range(
        const std::size_t node
    ) const {
        if (node > intervals_) {
            throw std::out_of_range("6-DoF state node is outside trajectory");
        }
        const auto begin = node * powered_descent_6dof_state_dimension;
        return {begin, begin + powered_descent_6dof_state_dimension};
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> control_range(
        const std::size_t interval
    ) const {
        if (interval >= intervals_) {
            throw std::out_of_range("6-DoF control interval is outside trajectory");
        }
        const auto begin = control_offset() +
            interval * powered_descent_6dof_control_dimension;
        return {begin, begin + powered_descent_6dof_control_dimension};
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> virtual_range(
        const std::size_t interval
    ) const {
        if (interval >= intervals_) {
            throw std::out_of_range("6-DoF virtual interval is outside trajectory");
        }
        const auto begin = virtual_offset() +
            interval * powered_descent_6dof_state_dimension;
        return {begin, begin + powered_descent_6dof_state_dimension};
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> epigraph_range(
        const std::size_t interval
    ) const {
        if (interval >= intervals_) {
            throw std::out_of_range("6-DoF epigraph interval is outside trajectory");
        }
        const auto begin = epigraph_offset() +
            interval * powered_descent_6dof_state_dimension;
        return {begin, begin + powered_descent_6dof_state_dimension};
    }

  private:
    std::size_t intervals_{0};
};

struct PoweredDescent6DOFCQPDecision {
    std::vector<PoweredDescent6DOFState> states;
    std::vector<PoweredDescent6DOFControl> controls;
    std::vector<PoweredDescent6DOFState> virtual_controls;
    std::vector<PoweredDescent6DOFState> virtual_epigraphs;
};

struct PoweredDescent6DOFCQPDiagnostics {
    double scalar_violation{0.0};
    double variable_violation{0.0};
    double cone_violation{0.0};
    double linearised_dynamics_defect{0.0};
    double nonlinear_euler_defect{0.0};
    double terminal_error{0.0};
    double quaternion_tangent_error{0.0};
    double virtual_control{0.0};

    [[nodiscard]] double convex_violation() const noexcept {
        return std::max({scalar_violation, variable_violation, cone_violation});
    }
};

class PoweredDescent6DOFCQP {
  public:
    explicit PoweredDescent6DOFCQP(
        PoweredDescent6DOF model = PoweredDescent6DOF{PoweredDescent6DOFConfig{}},
        PoweredDescent6DOFCQPConfig config = {}
    )
        : PoweredDescent6DOFCQP(std::move(model), config, build(config)) {}

    [[nodiscard]] const PoweredDescent6DOF& model() const noexcept { return model_; }
    [[nodiscard]] const PoweredDescent6DOFCQPConfig& config() const noexcept {
        return config_;
    }
    [[nodiscard]] const PoweredDescent6DOFCQPLayout& layout() const noexcept {
        return layout_;
    }
    [[nodiscard]] const CQPStructure& structure() const noexcept { return structure_; }

    [[nodiscard]] CQPValues values(
        std::span<const PoweredDescent6DOFState> reference_states,
        std::span<const PoweredDescent6DOFControl> reference_controls,
        const PoweredDescent6DOFState& initial_state,
        const PoweredDescent6DOFState& target_state,
        const double trust_radius = std::numeric_limits<double>::quiet_NaN()
    ) const {
        validate_reference(reference_states, reference_controls);
        validate_state(initial_state);
        validate_state(target_state);
        const double radius = std::isnan(trust_radius) ? config_.trust_radius : trust_radius;
        if (!std::isfinite(radius) || radius <= 0.0) {
            throw std::invalid_argument("6-DoF trust radius must be finite and positive");
        }

        CQPValues result;
        result.quadratic = quadratic_values_;
        result.scalar_constraint.assign(structure_.scalar_constraint().nonzeros(), 0.0);
        result.affine_cone.assign(structure_.affine_cone()->nonzeros(), 0.0);
        result.linear_objective.assign(layout_.variables(), 0.0);
        result.scalar_lower.assign(
            layout_.scalar_rows(),
            -std::numeric_limits<double>::infinity()
        );
        result.scalar_upper.assign(
            layout_.scalar_rows(),
            std::numeric_limits<double>::infinity()
        );
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
            const auto state = layout_.state_range(node);
            for (std::size_t component = 0;
                 component < powered_descent_6dof_state_dimension;
                 ++component) {
                result.linear_objective[state.first + component] =
                    -config_.state_tracking_weights[component] *
                    reference_states[node][component];
            }
        }
        for (std::size_t interval = 0; interval < layout_.intervals(); ++interval) {
            const auto control = layout_.control_range(interval);
            for (std::size_t component = 0;
                 component < powered_descent_6dof_control_dimension;
                 ++component) {
                result.linear_objective[control.first + component] =
                    -config_.control_tracking_weights[component] *
                    reference_controls[interval][component];
            }
            result.linear_objective[control.first + 6U] +=
                config_.fuel_weight * config_.step_seconds;
        }
        std::fill(
            result.linear_objective.begin() +
                static_cast<std::ptrdiff_t>(layout_.epigraph_offset()),
            result.linear_objective.end(),
            config_.virtual_l1_weight
        );

        for (std::size_t component = 0;
             component < powered_descent_6dof_state_dimension;
             ++component) {
            const auto row = layout_.initial_row_start() + component;
            scalar_index_.set(
                result.scalar_constraint,
                static_cast<Index>(row),
                static_cast<Index>(component),
                1.0
            );
            result.scalar_lower[row] = initial_state[component];
            result.scalar_upper[row] = initial_state[component];
        }

        for (std::size_t interval = 0; interval < layout_.intervals(); ++interval) {
            const auto jacobians = model_.numerical_jacobians(
                reference_states[interval],
                reference_controls[interval],
                config_.finite_difference_epsilon
            );
            const auto dynamics = model_.dynamics(
                reference_states[interval],
                reference_controls[interval]
            );
            PoweredDescent6DOFState offset = dynamics;
            for (std::size_t row_component = 0;
                 row_component < powered_descent_6dof_state_dimension;
                 ++row_component) {
                for (std::size_t column_component = 0;
                     column_component < powered_descent_6dof_state_dimension;
                     ++column_component) {
                    offset[row_component] -= jacobians.state[
                        powered_descent_6dof_state_index(row_component, column_component)
                    ] * reference_states[interval][column_component];
                }
                for (std::size_t column_component = 0;
                     column_component < powered_descent_6dof_control_dimension;
                     ++column_component) {
                    offset[row_component] -= jacobians.control[
                        powered_descent_6dof_control_index(row_component, column_component)
                    ] * reference_controls[interval][column_component];
                }
            }

            const auto state = layout_.state_range(interval);
            const auto next_state = layout_.state_range(interval + 1U);
            const auto control = layout_.control_range(interval);
            const auto virtual_control = layout_.virtual_range(interval);
            const auto row_start = layout_.dynamics_row_start() +
                interval * powered_descent_6dof_state_dimension;
            for (std::size_t row_component = 0;
                 row_component < powered_descent_6dof_state_dimension;
                 ++row_component) {
                const auto row = row_start + row_component;
                for (std::size_t column_component = 0;
                     column_component < powered_descent_6dof_state_dimension;
                     ++column_component) {
                    const double continuous = jacobians.state[
                        powered_descent_6dof_state_index(row_component, column_component)
                    ];
                    const double discrete =
                        (row_component == column_component ? 1.0 : 0.0) +
                        config_.step_seconds * continuous;
                    scalar_index_.set(
                        result.scalar_constraint,
                        static_cast<Index>(row),
                        static_cast<Index>(state.first + column_component),
                        -discrete
                    );
                }
                scalar_index_.set(
                    result.scalar_constraint,
                    static_cast<Index>(row),
                    static_cast<Index>(next_state.first + row_component),
                    1.0
                );
                for (std::size_t column_component = 0;
                     column_component < powered_descent_6dof_control_dimension;
                     ++column_component) {
                    scalar_index_.set(
                        result.scalar_constraint,
                        static_cast<Index>(row),
                        static_cast<Index>(control.first + column_component),
                        -config_.step_seconds * jacobians.control[
                            powered_descent_6dof_control_index(row_component, column_component)
                        ]
                    );
                }
                scalar_index_.set(
                    result.scalar_constraint,
                    static_cast<Index>(row),
                    static_cast<Index>(virtual_control.first + row_component),
                    -1.0
                );
                result.scalar_lower[row] = config_.step_seconds * offset[row_component];
                result.scalar_upper[row] = result.scalar_lower[row];
            }
        }

        const auto terminal = layout_.state_range(layout_.intervals());
        for (std::size_t component = 0; component < 13U; ++component) {
            const auto row = layout_.terminal_row_start() + component;
            scalar_index_.set(
                result.scalar_constraint,
                static_cast<Index>(row),
                static_cast<Index>(terminal.first + component),
                1.0
            );
            result.scalar_lower[row] = target_state[component];
            result.scalar_upper[row] = target_state[component];
        }

        for (std::size_t node = 0; node <= layout_.intervals(); ++node) {
            const auto state = layout_.state_range(node);
            const Quaternion quaternion = normalise_quaternion(Quaternion{
                reference_states[node][6],
                reference_states[node][7],
                reference_states[node][8],
                reference_states[node][9],
            });
            const auto row = layout_.quaternion_row_start() + node;
            for (std::size_t component = 0; component < 4U; ++component) {
                scalar_index_.set(
                    result.scalar_constraint,
                    static_cast<Index>(row),
                    static_cast<Index>(state.first + 6U + component),
                    quaternion[component]
                );
            }
            result.scalar_lower[row] = 1.0;
            result.scalar_upper[row] = 1.0;
        }

        for (std::size_t flat = 0; flat < layout_.virtual_count(); ++flat) {
            const auto positive_row = layout_.virtual_epigraph_row_start() + 2U * flat;
            const auto negative_row = positive_row + 1U;
            const auto virtual_column = layout_.virtual_offset() + flat;
            const auto epigraph_column = layout_.epigraph_offset() + flat;
            scalar_index_.set(
                result.scalar_constraint,
                static_cast<Index>(positive_row),
                static_cast<Index>(virtual_column),
                1.0
            );
            scalar_index_.set(
                result.scalar_constraint,
                static_cast<Index>(positive_row),
                static_cast<Index>(epigraph_column),
                -1.0
            );
            scalar_index_.set(
                result.scalar_constraint,
                static_cast<Index>(negative_row),
                static_cast<Index>(virtual_column),
                -1.0
            );
            scalar_index_.set(
                result.scalar_constraint,
                static_cast<Index>(negative_row),
                static_cast<Index>(epigraph_column),
                -1.0
            );
            result.scalar_upper[positive_row] = 0.0;
            result.scalar_upper[negative_row] = 0.0;
        }

        for (std::size_t interval = 0; interval < layout_.intervals(); ++interval) {
            const auto control = layout_.control_range(interval);
            const auto start = layout_.thrust_cone_row_start() + 4U * interval;
            for (std::size_t component = 0; component < 3U; ++component) {
                affine_index_.set(
                    result.affine_cone,
                    static_cast<Index>(start + component),
                    static_cast<Index>(control.first + component),
                    1.0
                );
            }
            affine_index_.set(
                result.affine_cone,
                static_cast<Index>(start + 3U),
                static_cast<Index>(control.first + 6U),
                1.0
            );
        }

        for (std::size_t node = 0; node <= layout_.intervals(); ++node) {
            const auto state = layout_.state_range(node);
            const auto start = layout_.angular_rate_cone_row_start() + 4U * node;
            for (std::size_t component = 0; component < 3U; ++component) {
                affine_index_.set(
                    result.affine_cone,
                    static_cast<Index>(start + component),
                    static_cast<Index>(state.first + 10U + component),
                    1.0
                );
            }
            result.affine_offset[start + 3U] = model_.config().maximum_angular_rate;
        }

        for (std::size_t interval = 0; interval < layout_.intervals(); ++interval) {
            const auto state = layout_.state_range(interval);
            const auto control = layout_.control_range(interval);
            const auto start = layout_.stage_trust_cone_row_start() + 22U * interval;
            for (std::size_t component = 0;
                 component < powered_descent_6dof_state_dimension;
                 ++component) {
                const double scale_value = config_.state_trust_scales[component];
                affine_index_.set(
                    result.affine_cone,
                    static_cast<Index>(start + component),
                    static_cast<Index>(state.first + component),
                    scale_value
                );
                result.affine_offset[start + component] =
                    -scale_value * reference_states[interval][component];
            }
            for (std::size_t component = 0;
                 component < powered_descent_6dof_control_dimension;
                 ++component) {
                const auto row = start + powered_descent_6dof_state_dimension + component;
                const double scale_value = config_.control_trust_scales[component];
                affine_index_.set(
                    result.affine_cone,
                    static_cast<Index>(row),
                    static_cast<Index>(control.first + component),
                    scale_value
                );
                result.affine_offset[row] =
                    -scale_value * reference_controls[interval][component];
            }
            result.affine_offset[start + 21U] = radius;
        }

        const auto terminal_trust_start = layout_.terminal_trust_cone_row_start();
        for (std::size_t component = 0;
             component < powered_descent_6dof_state_dimension;
             ++component) {
            const double scale_value = config_.state_trust_scales[component];
            affine_index_.set(
                result.affine_cone,
                static_cast<Index>(terminal_trust_start + component),
                static_cast<Index>(terminal.first + component),
                scale_value
            );
            result.affine_offset[terminal_trust_start + component] =
                -scale_value * reference_states.back()[component];
        }
        result.affine_offset[terminal_trust_start + 14U] = radius;

        for (std::size_t node = 0; node <= layout_.intervals(); ++node) {
            const auto state = layout_.state_range(node);
            result.variable_lower[state.first + 2U] = 0.0;
            result.variable_lower[state.first + 13U] = model_.config().minimum_mass;
            for (std::size_t component = 0; component < 4U; ++component) {
                result.variable_lower[state.first + 6U + component] = -1.0;
                result.variable_upper[state.first + 6U + component] = 1.0;
            }
        }
        for (std::size_t interval = 0; interval < layout_.intervals(); ++interval) {
            const auto control = layout_.control_range(interval);
            for (std::size_t axis = 0; axis < 3U; ++axis) {
                result.variable_lower[control.first + 3U + axis] =
                    -model_.config().maximum_torque[axis];
                result.variable_upper[control.first + 3U + axis] =
                    model_.config().maximum_torque[axis];
            }
            result.variable_lower[control.first + 6U] = model_.config().minimum_sigma;
            result.variable_upper[control.first + 6U] = model_.config().maximum_thrust;
        }
        std::fill(
            result.variable_lower.begin() +
                static_cast<std::ptrdiff_t>(layout_.epigraph_offset()),
            result.variable_lower.end(),
            0.0
        );

        validate_values(structure_, result);
        return result;
    }

    [[nodiscard]] std::vector<double> reference_decision(
        std::span<const PoweredDescent6DOFState> states,
        std::span<const PoweredDescent6DOFControl> controls
    ) const {
        validate_reference(states, controls);
        std::vector<double> decision(layout_.variables(), 0.0);
        for (std::size_t node = 0; node < states.size(); ++node) {
            copy_to_decision(states[node], layout_.state_range(node), decision);
        }
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            copy_to_decision(controls[interval], layout_.control_range(interval), decision);
        }
        return decision;
    }

    [[nodiscard]] PoweredDescent6DOFCQPDecision decode(
        std::span<const double> decision
    ) const {
        if (decision.size() != layout_.variables()) {
            throw std::invalid_argument("6-DoF decision has an incompatible size");
        }
        PoweredDescent6DOFCQPDecision result;
        result.states.resize(layout_.intervals() + 1U);
        result.controls.resize(layout_.intervals());
        result.virtual_controls.resize(layout_.intervals());
        result.virtual_epigraphs.resize(layout_.intervals());
        for (std::size_t node = 0; node < result.states.size(); ++node) {
            copy_from_decision(decision, layout_.state_range(node), result.states[node]);
        }
        for (std::size_t interval = 0; interval < layout_.intervals(); ++interval) {
            copy_from_decision(decision, layout_.control_range(interval), result.controls[interval]);
            copy_from_decision(
                decision,
                layout_.virtual_range(interval),
                result.virtual_controls[interval]
            );
            copy_from_decision(
                decision,
                layout_.epigraph_range(interval),
                result.virtual_epigraphs[interval]
            );
        }
        return result;
    }

    [[nodiscard]] PoweredDescent6DOFCQPDiagnostics diagnostics(
        std::span<const double> decision,
        const CQPValues& values
    ) const {
        validate_values(structure_, values);
        const auto decoded = decode(decision);
        CscOperator scalar_operator{structure_.scalar_constraint(), values.scalar_constraint};
        const auto scalar_activity = scalar_operator.multiply(decision);
        PoweredDescent6DOFCQPDiagnostics result{};
        for (std::size_t row = 0; row < scalar_activity.size(); ++row) {
            result.scalar_violation = std::max(
                result.scalar_violation,
                std::max({
                    values.scalar_lower[row] - scalar_activity[row],
                    scalar_activity[row] - values.scalar_upper[row],
                    0.0,
                })
            );
        }
        for (std::size_t index = 0; index < decision.size(); ++index) {
            result.variable_violation = std::max(
                result.variable_violation,
                std::max({
                    values.variable_lower[index] - decision[index],
                    decision[index] - values.variable_upper[index],
                    0.0,
                })
            );
        }
        CscOperator affine_operator{*structure_.affine_cone(), values.affine_cone};
        auto affine_activity = affine_operator.multiply(decision);
        for (std::size_t row = 0; row < affine_activity.size(); ++row) {
            affine_activity[row] += values.affine_offset[row];
        }
        for (const auto& cone : structure_.affine_cones()) {
            double norm_squared = 0.0;
            for (Index local = 0; local < cone.slot_count() - 1; ++local) {
                const double value = affine_activity[
                    static_cast<std::size_t>(cone.start + local)
                ];
                norm_squared += value * value;
            }
            result.cone_violation = std::max(
                result.cone_violation,
                std::sqrt(norm_squared) - affine_activity[
                    static_cast<std::size_t>(cone.stop() - 1)
                ]
            );
        }
        result.cone_violation = std::max(0.0, result.cone_violation);

        for (std::size_t interval = 0; interval < layout_.intervals(); ++interval) {
            auto predicted = decoded.states[interval];
            const auto derivative = model_.dynamics(
                decoded.states[interval],
                decoded.controls[interval]
            );
            for (std::size_t component = 0;
                 component < powered_descent_6dof_state_dimension;
                 ++component) {
                predicted[component] += config_.step_seconds * derivative[component];
                result.nonlinear_euler_defect = std::max(
                    result.nonlinear_euler_defect,
                    std::abs(decoded.states[interval + 1U][component] - predicted[component])
                );
                result.virtual_control = std::max(
                    result.virtual_control,
                    std::abs(decoded.virtual_controls[interval][component])
                );
            }
        }
        const auto dynamics_start = layout_.dynamics_row_start();
        const auto dynamics_stop = dynamics_start +
            layout_.intervals() * powered_descent_6dof_state_dimension;
        for (std::size_t row = dynamics_start; row < dynamics_stop; ++row) {
            result.linearised_dynamics_defect = std::max(
                result.linearised_dynamics_defect,
                std::abs(scalar_activity[row] - values.scalar_lower[row])
            );
        }
        for (std::size_t component = 0; component < 13U; ++component) {
            const auto row = layout_.terminal_row_start() + component;
            result.terminal_error = std::max(
                result.terminal_error,
                std::abs(scalar_activity[row] - values.scalar_lower[row])
            );
        }
        for (std::size_t node = 0; node <= layout_.intervals(); ++node) {
            const auto row = layout_.quaternion_row_start() + node;
            result.quaternion_tangent_error = std::max(
                result.quaternion_tangent_error,
                std::abs(scalar_activity[row] - 1.0)
            );
        }
        return result;
    }

  private:
    class ValueIndex {
      public:
        explicit ValueIndex(const CscStructure& structure) {
            const auto offsets = structure.offsets();
            const auto indices = structure.indices();
            for (Index column = 0; column < structure.columns(); ++column) {
                const auto begin = static_cast<std::size_t>(
                    offsets[static_cast<std::size_t>(column)]
                );
                const auto end = static_cast<std::size_t>(
                    offsets[static_cast<std::size_t>(column) + 1U]
                );
                for (std::size_t position = begin; position < end; ++position) {
                    positions_.emplace(key(indices[position], column), position);
                }
            }
        }

        void set(
            std::vector<double>& values,
            const Index row,
            const Index column,
            const double value
        ) const {
            const auto found = positions_.find(key(row, column));
            if (found == positions_.end()) {
                throw std::logic_error("6-DoF update requested an absent fixed-pattern entry");
            }
            values[found->second] = value;
        }

      private:
        [[nodiscard]] static std::uint64_t key(
            const Index row,
            const Index column
        ) noexcept {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(column)) << 32U) |
                static_cast<std::uint32_t>(row);
        }

        std::unordered_map<std::uint64_t, std::size_t> positions_;
    };

    struct BuildData {
        CQPStructure structure;
        std::vector<double> quadratic_values;
    };

    PoweredDescent6DOFCQP(
        PoweredDescent6DOF model,
        PoweredDescent6DOFCQPConfig config,
        BuildData build_data
    )
        : model_(std::move(model)),
          config_(std::move(config)),
          layout_(config_.intervals),
          structure_(std::move(build_data.structure)),
          quadratic_values_(std::move(build_data.quadratic_values)),
          scalar_index_(structure_.scalar_constraint()),
          affine_index_(*structure_.affine_cone()) {
        config_.validate();
    }

    [[nodiscard]] static BuildData build(const PoweredDescent6DOFCQPConfig& config) {
        config.validate();
        const PoweredDescent6DOFCQPLayout layout{config.intervals};
        CscBuilder quadratic_builder{
            static_cast<Index>(layout.variables()),
            static_cast<Index>(layout.variables()),
        };
        for (std::size_t node = 0; node <= layout.intervals(); ++node) {
            const auto state = layout.state_range(node);
            for (std::size_t component = 0;
                 component < powered_descent_6dof_state_dimension;
                 ++component) {
                quadratic_builder.add(
                    static_cast<Index>(state.first + component),
                    static_cast<Index>(state.first + component),
                    config.state_tracking_weights[component]
                );
            }
        }
        for (std::size_t interval = 0; interval < layout.intervals(); ++interval) {
            const auto control = layout.control_range(interval);
            const auto virtual_control = layout.virtual_range(interval);
            const auto epigraph = layout.epigraph_range(interval);
            for (std::size_t component = 0;
                 component < powered_descent_6dof_control_dimension;
                 ++component) {
                quadratic_builder.add(
                    static_cast<Index>(control.first + component),
                    static_cast<Index>(control.first + component),
                    config.control_tracking_weights[component]
                );
            }
            for (std::size_t component = 0;
                 component < powered_descent_6dof_state_dimension;
                 ++component) {
                quadratic_builder.add(
                    static_cast<Index>(virtual_control.first + component),
                    static_cast<Index>(virtual_control.first + component),
                    config.virtual_quadratic_weight
                );
                quadratic_builder.add(
                    static_cast<Index>(epigraph.first + component),
                    static_cast<Index>(epigraph.first + component),
                    config.virtual_epigraph_regularisation
                );
            }
        }

        CscBuilder scalar_builder{
            static_cast<Index>(layout.scalar_rows()),
            static_cast<Index>(layout.variables()),
        };
        for (std::size_t component = 0;
             component < powered_descent_6dof_state_dimension;
             ++component) {
            scalar_builder.add(
                static_cast<Index>(layout.initial_row_start() + component),
                static_cast<Index>(component)
            );
        }
        for (std::size_t interval = 0; interval < layout.intervals(); ++interval) {
            const auto state = layout.state_range(interval);
            const auto next_state = layout.state_range(interval + 1U);
            const auto control = layout.control_range(interval);
            const auto virtual_control = layout.virtual_range(interval);
            const auto row_start = layout.dynamics_row_start() +
                interval * powered_descent_6dof_state_dimension;
            for (std::size_t row_component = 0;
                 row_component < powered_descent_6dof_state_dimension;
                 ++row_component) {
                const auto row = row_start + row_component;
                for (std::size_t column_component = 0;
                     column_component < powered_descent_6dof_state_dimension;
                     ++column_component) {
                    scalar_builder.add(
                        static_cast<Index>(row),
                        static_cast<Index>(state.first + column_component)
                    );
                }
                scalar_builder.add(
                    static_cast<Index>(row),
                    static_cast<Index>(next_state.first + row_component)
                );
                for (std::size_t column_component = 0;
                     column_component < powered_descent_6dof_control_dimension;
                     ++column_component) {
                    scalar_builder.add(
                        static_cast<Index>(row),
                        static_cast<Index>(control.first + column_component)
                    );
                }
                scalar_builder.add(
                    static_cast<Index>(row),
                    static_cast<Index>(virtual_control.first + row_component)
                );
            }
        }
        const auto terminal = layout.state_range(layout.intervals());
        for (std::size_t component = 0; component < 13U; ++component) {
            scalar_builder.add(
                static_cast<Index>(layout.terminal_row_start() + component),
                static_cast<Index>(terminal.first + component)
            );
        }
        for (std::size_t node = 0; node <= layout.intervals(); ++node) {
            const auto state = layout.state_range(node);
            const auto row = layout.quaternion_row_start() + node;
            for (std::size_t component = 0; component < 4U; ++component) {
                scalar_builder.add(
                    static_cast<Index>(row),
                    static_cast<Index>(state.first + 6U + component)
                );
            }
        }
        for (std::size_t flat = 0; flat < layout.virtual_count(); ++flat) {
            const auto positive_row = layout.virtual_epigraph_row_start() + 2U * flat;
            const auto negative_row = positive_row + 1U;
            const auto virtual_column = layout.virtual_offset() + flat;
            const auto epigraph_column = layout.epigraph_offset() + flat;
            scalar_builder.add(
                static_cast<Index>(positive_row),
                static_cast<Index>(virtual_column)
            );
            scalar_builder.add(
                static_cast<Index>(positive_row),
                static_cast<Index>(epigraph_column)
            );
            scalar_builder.add(
                static_cast<Index>(negative_row),
                static_cast<Index>(virtual_column)
            );
            scalar_builder.add(
                static_cast<Index>(negative_row),
                static_cast<Index>(epigraph_column)
            );
        }

        CscBuilder affine_builder{
            static_cast<Index>(layout.affine_rows()),
            static_cast<Index>(layout.variables()),
        };
        for (std::size_t interval = 0; interval < layout.intervals(); ++interval) {
            const auto control = layout.control_range(interval);
            const auto start = layout.thrust_cone_row_start() + 4U * interval;
            for (std::size_t component = 0; component < 3U; ++component) {
                affine_builder.add(
                    static_cast<Index>(start + component),
                    static_cast<Index>(control.first + component)
                );
            }
            affine_builder.add(
                static_cast<Index>(start + 3U),
                static_cast<Index>(control.first + 6U)
            );
        }
        for (std::size_t node = 0; node <= layout.intervals(); ++node) {
            const auto state = layout.state_range(node);
            const auto start = layout.angular_rate_cone_row_start() + 4U * node;
            for (std::size_t component = 0; component < 3U; ++component) {
                affine_builder.add(
                    static_cast<Index>(start + component),
                    static_cast<Index>(state.first + 10U + component)
                );
            }
        }
        for (std::size_t interval = 0; interval < layout.intervals(); ++interval) {
            const auto state = layout.state_range(interval);
            const auto control = layout.control_range(interval);
            const auto start = layout.stage_trust_cone_row_start() + 22U * interval;
            for (std::size_t component = 0;
                 component < powered_descent_6dof_state_dimension;
                 ++component) {
                affine_builder.add(
                    static_cast<Index>(start + component),
                    static_cast<Index>(state.first + component)
                );
            }
            for (std::size_t component = 0;
                 component < powered_descent_6dof_control_dimension;
                 ++component) {
                affine_builder.add(
                    static_cast<Index>(start + powered_descent_6dof_state_dimension + component),
                    static_cast<Index>(control.first + component)
                );
            }
        }
        const auto terminal_trust = layout.terminal_trust_cone_row_start();
        for (std::size_t component = 0;
             component < powered_descent_6dof_state_dimension;
             ++component) {
            affine_builder.add(
                static_cast<Index>(terminal_trust + component),
                static_cast<Index>(terminal.first + component)
            );
        }

        auto quadratic = quadratic_builder.build();
        auto scalar = scalar_builder.build();
        auto affine = affine_builder.build();
        std::vector<ConeBlock> cones;
        for (std::size_t interval = 0; interval < layout.intervals(); ++interval) {
            cones.emplace_back(
                ConeKind::second_order,
                static_cast<Index>(layout.thrust_cone_row_start() + 4U * interval),
                2
            );
        }
        for (std::size_t node = 0; node <= layout.intervals(); ++node) {
            cones.emplace_back(
                ConeKind::second_order,
                static_cast<Index>(layout.angular_rate_cone_row_start() + 4U * node),
                2
            );
        }
        for (std::size_t interval = 0; interval < layout.intervals(); ++interval) {
            cones.emplace_back(
                ConeKind::second_order,
                static_cast<Index>(layout.stage_trust_cone_row_start() + 22U * interval),
                20
            );
        }
        cones.emplace_back(
            ConeKind::second_order,
            static_cast<Index>(layout.terminal_trust_cone_row_start()),
            13
        );
        auto quadratic_values = quadratic.values;
        CQPStructure structure{
            std::move(quadratic.structure),
            std::move(scalar.structure),
            std::optional<CscStructure>{std::move(affine.structure)},
            std::move(cones),
        };
        return BuildData{std::move(structure), std::move(quadratic_values)};
    }

    void validate_reference(
        std::span<const PoweredDescent6DOFState> states,
        std::span<const PoweredDescent6DOFControl> controls
    ) const {
        if (states.size() != layout_.intervals() + 1U ||
            controls.size() != layout_.intervals()) {
            throw std::invalid_argument("6-DoF reference dimensions are incompatible");
        }
        for (const auto& state : states) {
            validate_state(state);
            if (state[13] <= 0.0) {
                throw std::invalid_argument("6-DoF reference mass must be positive");
            }
        }
        for (const auto& control : controls) {
            for (const auto value : control) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("6-DoF reference control must be finite");
                }
            }
        }
    }

    static void validate_state(const PoweredDescent6DOFState& state) {
        for (const auto value : state) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("6-DoF state must be finite");
            }
        }
        if (quaternion_norm(Quaternion{state[6], state[7], state[8], state[9]}) <= 0.0) {
            throw std::invalid_argument("6-DoF quaternion must be non-zero");
        }
    }

    template <std::size_t Size>
    static void copy_to_decision(
        const std::array<double, Size>& source,
        const std::pair<std::size_t, std::size_t> range,
        std::vector<double>& destination
    ) {
        if (range.second - range.first != Size) {
            throw std::logic_error("6-DoF fixed decision range has an unexpected size");
        }
        std::copy(
            source.begin(),
            source.end(),
            destination.begin() + static_cast<std::ptrdiff_t>(range.first)
        );
    }

    template <std::size_t Size>
    static void copy_from_decision(
        std::span<const double> source,
        const std::pair<std::size_t, std::size_t> range,
        std::array<double, Size>& destination
    ) {
        if (range.second - range.first != Size) {
            throw std::logic_error("6-DoF fixed decision range has an unexpected size");
        }
        std::copy(
            source.begin() + static_cast<std::ptrdiff_t>(range.first),
            source.begin() + static_cast<std::ptrdiff_t>(range.second),
            destination.begin()
        );
    }

    PoweredDescent6DOF model_;
    PoweredDescent6DOFCQPConfig config_;
    PoweredDescent6DOFCQPLayout layout_;
    CQPStructure structure_;
    std::vector<double> quadratic_values_;
    ValueIndex scalar_index_;
    ValueIndex affine_index_;
};

}  // namespace spacepdhcg::core
