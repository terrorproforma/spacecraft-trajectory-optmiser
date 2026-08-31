#pragma once

#include "spacepdhcg/core/fixed_cqp.hpp"
#include "spacepdhcg/dynamics/low_thrust_two_body.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace spacepdhcg::transcription {

using dynamics::LowThrustControl;
using dynamics::LowThrustState;
using dynamics::LowThrustTwoBodyModel;

struct LowThrustScvxConfig {
    std::size_t intervals{100U};
    double step_seconds{60.0};
    double trust_radius{1.0};
    double virtual_l1_weight{1.0e6};
    double virtual_quadratic_weight{1.0e-8};
    double virtual_epigraph_regularisation{1.0e-10};
    double fuel_weight{1.0};
    std::array<double, 7U> state_tracking_weights{
        1.0e-6,
        1.0e-6,
        1.0e-6,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-8,
    };
    std::array<double, 4U> control_tracking_weights{
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e-3,
    };
    std::array<double, 7U> state_trust_scales{
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0,
        1.0,
        1.0,
        1.0e-3,
    };
    std::array<double, 4U> control_trust_scales{
        1.0,
        1.0,
        1.0,
        1.0,
    };

    void validate() const {
        if (intervals < 2U) {
            throw std::invalid_argument("low-thrust transcription needs at least two intervals");
        }
        require_positive(step_seconds, "low-thrust step duration must be positive");
        require_positive(trust_radius, "low-thrust trust radius must be positive");
        require_nonnegative(virtual_l1_weight, "virtual L1 weight must be non-negative");
        require_nonnegative(
            virtual_quadratic_weight,
            "virtual quadratic weight must be non-negative"
        );
        require_nonnegative(
            virtual_epigraph_regularisation,
            "virtual epigraph regularisation must be non-negative"
        );
        require_nonnegative(fuel_weight, "fuel weight must be non-negative");
        require_positive_array(state_tracking_weights, "state weights must be positive");
        require_positive_array(control_tracking_weights, "control weights must be positive");
        require_positive_array(state_trust_scales, "state trust scales must be positive");
        require_positive_array(control_trust_scales, "control trust scales must be positive");
    }

  private:
    static void require_positive(double value, const char* message) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw std::invalid_argument(message);
        }
    }
    static void require_nonnegative(double value, const char* message) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument(message);
        }
    }
    template <std::size_t Size>
    static void require_positive_array(
        const std::array<double, Size>& values,
        const char* message
    ) {
        for (const auto value : values) {
            require_positive(value, message);
        }
    }
};

struct LowThrustRange {
    std::size_t start{0U};
    std::size_t size{0U};
    [[nodiscard]] std::size_t stop() const noexcept { return start + size; }
};

struct LowThrustScvxLayout {
    explicit LowThrustScvxLayout(std::size_t interval_count) : intervals(interval_count) {
        if (intervals < 2U) {
            throw std::invalid_argument("low-thrust layout needs at least two intervals");
        }
    }

    std::size_t intervals{0U};

    [[nodiscard]] std::size_t state_count() const noexcept { return 7U * (intervals + 1U); }
    [[nodiscard]] std::size_t control_count() const noexcept { return 4U * intervals; }
    [[nodiscard]] std::size_t virtual_count() const noexcept { return 7U * intervals; }
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

    [[nodiscard]] LowThrustRange initial_rows() const noexcept { return {0U, 7U}; }
    [[nodiscard]] LowThrustRange dynamics_rows() const noexcept {
        return {initial_rows().stop(), 7U * intervals};
    }
    [[nodiscard]] LowThrustRange terminal_rows() const noexcept {
        return {dynamics_rows().stop(), 6U};
    }
    [[nodiscard]] LowThrustRange virtual_epigraph_rows() const noexcept {
        return {terminal_rows().stop(), 14U * intervals};
    }
    [[nodiscard]] LowThrustRange radial_rows() const noexcept {
        return {virtual_epigraph_rows().stop(), intervals + 1U};
    }
    [[nodiscard]] std::size_t scalar_rows() const noexcept { return radial_rows().stop(); }

    [[nodiscard]] LowThrustRange thrust_rows() const noexcept { return {0U, 4U * intervals}; }
    [[nodiscard]] LowThrustRange stage_trust_rows() const noexcept {
        return {thrust_rows().stop(), 12U * intervals};
    }
    [[nodiscard]] LowThrustRange terminal_trust_rows() const noexcept {
        return {stage_trust_rows().stop(), 8U};
    }
    [[nodiscard]] std::size_t affine_rows() const noexcept {
        return terminal_trust_rows().stop();
    }

    [[nodiscard]] LowThrustRange state(std::size_t node) const {
        if (node > intervals) {
            throw std::out_of_range("low-thrust state node is outside the trajectory");
        }
        return {7U * node, 7U};
    }
    [[nodiscard]] LowThrustRange control(std::size_t interval) const {
        validate_interval(interval);
        return {control_offset() + 4U * interval, 4U};
    }
    [[nodiscard]] LowThrustRange virtual_control(std::size_t interval) const {
        validate_interval(interval);
        return {virtual_offset() + 7U * interval, 7U};
    }
    [[nodiscard]] LowThrustRange epigraph(std::size_t interval) const {
        validate_interval(interval);
        return {epigraph_offset() + 7U * interval, 7U};
    }

  private:
    void validate_interval(std::size_t interval) const {
        if (interval >= intervals) {
            throw std::out_of_range("low-thrust interval is outside the trajectory");
        }
    }
};

struct LowThrustConvexDiagnostics {
    double scalar_violation_inf{0.0};
    double variable_violation_inf{0.0};
    double cone_violation_inf{0.0};
    double linearised_dynamics_defect_inf{0.0};
    double terminal_error_inf{0.0};
    double radial_linearisation_error_inf{0.0};
    double virtual_control_inf{0.0};

    [[nodiscard]] double maximum_violation() const noexcept {
        return std::max({scalar_violation_inf, variable_violation_inf, cone_violation_inf});
    }
};

class LowThrustSubproblem {
  public:
    explicit LowThrustSubproblem(
        LowThrustTwoBodyModel model = {},
        LowThrustScvxConfig config = {}
    )
        : model_(std::move(model)), config_(config), layout_(config_.intervals) {
        config_.validate();
        auto [quadratic, scalar, affine] = build_patterns();
        structure_ = core::FixedStructure{
            std::move(quadratic),
            std::move(scalar),
            std::move(affine),
            cone_blocks(),
            {},
        };
        structure_.validate();
        quadratic_index_ = ValueIndex(structure_.quadratic);
        scalar_index_ = ValueIndex(structure_.scalar_constraint);
        affine_index_ = ValueIndex(*structure_.affine_cone);
        quadratic_values_ = make_quadratic_values();
    }

    [[nodiscard]] const LowThrustTwoBodyModel& model() const noexcept { return model_; }
    [[nodiscard]] const LowThrustScvxConfig& config() const noexcept { return config_; }
    [[nodiscard]] const LowThrustScvxLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] const core::FixedStructure& structure() const noexcept { return structure_; }

    [[nodiscard]] core::NumericValues values(
        const std::vector<LowThrustState>& reference_states,
        const std::vector<LowThrustControl>& reference_controls,
        const LowThrustState& initial_state,
        const LowThrustState& target_state,
        double trust_radius = -1.0
    ) const {
        validate_reference(reference_states, reference_controls);
        validate_state(initial_state, "initial state");
        validate_state(target_state, "target state");
        const auto radius = trust_radius > 0.0 ? trust_radius : config_.trust_radius;
        if (!std::isfinite(radius) || radius <= 0.0) {
            throw std::invalid_argument("low-thrust trust radius must be finite and positive");
        }

        core::NumericValues result{};
        result.quadratic = quadratic_values_;
        result.scalar_constraint.assign(structure_.scalar_constraint.nonzeros(), 0.0);
        result.affine_cone.assign(structure_.affine_cone->nonzeros(), 0.0);
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

        fill_objective(result, reference_states, reference_controls);
        fill_initial(result, initial_state);
        fill_dynamics(result, reference_states, reference_controls);
        fill_terminal(result, target_state);
        fill_virtual_epigraph(result);
        fill_radial_halfspaces(result, reference_states);
        fill_affine_cones(result, reference_states, reference_controls, radius);
        fill_variable_bounds(result);
        result.validate(structure_);
        return result;
    }

    [[nodiscard]] core::FixedCQP problem(
        const std::vector<LowThrustState>& reference_states,
        const std::vector<LowThrustControl>& reference_controls,
        const LowThrustState& initial_state,
        const LowThrustState& target_state,
        double trust_radius = -1.0
    ) const {
        return core::FixedCQP(
            structure_,
            values(
                reference_states,
                reference_controls,
                initial_state,
                target_state,
                trust_radius
            )
        );
    }

    [[nodiscard]] std::vector<double> reference_decision(
        const std::vector<LowThrustState>& states,
        const std::vector<LowThrustControl>& controls
    ) const {
        validate_reference(states, controls);
        std::vector<double> decision(layout_.variables(), 0.0);
        for (std::size_t node = 0; node < states.size(); ++node) {
            const auto range = layout_.state(node);
            std::copy(
                states[node].begin(),
                states[node].end(),
                decision.begin() + static_cast<std::ptrdiff_t>(range.start)
            );
        }
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            const auto range = layout_.control(interval);
            std::copy(
                controls[interval].begin(),
                controls[interval].end(),
                decision.begin() + static_cast<std::ptrdiff_t>(range.start)
            );
        }
        return decision;
    }

    [[nodiscard]] LowThrustConvexDiagnostics diagnostics(
        const std::vector<double>& decision,
        const core::NumericValues& values
    ) const {
        if (decision.size() != layout_.variables()) {
            throw std::invalid_argument("low-thrust decision vector has the wrong size");
        }
        values.validate(structure_);
        const auto scalar = matvec(structure_.scalar_constraint, values.scalar_constraint, decision);
        const auto affine = matvec(*structure_.affine_cone, values.affine_cone, decision);
        LowThrustConvexDiagnostics result{};
        for (std::size_t row = 0; row < scalar.size(); ++row) {
            result.scalar_violation_inf = std::max(
                result.scalar_violation_inf,
                std::max(
                    values.scalar_lower[row] - scalar[row],
                    scalar[row] - values.scalar_upper[row]
                )
            );
        }
        for (std::size_t variable = 0; variable < decision.size(); ++variable) {
            result.variable_violation_inf = std::max(
                result.variable_violation_inf,
                std::max(
                    values.variable_lower[variable] - decision[variable],
                    decision[variable] - values.variable_upper[variable]
                )
            );
        }
        for (const auto& cone : structure_.affine_cones) {
            const auto slots = static_cast<std::size_t>(cone.vector_dimension) + 2U;
            double norm_squared{0.0};
            for (std::size_t local = 0; local + 1U < slots; ++local) {
                const auto row = static_cast<std::size_t>(cone.start) + local;
                const auto value = affine[row] + values.affine_offset[row];
                norm_squared += value * value;
            }
            const auto scalar_row = static_cast<std::size_t>(cone.start) + slots - 1U;
            result.cone_violation_inf = std::max(
                result.cone_violation_inf,
                std::sqrt(norm_squared)
                    - (affine[scalar_row] + values.affine_offset[scalar_row])
            );
        }
        result.scalar_violation_inf = std::max(result.scalar_violation_inf, 0.0);
        result.variable_violation_inf = std::max(result.variable_violation_inf, 0.0);
        result.cone_violation_inf = std::max(result.cone_violation_inf, 0.0);
        for (std::size_t local = 0; local < layout_.dynamics_rows().size; ++local) {
            const auto row = layout_.dynamics_rows().start + local;
            result.linearised_dynamics_defect_inf = std::max(
                result.linearised_dynamics_defect_inf,
                std::abs(scalar[row] - values.scalar_lower[row])
            );
        }
        for (std::size_t local = 0; local < layout_.terminal_rows().size; ++local) {
            const auto row = layout_.terminal_rows().start + local;
            result.terminal_error_inf = std::max(
                result.terminal_error_inf,
                std::abs(scalar[row] - values.scalar_lower[row])
            );
        }
        for (std::size_t local = 0; local < layout_.radial_rows().size; ++local) {
            const auto row = layout_.radial_rows().start + local;
            result.radial_linearisation_error_inf = std::max(
                result.radial_linearisation_error_inf,
                std::max(values.scalar_lower[row] - scalar[row], 0.0)
            );
        }
        for (std::size_t variable = layout_.virtual_offset();
             variable < layout_.epigraph_offset();
             ++variable) {
            result.virtual_control_inf = std::max(
                result.virtual_control_inf,
                std::abs(decision[variable])
            );
        }
        return result;
    }

  private:
    class ValueIndex {
      public:
        ValueIndex() = default;
        explicit ValueIndex(const core::CscPattern& pattern) {
            for (Index column = 0; column < pattern.columns; ++column) {
                const auto begin = static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column)]);
                const auto end = static_cast<std::size_t>(
                    pattern.offsets[static_cast<std::size_t>(column) + 1U]
                );
                for (std::size_t position = begin; position < end; ++position) {
                    positions_[{pattern.indices[position], column}] = position;
                }
            }
        }
        void set(std::vector<double>& values, std::size_t row, std::size_t column, double value) const {
            const auto iterator = positions_.find(
                {static_cast<Index>(row), static_cast<Index>(column)}
            );
            if (iterator == positions_.end()) {
                throw std::logic_error(
                    "low-thrust coefficient (" + std::to_string(row) + ", "
                    + std::to_string(column) + ") is absent from the fixed pattern"
                );
            }
            values[iterator->second] = value;
        }

      private:
        std::map<std::pair<Index, Index>, std::size_t> positions_{};
    };

    LowThrustTwoBodyModel model_{};
    LowThrustScvxConfig config_{};
    LowThrustScvxLayout layout_;
    core::FixedStructure structure_{};
    ValueIndex quadratic_index_{};
    ValueIndex scalar_index_{};
    ValueIndex affine_index_{};
    std::vector<double> quadratic_values_{};

    [[nodiscard]] std::tuple<core::CscPattern, core::CscPattern, core::CscPattern>
    build_patterns() const {
        std::set<std::pair<std::size_t, std::size_t>> q{};
        std::set<std::pair<std::size_t, std::size_t>> a{};
        std::set<std::pair<std::size_t, std::size_t>> f{};
        for (std::size_t variable = 0; variable < layout_.variables(); ++variable) {
            q.insert({variable, variable});
        }
        for (std::size_t component = 0; component < 7U; ++component) {
            a.insert({layout_.initial_rows().start + component, component});
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto row_start = layout_.dynamics_rows().start + 7U * interval;
            const auto state = layout_.state(interval);
            const auto next_state = layout_.state(interval + 1U);
            const auto control = layout_.control(interval);
            const auto virtual_control = layout_.virtual_control(interval);
            for (std::size_t row_offset = 0; row_offset < 7U; ++row_offset) {
                const auto row = row_start + row_offset;
                for (std::size_t column = state.start; column < state.stop(); ++column) {
                    a.insert({row, column});
                }
                a.insert({row, next_state.start + row_offset});
                for (std::size_t column = control.start; column < control.stop(); ++column) {
                    a.insert({row, column});
                }
                a.insert({row, virtual_control.start + row_offset});
            }
        }
        const auto terminal_state = layout_.state(layout_.intervals);
        for (std::size_t component = 0; component < 6U; ++component) {
            a.insert({layout_.terminal_rows().start + component, terminal_state.start + component});
        }
        for (std::size_t flat = 0; flat < layout_.virtual_count(); ++flat) {
            const auto positive = layout_.virtual_epigraph_rows().start + 2U * flat;
            const auto negative = positive + 1U;
            const auto virtual_variable = layout_.virtual_offset() + flat;
            const auto epigraph_variable = layout_.epigraph_offset() + flat;
            a.insert({positive, virtual_variable});
            a.insert({positive, epigraph_variable});
            a.insert({negative, virtual_variable});
            a.insert({negative, epigraph_variable});
        }
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            const auto radial_row = layout_.radial_rows().start + node;
            a.insert({radial_row, state.start});
            a.insert({radial_row, state.start + 1U});
            a.insert({radial_row, state.start + 2U});
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            const auto thrust_start = layout_.thrust_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 4U; ++component) {
                f.insert({thrust_start + component, control.start + component});
            }
            const auto state = layout_.state(interval);
            const auto trust_start = layout_.stage_trust_rows().start + 12U * interval;
            for (std::size_t component = 0; component < 7U; ++component) {
                f.insert({trust_start + component, state.start + component});
            }
            for (std::size_t component = 0; component < 4U; ++component) {
                f.insert({trust_start + 7U + component, control.start + component});
            }
        }
        const auto terminal_trust = layout_.terminal_trust_rows().start;
        for (std::size_t component = 0; component < 7U; ++component) {
            f.insert({terminal_trust + component, terminal_state.start + component});
        }
        return {
            make_pattern(layout_.variables(), layout_.variables(), q),
            make_pattern(layout_.scalar_rows(), layout_.variables(), a),
            make_pattern(layout_.affine_rows(), layout_.variables(), f),
        };
    }

    [[nodiscard]] std::vector<ConeBlockDescriptor> cone_blocks() const {
        std::vector<ConeBlockDescriptor> blocks{};
        blocks.reserve(2U * layout_.intervals + 1U);
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            blocks.push_back(
                ConeBlockDescriptor{
                    ConeKind::second_order,
                    static_cast<Index>(layout_.thrust_rows().start + 4U * interval),
                    2,
                    0.0,
                }
            );
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            blocks.push_back(
                ConeBlockDescriptor{
                    ConeKind::second_order,
                    static_cast<Index>(layout_.stage_trust_rows().start + 12U * interval),
                    10,
                    0.0,
                }
            );
        }
        blocks.push_back(
            ConeBlockDescriptor{
                ConeKind::second_order,
                static_cast<Index>(layout_.terminal_trust_rows().start),
                6,
                0.0,
            }
        );
        return blocks;
    }

    [[nodiscard]] std::vector<double> make_quadratic_values() const {
        std::vector<double> values(structure_.quadratic.nonzeros(), 0.0);
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            for (std::size_t component = 0; component < 7U; ++component) {
                quadratic_index_.set(
                    values,
                    state.start + component,
                    state.start + component,
                    config_.state_tracking_weights[component]
                );
            }
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            const auto virtual_control = layout_.virtual_control(interval);
            const auto epigraph = layout_.epigraph(interval);
            for (std::size_t component = 0; component < 4U; ++component) {
                quadratic_index_.set(
                    values,
                    control.start + component,
                    control.start + component,
                    config_.control_tracking_weights[component]
                );
            }
            for (std::size_t component = 0; component < 7U; ++component) {
                quadratic_index_.set(
                    values,
                    virtual_control.start + component,
                    virtual_control.start + component,
                    config_.virtual_quadratic_weight
                );
                quadratic_index_.set(
                    values,
                    epigraph.start + component,
                    epigraph.start + component,
                    config_.virtual_epigraph_regularisation
                );
            }
        }
        return values;
    }

    void fill_objective(
        core::NumericValues& values,
        const std::vector<LowThrustState>& states,
        const std::vector<LowThrustControl>& controls
    ) const {
        for (std::size_t node = 0; node < states.size(); ++node) {
            const auto range = layout_.state(node);
            for (std::size_t component = 0; component < 7U; ++component) {
                values.linear_objective[range.start + component] =
                    -states[node][component] * config_.state_tracking_weights[component];
            }
        }
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            const auto range = layout_.control(interval);
            for (std::size_t component = 0; component < 4U; ++component) {
                values.linear_objective[range.start + component] =
                    -controls[interval][component] * config_.control_tracking_weights[component];
            }
            values.linear_objective[range.start + 3U] +=
                config_.fuel_weight * config_.step_seconds;
        }
        for (std::size_t variable = layout_.epigraph_offset();
             variable < layout_.variables();
             ++variable) {
            values.linear_objective[variable] = config_.virtual_l1_weight;
        }
    }

    void fill_initial(core::NumericValues& values, const LowThrustState& initial) const {
        for (std::size_t component = 0; component < 7U; ++component) {
            const auto row = layout_.initial_rows().start + component;
            scalar_index_.set(values.scalar_constraint, row, component, 1.0);
            values.scalar_lower[row] = initial[component];
            values.scalar_upper[row] = initial[component];
        }
    }

    void fill_dynamics(
        core::NumericValues& values,
        const std::vector<LowThrustState>& states,
        const std::vector<LowThrustControl>& controls
    ) const {
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto linearisation = model_.linearised_euler_dynamics(
                states[interval],
                controls[interval],
                config_.step_seconds
            );
            const auto row_start = layout_.dynamics_rows().start + 7U * interval;
            const auto state = layout_.state(interval);
            const auto next_state = layout_.state(interval + 1U);
            const auto control = layout_.control(interval);
            const auto virtual_control = layout_.virtual_control(interval);
            for (std::size_t row_offset = 0; row_offset < 7U; ++row_offset) {
                const auto row = row_start + row_offset;
                for (std::size_t column = 0; column < 7U; ++column) {
                    scalar_index_.set(
                        values.scalar_constraint,
                        row,
                        state.start + column,
                        -linearisation.state[row_offset * 7U + column]
                    );
                }
                scalar_index_.set(
                    values.scalar_constraint,
                    row,
                    next_state.start + row_offset,
                    1.0
                );
                for (std::size_t column = 0; column < 4U; ++column) {
                    scalar_index_.set(
                        values.scalar_constraint,
                        row,
                        control.start + column,
                        -linearisation.control[row_offset * 4U + column]
                    );
                }
                scalar_index_.set(
                    values.scalar_constraint,
                    row,
                    virtual_control.start + row_offset,
                    -1.0
                );
                values.scalar_lower[row] = linearisation.offset[row_offset];
                values.scalar_upper[row] = linearisation.offset[row_offset];
            }
        }
    }

    void fill_terminal(core::NumericValues& values, const LowThrustState& target) const {
        const auto final_state = layout_.state(layout_.intervals);
        for (std::size_t component = 0; component < 6U; ++component) {
            const auto row = layout_.terminal_rows().start + component;
            scalar_index_.set(
                values.scalar_constraint,
                row,
                final_state.start + component,
                1.0
            );
            values.scalar_lower[row] = target[component];
            values.scalar_upper[row] = target[component];
        }
    }

    void fill_virtual_epigraph(core::NumericValues& values) const {
        for (std::size_t flat = 0; flat < layout_.virtual_count(); ++flat) {
            const auto positive = layout_.virtual_epigraph_rows().start + 2U * flat;
            const auto negative = positive + 1U;
            const auto virtual_variable = layout_.virtual_offset() + flat;
            const auto epigraph_variable = layout_.epigraph_offset() + flat;
            scalar_index_.set(values.scalar_constraint, positive, virtual_variable, 1.0);
            scalar_index_.set(values.scalar_constraint, positive, epigraph_variable, -1.0);
            scalar_index_.set(values.scalar_constraint, negative, virtual_variable, -1.0);
            scalar_index_.set(values.scalar_constraint, negative, epigraph_variable, -1.0);
            values.scalar_upper[positive] = 0.0;
            values.scalar_upper[negative] = 0.0;
        }
    }

    void fill_radial_halfspaces(
        core::NumericValues& values,
        const std::vector<LowThrustState>& states
    ) const {
        for (std::size_t node = 0; node < states.size(); ++node) {
            const auto state = layout_.state(node);
            const auto row = layout_.radial_rows().start + node;
            const auto radius = std::sqrt(
                states[node][0U] * states[node][0U]
                + states[node][1U] * states[node][1U]
                + states[node][2U] * states[node][2U]
            );
            if (radius <= 0.0) {
                throw std::invalid_argument("low-thrust reference radius must be positive");
            }
            for (std::size_t component = 0; component < 3U; ++component) {
                scalar_index_.set(
                    values.scalar_constraint,
                    row,
                    state.start + component,
                    states[node][component] / radius
                );
            }
            values.scalar_lower[row] = model_.config().minimum_radius;
        }
    }

    void fill_affine_cones(
        core::NumericValues& values,
        const std::vector<LowThrustState>& states,
        const std::vector<LowThrustControl>& controls,
        double radius
    ) const {
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            const auto thrust_start = layout_.thrust_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 4U; ++component) {
                affine_index_.set(
                    values.affine_cone,
                    thrust_start + component,
                    control.start + component,
                    1.0
                );
            }
            const auto state = layout_.state(interval);
            const auto trust_start = layout_.stage_trust_rows().start + 12U * interval;
            for (std::size_t component = 0; component < 7U; ++component) {
                const auto scale = config_.state_trust_scales[component];
                affine_index_.set(
                    values.affine_cone,
                    trust_start + component,
                    state.start + component,
                    scale
                );
                values.affine_offset[trust_start + component] =
                    -scale * states[interval][component];
            }
            for (std::size_t component = 0; component < 4U; ++component) {
                const auto row = trust_start + 7U + component;
                const auto scale = config_.control_trust_scales[component];
                affine_index_.set(
                    values.affine_cone,
                    row,
                    control.start + component,
                    scale
                );
                values.affine_offset[row] = -scale * controls[interval][component];
            }
            values.affine_offset[trust_start + 11U] = radius;
        }
        const auto final_state = layout_.state(layout_.intervals);
        const auto terminal_start = layout_.terminal_trust_rows().start;
        for (std::size_t component = 0; component < 7U; ++component) {
            const auto scale = config_.state_trust_scales[component];
            affine_index_.set(
                values.affine_cone,
                terminal_start + component,
                final_state.start + component,
                scale
            );
            values.affine_offset[terminal_start + component] =
                -scale * states.back()[component];
        }
        values.affine_offset[terminal_start + 7U] = radius;
    }

    void fill_variable_bounds(core::NumericValues& values) const {
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            values.variable_lower[state.start + 6U] = model_.config().minimum_mass;
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            values.variable_lower[control.start + 3U] = 0.0;
            values.variable_upper[control.start + 3U] = model_.config().maximum_thrust;
        }
        for (std::size_t variable = layout_.epigraph_offset();
             variable < layout_.variables();
             ++variable) {
            values.variable_lower[variable] = 0.0;
        }
    }

    void validate_reference(
        const std::vector<LowThrustState>& states,
        const std::vector<LowThrustControl>& controls
    ) const {
        if (states.size() != layout_.intervals + 1U || controls.size() != layout_.intervals) {
            throw std::invalid_argument("low-thrust reference trajectory has the wrong horizon");
        }
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            static_cast<void>(model_.dynamics(states[interval], controls[interval]));
        }
        validate_state(states.back(), "final reference state");
    }

    static void validate_state(const LowThrustState& state, const char* name) {
        for (const auto value : state) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(std::string(name) + " must be finite");
            }
        }
        if (state[6U] <= 0.0) {
            throw std::invalid_argument(std::string(name) + " must have positive mass");
        }
    }

    static core::CscPattern make_pattern(
        std::size_t rows,
        std::size_t columns,
        const std::set<std::pair<std::size_t, std::size_t>>& entries
    ) {
        std::vector<std::pair<std::size_t, std::size_t>> ordered(entries.begin(), entries.end());
        std::sort(
            ordered.begin(),
            ordered.end(),
            [](const auto& left, const auto& right) {
                return std::tie(left.second, left.first) < std::tie(right.second, right.first);
            }
        );
        core::CscPattern pattern{};
        pattern.rows = static_cast<Index>(rows);
        pattern.columns = static_cast<Index>(columns);
        pattern.offsets.assign(columns + 1U, 0);
        pattern.indices.reserve(ordered.size());
        for (const auto& [row, column] : ordered) {
            ++pattern.offsets[column + 1U];
            pattern.indices.push_back(static_cast<Index>(row));
        }
        for (std::size_t column = 0; column < columns; ++column) {
            pattern.offsets[column + 1U] += pattern.offsets[column];
        }
        pattern.validate();
        return pattern;
    }

    static std::vector<double> matvec(
        const core::CscPattern& pattern,
        const std::vector<double>& values,
        const std::vector<double>& vector
    ) {
        std::vector<double> result(static_cast<std::size_t>(pattern.rows), 0.0);
        for (Index column = 0; column < pattern.columns; ++column) {
            const auto begin = static_cast<std::size_t>(pattern.offsets[static_cast<std::size_t>(column)]);
            const auto end = static_cast<std::size_t>(
                pattern.offsets[static_cast<std::size_t>(column) + 1U]
            );
            for (std::size_t position = begin; position < end; ++position) {
                result[static_cast<std::size_t>(pattern.indices[position])] +=
                    values[position] * vector[static_cast<std::size_t>(column)];
            }
        }
        return result;
    }
};

}  // namespace spacepdhcg::transcription
