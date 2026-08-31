#pragma once

#include "spacepdhcg/core/fixed_cqp.hpp"
#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"
#include "spacepdhcg/transcription/discrete_flow_linearisation.hpp"
#include "spacepdhcg/transcription/discretisation.hpp"

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

using dynamics::PoweredDescent6DofControl;
using dynamics::PoweredDescent6DofModel;
using dynamics::PoweredDescent6DofState;

struct PoweredDescent6DofScvxConfig {
    std::size_t intervals{10U};
    double step_seconds{0.5};
    double trust_radius{1.0};
    double virtual_l1_weight{1.0e5};
    double virtual_quadratic_weight{1.0e-8};
    double virtual_epigraph_regularisation{1.0e-10};
    double fuel_weight{1.0e-3};
    DiscretisationMethod discretisation{DiscretisationMethod::forward_euler};
    double finite_difference_relative_step{1.0e-6};
    std::array<double, 14U> state_tracking_weights{
        1.0e-4,
        1.0e-4,
        1.0e-4,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-8,
    };
    std::array<double, 7U> control_tracking_weights{
        1.0e-8,
        1.0e-8,
        1.0e-8,
        1.0e-6,
        1.0e-6,
        1.0e-6,
        1.0e-8,
    };
    std::array<double, 14U> state_trust_scales{
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
    std::array<double, 7U> control_trust_scales{
        1.0 / 15'000.0,
        1.0 / 15'000.0,
        1.0 / 15'000.0,
        1.0 / 2'000.0,
        1.0 / 2'000.0,
        1.0 / 2'000.0,
        1.0 / 15'000.0,
    };

    void validate() const {
        if (intervals < 2U) {
            throw std::invalid_argument("6-DoF transcription needs at least two intervals");
        }
        require_positive(step_seconds, "6-DoF step duration must be positive");
        require_positive(trust_radius, "6-DoF trust radius must be positive");
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
        require_positive(
            finite_difference_relative_step,
            "finite-difference relative step must be positive"
        );
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

struct PoweredDescent6DofRange {
    std::size_t start{0U};
    std::size_t size{0U};

    [[nodiscard]] std::size_t stop() const noexcept { return start + size; }
};

struct PoweredDescent6DofScvxLayout {
    explicit PoweredDescent6DofScvxLayout(std::size_t interval_count)
        : intervals(interval_count) {
        if (intervals < 2U) {
            throw std::invalid_argument("6-DoF layout needs at least two intervals");
        }
    }

    std::size_t intervals{0U};

    [[nodiscard]] std::size_t state_count() const noexcept { return (intervals + 1U) * 14U; }
    [[nodiscard]] std::size_t control_count() const noexcept { return intervals * 7U; }
    [[nodiscard]] std::size_t virtual_count() const noexcept { return intervals * 14U; }
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

    [[nodiscard]] PoweredDescent6DofRange initial_rows() const noexcept { return {0U, 14U}; }
    [[nodiscard]] PoweredDescent6DofRange dynamics_rows() const noexcept {
        return {initial_rows().stop(), 14U * intervals};
    }
    [[nodiscard]] PoweredDescent6DofRange terminal_rows() const noexcept {
        return {dynamics_rows().stop(), 13U};
    }
    [[nodiscard]] PoweredDescent6DofRange virtual_epigraph_rows() const noexcept {
        return {terminal_rows().stop(), 28U * intervals};
    }
    [[nodiscard]] PoweredDescent6DofRange tilt_rows() const noexcept {
        return {virtual_epigraph_rows().stop(), intervals};
    }
    [[nodiscard]] PoweredDescent6DofRange quaternion_rows() const noexcept {
        return {tilt_rows().stop(), intervals + 1U};
    }
    [[nodiscard]] std::size_t scalar_rows() const noexcept { return quaternion_rows().stop(); }

    [[nodiscard]] PoweredDescent6DofRange thrust_rows() const noexcept {
        return {0U, 4U * intervals};
    }
    [[nodiscard]] PoweredDescent6DofRange torque_rows() const noexcept {
        return {thrust_rows().stop(), 4U * intervals};
    }
    [[nodiscard]] PoweredDescent6DofRange glide_rows() const noexcept {
        return {torque_rows().stop(), 3U * (intervals + 1U)};
    }
    [[nodiscard]] PoweredDescent6DofRange angular_rate_rows() const noexcept {
        return {glide_rows().stop(), 4U * (intervals + 1U)};
    }
    [[nodiscard]] PoweredDescent6DofRange stage_trust_rows() const noexcept {
        return {angular_rate_rows().stop(), 22U * intervals};
    }
    [[nodiscard]] PoweredDescent6DofRange terminal_trust_rows() const noexcept {
        return {stage_trust_rows().stop(), 15U};
    }
    [[nodiscard]] std::size_t affine_rows() const noexcept {
        return terminal_trust_rows().stop();
    }

    [[nodiscard]] PoweredDescent6DofRange state(std::size_t node) const {
        if (node > intervals) {
            throw std::out_of_range("6-DoF state node is outside the trajectory");
        }
        return {14U * node, 14U};
    }
    [[nodiscard]] PoweredDescent6DofRange control(std::size_t interval) const {
        validate_interval(interval);
        return {control_offset() + 7U * interval, 7U};
    }
    [[nodiscard]] PoweredDescent6DofRange virtual_control(std::size_t interval) const {
        validate_interval(interval);
        return {virtual_offset() + 14U * interval, 14U};
    }
    [[nodiscard]] PoweredDescent6DofRange epigraph(std::size_t interval) const {
        validate_interval(interval);
        return {epigraph_offset() + 14U * interval, 14U};
    }

  private:
    void validate_interval(std::size_t interval) const {
        if (interval >= intervals) {
            throw std::out_of_range("6-DoF interval is outside the trajectory");
        }
    }
};

struct PoweredDescent6DofConvexDiagnostics {
    double scalar_violation_inf{0.0};
    double variable_violation_inf{0.0};
    double cone_violation_inf{0.0};
    double linearised_dynamics_defect_inf{0.0};
    double terminal_error_inf{0.0};
    double quaternion_linearisation_error_inf{0.0};
    double virtual_control_inf{0.0};

    [[nodiscard]] double maximum_violation() const noexcept {
        return std::max({scalar_violation_inf, variable_violation_inf, cone_violation_inf});
    }
};

class PoweredDescent6DofSubproblem {
  public:
    explicit PoweredDescent6DofSubproblem(
        PoweredDescent6DofModel model = PoweredDescent6DofModel{},
        PoweredDescent6DofScvxConfig config = PoweredDescent6DofScvxConfig{}
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

    [[nodiscard]] const PoweredDescent6DofModel& model() const noexcept { return model_; }
    [[nodiscard]] const PoweredDescent6DofScvxConfig& config() const noexcept {
        return config_;
    }
    [[nodiscard]] const PoweredDescent6DofScvxLayout& layout() const noexcept {
        return layout_;
    }
    [[nodiscard]] const core::FixedStructure& structure() const noexcept { return structure_; }

    [[nodiscard]] core::NumericValues values(
        const std::vector<PoweredDescent6DofState>& reference_states,
        const std::vector<PoweredDescent6DofControl>& reference_controls,
        const PoweredDescent6DofState& initial_state,
        const PoweredDescent6DofState& target_state,
        double trust_radius = -1.0
    ) const {
        validate_reference(reference_states, reference_controls);
        validate_state(initial_state, "initial state");
        validate_state(target_state, "target state");
        const auto radius = trust_radius > 0.0 ? trust_radius : config_.trust_radius;
        if (!std::isfinite(radius) || radius <= 0.0) {
            throw std::invalid_argument("6-DoF trust radius must be finite and positive");
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
        fill_tilt(result);
        fill_quaternion_linearisation(result, reference_states);
        fill_affine_cones(result, reference_states, reference_controls, radius);
        fill_variable_bounds(result);
        result.validate(structure_);
        return result;
    }

    [[nodiscard]] core::FixedCQP problem(
        const std::vector<PoweredDescent6DofState>& reference_states,
        const std::vector<PoweredDescent6DofControl>& reference_controls,
        const PoweredDescent6DofState& initial_state,
        const PoweredDescent6DofState& target_state,
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
        const std::vector<PoweredDescent6DofState>& states,
        const std::vector<PoweredDescent6DofControl>& controls
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

    [[nodiscard]] PoweredDescent6DofConvexDiagnostics diagnostics(
        const std::vector<double>& decision,
        const core::NumericValues& values
    ) const {
        if (decision.size() != layout_.variables()) {
            throw std::invalid_argument("6-DoF decision vector has the wrong size");
        }
        values.validate(structure_);
        const auto scalar = matvec(structure_.scalar_constraint, values.scalar_constraint, decision);
        const auto affine = matvec(*structure_.affine_cone, values.affine_cone, decision);
        PoweredDescent6DofConvexDiagnostics result{};
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
        for (std::size_t local = 0; local < layout_.quaternion_rows().size; ++local) {
            const auto row = layout_.quaternion_rows().start + local;
            result.quaternion_linearisation_error_inf = std::max(
                result.quaternion_linearisation_error_inf,
                std::abs(scalar[row] - values.scalar_lower[row])
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
                    "6-DoF coefficient (" + std::to_string(row) + ", "
                    + std::to_string(column) + ") is absent from the fixed pattern"
                );
            }
            values[iterator->second] = value;
        }

      private:
        std::map<std::pair<Index, Index>, std::size_t> positions_{};
    };

    PoweredDescent6DofModel model_{};
    PoweredDescent6DofScvxConfig config_{};
    PoweredDescent6DofScvxLayout layout_;
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
        for (std::size_t component = 0; component < 14U; ++component) {
            a.insert({layout_.initial_rows().start + component, component});
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto row_start = layout_.dynamics_rows().start + 14U * interval;
            const auto state = layout_.state(interval);
            const auto next_state = layout_.state(interval + 1U);
            const auto control = layout_.control(interval);
            const auto virtual_control = layout_.virtual_control(interval);
            for (std::size_t row_offset = 0; row_offset < 14U; ++row_offset) {
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
        for (std::size_t component = 0; component < 13U; ++component) {
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
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            const auto tilt_row = layout_.tilt_rows().start + interval;
            a.insert({tilt_row, control.start + 2U});
            a.insert({tilt_row, control.start + 6U});
            const auto thrust_start = layout_.thrust_rows().start + 4U * interval;
            f.insert({thrust_start, control.start});
            f.insert({thrust_start + 1U, control.start + 1U});
            f.insert({thrust_start + 2U, control.start + 2U});
            f.insert({thrust_start + 3U, control.start + 6U});
            const auto torque_start = layout_.torque_rows().start + 4U * interval;
            f.insert({torque_start, control.start + 3U});
            f.insert({torque_start + 1U, control.start + 4U});
            f.insert({torque_start + 2U, control.start + 5U});
        }
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            const auto quaternion_row = layout_.quaternion_rows().start + node;
            for (std::size_t component = 0; component < 4U; ++component) {
                a.insert({quaternion_row, state.start + 6U + component});
            }
            const auto glide_start = layout_.glide_rows().start + 3U * node;
            f.insert({glide_start, state.start});
            f.insert({glide_start + 1U, state.start + 1U});
            f.insert({glide_start + 2U, state.start + 2U});
            const auto rate_start = layout_.angular_rate_rows().start + 4U * node;
            f.insert({rate_start, state.start + 10U});
            f.insert({rate_start + 1U, state.start + 11U});
            f.insert({rate_start + 2U, state.start + 12U});
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto state = layout_.state(interval);
            const auto control = layout_.control(interval);
            const auto trust_start = layout_.stage_trust_rows().start + 22U * interval;
            for (std::size_t component = 0; component < 14U; ++component) {
                f.insert({trust_start + component, state.start + component});
            }
            for (std::size_t component = 0; component < 7U; ++component) {
                f.insert({trust_start + 14U + component, control.start + component});
            }
        }
        const auto terminal_trust = layout_.terminal_trust_rows().start;
        for (std::size_t component = 0; component < 14U; ++component) {
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
        blocks.reserve(5U * layout_.intervals + 3U);
        append_uniform_cones(blocks, layout_.thrust_rows().start, layout_.intervals, 4U, 2);
        append_uniform_cones(blocks, layout_.torque_rows().start, layout_.intervals, 4U, 2);
        append_uniform_cones(
            blocks,
            layout_.glide_rows().start,
            layout_.intervals + 1U,
            3U,
            1
        );
        append_uniform_cones(
            blocks,
            layout_.angular_rate_rows().start,
            layout_.intervals + 1U,
            4U,
            2
        );
        append_uniform_cones(
            blocks,
            layout_.stage_trust_rows().start,
            layout_.intervals,
            22U,
            20
        );
        blocks.push_back(
            ConeBlockDescriptor{
                ConeKind::second_order,
                static_cast<Index>(layout_.terminal_trust_rows().start),
                13,
                0.0,
            }
        );
        return blocks;
    }

    static void append_uniform_cones(
        std::vector<ConeBlockDescriptor>& blocks,
        std::size_t start,
        std::size_t count,
        std::size_t stride,
        Index vector_dimension
    ) {
        for (std::size_t item = 0; item < count; ++item) {
            blocks.push_back(
                ConeBlockDescriptor{
                    ConeKind::second_order,
                    static_cast<Index>(start + stride * item),
                    vector_dimension,
                    0.0,
                }
            );
        }
    }

    [[nodiscard]] std::vector<double> make_quadratic_values() const {
        std::vector<double> values(structure_.quadratic.nonzeros(), 0.0);
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            for (std::size_t component = 0; component < 14U; ++component) {
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
            for (std::size_t component = 0; component < 7U; ++component) {
                quadratic_index_.set(
                    values,
                    control.start + component,
                    control.start + component,
                    config_.control_tracking_weights[component]
                );
            }
            for (std::size_t component = 0; component < 14U; ++component) {
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
        const std::vector<PoweredDescent6DofState>& states,
        const std::vector<PoweredDescent6DofControl>& controls
    ) const {
        for (std::size_t node = 0; node < states.size(); ++node) {
            const auto range = layout_.state(node);
            for (std::size_t component = 0; component < 14U; ++component) {
                values.linear_objective[range.start + component] =
                    -states[node][component] * config_.state_tracking_weights[component];
            }
        }
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            const auto range = layout_.control(interval);
            for (std::size_t component = 0; component < 7U; ++component) {
                values.linear_objective[range.start + component] =
                    -controls[interval][component] * config_.control_tracking_weights[component];
            }
            values.linear_objective[range.start + 6U] +=
                config_.fuel_weight * config_.step_seconds;
        }
        for (std::size_t variable = layout_.epigraph_offset();
             variable < layout_.variables();
             ++variable) {
            values.linear_objective[variable] = config_.virtual_l1_weight;
        }
    }

    void fill_initial(core::NumericValues& values, const PoweredDescent6DofState& initial) const {
        for (std::size_t component = 0; component < 14U; ++component) {
            const auto row = layout_.initial_rows().start + component;
            scalar_index_.set(values.scalar_constraint, row, component, 1.0);
            values.scalar_lower[row] = initial[component];
            values.scalar_upper[row] = initial[component];
        }
    }

    void fill_dynamics(
        core::NumericValues& values,
        const std::vector<PoweredDescent6DofState>& states,
        const std::vector<PoweredDescent6DofControl>& controls
    ) const {
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto linearisation = linearise_discrete_flow<14U, 7U>(
                model_,
                states[interval],
                controls[interval],
                config_.step_seconds,
                config_.discretisation,
                config_.finite_difference_relative_step
            );
            const auto row_start = layout_.dynamics_rows().start + 14U * interval;
            const auto state = layout_.state(interval);
            const auto next_state = layout_.state(interval + 1U);
            const auto control = layout_.control(interval);
            const auto virtual_control = layout_.virtual_control(interval);
            for (std::size_t row_offset = 0; row_offset < 14U; ++row_offset) {
                const auto row = row_start + row_offset;
                for (std::size_t column = 0; column < 14U; ++column) {
                    scalar_index_.set(
                        values.scalar_constraint,
                        row,
                        state.start + column,
                        -linearisation.state[row_offset * 14U + column]
                    );
                }
                scalar_index_.set(
                    values.scalar_constraint,
                    row,
                    next_state.start + row_offset,
                    1.0
                );
                for (std::size_t column = 0; column < 7U; ++column) {
                    scalar_index_.set(
                        values.scalar_constraint,
                        row,
                        control.start + column,
                        -linearisation.control[row_offset * 7U + column]
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

    void fill_terminal(core::NumericValues& values, const PoweredDescent6DofState& target) const {
        const auto final_state = layout_.state(layout_.intervals);
        for (std::size_t component = 0; component < 13U; ++component) {
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

    void fill_tilt(core::NumericValues& values) const {
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto row = layout_.tilt_rows().start + interval;
            const auto control = layout_.control(interval);
            scalar_index_.set(values.scalar_constraint, row, control.start + 2U, -1.0);
            scalar_index_.set(
                values.scalar_constraint,
                row,
                control.start + 6U,
                model_.config().tilt_cosine()
            );
            values.scalar_upper[row] = 0.0;
        }
    }

    void fill_quaternion_linearisation(
        core::NumericValues& values,
        const std::vector<PoweredDescent6DofState>& states
    ) const {
        for (std::size_t node = 0; node < states.size(); ++node) {
            const auto row = layout_.quaternion_rows().start + node;
            const auto state = layout_.state(node);
            double norm_squared{0.0};
            for (std::size_t component = 0; component < 4U; ++component) {
                const auto reference = states[node][6U + component];
                norm_squared += reference * reference;
                scalar_index_.set(
                    values.scalar_constraint,
                    row,
                    state.start + 6U + component,
                    2.0 * reference
                );
            }
            values.scalar_lower[row] = 1.0 + norm_squared;
            values.scalar_upper[row] = 1.0 + norm_squared;
        }
    }

    void fill_affine_cones(
        core::NumericValues& values,
        const std::vector<PoweredDescent6DofState>& states,
        const std::vector<PoweredDescent6DofControl>& controls,
        double radius
    ) const {
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            const auto thrust_start = layout_.thrust_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 3U; ++component) {
                affine_index_.set(
                    values.affine_cone,
                    thrust_start + component,
                    control.start + component,
                    1.0
                );
            }
            affine_index_.set(
                values.affine_cone,
                thrust_start + 3U,
                control.start + 6U,
                1.0
            );
            const auto torque_start = layout_.torque_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 3U; ++component) {
                affine_index_.set(
                    values.affine_cone,
                    torque_start + component,
                    control.start + 3U + component,
                    1.0
                );
            }
            values.affine_offset[torque_start + 3U] = model_.config().maximum_torque;
        }
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            const auto glide_start = layout_.glide_rows().start + 3U * node;
            affine_index_.set(values.affine_cone, glide_start, state.start, 1.0);
            affine_index_.set(values.affine_cone, glide_start + 1U, state.start + 1U, 1.0);
            affine_index_.set(
                values.affine_cone,
                glide_start + 2U,
                state.start + 2U,
                model_.config().glide_slope_tangent()
            );
            const auto rate_start = layout_.angular_rate_rows().start + 4U * node;
            for (std::size_t component = 0; component < 3U; ++component) {
                affine_index_.set(
                    values.affine_cone,
                    rate_start + component,
                    state.start + 10U + component,
                    1.0
                );
            }
            values.affine_offset[rate_start + 3U] = model_.config().maximum_angular_rate;
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto state = layout_.state(interval);
            const auto control = layout_.control(interval);
            const auto trust_start = layout_.stage_trust_rows().start + 22U * interval;
            for (std::size_t component = 0; component < 14U; ++component) {
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
            for (std::size_t component = 0; component < 7U; ++component) {
                const auto row = trust_start + 14U + component;
                const auto scale = config_.control_trust_scales[component];
                affine_index_.set(
                    values.affine_cone,
                    row,
                    control.start + component,
                    scale
                );
                values.affine_offset[row] = -scale * controls[interval][component];
            }
            values.affine_offset[trust_start + 21U] = radius;
        }
        const auto final_state = layout_.state(layout_.intervals);
        const auto terminal_start = layout_.terminal_trust_rows().start;
        for (std::size_t component = 0; component < 14U; ++component) {
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
        values.affine_offset[terminal_start + 14U] = radius;
    }

    void fill_variable_bounds(core::NumericValues& values) const {
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            values.variable_lower[state.start + 2U] = 0.0;
            values.variable_lower[state.start + 13U] = model_.config().minimum_mass;
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            values.variable_lower[control.start + 6U] = model_.config().minimum_sigma;
            values.variable_upper[control.start + 6U] = model_.config().maximum_thrust;
        }
        for (std::size_t variable = layout_.epigraph_offset();
             variable < layout_.variables();
             ++variable) {
            values.variable_lower[variable] = 0.0;
        }
    }

    void validate_reference(
        const std::vector<PoweredDescent6DofState>& states,
        const std::vector<PoweredDescent6DofControl>& controls
    ) const {
        if (states.size() != layout_.intervals + 1U || controls.size() != layout_.intervals) {
            throw std::invalid_argument("6-DoF reference trajectory has the wrong horizon");
        }
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            static_cast<void>(model_.dynamics(states[interval], controls[interval]));
        }
        validate_state(states.back(), "final reference state");
    }

    static void validate_state(const PoweredDescent6DofState& state, const char* name) {
        for (const auto value : state) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(std::string(name) + " must be finite");
            }
        }
        if (state[13U] <= 0.0) {
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
