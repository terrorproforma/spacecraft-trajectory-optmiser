#pragma once

#include "spacepdhcg/core/fixed_cqp.hpp"
#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
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
#include <utility>
#include <vector>

namespace spacepdhcg::transcription {

using dynamics::PoweredDescent3DofModel;
using dynamics::PoweredDescentControl;
using dynamics::PoweredDescentState;

struct PoweredDescentScvxConfig {
    std::size_t intervals{10U};
    double step_seconds{2.0};
    double trust_radius{1.0};
    double virtual_l1_weight{1.0e5};
    double virtual_quadratic_weight{1.0e-8};
    double virtual_epigraph_regularisation{1.0e-10};
    double fuel_weight{1.0e-3};
    DiscretisationMethod discretisation{DiscretisationMethod::forward_euler};
    double finite_difference_relative_step{1.0e-6};
    std::array<double, 7U> state_tracking_weights{
        1.0e-4,
        1.0e-4,
        1.0e-4,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-8,
    };
    std::array<double, 4U> control_tracking_weights{
        1.0e-8,
        1.0e-8,
        1.0e-8,
        1.0e-8,
    };
    std::array<double, 7U> state_trust_scales{
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e-2,
        1.0e-2,
        1.0e-2,
        1.0e-3,
    };
    std::array<double, 4U> control_trust_scales{
        1.0 / 15'000.0,
        1.0 / 15'000.0,
        1.0 / 15'000.0,
        1.0 / 15'000.0,
    };

    void validate() const {
        if (intervals < 2U) {
            throw std::invalid_argument("powered-descent transcription needs at least two intervals");
        }
        require_positive(step_seconds, "step duration must be positive");
        require_positive(trust_radius, "trust radius must be positive");
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
        require_positive_vector(state_tracking_weights, "state tracking weights must be positive");
        require_positive_vector(
            control_tracking_weights,
            "control tracking weights must be positive"
        );
        require_positive_vector(state_trust_scales, "state trust scales must be positive");
        require_positive_vector(control_trust_scales, "control trust scales must be positive");
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
    static void require_positive_vector(
        const std::array<double, Size>& values,
        const char* message
    ) {
        for (const auto value : values) {
            require_positive(value, message);
        }
    }
};

struct IndexRange {
    std::size_t start{0U};
    std::size_t size{0U};

    [[nodiscard]] std::size_t stop() const noexcept { return start + size; }
};

struct PoweredDescentScvxLayout {
    explicit PoweredDescentScvxLayout(std::size_t interval_count) : intervals(interval_count) {
        if (intervals < 2U) {
            throw std::invalid_argument("powered-descent layout needs at least two intervals");
        }
    }

    std::size_t intervals{0U};

    [[nodiscard]] std::size_t state_count() const noexcept { return (intervals + 1U) * 7U; }
    [[nodiscard]] std::size_t control_count() const noexcept { return intervals * 4U; }
    [[nodiscard]] std::size_t virtual_count() const noexcept { return intervals * 7U; }
    [[nodiscard]] std::size_t virtual_epigraph_count() const noexcept {
        return virtual_count();
    }
    [[nodiscard]] std::size_t control_offset() const noexcept { return state_count(); }
    [[nodiscard]] std::size_t virtual_offset() const noexcept {
        return control_offset() + control_count();
    }
    [[nodiscard]] std::size_t virtual_epigraph_offset() const noexcept {
        return virtual_offset() + virtual_count();
    }
    [[nodiscard]] std::size_t variables() const noexcept {
        return virtual_epigraph_offset() + virtual_epigraph_count();
    }

    [[nodiscard]] IndexRange initial_rows() const noexcept { return {0U, 7U}; }
    [[nodiscard]] IndexRange dynamics_rows() const noexcept {
        return {initial_rows().stop(), intervals * 7U};
    }
    [[nodiscard]] IndexRange terminal_rows() const noexcept {
        return {dynamics_rows().stop(), 6U};
    }
    [[nodiscard]] IndexRange virtual_epigraph_rows() const noexcept {
        return {terminal_rows().stop(), 2U * virtual_count()};
    }
    [[nodiscard]] IndexRange tilt_rows() const noexcept {
        return {virtual_epigraph_rows().stop(), intervals};
    }
    [[nodiscard]] std::size_t scalar_rows() const noexcept { return tilt_rows().stop(); }

    [[nodiscard]] IndexRange thrust_cone_rows() const noexcept { return {0U, 4U * intervals}; }
    [[nodiscard]] IndexRange glide_cone_rows() const noexcept {
        return {thrust_cone_rows().stop(), 3U * (intervals + 1U)};
    }
    [[nodiscard]] IndexRange stage_trust_cone_rows() const noexcept {
        return {glide_cone_rows().stop(), 12U * intervals};
    }
    [[nodiscard]] IndexRange terminal_trust_cone_rows() const noexcept {
        return {stage_trust_cone_rows().stop(), 8U};
    }
    [[nodiscard]] std::size_t affine_rows() const noexcept {
        return terminal_trust_cone_rows().stop();
    }

    [[nodiscard]] IndexRange state(std::size_t node) const {
        if (node > intervals) {
            throw std::out_of_range("powered-descent state node is outside the trajectory");
        }
        return {7U * node, 7U};
    }
    [[nodiscard]] IndexRange control(std::size_t interval) const {
        validate_interval(interval);
        return {control_offset() + 4U * interval, 4U};
    }
    [[nodiscard]] IndexRange virtual_control(std::size_t interval) const {
        validate_interval(interval);
        return {virtual_offset() + 7U * interval, 7U};
    }
    [[nodiscard]] IndexRange virtual_epigraph(std::size_t interval) const {
        validate_interval(interval);
        return {virtual_epigraph_offset() + 7U * interval, 7U};
    }

  private:
    void validate_interval(std::size_t interval) const {
        if (interval >= intervals) {
            throw std::out_of_range("powered-descent interval is outside the trajectory");
        }
    }
};

struct PoweredDescentConvexDiagnostics {
    double scalar_violation_inf{0.0};
    double variable_violation_inf{0.0};
    double cone_violation_inf{0.0};
    double linearised_dynamics_defect_inf{0.0};
    double terminal_error_inf{0.0};
    double virtual_control_inf{0.0};

    [[nodiscard]] double maximum_violation() const noexcept {
        return std::max({scalar_violation_inf, variable_violation_inf, cone_violation_inf});
    }
};

class PoweredDescent3DofSubproblem {
  public:
    explicit PoweredDescent3DofSubproblem(
        PoweredDescent3DofModel model = PoweredDescent3DofModel{},
        PoweredDescentScvxConfig config = PoweredDescentScvxConfig{}
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
        if (!structure_.affine_cone.has_value()) {
            throw std::logic_error("powered-descent transcription requires affine cones");
        }
        affine_index_ = ValueIndex(*structure_.affine_cone);
        quadratic_values_ = make_quadratic_values();
    }

    [[nodiscard]] const PoweredDescent3DofModel& model() const noexcept { return model_; }
    [[nodiscard]] const PoweredDescentScvxConfig& config() const noexcept { return config_; }
    [[nodiscard]] const PoweredDescentScvxLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] const core::FixedStructure& structure() const noexcept { return structure_; }

    [[nodiscard]] core::NumericValues values(
        const std::vector<PoweredDescentState>& reference_states,
        const std::vector<PoweredDescentControl>& reference_controls,
        const PoweredDescentState& initial_state,
        const std::array<double, 3U>& target_position,
        const std::array<double, 3U>& target_velocity,
        double trust_radius = -1.0
    ) const {
        validate_reference(reference_states, reference_controls);
        const auto radius = trust_radius > 0.0 ? trust_radius : config_.trust_radius;
        if (!std::isfinite(radius) || radius <= 0.0) {
            throw std::invalid_argument("trust radius must be finite and positive");
        }
        for (const auto value : initial_state) {
            require_finite(value, "initial state must be finite");
        }
        for (const auto value : target_position) {
            require_finite(value, "target position must be finite");
        }
        for (const auto value : target_velocity) {
            require_finite(value, "target velocity must be finite");
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
        fill_terminal(result, target_position, target_velocity);
        fill_virtual_epigraph(result);
        fill_tilt(result);
        fill_affine_cones(result, reference_states, reference_controls, radius);
        fill_variable_bounds(result);
        result.validate(structure_);
        return result;
    }

    [[nodiscard]] core::FixedCQP problem(
        const std::vector<PoweredDescentState>& reference_states,
        const std::vector<PoweredDescentControl>& reference_controls,
        const PoweredDescentState& initial_state,
        const std::array<double, 3U>& target_position,
        const std::array<double, 3U>& target_velocity,
        double trust_radius = -1.0
    ) const {
        return core::FixedCQP(
            structure_,
            values(
                reference_states,
                reference_controls,
                initial_state,
                target_position,
                target_velocity,
                trust_radius
            )
        );
    }

    [[nodiscard]] std::vector<double> reference_decision(
        const std::vector<PoweredDescentState>& reference_states,
        const std::vector<PoweredDescentControl>& reference_controls
    ) const {
        validate_reference(reference_states, reference_controls);
        std::vector<double> decision(layout_.variables(), 0.0);
        for (std::size_t node = 0; node < reference_states.size(); ++node) {
            const auto range = layout_.state(node);
            std::copy(
                reference_states[node].begin(),
                reference_states[node].end(),
                decision.begin() + static_cast<std::ptrdiff_t>(range.start)
            );
        }
        for (std::size_t interval = 0; interval < reference_controls.size(); ++interval) {
            const auto range = layout_.control(interval);
            std::copy(
                reference_controls[interval].begin(),
                reference_controls[interval].end(),
                decision.begin() + static_cast<std::ptrdiff_t>(range.start)
            );
        }
        return decision;
    }

    [[nodiscard]] PoweredDescentConvexDiagnostics diagnostics(
        const std::vector<double>& decision,
        const core::NumericValues& values
    ) const {
        if (decision.size() != layout_.variables()) {
            throw std::invalid_argument("powered-descent decision vector has the wrong size");
        }
        values.validate(structure_);
        const auto scalar = matvec(structure_.scalar_constraint, values.scalar_constraint, decision);
        const auto affine = matvec(*structure_.affine_cone, values.affine_cone, decision);
        PoweredDescentConvexDiagnostics result{};
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
            const auto slots = cone_slot_count(cone);
            double norm_squared{0.0};
            for (std::size_t local = 0; local + 1U < slots; ++local) {
                const auto value = affine[static_cast<std::size_t>(cone.start) + local]
                                   + values.affine_offset[static_cast<std::size_t>(cone.start) + local];
                norm_squared += value * value;
            }
            const auto scalar_value =
                affine[static_cast<std::size_t>(cone.start) + slots - 1U]
                + values.affine_offset[static_cast<std::size_t>(cone.start) + slots - 1U];
            result.cone_violation_inf = std::max(
                result.cone_violation_inf,
                std::sqrt(norm_squared) - scalar_value
            );
        }
        result.scalar_violation_inf = std::max(result.scalar_violation_inf, 0.0);
        result.variable_violation_inf = std::max(result.variable_violation_inf, 0.0);
        result.cone_violation_inf = std::max(result.cone_violation_inf, 0.0);

        const auto dynamics_rows = layout_.dynamics_rows();
        for (std::size_t local = 0; local < dynamics_rows.size; ++local) {
            const auto row = dynamics_rows.start + local;
            result.linearised_dynamics_defect_inf = std::max(
                result.linearised_dynamics_defect_inf,
                std::abs(scalar[row] - values.scalar_lower[row])
            );
        }
        const auto terminal_rows = layout_.terminal_rows();
        for (std::size_t local = 0; local < terminal_rows.size; ++local) {
            const auto row = terminal_rows.start + local;
            result.terminal_error_inf = std::max(
                result.terminal_error_inf,
                std::abs(scalar[row] - values.scalar_lower[row])
            );
        }
        for (std::size_t variable = layout_.virtual_offset();
             variable < layout_.virtual_epigraph_offset();
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
                    "coefficient (" + std::to_string(row) + ", " + std::to_string(column)
                    + ") is absent from the fixed sparse pattern"
                );
            }
            values[iterator->second] = value;
        }

      private:
        std::map<std::pair<Index, Index>, std::size_t> positions_{};
    };

    PoweredDescent3DofModel model_{};
    PoweredDescentScvxConfig config_{};
    PoweredDescentScvxLayout layout_;
    core::FixedStructure structure_{};
    ValueIndex quadratic_index_{};
    ValueIndex scalar_index_{};
    ValueIndex affine_index_{};
    std::vector<double> quadratic_values_{};

    [[nodiscard]] std::tuple<core::CscPattern, core::CscPattern, core::CscPattern>
    build_patterns() const {
        std::set<std::pair<std::size_t, std::size_t>> quadratic_entries{};
        std::set<std::pair<std::size_t, std::size_t>> scalar_entries{};
        std::set<std::pair<std::size_t, std::size_t>> affine_entries{};
        for (std::size_t variable = 0; variable < layout_.variables(); ++variable) {
            quadratic_entries.insert({variable, variable});
        }
        for (std::size_t component = 0; component < 7U; ++component) {
            scalar_entries.insert({layout_.initial_rows().start + component, component});
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
                    scalar_entries.insert({row, column});
                }
                scalar_entries.insert({row, next_state.start + row_offset});
                for (std::size_t column = control.start; column < control.stop(); ++column) {
                    scalar_entries.insert({row, column});
                }
                scalar_entries.insert({row, virtual_control.start + row_offset});
            }
        }
        const auto terminal_state = layout_.state(layout_.intervals);
        for (std::size_t component = 0; component < 6U; ++component) {
            scalar_entries.insert(
                {layout_.terminal_rows().start + component, terminal_state.start + component}
            );
        }
        for (std::size_t flat = 0; flat < layout_.virtual_count(); ++flat) {
            const auto positive = layout_.virtual_epigraph_rows().start + 2U * flat;
            const auto negative = positive + 1U;
            const auto virtual_variable = layout_.virtual_offset() + flat;
            const auto epigraph_variable = layout_.virtual_epigraph_offset() + flat;
            scalar_entries.insert({positive, virtual_variable});
            scalar_entries.insert({positive, epigraph_variable});
            scalar_entries.insert({negative, virtual_variable});
            scalar_entries.insert({negative, epigraph_variable});
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            const auto tilt_row = layout_.tilt_rows().start + interval;
            scalar_entries.insert({tilt_row, control.start + 2U});
            scalar_entries.insert({tilt_row, control.start + 3U});
            const auto thrust_start = layout_.thrust_cone_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 4U; ++component) {
                affine_entries.insert({thrust_start + component, control.start + component});
            }
        }
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            const auto glide_start = layout_.glide_cone_rows().start + 3U * node;
            for (std::size_t component = 0; component < 3U; ++component) {
                affine_entries.insert({glide_start + component, state.start + component});
            }
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto state = layout_.state(interval);
            const auto control = layout_.control(interval);
            const auto trust_start = layout_.stage_trust_cone_rows().start + 12U * interval;
            for (std::size_t component = 0; component < 7U; ++component) {
                affine_entries.insert({trust_start + component, state.start + component});
            }
            for (std::size_t component = 0; component < 4U; ++component) {
                affine_entries.insert(
                    {trust_start + 7U + component, control.start + component}
                );
            }
        }
        const auto terminal_trust = layout_.terminal_trust_cone_rows().start;
        for (std::size_t component = 0; component < 7U; ++component) {
            affine_entries.insert({terminal_trust + component, terminal_state.start + component});
        }
        return {
            make_pattern(layout_.variables(), layout_.variables(), quadratic_entries),
            make_pattern(layout_.scalar_rows(), layout_.variables(), scalar_entries),
            make_pattern(layout_.affine_rows(), layout_.variables(), affine_entries),
        };
    }

    [[nodiscard]] std::vector<ConeBlockDescriptor> cone_blocks() const {
        std::vector<ConeBlockDescriptor> blocks{};
        blocks.reserve(2U * layout_.intervals + layout_.intervals + 2U);
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            blocks.push_back(
                ConeBlockDescriptor{
                    ConeKind::second_order,
                    static_cast<Index>(4U * interval),
                    2,
                    0.0,
                }
            );
        }
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            blocks.push_back(
                ConeBlockDescriptor{
                    ConeKind::second_order,
                    static_cast<Index>(layout_.glide_cone_rows().start + 3U * node),
                    1,
                    0.0,
                }
            );
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            blocks.push_back(
                ConeBlockDescriptor{
                    ConeKind::second_order,
                    static_cast<Index>(
                        layout_.stage_trust_cone_rows().start + 12U * interval
                    ),
                    10,
                    0.0,
                }
            );
        }
        blocks.push_back(
            ConeBlockDescriptor{
                ConeKind::second_order,
                static_cast<Index>(layout_.terminal_trust_cone_rows().start),
                6,
                0.0,
            }
        );
        return blocks;
    }

    [[nodiscard]] std::vector<double> make_quadratic_values() const {
        std::vector<double> values(structure_.quadratic.nonzeros(), 0.0);
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto range = layout_.state(node);
            for (std::size_t component = 0; component < 7U; ++component) {
                quadratic_index_.set(
                    values,
                    range.start + component,
                    range.start + component,
                    config_.state_tracking_weights[component]
                );
            }
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            for (std::size_t component = 0; component < 4U; ++component) {
                quadratic_index_.set(
                    values,
                    control.start + component,
                    control.start + component,
                    config_.control_tracking_weights[component]
                );
            }
            const auto virtual_control = layout_.virtual_control(interval);
            const auto epigraph = layout_.virtual_epigraph(interval);
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
        const std::vector<PoweredDescentState>& states,
        const std::vector<PoweredDescentControl>& controls
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
        for (std::size_t variable = layout_.virtual_epigraph_offset();
             variable < layout_.variables();
             ++variable) {
            values.linear_objective[variable] = config_.virtual_l1_weight;
        }
    }

    void fill_initial(core::NumericValues& values, const PoweredDescentState& initial) const {
        for (std::size_t component = 0; component < 7U; ++component) {
            const auto row = layout_.initial_rows().start + component;
            scalar_index_.set(values.scalar_constraint, row, component, 1.0);
            values.scalar_lower[row] = initial[component];
            values.scalar_upper[row] = initial[component];
        }
    }

    void fill_dynamics(
        core::NumericValues& values,
        const std::vector<PoweredDescentState>& states,
        const std::vector<PoweredDescentControl>& controls
    ) const {
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto linearisation = linearise_discrete_flow<7U, 4U>(
                model_,
                states[interval],
                controls[interval],
                config_.step_seconds,
                config_.discretisation,
                config_.finite_difference_relative_step
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

    void fill_terminal(
        core::NumericValues& values,
        const std::array<double, 3U>& position,
        const std::array<double, 3U>& velocity
    ) const {
        const auto terminal_state = layout_.state(layout_.intervals);
        for (std::size_t component = 0; component < 6U; ++component) {
            const auto row = layout_.terminal_rows().start + component;
            scalar_index_.set(
                values.scalar_constraint,
                row,
                terminal_state.start + component,
                1.0
            );
            const auto target = component < 3U ? position[component] : velocity[component - 3U];
            values.scalar_lower[row] = target;
            values.scalar_upper[row] = target;
        }
    }

    void fill_virtual_epigraph(core::NumericValues& values) const {
        for (std::size_t flat = 0; flat < layout_.virtual_count(); ++flat) {
            const auto positive = layout_.virtual_epigraph_rows().start + 2U * flat;
            const auto negative = positive + 1U;
            const auto virtual_variable = layout_.virtual_offset() + flat;
            const auto epigraph_variable = layout_.virtual_epigraph_offset() + flat;
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
                control.start + 3U,
                model_.config().tilt_cosine()
            );
            values.scalar_upper[row] = 0.0;
        }
    }

    void fill_affine_cones(
        core::NumericValues& values,
        const std::vector<PoweredDescentState>& states,
        const std::vector<PoweredDescentControl>& controls,
        double radius
    ) const {
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            const auto thrust_start = layout_.thrust_cone_rows().start + 4U * interval;
            for (std::size_t component = 0; component < 4U; ++component) {
                affine_index_.set(
                    values.affine_cone,
                    thrust_start + component,
                    control.start + component,
                    1.0
                );
            }
        }
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            const auto glide_start = layout_.glide_cone_rows().start + 3U * node;
            affine_index_.set(values.affine_cone, glide_start, state.start, 1.0);
            affine_index_.set(values.affine_cone, glide_start + 1U, state.start + 1U, 1.0);
            affine_index_.set(
                values.affine_cone,
                glide_start + 2U,
                state.start + 2U,
                model_.config().glide_slope_tangent()
            );
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto state = layout_.state(interval);
            const auto control = layout_.control(interval);
            const auto trust_start = layout_.stage_trust_cone_rows().start + 12U * interval;
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
        const auto terminal_state = layout_.state(layout_.intervals);
        const auto terminal_trust = layout_.terminal_trust_cone_rows().start;
        for (std::size_t component = 0; component < 7U; ++component) {
            const auto scale = config_.state_trust_scales[component];
            affine_index_.set(
                values.affine_cone,
                terminal_trust + component,
                terminal_state.start + component,
                scale
            );
            values.affine_offset[terminal_trust + component] =
                -scale * states.back()[component];
        }
        values.affine_offset[terminal_trust + 7U] = radius;
    }

    void fill_variable_bounds(core::NumericValues& values) const {
        for (std::size_t node = 0; node <= layout_.intervals; ++node) {
            const auto state = layout_.state(node);
            values.variable_lower[state.start + 2U] = 0.0;
            values.variable_lower[state.start + 6U] = model_.config().minimum_mass;
        }
        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {
            const auto control = layout_.control(interval);
            values.variable_lower[control.start + 3U] = model_.config().minimum_sigma;
            values.variable_upper[control.start + 3U] = model_.config().maximum_thrust;
        }
        for (std::size_t variable = layout_.virtual_epigraph_offset();
             variable < layout_.variables();
             ++variable) {
            values.variable_lower[variable] = 0.0;
        }
    }

    void validate_reference(
        const std::vector<PoweredDescentState>& states,
        const std::vector<PoweredDescentControl>& controls
    ) const {
        if (states.size() != layout_.intervals + 1U || controls.size() != layout_.intervals) {
            throw std::invalid_argument("reference trajectory has the wrong horizon");
        }
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            static_cast<void>(model_.dynamics(states[interval], controls[interval]));
        }
        const PoweredDescentControl zero_control{};
        static_cast<void>(model_.dynamics(states.back(), zero_control));
    }

    static void require_finite(double value, const char* message) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(message);
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
            if (row >= rows || column >= columns) {
                throw std::logic_error("sparse-pattern entry is outside its matrix");
            }
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
        if (values.size() != pattern.nonzeros()
            || vector.size() != static_cast<std::size_t>(pattern.columns)) {
            throw std::invalid_argument("sparse matrix-vector product has incompatible sizes");
        }
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

    static std::size_t cone_slot_count(const ConeBlockDescriptor& cone) {
        switch (cone.kind) {
            case ConeKind::second_order:
            case ConeKind::rotated_second_order:
                return static_cast<std::size_t>(cone.vector_dimension) + 2U;
            case ConeKind::exponential:
            case ConeKind::power:
                return 3U;
            case ConeKind::positive_semidefinite: {
                const auto order = static_cast<std::size_t>(cone.vector_dimension);
                return order * (order + 1U) / 2U;
            }
        }
        throw std::logic_error("unknown cone kind");
    }
};

}  // namespace spacepdhcg::transcription
